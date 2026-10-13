"""Build the TF-IDF utility matrix U (MeSH keywords x trial documents) and split it 8:1:1.

    U_ij = log10(1 + count of keyword i in document j) * log10(N / number of documents with keyword i)

Only keywords found in at least 0.5% of documents are kept (as in the paper), and documents
need at least two of them so that half can be hidden during evaluation.

Usage: python -m src.build_matrix
"""
import gzip
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fetch_data import DATA, TRIALS
from .mesh import Extractor, load_terms

MATRIX = DATA / "matrix.npz"
RESULTS = DATA.parent / "results"
TEXT_FIELDS = ["brief_title", "official_title", "brief_summary", "detailed_description", "eligibility"]
MIN_DF = 0.005
SEED = 42


def doc_keywords(ext, rec):
    found = Counter()
    for field in TEXT_FIELDS:
        found += ext.extract(rec[field])
    for condition in rec["conditions"]:
        found += ext.extract(condition)
    return found


def tfidf(counts, vocab, n_docs, df):
    """Dense U for the given vocabulary. counts is one Counter of keyword counts per document."""
    index = {ui: i for i, ui in enumerate(vocab)}
    U = np.zeros((len(vocab), len(counts)), np.float32)
    for j, c in enumerate(counts):
        for ui, cnt in c.items():
            if ui in index:
                U[index[ui], j] = np.log10(1 + cnt)
    idf = np.log10(n_docs / np.array([df[ui] for ui in vocab], np.float32))
    return U * idf[:, None], idf


def plot_distribution(U, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    a.hist(U.ravel(), bins=100, log=True)
    a.set(title="(A) TF-IDF values", xlabel="TF-IDF", ylabel="count (log scale)")
    b.hist((U > 0).sum(axis=0), bins=range(0, 101))
    b.set(title="(B) Keywords per document", xlabel="non-zero TF-IDF entries", ylabel="documents")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ext = Extractor(load_terms())
    ids, titles, counts = [], [], []
    with gzip.open(TRIALS, "rt") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["nct_id"])
            titles.append(rec["brief_title"])
            counts.append(doc_keywords(ext, rec))
            if len(counts) % 10000 == 0:
                print(f"  extracted keywords from {len(counts)} documents", flush=True)

    n_docs = len(counts)
    df = Counter(ui for c in counts for ui in c)
    vocab = sorted(ui for ui, d in df.items() if d >= MIN_DF * n_docs)
    U, idf = tfidf(counts, vocab, n_docs, df)

    keep = (U > 0).sum(axis=0) >= 2
    U, ids, titles = U[:, keep], np.array(ids)[keep], np.array(titles)[keep]
    split = np.random.default_rng(SEED).permutation(np.repeat([0, 1, 2], [8, 1, 1])[np.arange(U.shape[1]) % 10])

    np.savez_compressed(MATRIX, U=U, vocab=np.array(vocab), names=np.array([ext.names[ui] for ui in vocab]),
                        idf=idf, ids=ids, titles=titles, split=split)

    nnz = (U > 0).sum(axis=0)
    stats = {
        "documents_downloaded": n_docs,
        "distinct_mesh_descriptors_found": len(df),
        "keywords_kept_(df>=0.5%)": len(vocab),
        "documents_kept_(>=2_keywords)": int(keep.sum()),
        "train_dev_test": [int((split == s).sum()) for s in (0, 1, 2)],
        "keywords_per_document_median": float(np.median(nnz)),
        "keywords_per_document_mean": round(float(nnz.mean()), 2),
        "matrix_density": round(float((U > 0).mean()), 4),
        "most_common_keywords": [[ext.names[ui], d] for ui, d in df.most_common(15)],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data_stats.json").write_text(json.dumps(stats, indent=2))
    plot_distribution(U[:, split == 0], RESULTS / "data_distribution.png")
    train, dev, test = stats["train_dev_test"]
    print(f"{n_docs} trials, {len(df)} distinct MeSH keywords found, {len(vocab)} kept (in at least {MIN_DF:.1%} of trials)")
    print(f"{int(keep.sum())} trials with at least 2 keywords, median {np.median(nnz):.0f} keywords per trial")
    print(f"Split: {train} train, {dev} dev, {test} test. Details in {RESULTS / 'data_stats.json'}")


if __name__ == "__main__":
    main()
