"""Recommend extra MeSH keywords for a free-text query and find similar trials.

Usage: python -m src.recommend "pembrolizumab for metastatic melanoma after surgery"
"""
import argparse

import numpy as np

from .build_matrix import MATRIX
from .mesh import Extractor, load_terms
from .model import lift, predict
from .train import MODEL


class Recommender:
    def __init__(self):
        d, m = np.load(MATRIX), np.load(MODEL)
        self.vocab, self.names, self.idf = list(d["vocab"]), d["names"], d["idf"]
        self.index = {ui: i for i, ui in enumerate(self.vocab)}
        self.ids, self.titles = d["ids"], d["titles"]
        self.docs = d["U"]
        self.docs /= np.linalg.norm(self.docs, axis=0)  # unit columns, so a dot product is cosine similarity
        self.theta, self.b, self.W, self.model_name = m["theta"], m["b"], m["W"], str(m["name"])
        self.ext = Extractor(load_terms())

    def recommend(self, text, k=10, generic=False):
        """Returns (keyword counts found in text, query vector, expanded vector, [(keyword, score)]).

        By default keywords are ranked by lift, so the answer is specific to the query. With
        generic=True they are ranked by the full prediction, which favours common keywords.
        """
        found = self.ext.extract(text)
        x = np.zeros(len(self.vocab), np.float32)
        for ui, count in found.items():
            if ui in self.index:
                i = self.index[ui]
                x[i] = np.log10(1 + count) * self.idf[i]  # same TF-IDF as the training matrix
        if not x.any():  # nothing to recommend from; generic mode would just return popular keywords
            return found, x, x, []
        X = x[:, None]
        scores = (predict(self.theta, self.b, self.W, X) if generic else lift(self.theta, self.W, X))[:, 0]
        scores[x > 0] = 0  # only recommend keywords the query does not already have
        top = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]
        expanded = x.copy()
        expanded[top] = scores[top]
        return found, x, expanded, [(str(self.names[i]), float(scores[i])) for i in top]

    def similar_trials(self, vec, n=5):
        if not vec.any():
            return []
        sims = (vec / np.linalg.norm(vec)) @ self.docs
        return [(str(self.ids[j]), str(self.titles[j]), float(sims[j])) for j in np.argsort(-sims)[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="free-text description of a study")
    ap.add_argument("-k", type=int, default=10, help="number of keywords to recommend")
    ap.add_argument("--generic", action="store_true", help="rank by full prediction instead of lift")
    args = ap.parse_args()

    rec = Recommender()
    found, x, expanded, keywords = rec.recommend(args.query, args.k, args.generic)
    print(f"Model: {rec.model_name}\n\nMeSH keywords found in the query:")
    for ui in found:
        note = "" if ui in rec.index else "  (too rare, not used by the model)"
        print(f"  - {rec.ext.names[ui]}{note}")
    if not x.any():
        print("  none of them are in the model's vocabulary, so there is nothing to recommend from")
        return
    print("\nRecommended new keywords:")
    for i, (name, score) in enumerate(keywords, 1):
        print(f"  {i:2d}. {name:45s} {score:.3f}")
    for label, vec in (("query keywords only", x), ("query + recommended keywords", expanded)):
        print(f"\nMost similar trials using {label}:")
        for nct, title, sim in rec.similar_trials(vec):
            print(f"  {nct}  {sim:.2f}  {title[:90]}")


if __name__ == "__main__":
    main()
