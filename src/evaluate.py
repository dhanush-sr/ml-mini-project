"""Masked evaluation (paper section 4.4) and baseline recommenders.

For every test document, half of its keywords are hidden (set to 0). A method then scores all
keywords from the masked document, and we check how well it recovers the hidden ones:

* MSE  - mean squared error over every cell of the matrix.
* MSER - squared error divided by the number of cells that are non-zero in the truth or the
         prediction. The matrix is so sparse that plain MSE is always tiny.
* precision@K - of the K highest-scoring keywords that were not in the masked input, the share
         that really were in the document. Averaged over documents this is the paper's APTK.
         Picking K keywords at random instead gives APRK.
* mean IDF@K - how specific the recommended keywords are (our addition, see mean_idf_at_k).
"""
import numpy as np

from .model import cosine_similarity


def mask(X, frac=0.5, seed=0):
    """Copy of X with `frac` of each column's non-zero entries set to 0 (at least one is hidden)."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    for j in range(X.shape[1]):
        nz = np.flatnonzero(X[:, j])
        Xm[rng.choice(nz, size=max(1, int(len(nz) * frac)), replace=False), j] = 0
    return Xm


def mse(X, Xhat):
    return float(((Xhat - X) ** 2).mean())


def mser(X, Xhat):
    return float(((Xhat - X) ** 2).sum() / ((X > 0) | (Xhat > 0)).sum())


def top_k(Xm, scores, k):
    """Row indices of the k best new keywords per document, and whether each is really recommended.

    Candidates are keywords absent from the masked input Xm. A keyword only counts as
    recommended if its score is positive, matching the paper's definition.
    """
    s = np.where(Xm > 0, -np.inf, np.broadcast_to(scores, Xm.shape))
    top = np.argpartition(-s, k, axis=0)[:k]
    return top, np.take_along_axis(s, top, axis=0) > 0


def precision_at_k(X, Xm, scores, k):
    """Average precision of the top-k new keywords per document."""
    top, recommended = top_k(Xm, scores, k)
    return float(((np.take_along_axis(X, top, axis=0) > 0) & recommended).mean())


def mean_idf_at_k(Xm, scores, idf, k):
    """Average IDF of the top-k recommended keywords. Higher means more specific keywords;
    generic ones like "Patients" have low IDF. Precision alone rewards generic keywords."""
    top, recommended = top_k(Xm, scores, k)
    return float(idf[top][recommended].mean())


# Baselines. Each returns a score for every (keyword, document) cell given the masked input.

def random_scores(Xm, seed=0):
    """Uniform random choice of keywords (the paper's APRK)."""
    return np.random.default_rng(seed).random(Xm.shape, dtype=np.float32) + 1e-6


def popularity_scores(R):
    """Recommend the keywords that appear in the most training documents, whatever the input.
    Returns one column that broadcasts to every document."""
    return (R > 0).mean(axis=1, keepdims=True)


def knn_scores(R, Xm):
    """Item-based collaborative filtering: score_i = sum_f cos(i, f) * x_f."""
    S = cosine_similarity(R)
    np.fill_diagonal(S, 0)
    return S @ Xm


def svd_scores(R, Xm, rank):
    """Low-rank matrix factorisation: project the masked document onto the top `rank`
    left singular vectors of the training matrix and reconstruct it."""
    _, vecs = np.linalg.eigh(R @ R.T)  # eigenvectors of R R^T are R's left singular vectors
    V = vecs[:, -rank:]
    return V @ (V.T @ Xm)
