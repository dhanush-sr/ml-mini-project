"""Self-checks that need no downloaded data. Run: python -m pytest -q   (or: python tests/test_core.py)"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_matrix import tfidf
from src.evaluate import mask, mser, precision_at_k
from src.mesh import Extractor
from src.model import fit, gradients, loss, neighbour_weights


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    R = np.maximum(rng.normal(size=(6, 9)), 0)  # small non-negative "TF-IDF" matrix, float64
    W = neighbour_weights(R, tau=-1.0, k=2)
    theta, b = rng.normal(size=(6, 6)) * 0.3, rng.normal(size=(6, 1)) * 0.3
    R_in = R * (rng.random(R.shape) > 0.5)  # masked input, as in masked training
    for relu in (True, False):
        for X_in in (None, R_in):
            g_theta, g_b = gradients(theta, b, W, R, 0.1, relu, X_in)
            J = lambda t, bb: loss(t, bb, W, R, 0.1, relu, X_in)
            eps = 1e-6
            for (i, j) in [(0, 1), (2, 5), (4, 3)]:
                t = theta.copy(); t[i, j] += eps
                up = J(t, b)
                t[i, j] -= 2 * eps
                numeric = (up - J(t, b)) / (2 * eps)
                assert abs(numeric - g_theta[i, j]) < 1e-5, (relu, i, j, numeric, g_theta[i, j])
            bb = b.copy(); bb[2] += eps
            up = J(theta, bb)
            bb[2] -= 2 * eps
            assert abs((up - J(theta, bb)) / (2 * eps) - g_b[2, 0]) < 1e-5


def test_training_reduces_loss_from_zero_start():
    rng = np.random.default_rng(1)
    R = np.maximum(rng.normal(size=(8, 200)), 0).astype(np.float32)
    W = neighbour_weights(R, tau=-1.0, k=4)
    assert np.all(np.diag(W) == 0)
    _, _, losses = fit(R, W, alpha=0.1, lam=0.0, epochs=5, batch_size=32)
    assert losses[-1] < losses[0]


def test_tfidf_formula():
    counts = [Counter({"A": 3}), Counter({"A": 1, "B": 2}), Counter({"B": 1})]
    df = Counter(ui for c in counts for ui in c)
    U, idf = tfidf(counts, ["A", "B"], n_docs=4, df=df)
    assert np.isclose(U[0, 0], np.log10(4) * np.log10(4 / 2))
    assert np.isclose(U[1, 1], np.log10(3) * np.log10(4 / 2))
    assert U[1, 0] == 0


def test_extractor_longest_match_inversion_and_case():
    ext = Extractor({
        "D1": ["Neoplasms, Breast", "Breast Cancer"],
        "D2": ["Neoplasms"],
        "D3": ["World Health Organization", "WHO"],
        "D4": ["Lead"],
    })
    found = ext.extract("Breast neoplasms and breast cancer differ from other neoplasms in patients who follow WHO advice. This may lead to")
    # Longest match wins, "who" is not WHO, and the stop word "lead" is ignored.
    assert found == Counter({"D1": 2, "D2": 1, "D3": 1}), found


def test_mask_and_metrics():
    X = np.array([[1.0, 0], [2.0, 1.0], [0, 3.0], [4.0, 0]])
    Xm = mask(X, 0.5, seed=0)
    assert ((X > 0).sum(0) - (Xm > 0).sum(0)).tolist() == [1, 1]  # 3 -> hide 1, 2 -> hide 1
    assert np.all(Xm <= X)
    assert np.isclose(mser(X, X), 0)
    # Perfect scores: the hidden keyword is ranked first in every document.
    assert precision_at_k(X, Xm, X - Xm, k=1) == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
