"""Collaborative filtering model for keyword recommendation (Zhou, CS229 2020, sections 4.2-4.3).

The TF-IDF of keyword i in a document is predicted from the TF-IDF of the other keywords in
the same document:

    R_hat = ReLU((W * Theta) @ R + b)

W (m x m) is fixed and comes from cosine similarity between keyword rows of the training matrix.
Theta (m x m) and b (m x 1) are learned by mini-batch gradient descent on squared error with an
L2 penalty on Theta. With relu=False this is the paper's "naive" linear model.

Our addition is masked training (mask_frac > 0): each batch's input has a random share of its
entries set to 0 while the target stays complete. The model is then trained on the same task it
is tested on, recovering hidden keywords, instead of reconstructing full documents.
"""
import numpy as np


def cosine_similarity(R):
    """Cosine similarity between every pair of keyword rows of R (values in [0, 1] since R >= 0)."""
    norms = np.linalg.norm(R, axis=1, keepdims=True)
    Rn = R / np.where(norms == 0, 1, norms)
    return np.clip(Rn @ Rn.T, 0, 1)


def neighbour_weights(R, tau, k):
    """W_if = exp(tau * (1 - cos(R_i, R_f))^k), with W_ii = 0 so a keyword never predicts itself.

    tau < 0 and k > 0. Similar keywords get weights near 1, unrelated ones near exp(tau).
    """
    W = np.exp(tau * (1 - cosine_similarity(R)) ** k).astype(R.dtype)
    np.fill_diagonal(W, 0)
    return W


def predict(theta, b, W, X, relu=True):
    out = (W * theta) @ X + b
    return np.maximum(out, 0) if relu else out


def lift(theta, W, X):
    """How much the document's own keywords raise each prediction above the baseline b.

    Ranking by this instead of predict() leaves out b, which holds each keyword's average level
    and otherwise pushes generic keywords such as "Patients" to the top for every document.
    """
    return (W * theta) @ X


def loss(theta, b, W, X, lam=0.0, relu=True, X_in=None):
    """J = 1/(2n) * sum((R_hat - X)^2) + lam/2 * ||Theta||^2, where n = number of documents."""
    err = predict(theta, b, W, X if X_in is None else X_in, relu) - X
    return float((err ** 2).sum() / (2 * X.shape[1]) + lam / 2 * (theta ** 2).sum())


def gradients(theta, b, W, B, lam, relu=True, B_in=None):
    """Gradients of `loss` on a batch whose columns are documents.

    The model predicts the targets B from the inputs B_in (B itself unless masked training is on).
    """
    B_in = B if B_in is None else B_in
    Bhat = (W * theta) @ B_in + b
    G = Bhat - B
    if relu:
        # ReLU'(0) is taken as 1, as in the paper. With the usual ReLU'(0) = 0 the gradient
        # would be exactly zero at the starting point Theta = 0, b = 0, and nothing would train.
        G *= Bhat >= 0
    n = B.shape[1]
    return W * (G @ B_in.T) / n + lam * theta, G.sum(axis=1, keepdims=True) / n


def fit(R, W, alpha, lam, epochs, relu=True, mask_frac=0.0, batch_size=256, seed=0, on_epoch=None):
    """Mini-batch gradient descent (paper's Algorithm 1). Returns theta, b and the loss per epoch."""
    m, n = R.shape
    theta = np.zeros((m, m), R.dtype)
    b = np.zeros((m, 1), R.dtype)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            B = R[:, order[start:start + batch_size]]
            B_in = B * (rng.random(B.shape, dtype=B.dtype) >= mask_frac) if mask_frac else B
            g_theta, g_b = gradients(theta, b, W, B, lam, relu, B_in)
            theta -= alpha * g_theta
            b -= alpha * g_b
        losses.append(loss(theta, b, W, R, 0.0, relu))  # reported without the penalty, as in the paper
        if not np.isfinite(losses[-1]):
            break  # learning rate too large, training diverged
        if on_epoch:
            on_epoch(theta, b)
    return theta, b, losses
