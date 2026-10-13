"""Experiments. Steps 1, 2 and 5 follow the paper (section 5.2); steps 3 and 4 are our additions.

1. learning-rate search with tau=-1, k=4, lambda=1e-6, 30 epochs
2. grid search over tau, k, lambda at that learning rate, 30 epochs each
3. early stopping: the dev set also picks the number of epochs (the paper always trains 60). The
   setting with the best dev score at any epoch of the grid is retrained for up to 60 epochs, the
   same budget as the paper's final model, and the best epoch is kept.
4. masked training on top of early stopping, with lambda re-tuned
5. final models compared with baselines on the test set

Dev precision@3 (APTK) is recorded after every epoch, so each grid run gives both the paper's
score (after the last epoch) and the best score at any epoch. Only the dev set is used for any choice. Writes results/search.json, results/metrics.json, plots
in results/, and the model used by the demo to models/model.npz.

Usage: python -m src.train           full run (about 30 minutes on a laptop)
       python -m src.train --quick   reuse results/search.json and only do step 5 (about 2 minutes)
"""
import argparse
import itertools
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .build_matrix import MATRIX, RESULTS
from .evaluate import (knn_scores, mask, mean_idf_at_k, mse, mser, popularity_scores, precision_at_k,
                       random_scores, svd_scores)
from .model import fit, lift, neighbour_weights, predict

MODEL = MATRIX.parent.parent / "models" / "model.npz"
ALPHAS = [10, 1, 0.1, 0.01, 0.001]
TAUS = [-10, -1, -0.1, -0.001]
KS = [1, 4, 16, 1000]
LAMBDAS = [1e-4, 1e-2, 1e-1, 1]
SVD_RANKS = [20, 50, 100, 200]
SEARCH_EPOCHS, FINAL_EPOCHS, MASKED_EPOCHS, K, MASK_FRAC = 30, 60, 60, 3, 0.5

PAPER = "CF ReLU + L2 (paper protocol)"
EARLY = "CF ReLU + L2 + early stopping (ours)"
MASKED = "CF ReLU + L2 + early stopping + masked training (ours)"


def w_area(tau, k):
    """Area under exp(tau * (1 - x)^k) on [0, 1]. The paper uses it to summarise W: 1 means W is flat."""
    x = np.linspace(0, 1, 2001)
    return float(np.trapezoid(np.exp(tau * (1 - x) ** k), x))


def run(R, X, Xm, tau, k, lam, alpha, epochs, mask_frac=0.0):
    """Train one configuration and record dev APTK after every epoch."""
    t = time.time()
    W = neighbour_weights(R, tau, k)
    curve = []
    _, _, losses = fit(R, W, alpha, lam, epochs, mask_frac=mask_frac,
                       on_epoch=lambda theta, b: curve.append(precision_at_k(X, Xm, predict(theta, b, W, Xm), K)))
    finished = len(curve) == epochs
    best = int(np.argmax(curve)) if curve else 0
    res = {"tau": tau, "k": k, "lambda": lam, "alpha": alpha, "mask_frac": mask_frac, "area": w_area(tau, k),
           "losses": losses, "dev_curve": curve,
           "dev_aptk": curve[-1] if finished else None,  # paper protocol: score after the last epoch
           "best_epoch": best + 1 if finished else None,
           "best_dev_aptk": curve[best] if finished else None}
    print(f"  tau={tau:<7} k={k:<5} lambda={lam:<7} alpha={alpha:<6} mask={mask_frac}: " +
          (f"dev APTK {res['dev_aptk']:.4f} at the end, best {res['best_dev_aptk']:.4f} at epoch {res['best_epoch']}"
           if finished else "diverged") + f"  ({time.time() - t:.0f}s)", flush=True)
    return res


def search(R, Xd, Xdm):
    print("1. Learning-rate search")
    alpha_runs = [run(R, Xd, Xdm, -1.0, 4, 1e-6, a, SEARCH_EPOCHS) for a in ALPHAS]
    alpha = max((r for r in alpha_runs if r["dev_aptk"] is not None), key=lambda r: r["dev_aptk"])["alpha"]

    print(f"2. Grid search over tau, k, lambda (alpha={alpha})")
    grid = [run(R, Xd, Xdm, tau, k, lam, alpha, SEARCH_EPOCHS) for tau, k, lam in itertools.product(TAUS, KS, LAMBDAS)]
    done = [g for g in grid if g["dev_aptk"] is not None]
    paper = max(done, key=lambda g: g["dev_aptk"])
    early = max(done, key=lambda g: g["best_dev_aptk"])

    print(f"3. Early stopping: best grid setting retrained for up to {FINAL_EPOCHS} epochs")
    early = run(R, Xd, Xdm, early["tau"], early["k"], early["lambda"], alpha, FINAL_EPOCHS)

    print("4. Masked training with early stopping, lambda re-tuned")
    masked = [run(R, Xd, Xdm, early["tau"], early["k"], lam, alpha, MASKED_EPOCHS, MASK_FRAC) for lam in LAMBDAS]
    best_masked = max((g for g in masked if g["dev_aptk"] is not None), key=lambda g: g["best_dev_aptk"])

    svd = {r: precision_at_k(Xd, Xdm, svd_scores(R, Xdm, r), K) for r in SVD_RANKS}
    return {"alpha_runs": alpha_runs, "grid": grid, "early_run": early, "masked": masked,
            "chosen": {PAPER: {**paper, "epochs": FINAL_EPOCHS},
                       EARLY: {**early, "epochs": early["best_epoch"]},
                       MASKED: {**best_masked, "epochs": best_masked["best_epoch"]}},
            "svd_dev_aptk": svd, "svd_rank": max(svd, key=svd.get)}


def plot_search(s):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.3))
    for r in s["alpha_runs"]:
        if r["dev_aptk"] is None:
            continue
        a.plot(range(1, len(r["losses"]) + 1), r["losses"], label=f"alpha={r['alpha']}")
        b.plot(range(1, len(r["dev_curve"]) + 1), r["dev_curve"], label=f"alpha={r['alpha']}")
    a.set(xlabel="epoch", ylabel="training loss J", title="(A) Training loss")
    b.set(xlabel="epoch", ylabel="dev APTK (K=3)", title="(B) Dev precision of top-3 keywords")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "alpha_search.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for i, lam in enumerate(LAMBDAS):
        g = [x for x in s["grid"] if x["lambda"] == lam and x["dev_aptk"] is not None]
        ax.scatter([x["area"] for x in g], [x["dev_aptk"] for x in g], color=f"C{i}", s=18, label=f"lambda={lam}")
        ax.scatter([x["area"] for x in g], [x["best_dev_aptk"] for x in g], color=f"C{i}", s=18, facecolors="none")
    ax.set(xlabel="area under the W curve (1 = W is flat)", ylabel="dev APTK (K=3)",
           title="Grid search (filled: epoch 30, hollow: best epoch)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "grid_search.png", dpi=150)


def evaluate_all(R, X, Xm, s, idf, demo):
    """Train the final CF models and score every method on the masked test set.

    The demo model is also scored when ranked by lift (without the bias b), which is how the demo ranks.
    """
    scores, metrics, models = {}, {}, {}
    p = s["chosen"][PAPER]
    variants = {"CF naive, linear (paper 4.2)": {**p, "lambda": 0.0, "relu": False}}
    variants.update({name: {**c, "relu": True} for name, c in s["chosen"].items()})
    for name, c in variants.items():
        t = time.time()
        W = neighbour_weights(R, c["tau"], c["k"])
        theta, b, losses = fit(R, W, c["alpha"], c["lambda"], c["epochs"], relu=c["relu"], mask_frac=c["mask_frac"])
        P = predict(theta, b, W, Xm, c["relu"])
        scores[name], models[name] = P, (theta, b, W)
        metrics[name] = {"mse": mse(X, P), "mser": mser(X, P), "final_train_loss": losses[-1],
                         **{key: c[key] for key in ("tau", "k", "lambda", "alpha", "mask_frac", "epochs", "relu")}}
        print(f"  trained {name} ({c['epochs']} epochs) in {time.time() - t:.0f}s", flush=True)
    theta, b, W = models[demo]
    scores[f"{demo}, ranked by lift"] = lift(theta, W, Xm)
    scores["Random (paper's APRK)"] = random_scores(Xm)
    scores["Popularity"] = popularity_scores(R)
    scores["Item kNN, cosine"] = knn_scores(R, Xm)
    scores[f"SVD, rank {s['svd_rank']}"] = svd_scores(R, Xm, int(s["svd_rank"]))
    for name, S in scores.items():
        m = metrics.setdefault(name, {})
        m["precision_at_k"] = {k: precision_at_k(X, Xm, S, k) for k in range(1, 11)}
        m["mean_idf_at_3"] = mean_idf_at_k(Xm, S, idf, K)
    return metrics, models


def plot_precision(metrics):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for name, m in metrics.items():
        p = m["precision_at_k"]
        style = ":" if "naive" in name else "-" if name.startswith("CF") else "--"  # naive overlaps paper protocol
        ax.plot([int(k) for k in p], list(p.values()), style, marker="o", ms=3, label=name, zorder=3 if "naive" in name else 2)
    ax.set(xlabel="K (number of recommended keywords)", ylabel="test precision@K", title="Recovering hidden keywords")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(RESULTS / "precision_at_k.png", dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the searches and reuse results/search.json")
    args = ap.parse_args()

    d = np.load(MATRIX)
    U, split = d["U"], d["split"]
    R, Xd, Xt = U[:, split == 0], U[:, split == 1], U[:, split == 2]
    Xdm, Xtm = mask(Xd, MASK_FRAC, seed=1), mask(Xt, MASK_FRAC, seed=2)
    print(f"train {R.shape[1]}, dev {Xd.shape[1]}, test {Xt.shape[1]} documents, {R.shape[0]} keywords")

    search_file = RESULTS / "search.json"
    if args.quick:
        s = json.loads(search_file.read_text())
    else:
        s = search(R, Xd, Xdm)
        search_file.write_text(json.dumps(s, indent=1))
        plot_search(s)
    for name, c in s["chosen"].items():
        print(f"Chosen for {name}: tau={c['tau']} k={c['k']} lambda={c['lambda']} alpha={c['alpha']} "
              f"epochs={c['epochs']} (dev APTK {c['best_dev_aptk'] if name != PAPER else c['dev_aptk']:.4f})")

    # The demo uses the CF variant with the best dev score.
    dev = {PAPER: s["chosen"][PAPER]["dev_aptk"], EARLY: s["chosen"][EARLY]["best_dev_aptk"],
           MASKED: s["chosen"][MASKED]["best_dev_aptk"]}
    demo = max(dev, key=dev.get)

    print("5. Final training and test-set evaluation")
    metrics, models = evaluate_all(R, Xt, Xtm, s, d["idf"], demo)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=1))
    plot_precision(metrics)

    theta, b, W = models[demo]
    MODEL.parent.mkdir(exist_ok=True)
    np.savez(MODEL, theta=theta, b=b, W=W, name=demo)
    print(f"Saved '{demo}' to {MODEL}")

    print(f"\n{'method':70s} {'P@1':>6s} {'P@3':>6s} {'P@5':>6s} {'P@10':>6s} {'IDF@3':>6s} {'MSE':>8s} {'MSER':>8s}")
    for n, m in metrics.items():
        p = m["precision_at_k"]
        err = f"{m['mse']:8.5f} {m['mser']:8.5f}" if "mse" in m else ""
        print(f"{n:70s} {p[1]:6.3f} {p[3]:6.3f} {p[5]:6.3f} {p[10]:6.3f} {m['mean_idf_at_3']:6.3f} {err}")


if __name__ == "__main__":
    main()
