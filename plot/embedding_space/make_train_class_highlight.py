#!/usr/bin/env python3
"""Train-only ESM embedding map with a SINGLE first-cyclization class highlighted
in gold and no generated sequences — a clean per-class reference.

Uses the exact same train-only PCA + t-SNE layout as the all-class ESM map
(make_pca_tsne_fig.py / slide #315): same z-scoring, same SVD axes, same
t-SNE(init='pca', perplexity=30, random_state=0). Only the colouring changes —
target class gold, everything else grey. One PNG per requested class.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn")
from knn_first_cyclization import load_reference  # noqa: E402

OUTDIR = Path("/Users/soldatmat/Documents/terpene_synthases/dplm/run/"
              "class_predictor/slide306_eval")
GOLD = "#C9A227"
GRAY = "0.8"
TARGETS = [0, 1]


def compute_layout():
    """Train-only PCA + t-SNE — identical to make_pca_tsne_fig.py."""
    X, y, _ = load_reference()
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    Xc = Xz - Xz.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    vr = S ** 2 / (S ** 2).sum()
    pc1, pc2 = 100 * vr[0], 100 * vr[1]
    n_pca = min(50, Xc.shape[1])
    X50 = U[:, :n_pca] * S[:n_pca]
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=0).fit_transform(X50)
    return y, pcs, emb, pc1, pc2


def make_fig(y, pcs, emb, pc1, pc2, target, out):
    mask = y == target
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)
    for ax, coords, ttl in (
        (axes[0], pcs, f"PCA  (PC1 {pc1:.1f}%, PC2 {pc2:.1f}% var)"),
        (axes[1], emb, "t-SNE  (perplexity 30, PCA-50 init)"),
    ):
        ax.scatter(coords[~mask, 0], coords[~mask, 1], c=GRAY, s=8, alpha=0.4,
                   linewidths=0, zorder=1)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=GOLD, s=18, alpha=0.9,
                   linewidths=0, zorder=3)
        ax.set_title(ttl, fontsize=11)
        ax.tick_params(labelsize=8)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GOLD,
               markeredgecolor="none", markersize=8,
               label=f"class {target} (target)   n={int(mask.sum())}"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GRAY,
               markeredgecolor="none", markersize=7,
               label=f"other training classes   n={int((~mask).sum())}"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=9, frameon=True,
               framealpha=0.9)
    fig.suptitle(
        f"Training TPSs (n={len(y)}) in DPLM-150m embedding space "
        f"— first-cyclization class {target} highlighted (no generated seqs)",
        fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005,
             "train-only PCA + t-SNE — same layout/axes as the all-class ESM map",
             fontsize=9, style="italic", ha="left", va="bottom")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (class {target}: n={int(mask.sum())})")


def main():
    y, pcs, emb, pc1, pc2 = compute_layout()
    for t in TARGETS:
        make_fig(y, pcs, emb, pc1, pc2, t, OUTDIR / f"train_highlight_class{t}.png")


if __name__ == "__main__":
    main()
