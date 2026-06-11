#!/usr/bin/env python3
"""Project all training TPS embeddings to 2D by PCA and t-SNE, coloured by
first-cyclization class. Uses the SAME embeddings the kNN classifier loads.

Two side-by-side panels (left PCA, right t-SNE), shared categorical legend
parked outside the data area. Saves the PNG to the slide-306 eval dir and to
the NAS comparison dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE

# reuse the exact reference loader the kNN classifier uses — sourced from the
# CANONICAL standalone repo (github.com/soldatmat/tps-first-cyclization-knn).
# Its bundled data/ embeddings are md5-identical to the old dplm data-bin copy.
sys.path.insert(0, "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn")
from knn_first_cyclization import load_reference, N_CLASSES  # noqa: E402

OUT_PATHS = [
    Path("/Users/soldatmat/Documents/terpene_synthases/dplm/run/class_predictor/"
         "slide306_eval/train_tps_pca_tsne.png"),
    Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
         "knn_first_cyclization_fidelity_2026-06-09/train_tps_pca_tsne.png"),
]

# substrate type per first-cyclization class id (from the task spec)
SUBSTRATE = {}
for c in (0, 1, 2, 4, 10, 19):
    SUBSTRATE[c] = "sesqui"
for c in (5, 11, 20):
    SUBSTRATE[c] = "mono"
for c in (3, 6, 7, 8, 9, 13, 14, 15, 16):
    SUBSTRATE[c] = "di"
for c in (17, 18, 21):
    SUBSTRATE[c] = "sester"
for c in (12,):
    SUBSTRATE[c] = "sterol"


# substrate-type → hue family. Each type is one sequential colormap so classes
# within a type share a hue; classes across types are in clearly distinct hue
# families. Singleton types get a fixed, neutral colour that won't be confused
# with the multi-class families.
TYPE_ORDER = ["mono", "sesqui", "di", "sester", "sterol"]  # by carbon number of precursor (C10->C30)
TYPE_CMAP = {
    "sesqui": "Blues",
    "mono": "Greens",
    "di": "Reds",
    "sester": "Purples",
}
TYPE_FIXED = {"sterol": (0.40, 0.26, 0.13)}  # brown singleton (triterpene/sterol)


def build_class_colors():
    """Return {class_id: rgba}, coloured by substrate type.

    Within each multi-class family, ids are ordered ascending and assigned
    shades light→dark from a mid-dark slice of the family's sequential cmap
    (the very light end is skipped so points stay visible on white).
    """
    by_type = {t: sorted(c for c, tt in SUBSTRATE.items() if tt == t)
               for t in TYPE_ORDER}
    colors = {}
    for t in TYPE_ORDER:
        ids = by_type[t]
        if t in TYPE_FIXED:
            for cid in ids:
                colors[cid] = (*TYPE_FIXED[t], 1.0)
            continue
        cmap = plt.get_cmap(TYPE_CMAP[t])
        k = len(ids)
        shades = [0.70] if k == 1 else list(np.linspace(0.42, 0.92, k))
        for cid, s in zip(ids, shades):
            colors[cid] = cmap(s)
    return colors, by_type


def make_palette(n):
    """ListedColormap over class ids 0..n-1, grouped by substrate type."""
    colors, _ = build_class_colors()
    return ListedColormap([colors[c] for c in range(n)])


def main():
    X, y, groups = load_reference()
    print(f"loaded X={X.shape} y={y.shape} classes={sorted(set(y.tolist()))}")

    # z-score per dim (matches the kNN euclidean standardisation step)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    # ---- PCA via numpy SVD (center already done -> Xz is z-scored, mean~0) ----
    Xc = Xz - Xz.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]                      # projection onto first 2 PCs
    var = (S ** 2)
    var_ratio = var / var.sum()
    pc1_pct, pc2_pct = 100 * var_ratio[0], 100 * var_ratio[1]
    print(f"PCA var explained: PC1={pc1_pct:.1f}%  PC2={pc2_pct:.1f}%")

    # ---- t-SNE (PCA-50 init for speed; init='pca', perplexity 30) ----
    n_pca = min(50, Xc.shape[1])
    X50 = U[:, :n_pca] * S[:n_pca]
    tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=0)
    emb = tsne.fit_transform(X50)
    print(f"t-SNE done, embedding shape {emb.shape}")

    # ---- figure ----
    cmap = make_palette(N_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)

    for ax, coords, ttl in (
        (axes[0], pcs, f"PCA  (PC1 {pc1_pct:.1f}%, PC2 {pc2_pct:.1f}% var)"),
        (axes[1], emb, "t-SNE  (perplexity 30, PCA-50 init)"),
    ):
        ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap=cmap,
                   vmin=-0.5, vmax=N_CLASSES - 0.5, s=10, alpha=0.8,
                   linewidths=0)
        ax.set_title(ttl, fontsize=11)
        ax.tick_params(labelsize=8)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    # shared legend, parked to the right and GROUPED BY SUBSTRATE TYPE:
    # a bold header per type, then its class ids (same colour family).
    _, by_type = build_class_colors()
    handles, labels = [], []
    for t in TYPE_ORDER:
        ids = by_type[t]
        # type header: invisible handle, bold-ish text label
        handles.append(Line2D([0], [0], linestyle="", marker="", alpha=0))
        labels.append(f"$\\bf{{{t}}}$  ({len(ids)} cls)")
        for c in ids:
            handles.append(Line2D([0], [0], marker="o", linestyle="",
                                  markerfacecolor=cmap(c), markeredgecolor="none",
                                  markersize=7))
            labels.append(f"   {c:>2d}")
    fig.legend(handles=handles, labels=labels,
               title="first-cyclization class\n(grouped by substrate type)",
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=8, title_fontsize=9,
               ncol=1, frameon=True, framealpha=0.9, handletextpad=0.4,
               labelspacing=0.35)

    fig.suptitle(
        "Training TPSs (n=1349) in DPLM-150m mean-embedding space "
        "— coloured by first-cyclization class (hue = substrate type)",
        fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005, "same embeddings as the kNN first-cyclization classifier",
             fontsize=9, style="italic", ha="left", va="bottom")

    for p in OUT_PATHS:
        # NAS mirror is best-effort: skip cleanly when /Volumes/data isn't mounted
        if str(p).startswith("/Volumes/") and not Path("/Volumes/data").exists():
            print(f"skip (NAS not mounted): {p}")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"saved {p}")


if __name__ == "__main__":
    main()
