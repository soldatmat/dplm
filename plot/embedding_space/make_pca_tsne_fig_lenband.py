#!/usr/bin/env python3
"""All-class ESM-embedding map of the 1349 training TPSs (PCA + t-SNE), coloured
by first-cyclization class, with every point whose TRAINING amino-acid length is
in [280, 420] AA (~one structural domain) OUTLINED with a black edge ring.

Same layout/axes/colouring as the slide-#315 figure built by
``make_pca_tsne_fig.py`` (this is a NEW variant; it does not overwrite that
figure's PNG). Two panels (PCA left, t-SNE right), grouped-by-substrate-type
class legend parked outside right, plus one extra legend handle explaining the
outline.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.manifold import TSNE

# reuse the exact reference loader the kNN classifier uses — sourced from the
# CANONICAL standalone repo (github.com/soldatmat/tps-first-cyclization-knn).
sys.path.insert(0, "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn")
from knn_first_cyclization import load_reference, N_CLASSES  # noqa: E402

# reuse the palette / type machinery from the sibling all-class fig script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pca_tsne_fig import (  # noqa: E402
    build_class_colors,
    make_palette,
    TYPE_ORDER,
)

CSV_PATH = Path("/Users/soldatmat/Documents/terpene_synthases/dplm/data-bin/"
                "MARTS-DB/2026-04-12/TPS_first_cyclization.csv")

LEN_LO, LEN_HI = 280, 420  # ~ one structural domain

OUT_PATHS = [
    Path("/Users/soldatmat/Documents/terpene_synthases/dplm/run/class_predictor/"
         "slide306_eval/train_tps_pca_tsne_lenband.png"),
    Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
         "knn_first_cyclization_fidelity_2026-06-09/train_tps_pca_tsne_lenband.png"),
]


def build_id2len():
    """Map Enzyme_marts_ID -> sequence length (duplicate rows share the seq)."""
    id2len = {}
    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            eid = row["Enzyme_marts_ID"]
            seq = row["Aminoacid_sequence"]
            id2len[eid] = len(seq)
    return id2len


def main():
    X, y, groups = load_reference()
    print(f"loaded X={X.shape} y={y.shape} classes={sorted(set(y.tolist()))}")

    id2len = build_id2len()
    lengths = np.array([id2len[g] for g in groups])
    assert len(lengths) == len(y) == 1349, (
        f"length mismatch: lengths={len(lengths)} y={len(y)} (expected 1349)")
    in_band = (lengths >= LEN_LO) & (lengths <= LEN_HI)
    n_band = int(in_band.sum())
    print(f"in-band [{LEN_LO},{LEN_HI}] AA points: {n_band} / {len(y)}")

    # z-score per dim (matches the kNN euclidean standardisation step)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    # ---- PCA via numpy SVD ----
    Xc = Xz - Xz.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    var = (S ** 2)
    var_ratio = var / var.sum()
    pc1_pct, pc2_pct = 100 * var_ratio[0], 100 * var_ratio[1]
    print(f"PCA var explained: PC1={pc1_pct:.1f}%  PC2={pc2_pct:.1f}%")

    # ---- t-SNE (PCA-50 init, perplexity 30) ----
    n_pca = min(50, Xc.shape[1])
    X50 = U[:, :n_pca] * S[:n_pca]
    tsne = TSNE(n_components=2, init="pca", perplexity=30, random_state=0)
    emb = tsne.fit_transform(X50)
    print(f"t-SNE done, embedding shape {emb.shape}")

    # ---- figure ----
    cmap = make_palette(N_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)

    out = ~in_band
    for ax, coords, ttl in (
        (axes[0], pcs, f"PCA  (PC1 {pc1_pct:.1f}%, PC2 {pc2_pct:.1f}% var)"),
        (axes[1], emb, "t-SNE  (perplexity 30, PCA-50 init)"),
    ):
        # out-of-band: plain dots
        ax.scatter(coords[out, 0], coords[out, 1], c=y[out], cmap=cmap,
                   vmin=-0.5, vmax=N_CLASSES - 0.5, s=10, alpha=0.8,
                   linewidths=0, zorder=1)
        # in-band: same colour, black edge ring, on top
        ax.scatter(coords[in_band, 0], coords[in_band, 1], c=y[in_band],
                   cmap=cmap, vmin=-0.5, vmax=N_CLASSES - 0.5, s=14,
                   alpha=0.95, edgecolors="black", linewidths=0.7, zorder=3)
        ax.set_title(ttl, fontsize=11)
        ax.tick_params(labelsize=8)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")

    # shared legend, parked to the right and GROUPED BY SUBSTRATE TYPE
    _, by_type = build_class_colors()
    handles, labels = [], []
    for t in TYPE_ORDER:
        ids = by_type[t]
        handles.append(Line2D([0], [0], linestyle="", marker="", alpha=0))
        labels.append(f"$\\bf{{{t}}}$  ({len(ids)} cls)")
        for c in ids:
            handles.append(Line2D([0], [0], marker="o", linestyle="",
                                  markerfacecolor=cmap(c), markeredgecolor="none",
                                  markersize=7))
            labels.append(f"   {c:>2d}")
    # trailing entry explaining the outline ring
    handles.append(Line2D([0], [0], linestyle="", marker="", alpha=0))
    labels.append("")
    handles.append(Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor="white", markeredgecolor="black",
                          markeredgewidth=1.0, markersize=8))
    labels.append(f"outline = {LEN_LO}-{LEN_HI} AA\n(~1 structural domain)  n={n_band}")
    fig.legend(handles=handles, labels=labels,
               title="first-cyclization class\n(grouped by substrate type)",
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=8, title_fontsize=9,
               ncol=1, frameon=True, framealpha=0.9, handletextpad=0.4,
               labelspacing=0.35)

    fig.suptitle(
        "Training TPSs in DPLM-150m embedding space — by first-cyclization "
        "class; 280-420 AA outlined (~1 structural domain)",
        fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005, "same embeddings as the kNN first-cyclization classifier",
             fontsize=9, style="italic", ha="left", va="bottom")

    for p in OUT_PATHS:
        if str(p).startswith("/Volumes/") and not Path("/Volumes/data").exists():
            print(f"skip (NAS not mounted): {p}")
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"saved {p}")


if __name__ == "__main__":
    main()
