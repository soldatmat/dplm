#!/usr/bin/env python3
"""v2 of the BOS-conditioning map: identical to make_bos_conditioning_map.py but
the 22 class-id labels are de-overlapped with adjustText — overlapping labels are
offset and a thin leader line points back to the true point position.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from adjustText import adjust_text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import build_class_colors, make_palette, TYPE_ORDER, N_CLASSES, load_reference  # noqa

# Cloud = the EXACT slide-315 space: mean-pooled MARTS-DB embeddings via load_reference().
# Conditioning vectors = the per-class BOS-mean (22,640) the ClassEncoder is frozen at,
# standardized by the mean-cloud stats and projected into the mean-pooled PCA basis /
# co-embedded in a combined t-SNE.
PT = Path("/tmp/bos_class_mean_karolina.pt")
OUT = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
           "slide306_eval/bos_conditioning_map_meanspace.png")


def label_panel(ax, m2, cmap, ncls):
    # anchor dot stays ON the true point; the label text is what gets pushed away,
    # with a leader line from the displaced label back to the anchor.
    texts = []
    for ci in range(ncls):
        t = ax.text(m2[ci, 0], m2[ci, 1], str(ci), fontsize=8, fontweight="bold",
                    color="black", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.95,
                              edgecolor=cmap(ci), linewidth=1.2))
        texts.append(t)
    # aggressive de-overlap + always-drawn leader lines back to the true positions
    adjust_text(texts, x=m2[:, 0], y=m2[:, 1], ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7,
                                shrinkA=3, shrinkB=4),
                expand=(2.0, 2.4), force_text=(1.2, 1.6), force_static=(0.6, 0.8),
                force_pull=(0.02, 0.02), ensure_inside_axes=True,
                only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                min_arrow_len=0, max_move=40, iter_lim=600)


def main():
    # EXACT slide-315 cloud: mean-pooled MARTS-DB embeddings (same loader, z-score, SVD)
    Xref, _, _ = load_reference()
    X = np.asarray(Xref, dtype=np.float64)
    M = torch.load(PT, map_location="cpu", weights_only=False)
    M = (M.numpy() if hasattr(M, "numpy") else np.asarray(M)).astype(np.float64)
    ncls = M.shape[0]

    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd; Mz = (M - mu) / sd
    ctr = Xz.mean(0, keepdims=True); Xc = Xz - ctr
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    cloud_pca = U[:, :2] * S[:2]
    vr = S ** 2 / (S ** 2).sum(); p1, p2 = 100 * vr[0], 100 * vr[1]
    M_pca = (Mz - ctr) @ Vt[:2].T
    npc = min(50, Xc.shape[1])
    X50 = U[:, :npc] * S[:npc]; M50 = (Mz - ctr) @ Vt[:npc].T
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
        np.vstack([X50, M50]).astype(np.float32))
    cloud_t, M_t = emb[:len(X)], emb[len(X):]

    cmap = make_palette(N_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)
    for ax, c2, m2, ttl in (
        (axes[0], cloud_pca, M_pca, f"PCA  (PC1 {p1:.1f}%, PC2 {p2:.1f}% var)"),
        (axes[1], cloud_t, M_t, "t-SNE  (perplexity 30, PCA-50 init)"),
    ):
        ax.scatter(c2[:, 0], c2[:, 1], c="0.82", s=6, alpha=0.55, linewidths=0, zorder=1)
        for ci in range(ncls):
            ax.scatter(m2[ci, 0], m2[ci, 1], c=[cmap(ci)], s=110,
                       edgecolors="black", linewidths=0.7, zorder=3)
        label_panel(ax, m2, cmap, ncls)
        ax.set_title(ttl, fontsize=11); ax.tick_params(labelsize=8)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    _, by_type = build_class_colors()
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.82",
                      markeredgecolor="none", markersize=7)]
    labels = ["all MARTS-DB TPSs (mean-pooled, n=1349)"]
    for t in TYPE_ORDER:
        handles.append(Line2D([0], [0], linestyle="", marker="", alpha=0)); labels.append(f"$\\bf{{{t}}}$")
        for c in by_type[t]:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(c),
                                  markeredgecolor="black", markeredgewidth=0.5, markersize=8))
            labels.append(f"  class {c}")
    fig.legend(handles=handles, labels=labels, title="conditioning BOS-mean vectors\n(by substrate type)",
               loc="center left", bbox_to_anchor=(1.0, 0.5), bbox_transform=fig.transFigure,
               fontsize=7.5, title_fontsize=9, frameon=True, framealpha=0.9, labelspacing=0.3)
    fig.suptitle("Mean-pooled MARTS-DB TPS space (slide-315) + the 22 per-class BOS-mean conditioning vectors",
                 fontsize=12, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
