#!/usr/bin/env python3
"""Where do the conditioning vectors sit? Plot the MARTS-DB TPS BOS-embedding
cloud (grey) and overlay the 22 frozen per-class BOS-mean vectors that the
class-conditioning ClassEncoder is initialized to (and frozen at).

Cloud = per-TPS BOS embeddings (run_41V step-200000), n=1349, class-agnostic grey.
Overlay = the (22,640) `TPS_first_cyclization_embeddings_bos_class_mean.pt` the
model actually loads, coloured by first-cyclization class (carbon-ordered palette),
labelled with the class id. Same BOS space for both (honest overlay).

If the 22 class-means clump near the centre rather than spreading into distinct
class regions, that visually explains why the (frozen, low-capacity) conditioning
barely steers.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import build_class_colors, make_palette, TYPE_ORDER, N_CLASSES  # noqa

EMB = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/data-bin/MARTS-DB/2026-04-12/"
           "embeddings/dplm_150m_stage3_grid_run_41V_step200000/"
           "TPS_first_cyclization_embeddings_bos.csv")
PT = Path("/tmp/bos_class_mean_karolina.pt")  # the 22-class conditioning vectors (pulled from Karolina)
OUT = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
           "slide306_eval/bos_conditioning_map.png")


def main():
    cloud = pd.read_csv(EMB)
    X = cloud.drop(columns=["id"]).to_numpy(dtype=np.float64)        # (1349, 640) BOS
    M = torch.load(PT, map_location="cpu", weights_only=False)
    M = (M.numpy() if hasattr(M, "numpy") else np.asarray(M)).astype(np.float64)  # (22, 640)
    ncls = M.shape[0]
    print(f"cloud {X.shape}  conditioning means {M.shape}")

    # z-score on the cloud, project both
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd; Mz = (M - mu) / sd
    ctr = Xz.mean(0, keepdims=True)
    Xc = Xz - ctr
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    cloud_pca = U[:, :2] * S[:2]
    vr = S ** 2 / (S ** 2).sum(); p1, p2 = 100 * vr[0], 100 * vr[1]
    M_pca = (Mz - ctr) @ Vt[:2].T
    # how spread are the means vs the cloud? (diagnostic printed)
    print(f"PCA var PC1 {p1:.1f}% PC2 {p2:.1f}%; "
          f"means PC1-2 std=({M_pca[:,0].std():.2f},{M_pca[:,1].std():.2f}) "
          f"vs cloud std=({cloud_pca[:,0].std():.2f},{cloud_pca[:,1].std():.2f})")

    # t-SNE on cloud + means together
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
            ax.annotate(str(ci), (m2[ci, 0], m2[ci, 1]), fontsize=6.5,
                        ha="center", va="center", zorder=4, color="white", fontweight="bold")
        ax.set_title(ttl, fontsize=11); ax.tick_params(labelsize=8)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    # legend: grey cloud + per-type class grouping
    _, by_type = build_class_colors()
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.82",
                      markeredgecolor="none", markersize=7, label="all MARTS-DB TPSs (BOS, n=1349)")]
    labels = ["all MARTS-DB TPSs (BOS, n=1349)"]
    for t in TYPE_ORDER:
        handles.append(Line2D([0], [0], linestyle="", marker="", alpha=0)); labels.append(f"$\\bf{{{t}}}$")
        for c in by_type[t]:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(c),
                                  markeredgecolor="black", markeredgewidth=0.5, markersize=8))
            labels.append(f"  class {c}")
    fig.legend(handles=handles, labels=labels, title="conditioning BOS-mean vectors\n(by substrate type)",
               loc="center left", bbox_to_anchor=(1.0, 0.5), bbox_transform=fig.transFigure,
               fontsize=7.5, title_fontsize=9, frameon=True, framealpha=0.9, labelspacing=0.3)
    fig.suptitle("MARTS-DB TPS BOS-embedding space (grey) + the 22 frozen per-class BOS-mean "
                 "conditioning vectors", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005, "grey = per-TPS BOS embeddings (run_41V step-200000); large = the "
             "Embedding(22,640) the ClassEncoder is frozen at; same BOS space",
             fontsize=9, style="italic", ha="left", va="bottom")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
