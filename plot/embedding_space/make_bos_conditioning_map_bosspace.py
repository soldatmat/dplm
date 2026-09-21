#!/usr/bin/env python3
"""BOS-space conditioning map (honest same-representation version).

BOTH the grey cloud AND the 22 per-class conditioning vectors live in the SAME
BOS embedding space:
  * Cloud = the per-TPS BOS embeddings of the 1349 MARTS-DB training TPSs
    (run_41V step-200000 encoder), TPS_first_cyclization_embeddings_bos.csv.
  * 22 vectors = the frozen ClassEncoder init (TPS_first_cyclization_embeddings_bos
    _class_mean.pt). These are EXACTLY the per-class BOS-mean centroids of the
    cloud (verified: max |diff| ~7e-7 vs cloud means grouped by class) — so this
    is now an honest same-space plot: the 22 vectors ARE the per-class centroids.

3 panels: PCA | t-SNE | UMAP. "Base the space on TRAIN when possible":
  * PCA  : fit on the BOS cloud ONLY (z-score with cloud mu/sd, center, SVD),
           PROJECT the 22 means into that fixed basis. Reports PC1/PC2 var%.
  * UMAP : TRUE train-fit + transform. z-score (cloud stats) -> PCA-50 (cloud-fit)
           -> umap.UMAP(n_neighbors=15,min_dist=0.1,random_state=0).fit(cloud50),
           then reducer.transform of the 22 means through the IDENTICAL pipeline.
  * t-SNE: no out-of-sample transform, so recompute on cloud + 22 means COMBINED
           (z-scored with cloud stats, PCA-50 init, perplexity 30).

22 large carbon-ordered colored points with de-overlapped class-id labels
(adjustText leader lines) as in v2; grey BOS cloud background; one shared legend.

Run: /opt/miniconda3/bin/python3 make_bos_conditioning_map_bosspace.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import umap
from adjustText import adjust_text

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import build_class_colors, make_palette, TYPE_ORDER, N_CLASSES  # noqa

KNN = "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn"
LABELS_CSV = Path(f"{KNN}/data/TPS_first_cyclization.csv")
LABEL_COL = "First_cyclization_product_id"
GROUP_COL = "Enzyme_marts_ID"
BOS_CSV = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/data-bin/MARTS-DB/"
               "2026-04-12/embeddings/dplm_150m_stage3_grid_run_41V_step200000/"
               "TPS_first_cyclization_embeddings_bos.csv")
PT = Path("/tmp/bos_class_mean_karolina.pt")
OUT = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
           "slide306_eval/bos_conditioning_map_bosspace_3panel.png")


def load_bos_cloud():
    """Return (X [n,640] float64, y [n] int) — the BOS cloud and aligned labels."""
    labels = pd.read_csv(LABELS_CSV)
    bdf = pd.read_csv(BOS_CSV)
    if not (bdf["id"].values == labels[GROUP_COL].values).all():
        raise ValueError("BOS cloud `id` not aligned with labels group col")
    X = bdf.drop(columns=["id"]).to_numpy(dtype=np.float64)
    y = labels[LABEL_COL].to_numpy(dtype=np.int64)
    return X, y


def load_means():
    """Return the 22 per-class BOS-mean conditioning vectors (22,640) float64.

    Prefer the authoritative .pt; fall back to recomputing from the cloud."""
    if PT.exists():
        M = torch.load(PT, map_location="cpu", weights_only=False)
        M = (M.numpy() if hasattr(M, "numpy") else np.asarray(M)).astype(np.float64)
        return M, "karolina .pt"
    X, y = load_bos_cloud()
    ncls = int(y.max()) + 1
    M = np.vstack([X[y == c].mean(0) for c in range(ncls)])
    return M, "recomputed from cloud (pt unavailable)"


def label_panel(ax, m2, cmap, ncls):
    texts = []
    for ci in range(ncls):
        t = ax.text(m2[ci, 0], m2[ci, 1], str(ci), fontsize=8, fontweight="bold",
                    color="black", ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.95,
                              edgecolor=cmap(ci), linewidth=1.2))
        texts.append(t)
    adjust_text(texts, x=m2[:, 0], y=m2[:, 1], ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7,
                                shrinkA=3, shrinkB=4),
                expand=(2.0, 2.4), force_text=(1.2, 1.6), force_static=(0.6, 0.8),
                force_pull=(0.02, 0.02), ensure_inside_axes=True,
                only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
                min_arrow_len=0, max_move=40, iter_lim=600)


def main():
    X, y = load_bos_cloud()
    print(f"BOS cloud X={X.shape}  (n={X.shape[0]} TPSs, {X.shape[1]} dims)")
    M, src = load_means()
    ncls = M.shape[0]
    print(f"conditioning vectors M={M.shape}  source: {src}")

    # SANITY CHECK: M should equal the per-class BOS-mean centroids of the cloud.
    cloud_means = np.vstack([X[y == c].mean(0) for c in range(ncls)])
    maxdiff = float(np.abs(M - cloud_means).max())
    matched = np.allclose(M, cloud_means, rtol=1e-4, atol=1e-4)
    print(f"means-vs-cloud-centroids: max|diff|={maxdiff:.2e}  matched={matched}")

    # PCA fit on the BOS cloud ONLY; project the 22 means in.
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd; Mz = (M - mu) / sd
    ctr = Xz.mean(0, keepdims=True); Xc = Xz - ctr
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    cloud_pca = U[:, :2] * S[:2]
    vr = S ** 2 / (S ** 2).sum(); p1, p2 = 100 * vr[0], 100 * vr[1]
    M_pca = (Mz - ctr) @ Vt[:2].T
    print(f"PCA (BOS-cloud fit): PC1={p1:.1f}%  PC2={p2:.1f}%")

    npc = min(50, Xc.shape[1])
    X50 = U[:, :npc] * S[:npc]
    M50 = (Mz - ctr) @ Vt[:npc].T

    # t-SNE: combined recompute (cloud + 22 means).
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(
        np.vstack([X50, M50]).astype(np.float32))
    cloud_t, M_t = emb[:len(X)], emb[len(X):]

    # UMAP: TRUE train-fit on the cloud (PCA-50), transform the 22 means.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
    cloud_u = reducer.fit_transform(X50)
    M_u = reducer.transform(M50)

    cmap = make_palette(N_CLASSES)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True, dpi=200)
    panels = (
        (axes[0], cloud_pca, M_pca,
         f"PCA (train-fit, projected · PC1 {p1:.1f}%, PC2 {p2:.1f}%)", "PC1", "PC2"),
        (axes[1], cloud_t, M_t, "t-SNE (train+vectors recompute)", "t-SNE 1", "t-SNE 2"),
        (axes[2], cloud_u, M_u, "UMAP (train-fit, transformed)", "UMAP 1", "UMAP 2"),
    )
    for ax, c2, m2, ttl, xl, yl in panels:
        ax.scatter(c2[:, 0], c2[:, 1], c="0.82", s=6, alpha=0.55, linewidths=0, zorder=1)
        for ci in range(ncls):
            ax.scatter(m2[ci, 0], m2[ci, 1], c=[cmap(ci)], s=110,
                       edgecolors="black", linewidths=0.7, zorder=3)
        label_panel(ax, m2, cmap, ncls)
        ax.set_title(ttl, fontsize=10.5); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.tick_params(labelsize=8)

    _, by_type = build_class_colors()
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.82",
                      markeredgecolor="none", markersize=7)]
    labels = ["all MARTS-DB TPSs (BOS, n=1349)"]
    for t in TYPE_ORDER:
        handles.append(Line2D([0], [0], linestyle="", marker="", alpha=0)); labels.append(f"$\\bf{{{t}}}$")
        for c in by_type[t]:
            handles.append(Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(c),
                                  markeredgecolor="black", markeredgewidth=0.5, markersize=8))
            labels.append(f"  class {c}")
    fig.legend(handles=handles, labels=labels,
               title="per-class BOS-mean\nconditioning vectors\n(= class centroids)",
               loc="center left", bbox_to_anchor=(1.0, 0.5), bbox_transform=fig.transFigure,
               fontsize=7.5, title_fontsize=9, frameon=True, framealpha=0.9, labelspacing=0.3)
    fig.suptitle("The 22 per-class BOS-mean conditioning vectors IN the BOS embedding space "
                 "of MARTS-DB training TPSs (space fit on train) — PCA / t-SNE / UMAP",
                 fontsize=12.5, fontweight="bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image
    with Image.open(OUT) as im:
        w, h = im.size
    kb = OUT.stat().st_size / 1024
    print(f"saved {OUT}")
    print(f"{'OK' if kb > 50 else 'SMALL'}  {w}x{h}px  {kb:.1f}KB")
    print(f"PCA var: PC1={p1:.2f}% PC2={p2:.2f}% (expect BOS-space ~23.5/17.8, NOT mean-space ~41/17)")


if __name__ == "__main__":
    main()
