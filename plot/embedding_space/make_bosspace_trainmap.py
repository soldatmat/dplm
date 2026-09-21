#!/usr/bin/env python3
"""BOS-embedding-space version of slide #316's training-TPS map.

Slide #316 plots all 1349 MARTS-DB training TPSs in the DPLM-150m
SEQUENCE-MEAN embedding space as PCA + t-SNE, coloured by first-cyclization
class. This is the same plot but (a) in the BOS embedding space and (b) with a
UMAP panel added -> 3 panels PCA | t-SNE | UMAP on the BOS cloud, every point
coloured by first-cyclization class (carbon-ordered palette, substrate-type
legend). NO conditioning vectors -- this is the data cloud only.

Reuses make_pca_tsne_fig (palette/legend/PCA-SVD/t-SNE machinery) and the BOS
cloud loading + UMAP(n_neighbors=15,min_dist=0.1,random_state=0) pattern from
make_bos_conditioning_map_bosspace.

Run: /opt/miniconda3/bin/python3 make_bosspace_trainmap.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
import umap

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
OUT = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
           "slide306_eval/training_tps_bosspace_pca_tsne_umap.png")


def load_bos_cloud():
    """Return (X [n,640] float64, y [n] int) -- the BOS cloud + aligned labels.

    The BOS csv `id` order should equal the labels group col (both come from the
    same TPS_first_cyclization.csv). Verify; if not, join on id to align labels.
    """
    labels = pd.read_csv(LABELS_CSV)
    bdf = pd.read_csv(BOS_CSV)
    if (len(bdf) == len(labels)
            and (bdf["id"].values == labels[GROUP_COL].values).all()):
        print("id order matches labels group col 1:1 (no join needed)")
        X = bdf.drop(columns=["id"]).to_numpy(dtype=np.float64)
        y = labels[LABEL_COL].to_numpy(dtype=np.int64)
    else:
        print("id order MISMATCH -> joining BOS rows to labels on id")
        merged = bdf.merge(labels[[GROUP_COL, LABEL_COL]],
                           left_on="id", right_on=GROUP_COL, how="inner")
        if len(merged) != len(bdf):
            raise ValueError(f"join lost rows: {len(bdf)} BOS -> {len(merged)} matched")
        y = merged[LABEL_COL].to_numpy(dtype=np.int64)
        X = merged.drop(columns=["id", GROUP_COL, LABEL_COL]).to_numpy(dtype=np.float64)
    return X, y


def main():
    X, y = load_bos_cloud()
    print(f"BOS cloud X={X.shape}  y={y.shape}  "
          f"classes={sorted(set(y.tolist()))}")

    # ---- z-score per dim on the BOS cloud ----
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    ctr = Xz.mean(0, keepdims=True); Xc = Xz - ctr

    # ---- PCA via SVD ----
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :2] * S[:2]
    vr = S ** 2 / (S ** 2).sum()
    p1, p2 = 100 * vr[0], 100 * vr[1]
    print(f"PCA var explained (BOS cloud): PC1={p1:.2f}%  PC2={p2:.2f}% "
          f"(expect BOS-space ~23.5/17.8)")

    # ---- PCA-50 features (shared by t-SNE init + UMAP fit) ----
    npc = min(50, Xc.shape[1])
    X50 = U[:, :npc] * S[:npc]

    # ---- t-SNE (perplexity 30, PCA-50 init, random_state 0) ----
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=0).fit_transform(X50.astype(np.float32))
    print(f"t-SNE done, shape {emb.shape}")

    # ---- UMAP fit on the BOS cloud (PCA-50, consistent with t-SNE) ----
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
    umap_emb = reducer.fit_transform(X50)
    print(f"UMAP done, shape {umap_emb.shape}")

    # ---- figure: 3 panels PCA | t-SNE | UMAP, coloured by class ----
    cmap = make_palette(N_CLASSES)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True, dpi=200)
    panels = (
        (axes[0], pcs, f"PCA  (PC1 {p1:.1f}%, PC2 {p2:.1f}% var)", "PC1", "PC2"),
        (axes[1], emb, "t-SNE  (perplexity 30, PCA-50 init)", "t-SNE 1", "t-SNE 2"),
        (axes[2], umap_emb, "UMAP  (n_neighbors 15, min_dist 0.1)", "UMAP 1", "UMAP 2"),
    )
    for ax, coords, ttl, xl, yl in panels:
        ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap=cmap,
                   vmin=-0.5, vmax=N_CLASSES - 0.5, s=10, alpha=0.8, linewidths=0)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.tick_params(labelsize=8)

    # shared substrate-type-grouped legend, parked OUTSIDE the data area
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
    fig.legend(handles=handles, labels=labels,
               title="first-cyclization class\n(grouped by substrate type)",
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=8, title_fontsize=9,
               ncol=1, frameon=True, framealpha=0.9, handletextpad=0.4,
               labelspacing=0.35)

    fig.suptitle(
        "Training TPSs (n=1349) in the DPLM-150m BOS embedding space "
        "— PCA / t-SNE / UMAP, coloured by first-cyclization class "
        "(hue = substrate type)",
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


if __name__ == "__main__":
    main()
