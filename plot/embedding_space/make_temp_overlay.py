#!/usr/bin/env python3
"""Slide-342 twin for run_41V sampled at a non-default temperature.

Same train-only PCA / t-SNE / UMAP overlay method as make_baseline_overlay.py
(slide 342), but for run_41V (step-200000) generations at a given temperature
(T=2.0, T=5.0). The slide-342 baseline is T=1.0; this shows how raising the
sampling temperature spreads generations across / off the TPS manifold.

Embeddings are mean-pooled with the run_41V step-200000 encoder (same space).

Usage:
  /opt/miniconda3/bin/python3 make_temp_overlay.py --temp 2.0 \
      --emb_csv <path>/emb/T2.0_labeled_embeddings_mean.csv \
      --out base/run41V_t2.0_overlay_pca_tsne_umap.png
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import load_reference  # noqa: E402
from make_baseline_overlay import (  # noqa: E402
    build_projection, project_pca2, fit_combined_tsne, fit_train_umap,
)


def load_gen(emb_csv: Path):
    df = pd.read_csv(emb_csv)
    feat_cols = [c for c in df.columns if c != "id"]
    return df[feat_cols].to_numpy(dtype=np.float64)


def draw_figure(out_png, proj, cloud_tsne, base_tsne, base_pca,
                cloud_umap, base_umap, n_base, temp, color):
    fig, (axP, axT, axU) = plt.subplots(1, 3, figsize=(17, 5.3),
                                        constrained_layout=True, dpi=200)
    panels = [
        (axP, proj["cloud_pca"], base_pca,
         f"PCA (train-fit, projected · PC1 {proj['pc1']:.1f}%, "
         f"PC2 {proj['pc2']:.1f}%)", "PC1", "PC2"),
        (axT, cloud_tsne, base_tsne,
         "t-SNE (train+gen recompute)", "t-SNE 1", "t-SNE 2"),
        (axU, cloud_umap, base_umap,
         "UMAP (train-fit, transformed)", "UMAP 1", "UMAP 2"),
    ]
    for ax, cloud, base, ttl, xl, yl in panels:
        ax.scatter(cloud[:, 0], cloud[:, 1], c="0.8", s=6, alpha=0.35,
                   linewidths=0, zorder=1)
        ax.scatter(base[:, 0], base[:, 1], c=color, s=45, alpha=0.9,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.tick_params(labelsize=8)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.8",
               markeredgecolor="none", markersize=8),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
               markeredgecolor="black", markeredgewidth=0.5, markersize=9),
    ]
    labels = [
        "MARTS-DB training TPSs (n=1349)",
        f"run_41V gen seqs, T={temp}\n(n={n_base})",
    ]
    fig.legend(handles=handles, labels=labels, loc="center left",
               bbox_to_anchor=(1.0, 0.5), bbox_transform=fig.transFigure,
               fontsize=9, frameon=True, framealpha=0.9, labelspacing=0.8)
    fig.suptitle(
        f"run_41V (step-200000) at temperature T={temp}: where generated seqs land "
        f"in the MARTS-DB TPS embedding space — space fit on train, generated "
        f"projected in (cf. slide 342 = T=1.0)",
        fontsize=12.5, fontweight="bold")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", required=True)
    ap.add_argument("--emb_csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--color", default="#7b2cbf")  # purple
    args = ap.parse_args()

    X, _, _ = load_reference()
    X = np.asarray(X, dtype=np.float64)
    proj = build_projection(X)
    print(f"train-only PCA var: PC1={proj['pc1']:.2f}%  PC2={proj['pc2']:.2f}%")

    G = load_gen(args.emb_csv)
    print(f"loaded T={args.temp} gen embeddings: {G.shape}")

    base_pca = project_pca2(proj, G)
    cloud_tsne, base_tsne = fit_combined_tsne(proj, G)
    cloud_umap, base_umap = fit_train_umap(proj, G)

    tc = proj["cloud_pca"].mean(0)
    train_rad = np.median(np.linalg.norm(proj["cloud_pca"] - tc, axis=1))
    base_rad = np.median(np.linalg.norm(base_pca - tc, axis=1))
    print(f"median |PC1,PC2| from train centroid: train={train_rad:.2f}  "
          f"T{args.temp}-gen={base_rad:.2f}  (ratio {base_rad/train_rad:.1f}x)")

    draw_figure(args.out, proj, cloud_tsne, base_tsne, base_pca,
                cloud_umap, base_umap, G.shape[0], args.temp, args.color)

    from PIL import Image
    with Image.open(args.out) as im:
        w, h = im.size
    kb = args.out.stat().st_size / 1024
    print(f"{'OK' if kb > 50 else 'SMALL'}  {w}x{h}px  {kb:.1f}KB  {args.out}")
    print(f"PCA var check ≈ 41.4/17.3: PC1={proj['pc1']:.2f}% PC2={proj['pc2']:.2f}%")


if __name__ == "__main__":
    main()
