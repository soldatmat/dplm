#!/usr/bin/env python3
"""Slide-342 twin for the PRETRAINED (un-finetuned) DPLM-150m base model.

Show WHERE in the MARTS-DB training TPS embedding space the raw pretrained
`airkingbd/dplm_150m` model samples — the starting point that run_41V was
LoRA-stage3 fine-tuned from — in three projections: PCA | t-SNE | UMAP.

Base-gen = 50 unconditional seqs @ L=350 generated with the SAME procedure as
run_41V's baseline (temperature 1.0, gumbel_argmax, max_iter 500), then
mean-pooled with the run_41V step-200000 encoder — the SAME embedding space as
the slide-342 training cloud and run_41V baseline points.

METHOD is byte-identical to make_baseline_overlay.py (slide 342):
  * PCA  : fit on the MARTS-DB TRAINING embeddings ONLY; project base-gen points
           into that fixed basis out-of-sample. var% must be ≈ PC1 41.4% /
           PC2 17.3% (slide-315 train-only basis) — correctness check.
  * UMAP : train-only fit on PCA-50; base-gen transformed through the same
           PCA-50 + reducer.transform.
  * t-SNE: recompute on train+base-gen combined (no out-of-sample transform).

Expectation: the un-finetuned base model has not seen the TPS manifold, so its
generations may land FAR from the train cloud — that distance is the point of
this slide (it quantifies how much the 41V fine-tuning pulled generations onto
the TPS manifold). Autoscale is intentionally left on so that distance shows.

Run: /opt/miniconda3/bin/python3 make_base150m_overlay.py
"""
from __future__ import annotations

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

EMB_CSV = Path(
    "/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
    "slide306_eval/base150m_2026-06-15/base150m_labeled_embeddings_mean.csv"
)
OUT_PNG = HERE / "base150m_overlay_pca_tsne_umap.png"
HIGHLIGHT = "darkorange"


def load_basegen(emb_csv: Path):
    """All rows are base-gen points (no prefix filter)."""
    df = pd.read_csv(emb_csv)
    feat_cols = [c for c in df.columns if c != "id"]
    return df[feat_cols].to_numpy(dtype=np.float64)


def draw_figure(out_png, proj, cloud_tsne, base_tsne, base_pca,
                cloud_umap, base_umap, n_base):
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
        ax.scatter(base[:, 0], base[:, 1], c=HIGHLIGHT, s=45, alpha=0.9,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.tick_params(labelsize=8)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.8",
               markeredgecolor="none", markersize=8),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=HIGHLIGHT,
               markeredgecolor="black", markeredgewidth=0.5, markersize=9),
    ]
    labels = [
        "MARTS-DB training TPSs (n=1349)",
        f"pretrained DPLM-150m base\n(no TPS fine-tuning) gen seqs (n={n_base})",
    ]
    fig.legend(handles=handles, labels=labels, loc="center left",
               bbox_to_anchor=(1.0, 0.5), bbox_transform=fig.transFigure,
               fontsize=9, frameon=True, framealpha=0.9, labelspacing=0.8)

    fig.suptitle(
        "Pretrained DPLM-150m base (run_41V's un-finetuned starting point): where "
        "its generated seqs land in the MARTS-DB TPS embedding space — space fit "
        "on train, generated projected in",
        fontsize=12.5, fontweight="bold")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    X, _, _ = load_reference()
    X = np.asarray(X, dtype=np.float64)
    print(f"loaded reference X={X.shape}")
    proj = build_projection(X)
    print(f"train-only PCA var: PC1={proj['pc1']:.2f}%  PC2={proj['pc2']:.2f}%")

    G = load_basegen(EMB_CSV)
    print(f"loaded base-gen embeddings: {G.shape}")

    base_pca = project_pca2(proj, G)
    cloud_tsne, base_tsne = fit_combined_tsne(proj, G)
    cloud_umap, base_umap = fit_train_umap(proj, G)

    # how far out does the base model land? (PCA-space distance, train sigma units)
    train_pca = proj["cloud_pca"]
    tc = train_pca.mean(0)
    train_rad = np.median(np.linalg.norm(train_pca - tc, axis=1))
    base_rad = np.median(np.linalg.norm(base_pca - tc, axis=1))
    print(f"median |PC1,PC2| from train centroid: train={train_rad:.2f}  "
          f"base-gen={base_rad:.2f}  (ratio {base_rad/train_rad:.1f}x)")

    draw_figure(OUT_PNG, proj, cloud_tsne, base_tsne, base_pca,
                cloud_umap, base_umap, G.shape[0])

    print("\n--- verification ---")
    from PIL import Image
    with Image.open(OUT_PNG) as im:
        w, h = im.size
    kb = OUT_PNG.stat().st_size / 1024
    print(f"{'OK' if kb > 50 else 'SMALL'}  {w}x{h}px  {kb:7.1f}KB  {OUT_PNG}")
    print(f"PCA var (train-only): PC1={proj['pc1']:.2f}% PC2={proj['pc2']:.2f}% "
          f"(check ≈ 41.4 / 17.3)")


if __name__ == "__main__":
    main()
