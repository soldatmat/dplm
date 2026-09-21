#!/usr/bin/env python3
"""Show WHERE in the MARTS-DB training TPS embedding space the UNCONDITIONAL
baseline (run_41V) model samples, in three projections — PCA | t-SNE | UMAP.

Baseline = the 50 generated seqs whose id starts with BASELINE_run41_V in the
slide306 combined-mean embeddings CSV. These are mean-pooled embeddings from the
run_41V step-200000 encoder — the SAME space as the slide-315 training cloud.

METHOD ("compute the space on TRAIN only, project the generated data in"):
  * PCA  : fit on the MARTS-DB TRAINING embeddings ONLY (z-score per dim with
           TRAIN mu/sd, center, np.linalg.svd); PROJECT the baseline points into
           that fixed basis (base_pca = (base_z - train_ctr) @ Vt[:2].T using the
           TRAIN mu/sd). NOT refit on the baseline. var% must be ≈ PC1 41.4% /
           PC2 17.3% (slide-315 train-only basis) — correctness check.
  * UMAP : TRUE train-only fit + projection. z-score with train mu/sd, reduce to
           PCA-50 (train-fit), fit umap.UMAP(n_neighbors=15, min_dist=0.1,
           random_state=0) on those 50 train dims, then transform the baseline
           through the IDENTICAL PCA-50 + reducer.transform.
  * t-SNE: NO out-of-sample transform exists, so recompute on train+baseline
           COMBINED (z-scored with train stats, PCA-50 init, perplexity 30) —
           panel title flags this as a combined recompute, not a projection.

Unconditional generation => NO per-class colouring of generated points; the 50
baseline points get ONE bold highlight colour to show WHERE they land.

Run: /opt/miniconda3/bin/python3 make_baseline_overlay.py
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

from sklearn.manifold import TSNE
import umap

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import load_reference  # noqa: E402

EMB_CSV = Path(
    "/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
    "slide306_eval/slide306_10runs_combined_embeddings_mean.csv"
)
OUT_PNG = HERE / "baseline_run41V_overlay_pca_tsne_umap.png"

BASELINE_PREFIX = "BASELINE_run41_V"
HIGHLIGHT = "crimson"


# --------------------------------------------------------------------------- #
def load_baseline(emb_csv: Path):
    df = pd.read_csv(emb_csv)
    mask = df["id"].str.startswith(BASELINE_PREFIX)
    sub = df.loc[mask]
    feat_cols = [c for c in df.columns if c != "id"]
    feats = sub[feat_cols].to_numpy(dtype=np.float64)
    return feats


def build_projection(X):
    """Train-only z-score + PCA basis (numpy SVD)."""
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    ctr = Xz.mean(0, keepdims=True)
    Xc = Xz - ctr
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    cloud_pca = U[:, :2] * S[:2]
    vr = S ** 2 / (S ** 2).sum()
    return {
        "mu": mu, "sd": sd, "ctr": ctr, "U": U, "S": S, "Vt": Vt,
        "cloud_pca": cloud_pca, "pc1": 100 * vr[0], "pc2": 100 * vr[1],
        "ncol": Xc.shape[1],
    }


def project_pca2(proj, G):
    Gz = (G - proj["mu"]) / proj["sd"]
    return (Gz - proj["ctr"]) @ proj["Vt"][:2].T


def train_pca50(proj):
    npc = min(50, proj["ncol"])
    return proj["U"][:, :npc] * proj["S"][:npc]


def project_pca50(proj, G):
    npc = min(50, proj["ncol"])
    Gz = (G - proj["mu"]) / proj["sd"]
    return (Gz - proj["ctr"]) @ proj["Vt"][:npc].T


def fit_combined_tsne(proj, G):
    """t-SNE recomputed on train(50pc) + baseline(50pc). Returns (cloud, base)."""
    train50 = train_pca50(proj)
    base50 = project_pca50(proj, G)
    stacked = np.vstack([train50, base50]).astype(np.float32)
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=0).fit_transform(stacked)
    n = train50.shape[0]
    return emb[:n], emb[n:]


def fit_train_umap(proj, G):
    """TRUE train-only UMAP fit on PCA-50; baseline transformed through the same
    PCA-50 + reducer.transform. Returns (cloud, base)."""
    train50 = train_pca50(proj)
    base50 = project_pca50(proj, G)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
    cloud_umap = reducer.fit_transform(train50)
    base_umap = reducer.transform(base50)
    return cloud_umap, base_umap


# --------------------------------------------------------------------------- #
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
        f"unconditional baseline run_41V\ngenerated seqs (n={n_base})",
    ]
    fig.legend(handles=handles, labels=labels, loc="center left",
               bbox_to_anchor=(1.0, 0.5), bbox_transform=fig.transFigure,
               fontsize=9, frameon=True, framealpha=0.9, labelspacing=0.8)

    fig.suptitle(
        "Unconditional baseline (run_41V): where the generated seqs land in the "
        "MARTS-DB TPS embedding space — space fit on train, generated projected in",
        fontsize=12.5, fontweight="bold")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
def main():
    X, _, _ = load_reference()
    X = np.asarray(X, dtype=np.float64)
    print(f"loaded reference X={X.shape}")
    proj = build_projection(X)
    print(f"train-only PCA var: PC1={proj['pc1']:.2f}%  PC2={proj['pc2']:.2f}%")

    G = load_baseline(EMB_CSV)
    print(f"loaded baseline embeddings: {G.shape}")

    base_pca = project_pca2(proj, G)
    cloud_tsne, base_tsne = fit_combined_tsne(proj, G)
    cloud_umap, base_umap = fit_train_umap(proj, G)

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
