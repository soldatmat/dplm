#!/usr/bin/env python3
"""Overlay the MULTICLASS STEERING SWEEP generated-sequence embeddings onto the
training-TPS reference 2D space (PCA + t-SNE), coloured by their CONDITIONING
(target) class, on the grey MARTS-DB training cloud.

Sweep = 3 mini models {MINI_QVK, MINI_V15to28, MINI_ltm0} x 6 conditioning
classes {0,1,5,9,12,17} x 50 seqs = 900 generations.

METHOD (per the established slide-315 pattern):
  * PCA is fit on the MARTS-DB TRAINING sequences ONLY (z-score per dim with
    train mu/sd, center, np.linalg.svd; cloud_pca = U[:,:2]*S[:2]); the generated
    sequences are PROJECTED into that fixed basis (gen_pca = (gen_z - ctr) @ Vt[:2].T,
    gen_z standardized with the TRAIN mu/sd). PCA is NOT refit on the gen points.
  * t-SNE is recomputed on train(50pc) + generated(50pc) TOGETHER (combined fit),
    matching make_gen_overlay_figs.py / make_bos_conditioning_map_v2.py.

One figure per model (3 -> 3 slides): a 2-panel PCA + t-SNE with the grey
training cloud + that model's 300 generated points coloured by target class,
using the shared carbon-ordered palette make_palette(N_CLASSES).

Run: /opt/miniconda3/bin/python3 make_gen_multiclass_overlay.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import (  # noqa: E402
    build_class_colors, make_palette, TYPE_ORDER, N_CLASSES, load_reference,
)
from sklearn.manifold import TSNE  # noqa: E402

EMB_CSV = Path(
    "/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
    "slide306_eval/multiclass_sweep_2026-06-12/emb/"
    "multiclass_combined_embeddings_mean.csv"
)
OUT_DIR = Path(
    "/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
    "slide306_eval/multiclass_sweep_2026-06-12"
)

MODEL_ORDER = ["MINI_QVK", "MINI_V15to28", "MINI_ltm0"]
SWEEP_CLASSES = [0, 1, 5, 9, 12, 17]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_gen(emb_csv: Path):
    """Return DataFrame with parsed model/target_class + (n,640) feature array."""
    df = pd.read_csv(emb_csv)
    lab = df["id"].map(lambda x: x.split("__SEQUENCE")[0])
    df["__model"] = lab.map(lambda L: L.rsplit("_class", 1)[0])
    df["__class"] = lab.map(lambda L: int(L.rsplit("_class", 1)[1]))
    feat_cols = [c for c in df.columns if c not in ("id", "__model", "__class")]
    feats = df[feat_cols].to_numpy(dtype=np.float64)
    return df, feats


# --------------------------------------------------------------------------- #
# Projection (train-only PCA, exact out-of-sample for gen)
# --------------------------------------------------------------------------- #
def build_projection(X):
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


def project_pca50(proj, G):
    npc = min(50, proj["ncol"])
    Gz = (G - proj["mu"]) / proj["sd"]
    return (Gz - proj["ctr"]) @ proj["Vt"][:npc].T


def fit_combined_tsne(proj, G):
    """Combined t-SNE on train(50pc) + this model's gen(50pc). Returns
    (cloud_tsne, gen_tsne)."""
    npc = min(50, proj["ncol"])
    train50 = proj["U"][:, :npc] * proj["S"][:npc]
    gen50 = project_pca50(proj, G)
    stacked = np.vstack([train50, gen50]).astype(np.float32)
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=0).fit_transform(stacked)
    n = train50.shape[0]
    return emb[:n], emb[n:]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def draw_figure(out_png, model, proj, cloud_tsne, gen_tsne,
                gen_pca, gen_classes):
    cmap = make_palette(N_CLASSES)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.3, 6.0),
                                   constrained_layout=True)

    for ax, cloud, gen in (
        (axL, proj["cloud_pca"], gen_pca),
        (axR, cloud_tsne, gen_tsne),
    ):
        # grey MARTS-DB training cloud as background
        ax.scatter(cloud[:, 0], cloud[:, 1], c="0.8", s=6, alpha=0.35,
                   linewidths=0, zorder=1)
        # generated points coloured by conditioning (target) class
        ax.scatter(gen[:, 0], gen[:, 1], c=gen_classes, cmap=cmap,
                   vmin=-0.5, vmax=N_CLASSES - 0.5, s=22, alpha=0.9,
                   edgecolors="black", linewidths=0.3, zorder=3)

    axL.set_title(f"PCA  (PC1 {proj['pc1']:.1f}%, PC2 {proj['pc2']:.1f}% var)",
                  fontsize=11)
    axL.set_xlabel("PC1"); axL.set_ylabel("PC2")
    axR.set_title("t-SNE  (perplexity 30, PCA-50 init)", fontsize=11)
    axR.set_xlabel("t-SNE 1"); axR.set_ylabel("t-SNE 2")
    for ax in (axL, axR):
        ax.tick_params(labelsize=8)

    # shared legend, parked outside the data area
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.8",
                      markeredgecolor="none", markersize=7)]
    labels = ["MARTS-DB training TPSs (n=1349)"]
    for c in SWEEP_CLASSES:
        handles.append(Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=cmap(c), markeredgecolor="black",
                              markeredgewidth=0.4, markersize=8))
        labels.append(f"target class {c}")
    fig.legend(handles=handles, labels=labels,
               title="generated (coloured by\nconditioning class)",
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=8, title_fontsize=9,
               frameon=True, framealpha=0.9, labelspacing=0.35)

    fig.suptitle(
        f"{model}: 300 generated seqs (6 conditioning classes x 50) "
        "in MARTS-DB TPS embedding space",
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

    df, feats = load_gen(EMB_CSV)
    print(f"loaded gen embeddings: {feats.shape}")

    written = []
    for model in MODEL_ORDER:
        mask = (df["__model"] == model).to_numpy()
        G = feats[mask]
        gen_classes = df.loc[mask, "__class"].to_numpy()
        print(f"\n{model}: {G.shape[0]} gen points; "
              f"classes={sorted(set(gen_classes.tolist()))}")

        gen_pca = project_pca2(proj, G)
        cloud_tsne, gen_tsne = fit_combined_tsne(proj, G)

        out = OUT_DIR / f"gen_multiclass_overlay_{model}.png"
        draw_figure(out, model, proj, cloud_tsne, gen_tsne,
                    gen_pca, gen_classes)
        written.append(out)

    print("\n--- verification ---")
    from PIL import Image
    for p in written:
        with Image.open(p) as im:
            w, h = im.size
        kb = p.stat().st_size / 1024
        print(f"{'OK' if kb > 50 else 'SMALL'}  {w}x{h}px  {kb:7.1f}KB  {p}")

    print(f"\nPCA var (train-only): PC1={proj['pc1']:.2f}% "
          f"PC2={proj['pc2']:.2f}%")


if __name__ == "__main__":
    main()
