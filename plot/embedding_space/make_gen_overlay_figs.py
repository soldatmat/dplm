#!/usr/bin/env python3
"""Overlay DPLM-GENERATED sequence embeddings onto the training-TPS reference
2D space (PCA + t-SNE), to see where generated sequences land.

Produces 5 PNGs (each = PCA panel left + t-SNE panel right), plus copies them
to the NAS bundle dir if /Volumes/data is mounted.

Projection is an EXACT out-of-sample projection: train axes are fixed (mean/std
standardisation + centering + SVD on the train reference), and generated points
are pushed through the same transform. Generated embeddings were extracted with
the SAME encoder (run_41 V step-200000) as the reference, so they share the
640-d space and this projection is valid.

Run: python3 make_gen_overlay_figs.py
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

# t-SNE's PCA-init does a float32 matmul that can raise spurious overflow
# RuntimeWarnings on macOS/Accelerate; they're harmless. Silence them.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# kNN reference loader from the CANONICAL standalone repo (data md5-identical).
sys.path.insert(0, "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn")
from knn_first_cyclization import load_reference, N_CLASSES  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

# This script lives in dplm/plot/embedding_space/, but its input CSVs and output
# PNGs live with the rest of the generated-seq eval data (slide306_eval/).
HERE = Path(__file__).resolve().parent
DATA_DIR = Path("/Users/soldatmat/Documents/terpene_synthases/dplm/run/"
                "class_predictor/slide306_eval")
CLASS0_EMB = DATA_DIR / "slide306_10runs_combined_embeddings_mean.csv"
CLASS1_EMB = DATA_DIR / "class1_mini_combined_embeddings_mean.csv"
CLASS0_FID = DATA_DIR / "slide306_fidelity_summary.csv"
CLASS1_FID = DATA_DIR / "class1_fidelity_summary.csv"

NAS_DIR = Path(
    "/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
    "knn_first_cyclization_fidelity_2026-06-09"
)

# vivid colors for conditional models (skip gray/black/gold)
COND_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]
GOLD = "#C9A227"
BASELINE_LABEL = "BASELINE_run41_V"

FOOTNOTE = (
    "generated-seq embeddings via the run_41 V step-200000 encoder "
    "— same space as the kNN reference"
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_gen(emb_csv: Path):
    """Return dict run_label -> ndarray (n,640), preserving file order."""
    df = pd.read_csv(emb_csv)
    df["__run"] = df["id"].map(lambda x: x.split("__SEQUENCE")[0])
    feat_cols = [c for c in df.columns if c not in ("id", "__run")]
    out = {}
    for run, sub in df.groupby("__run", sort=False):
        out[run] = sub[feat_cols].to_numpy(dtype=np.float32)
    return out


def load_fid(fid_csv: Path):
    """Return dict label -> {display, arch, ...row}."""
    df = pd.read_csv(fid_csv)
    return {r["label"]: r.to_dict() for _, r in df.iterrows()}


# --------------------------------------------------------------------------- #
# Projection (fixed train axes, exact out-of-sample)
# --------------------------------------------------------------------------- #
def build_projection(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    center = Xz.mean(0, keepdims=True)
    Xc = Xz - center
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    train2d = U[:, :2] * S[:2]
    var_ratio = S ** 2 / (S ** 2).sum()
    pc1 = 100 * var_ratio[0]
    pc2 = 100 * var_ratio[1]
    proj = {
        "mu": mu, "sd": sd, "center": center,
        "Vt": Vt, "U": U, "S": S,
        "train2d": train2d, "pc1": pc1, "pc2": pc2, "Xc": Xc,
    }
    return proj


def project_pca(proj, G):
    Gz = (G - proj["mu"]) / proj["sd"]
    Gc = Gz - proj["center"]
    return Gc @ proj["Vt"][:2].T


def project_pca50(proj, G):
    Gz = (G - proj["mu"]) / proj["sd"]
    Gc = Gz - proj["center"]
    n_pca = min(50, proj["Xc"].shape[1])
    return Gc @ proj["Vt"][:n_pca].T


def fit_tsne(proj, gen_groups_order, gen_arrays):
    """Fit t-SNE ONCE on train(50pc) + all generated(50pc) for this class.

    gen_groups_order: list of run labels (defines row order of generated block).
    gen_arrays: dict label -> (n,640) raw embeddings.
    Returns (train_tsne (n_train,2), dict label -> (n,2)).
    """
    n_pca = min(50, proj["Xc"].shape[1])
    train50 = proj["U"][:, :n_pca] * proj["S"][:n_pca]
    gen50_blocks = []
    counts = []
    for lab in gen_groups_order:
        g50 = project_pca50(proj, gen_arrays[lab])
        gen50_blocks.append(g50)
        counts.append(g50.shape[0])
    stacked = np.vstack([train50] + gen50_blocks).astype(np.float32)
    emb = TSNE(
        n_components=2, init="pca", perplexity=30, random_state=0
    ).fit_transform(stacked)
    n_train = train50.shape[0]
    train_t = emb[:n_train]
    out = {}
    off = n_train
    for lab, c in zip(gen_groups_order, counts):
        out[lab] = emb[off:off + c]
        off += c
    return train_t, out


# --------------------------------------------------------------------------- #
# Figure drawing
# --------------------------------------------------------------------------- #
def draw_figure(out_png, suptitle, target_class,
                proj, train_tsne,
                series, fid):
    """series: ordered list of run labels to overlay (baseline first).
    fid: dict label -> fidelity row (for display/arch).
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)

    train2d = proj["train2d"]
    ymask_target = (Y_ALL == target_class)

    # ---- train background ----
    for ax, coords in ((axL, train2d), (axR, train_tsne)):
        ax.scatter(coords[~ymask_target, 0], coords[~ymask_target, 1],
                   c="0.8", s=6, alpha=0.35, linewidths=0, zorder=1)
        ax.scatter(coords[ymask_target, 0], coords[ymask_target, 1],
                   c=GOLD, s=16, alpha=0.6, linewidths=0, zorder=2)

    # legend handles
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="0.8",
               markersize=6, alpha=0.6, label="training: non-target classes"),
        Line2D([0], [0], marker="o", linestyle="", color=GOLD,
               markersize=8, alpha=0.9, label=f"training: class {target_class} (target)"),
    ]

    cond_i = 0
    for lab in series:
        g_pca = project_pca(proj, GEN_ARRAYS[lab])
        g_tsne = TSNE_COORDS[lab]
        if lab == BASELINE_LABEL:
            for ax, coords in ((axL, g_pca), (axR, g_tsne)):
                ax.scatter(coords[:, 0], coords[:, 1], c="black", marker="X",
                           s=30, edgecolors="white", linewidths=0.4, zorder=4)
            handles.append(Line2D([0], [0], marker="X", linestyle="", color="black",
                                   markersize=7, markeredgecolor="white",
                                   label="baseline run_41 V (uncond.)"))
        else:
            color = COND_COLORS[cond_i % len(COND_COLORS)]
            cond_i += 1
            row = fid[lab]
            lbl = f"{row['display']} ({row['arch']})"
            for ax, coords in ((axL, g_pca), (axR, g_tsne)):
                ax.scatter(coords[:, 0], coords[:, 1], c=color, marker="o",
                           s=26, edgecolors="black", linewidths=0.3, alpha=0.9,
                           zorder=3)
            handles.append(Line2D([0], [0], marker="o", linestyle="", color=color,
                                   markersize=7, markeredgecolor="black",
                                   alpha=0.9, label=lbl))

    axL.set_title(f"PCA (PC1 {proj['pc1']:.1f}%, PC2 {proj['pc2']:.1f}% var)")
    axL.set_xlabel("PC1")
    axL.set_ylabel("PC2")
    axR.set_title("t-SNE (perplexity 30, PCA-50 init)")
    axR.set_xlabel("t-SNE 1")
    axR.set_ylabel("t-SNE 2")

    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=8, frameon=True)
    fig.text(0.0, 0.0, FOOTNOTE, fontsize=7, style="italic",
             ha="left", va="bottom")

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
def main():
    global Y_ALL, GEN_ARRAYS, TSNE_COORDS

    # ---- reference ----
    X, y, _ = load_reference()
    Y_ALL = y
    proj0 = build_projection(X)

    fid0 = load_fid(CLASS0_FID)
    fid1 = load_fid(CLASS1_FID)

    gen0 = load_gen(CLASS0_EMB)   # 10 runs, 50 each
    gen1 = load_gen(CLASS1_EMB)   # 3 mini runs, 50 each

    # ---- best-per-architecture by frac_class0 (CA / PRE / MINI) ----
    fdf = pd.read_csv(CLASS0_FID)
    best_per_arch = {}
    for arch in ("CA", "PRE", "MINI"):
        sub = fdf[fdf["arch"] == arch]
        best = sub.loc[sub["frac_class0"].idxmax(), "label"]
        best_per_arch[arch] = best
    print("best-per-arch (frac_class0):", best_per_arch)

    written = []

    # ============================ CLASS 0 ============================
    # one shared t-SNE layout: train + ALL 500 class-0 generated
    GEN_ARRAYS = dict(gen0)  # for class-0 figures
    order0 = list(gen0.keys())  # file order
    train_t0, tsne0 = fit_tsne(proj0, order0, gen0)
    TSNE_COORDS = tsne0

    # A) overview: baseline + best-per-arch
    seriesA = [BASELINE_LABEL, best_per_arch["CA"], best_per_arch["PRE"], best_per_arch["MINI"]]
    written.append(draw_figure(
        DATA_DIR / "gen_overlay_class0_overview.png",
        "Generated sequences (class_ids=[0]) in train TPS embedding space "
        "— baseline vs best-per-architecture",
        0, proj0, train_t0, seriesA, fid0))

    # B) CA variants
    seriesB = [BASELINE_LABEL, "CA_FT_rand", "CA_FT_orig", "CA_ALLadap_orig"]
    written.append(draw_figure(
        DATA_DIR / "gen_overlay_class0_CA.png",
        "Generated sequences (class_ids=[0]) in train TPS embedding space "
        "— cross-attention (CA) variants",
        0, proj0, train_t0, seriesB, fid0))

    # C) PRE variants
    seriesC = [BASELINE_LABEL, "PRE_QVKO", "PRE_V29", "PRE_V_default"]
    written.append(draw_figure(
        DATA_DIR / "gen_overlay_class0_PRE.png",
        "Generated sequences (class_ids=[0]) in train TPS embedding space "
        "— prepend variants",
        0, proj0, train_t0, seriesC, fid0))

    # D) MINI variants
    seriesD = [BASELINE_LABEL, "MINI_QVK", "MINI_V15to28", "MINI_ltm0"]
    written.append(draw_figure(
        DATA_DIR / "gen_overlay_class0_MINI.png",
        "Generated sequences (class_ids=[0]) in train TPS embedding space "
        "— mini-CA variants",
        0, proj0, train_t0, seriesD, fid0))

    # ============================ CLASS 1 ============================
    # baseline reused from class-0 CSV; t-SNE fit on train + 150 mini + 50 baseline
    gen1_full = dict(gen1)
    gen1_full[BASELINE_LABEL] = gen0[BASELINE_LABEL]
    GEN_ARRAYS = gen1_full
    order1 = list(gen1.keys()) + [BASELINE_LABEL]
    train_t1, tsne1 = fit_tsne(proj0, order1, gen1_full)
    TSNE_COORDS = tsne1

    # E) class1 MINI
    seriesE = [BASELINE_LABEL, "MINI_QVK", "MINI_V15to28", "MINI_ltm0_step10k"]
    written.append(draw_figure(
        DATA_DIR / "gen_overlay_class1_MINI.png",
        "Generated sequences (class_ids=[1]) in train TPS embedding space "
        "— mini-CA variants (CA/prepend unavailable: pre-merge)",
        1, proj0, train_t1, seriesE, fid1))

    # ---- verify ----
    print("\n--- verification ---")
    for p in written:
        kb = p.stat().st_size / 1024
        ok = "OK" if kb > 50 else "TOO SMALL"
        print(f"{ok}  {kb:8.1f} KB  {p}")

    # ---- NAS copies ----
    nas_written = False
    if Path("/Volumes/data").exists():
        try:
            NAS_DIR.mkdir(parents=True, exist_ok=True)
            import shutil
            for p in written:
                shutil.copy2(p, NAS_DIR / p.name)
            nas_written = True
            print(f"\nNAS copies written to {NAS_DIR}")
        except Exception as e:
            print(f"\nNAS copy FAILED: {e}")
    else:
        print("\n/Volumes/data not mounted — skipped NAS copies")

    print(f"\nNAS_WRITTEN={nas_written}")
    print(f"PCA var: pc1={proj0['pc1']:.2f}% pc2={proj0['pc2']:.2f}%")


if __name__ == "__main__":
    main()
