#!/usr/bin/env python3
"""Train-only ESM embedding map with a SINGLE first-cyclization class highlighted
in gold (no generated seqs), PLUS the subset of that class's TRAINING sequences
whose amino-acid length is in [280, 420] (~one structural domain) marked in a
contrasting blue.

Reuses the exact train-only PCA + t-SNE layout from make_train_class_highlight.py
(imports its `compute_layout`). Lengths come from the training CSV, mapped
per-row via Enzyme_marts_ID -> sequence length. One PNG per class in {0, 1}.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Reuse the canonical layout + palette + reference loader.
from make_train_class_highlight import compute_layout, GOLD, GRAY, OUTDIR  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn")
from knn_first_cyclization import load_reference  # noqa: E402

CSV = Path("/Users/soldatmat/Documents/terpene_synthases/dplm/data-bin/"
           "MARTS-DB/2026-04-12/TPS_first_cyclization.csv")
BLUE = "#1f77b4"
TARGETS = [0, 1]
LO, HI = 280, 420

# Best-effort NAS mirror (often unmounted off-VPN).
NAS_DIR = Path("/Volumes/data/Users/Matous/terpene_synthases/dplm/run/"
               "class_predictor/slide306_eval")


def build_lengths(groups):
    """Map each row's Enzyme_marts_ID -> AA sequence length (rows aligned w/ y)."""
    df = pd.read_csv(CSV, usecols=["Enzyme_marts_ID", "Aminoacid_sequence"])
    id2len = {i: len(s) for i, s in zip(df["Enzyme_marts_ID"], df["Aminoacid_sequence"])}
    return np.array([id2len[g] for g in groups])


def make_fig(y, pcs, emb, pc1, pc2, lengths, target, fname):
    in_class = y == target
    in_band = in_class & (lengths >= LO) & (lengths <= HI)
    cls_other = in_class & ~in_band
    rest = ~in_class

    n_band = int(in_band.sum())
    n_other = int(cls_other.sum())
    n_rest = int(rest.sum())

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 6.0), constrained_layout=True)
    for ax, coords, ttl in (
        (axes[0], pcs, f"PCA  (PC1 {pc1:.1f}%, PC2 {pc2:.1f}% var)"),
        (axes[1], emb, "t-SNE  (perplexity 30, PCA-50 init)"),
    ):
        ax.scatter(coords[rest, 0], coords[rest, 1], c=GRAY, s=8, alpha=0.4,
                   linewidths=0, zorder=1)
        ax.scatter(coords[cls_other, 0], coords[cls_other, 1], c=GOLD, s=18,
                   alpha=0.9, linewidths=0, zorder=3)
        ax.scatter(coords[in_band, 0], coords[in_band, 1], c=BLUE, s=26,
                   alpha=0.95, edgecolors="black", linewidths=0.3, zorder=4)
        ax.set_title(ttl, fontsize=11)
        ax.tick_params(labelsize=8)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[1].set_xlabel("t-SNE 1"); axes[1].set_ylabel("t-SNE 2")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=BLUE,
               markeredgecolor="black", markeredgewidth=0.3, markersize=8,
               label=f"class {target}, 280-420 AA   n={n_band}"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GOLD,
               markeredgecolor="none", markersize=8,
               label=f"class {target}, other lengths   n={n_other}"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GRAY,
               markeredgecolor="none", markersize=7,
               label=f"other training classes   n={n_rest}"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               bbox_transform=fig.transFigure, fontsize=9, frameon=True,
               framealpha=0.9)
    fig.suptitle(
        f"Training TPSs — class {target} highlighted; 280-420 AA "
        f"(~1 structural domain) in blue (no generated seqs)",
        fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.005,
             "train-only PCA + t-SNE; AA length of the training sequence; "
             "band 280-420",
             fontsize=9, style="italic", ha="left", va="bottom")

    _save(fig, OUTDIR / fname)
    _save(fig, NAS_DIR / fname, guard=True)
    plt.close(fig)
    print(f"class {target}: total={int(in_class.sum())}  "
          f"in_band[280,420]={n_band}  outside={n_other}")
    return int(in_class.sum()), n_band, n_other


def _save(fig, out, guard=False):
    if guard and not Path("/Volumes/data").exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


def main():
    _, y, groups = load_reference()
    lengths = build_lengths(groups)
    y2, pcs, emb, pc1, pc2 = compute_layout()
    assert len(lengths) == len(y) == len(y2) == 1349, \
        f"length mismatch: {len(lengths)} / {len(y)} / {len(y2)}"
    assert (y == y2).all(), "row order mismatch between load_reference and compute_layout"
    n_all_band = int(((lengths >= LO) & (lengths <= HI)).sum())
    print(f"all training rows in [280,420]: {n_all_band} / {len(y)}")
    for t in TARGETS:
        make_fig(y, pcs, emb, pc1, pc2, lengths, t,
                 f"train_highlight_class{t}_lenband.png")


if __name__ == "__main__":
    main()
