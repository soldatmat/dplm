#!/usr/bin/env python3
"""Sorted per-architecture struct-vs-seq isTPS bar charts.

One figure per architecture (cross-attention / prepend / mini cross-attn),
runs sorted best->worst by structure-augmented best-median isTPS. Each column
is two-tone in a single hue: dark lower segment = sequence-only median,
light cap = the gain from adding ESMFold structure (total height = structure
median). No extra width vs the seq-only slides — both numbers in one column.
"""
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SUMMARY = Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
               "structure_vs_sequence_3arch_2026-06-08/struct_vs_seq_summary.csv")
OUTDIR = SUMMARY.parent

# label -> short display name (matches the seq-only slide legends)
DISPLAY = {
    "CA_FT_rand": "FT, rand", "CA_FT_orig": "FT, orig", "CA_Vca_LN_orig": "Vca+LN",
    "CA_ALLadap_rand": "ALLadap rand", "CA_ALLadap_orig": "ALLadap orig",
    "CA_Vca_2LN_only": "Vca+2LN", "CA_ALLadap_allV": "ALLadap+allV",
    "PRE_QV": "QV", "PRE_V29": "V29", "PRE_V15to29": "V15-29", "PRE_V_default": "V",
    "PRE_V_lr1e4": "V lr1e-4", "PRE_V_lr1e2": "V lr1e-2", "PRE_V_r2a4": "V r=2",
    "PRE_QVK": "QVK", "PRE_QVKO": "QVKO", "PRE_QVK_lmhead": "QVK+lmhead",
    "PRE_K": "K", "PRE_Q": "Q",
    "MINI_ltm0": "ltm0", "MINI_V": "V", "MINI_Q": "Q", "MINI_K": "K",
    "MINI_QV": "QV", "MINI_QVK": "QVK", "MINI_V15to28": "V15-28",
    "BASELINE_run41_V": "run_41 V\n(baseline)",
}

# architecture -> (groups to include, title, output stem)
# run_41 V baseline is the common init checkpoint for all three architectures
# (init_weights + class-encoder embeddings derive from run_41V step 200000),
# so it is shown as a reference bar on every panel.
_BASE = "slide 266  baseline (run_41 V)"
ARCHES = {
    "CA": (["slide 247  cross-attention", _BASE],
           "isTPS comparison – cross-attention conditioning (sorted, +structure)",
           "struct_vs_seq_sorted_CA"),
    "PRE": (["slide 248  prepend", _BASE],
            "isTPS comparison – prepend conditioning (sorted, +structure)",
            "struct_vs_seq_sorted_PRE"),
    "MINI": (["slide 266  mini cross-attn", _BASE],
             "isTPS comparison – mini cross-attention conditioning (sorted, +structure)",
             "struct_vs_seq_sorted_MINI"),
}

DARK = "#1f4e79"   # sequence-only
LIGHT = "#9dc3e6"  # + structure gain

# trainable params per run (from each run's train.log "trainable params" line;
# see comparison/trainable_params_overview/data.csv. run_41 V baseline = 38,400).
PARAMS = {
    "CA_FT_rand": 4923520, "CA_FT_orig": 4923520, "CA_Vca_2LN_only": 412800,
    "CA_ALLadap_allV": 52480, "CA_ALLadap_rand": 14080, "CA_ALLadap_orig": 14080,
    "CA_Vca_LN_orig": 3840,
    "PRE_QVKO": 153600, "PRE_QVK_lmhead": 116480, "PRE_QVK": 115200, "PRE_QV": 76800,
    "PRE_V_r2a4": 76800, "PRE_V_default": 38400, "PRE_V_lr1e4": 38400,
    "PRE_V_lr1e2": 38400, "PRE_Q": 38400, "PRE_K": 38400, "PRE_V15to29": 19200,
    "PRE_V29": 1280,
    "MINI_QVK": 453280, "MINI_QV": 414880, "MINI_V": 376480, "MINI_Q": 376480,
    "MINI_K": 376480, "MINI_ltm0": 361840, "MINI_V15to28": 350880,
    "BASELINE_run41_V": 38400,
}


def fmtp(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    return f"{n / 1e3:.1f}k"


def load():
    rows = []
    with open(SUMMARY) as f:
        for r in csv.DictReader(f):
            rows.append((r["group"], r["label"], float(r["seq_median"]), float(r["struct_median"])))
    return rows


def make_fig(rows, groups, title, stem, with_params=False):
    data = [(lab, seq, st) for g, lab, seq, st in rows if g in groups]
    data.sort(key=lambda d: d[2], reverse=True)  # by structure median desc
    labels = [l for l, _, _ in data]
    names = [DISPLAY.get(l, l) for l, _, _ in data]
    seq = np.array([d[1] for d in data])
    st = np.array([d[2] for d in data])
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(13.33, 6.8), constrained_layout=True)
    width = 0.7
    ax.bar(x, seq, width, color=DARK, edgecolor="white", linewidth=0.6, zorder=2)
    ax.bar(x, st - seq, width, bottom=seq, color=LIGHT,
           edgecolor="white", linewidth=0.6, zorder=2)

    for xi, s, t in zip(x, seq, st):
        # structure value above the bar (bold, white bbox)
        ax.text(xi, t + 0.008, f"{t:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        # seq-only value at the colour boundary, white text on the dark segment
        ax.text(xi, s - 0.006, f"{s:.3f}", ha="center", va="top",
                fontsize=10.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=13)
    ax.set_ylim(0.80, 1.0)
    ax.set_yticks(np.arange(0.80, 1.001, 0.05))
    ax.set_ylabel("best-median isTPS", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    if with_params:
        # trainable-param count as a row under each column (axes-fraction y so it
        # never collides with data); caption to its left.
        for xi, lab in zip(x, labels):
            ax.text(xi, -0.30, fmtp(PARAMS[lab]), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=11.5, color="#333333", fontweight="bold")
        ax.text(-0.012, -0.30, "trainable\nparams:", transform=ax.transAxes,
                ha="right", va="top", fontsize=10.5, color="#333333", style="italic")

    handles = [Patch(facecolor=LIGHT, edgecolor="white", label="+ ESMFold structure (cap = gain)"),
               Patch(facecolor=DARK, edgecolor="white", label="sequence-only EE")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=2, frameon=True, framealpha=0.9, fontsize=13)

    fig.suptitle(title + "   ·   n=50 seqs, best-median ckpt per run",
                 fontsize=15, fontweight="bold")
    out = OUTDIR / f"{stem}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    print(f"{out.name}: {Image.open(out).size}  ({len(data)} runs)")
    return out


def main():
    rows = load()
    for key, (groups, title, stem) in ARCHES.items():
        make_fig(rows, groups, title, stem)                       # no-param version
        make_fig(rows, groups, title, stem + "_params", with_params=True)


if __name__ == "__main__":
    main()
