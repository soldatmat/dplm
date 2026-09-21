#!/usr/bin/env python3
"""Cross-architecture struct-vs-seq isTPS: best 3 runs per arch + run_41 V baseline.

10 bars, sorted best->worst by structure-augmented best-median isTPS. Colour =
architecture (4 families incl. baseline); within each bar the darker lower
segment is the sequence-only median and the lighter cap is the gain from adding
ESMFold structure (bar top = structure median). Trainable-params row underneath.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUTDIR = Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
              "structure_vs_sequence_3arch_2026-06-08")
SUMMARY = OUTDIR / "struct_vs_seq_summary.csv"

# label -> (display name, architecture key)
SEL = {
    "BASELINE_run41_V": ("run_41 V\n(baseline)", "BASE"),
    "CA_FT_rand": ("FT, rand", "CA"),
    "CA_FT_orig": ("FT, orig", "CA"),
    "CA_ALLadap_orig": ("ALLadap orig", "CA"),
    "PRE_QVKO": ("QVKO", "PRE"),
    "PRE_V29": ("V29", "PRE"),
    "PRE_V_default": ("V", "PRE"),
    "MINI_QVK": ("QVK", "MINI"),
    "MINI_V15to28": ("V15-28", "MINI"),
    "MINI_ltm0": ("ltm0", "MINI"),
}

# architecture -> (dark = seq-only, light = +structure cap, legend label)
ARCH = {
    "CA":   ("#1f4e79", "#9dc3e6", "Cross-attention"),
    "PRE":  ("#1e6b2e", "#a5d6a7", "Prepend"),
    "MINI": ("#c55a11", "#f4b183", "Mini cross-attn"),
    "BASE": ("#5b2d8e", "#c3a6e0", "run_41 V baseline (uncond.)"),
}
ORDER = ["CA", "PRE", "MINI", "BASE"]

PARAMS = {
    "CA_FT_rand": 4923520, "CA_FT_orig": 4923520, "CA_ALLadap_orig": 14080,
    "PRE_QVKO": 153600, "PRE_V29": 1280, "PRE_V_default": 38400,
    "MINI_QVK": 453280, "MINI_V15to28": 350880, "MINI_ltm0": 361840,
    "BASELINE_run41_V": 38400,
}


def fmtp(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    return f"{n / 1e3:.1f}k"


def main():
    rows = []
    for r in csv.DictReader(open(SUMMARY)):
        if r["label"] in SEL:
            disp, arch = SEL[r["label"]]
            rows.append((r["label"], disp, arch, float(r["seq_median"]), float(r["struct_median"])))
    rows.sort(key=lambda d: d[4], reverse=True)  # by structure median desc

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.33, 6.8), constrained_layout=True)
    width = 0.72
    for xi, (lab, disp, arch, seq, st) in zip(x, rows):
        dark, light, _ = ARCH[arch]
        ax.bar(xi, seq, width, color=dark, edgecolor="white", linewidth=0.6, zorder=2)
        ax.bar(xi, st - seq, width, bottom=seq, color=light, edgecolor="white", linewidth=0.6, zorder=2)
        ax.text(xi, st + 0.008, f"{st:.3f}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        ax.text(xi, seq - 0.006, f"{seq:.3f}", ha="center", va="top", fontsize=10.5, color="white")
        ax.text(xi, -0.30, fmtp(PARAMS[lab]), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11.5, color="#333333", fontweight="bold")

    ax.text(-0.012, -0.30, "trainable\nparams:", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color="#333333", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([d for _, d, _, _, _ in rows], rotation=40, ha="right", fontsize=13)
    ax.set_ylim(0.80, 1.0)
    ax.set_yticks(np.arange(0.80, 1.001, 0.05))
    ax.set_ylabel("best-median isTPS", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    handles = [Patch(facecolor=ARCH[a][0], edgecolor="white", label=ARCH[a][2]) for a in ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.40),
              ncol=4, frameon=True, framealpha=0.9, fontsize=11,
              title="per bar:  dark = sequence-only    ·    light cap = + ESMFold structure",
              title_fontsize=10.5)

    fig.suptitle("isTPS comparison — best runs per architecture + baseline (sorted, +structure)"
                 "   ·   n=50 seqs, best-median ckpt per run", fontsize=14, fontweight="bold")
    out = OUTDIR / "struct_vs_seq_crossarch.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    print(f"{out.name}: {Image.open(out).size}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
