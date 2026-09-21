#!/usr/bin/env python3
"""Cross-architecture struct-vs-seq isTPS WITH bootstrap 95% CIs (slide-307 variant).

Same 10-bar layout as plot_struct_vs_seq_crossarch.py (colour = architecture;
dark lower segment = sequence-only median, light cap = gain from ESMFold
structure), but each median now carries a bootstrap 95% confidence interval.

For every run the bootstrap resamples the 50 per-sequence isTPS scores WITH
replacement B times, recomputes the median each time, and takes the 2.5/97.5
percentiles. Two whiskers per bar: one on the seq-only median (top of the dark
segment) and one on the structure median (bar top).

Per-sequence inputs:
  seq-only  : training/<run>/enzyme_explorer_validation/<step>/
              generated_sequences_enzyme_explorer_sequence_only.csv  (isTPS col)
  structure : <3arch>/inputs/<label>/sequences_enzyme_explorer.csv     (isTPS col)
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

TRAIN = Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/training")
OUTDIR = Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
              "structure_vs_sequence_3arch_2026-06-08")
SUMMARY = OUTDIR / "struct_vs_seq_summary.csv"
MANIFEST = OUTDIR / "manifest.csv"

B = 10000     # bootstrap resamples
SEED = 0

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


def isTPS(fp):
    df = pd.read_csv(fp)
    df.columns = [c.strip() for c in df.columns]
    col = [c for c in df.columns if c.lower() == "istps"][0]
    return df[col].astype(float).to_numpy()


def boot_ci(rng, x, b=B, q=(2.5, 97.5)):
    """Bootstrap CI of the median: resample n WITH replacement, b times."""
    boots = rng.choice(x, size=(b, len(x)), replace=True)
    med = np.median(boots, axis=1)
    return np.percentile(med, q)


def main():
    man = {r["label"]: r for r in csv.DictReader(open(MANIFEST))}
    rng = np.random.default_rng(SEED)

    rows = []
    for lab, (disp, arch) in SEL.items():
        m = man[lab]
        so = isTPS(TRAIN / m["run_folder"] / "enzyme_explorer_validation"
                   / m["step_name"] / "generated_sequences_enzyme_explorer_sequence_only.csv")
        st = isTPS(OUTDIR / "inputs" / lab / "sequences_enzyme_explorer.csv")
        so_med, st_med = float(np.median(so)), float(np.median(st))
        so_lo, so_hi = boot_ci(rng, so)
        st_lo, st_hi = boot_ci(rng, st)
        rows.append(dict(lab=lab, disp=disp, arch=arch, so=so_med, st=st_med,
                         so_lo=so_lo, so_hi=so_hi, st_lo=st_lo, st_hi=st_hi))
    rows.sort(key=lambda d: d["st"], reverse=True)  # by structure median desc

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.33, 6.8), constrained_layout=True)
    width = 0.72
    for xi, r in zip(x, rows):
        dark, light, _ = ARCH[r["arch"]]
        ax.bar(xi, r["so"], width, color=dark, edgecolor="white", linewidth=0.6, zorder=2)
        ax.bar(xi, r["st"] - r["so"], width, bottom=r["so"], color=light,
               edgecolor="white", linewidth=0.6, zorder=2)
        # bootstrap whiskers: structure (bar top) and seq-only (dark-segment top)
        ax.errorbar(xi, r["st"], yerr=[[r["st"] - r["st_lo"]], [r["st_hi"] - r["st"]]],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=6)
        ax.errorbar(xi, r["so"], yerr=[[r["so"] - r["so_lo"]], [r["so_hi"] - r["so"]]],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=6)
        # numeric labels clear of the whiskers
        ax.text(xi, r["st_hi"] + 0.006, f"{r['st']:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        ax.text(xi, r["so_lo"] - 0.006, f"{r['so']:.3f}", ha="center", va="top",
                fontsize=10, color="white")
        ax.text(xi, -0.30, fmtp(PARAMS[r["lab"]]), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11.5, color="#333333", fontweight="bold")

    ax.text(-0.012, -0.30, "trainable\nparams:", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color="#333333", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([r["disp"] for r in rows], rotation=40, ha="right", fontsize=13)
    ax.set_ylim(0.80, 1.0)
    ax.set_yticks(np.arange(0.80, 1.001, 0.05))
    ax.set_ylabel("median isTPS", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    handles = [Patch(facecolor=ARCH[a][0], edgecolor="white", label=ARCH[a][2]) for a in ORDER]
    handles.append(Line2D([0], [0], color="black", lw=1.5, marker="_",
                          label=f"bootstrap 95% CI ({B:,} resamples, with replacement)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.40),
              ncol=3, frameon=True, framealpha=0.9, fontsize=10.5,
              title="per bar:  dark = sequence-only    ·    light cap = + ESMFold structure",
              title_fontsize=10.5)

    fig.suptitle("isTPS comparison — best runs per architecture + baseline  ·  "
                 "bootstrap 95% CI on each median\n"
                 "n=50 seqs, best-median ckpt per run  ·  50 seqs resampled with replacement",
                 fontsize=13.5, fontweight="bold")
    out = OUTDIR / "struct_vs_seq_crossarch_bootstrap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    from PIL import Image
    print(f"{out.name}: {Image.open(out).size}  ({len(rows)} runs, B={B})")

    # also print the CIs for the record
    print(f"\n{'run':18s} {'seq med [95% CI]':28s} {'struct med [95% CI]'}")
    for r in rows:
        print(f"{r['lab']:18s} {r['so']:.3f} [{r['so_lo']:.3f},{r['so_hi']:.3f}]"
              f"      {r['st']:.3f} [{r['st_lo']:.3f},{r['st_hi']:.3f}]")


if __name__ == "__main__":
    main()
