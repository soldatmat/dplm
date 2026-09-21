"""n=500 (full) vs n=50 (subsample) distribution comparison for run_41_V step 70k.

Mirrors the seq-side panels of slide 256 (seq-only isTPS, max seq identity to
MARTS-DB, canonical class-I motif presence) and shows how much a 50-sequence
subsample *wobbles* relative to the full n=500 estimate.

For each metric:
  - LEFT  : the full n=500 per-sequence distribution (the reference "truth").
  - RIGHT : the sampling distribution of the n=50 estimator -- the statistic
            (median for continuous metrics, fraction for the binary motif
            metrics) recomputed over K random subsamples of size 50 drawn
            WITHOUT replacement from the 500. The n=500 truth (solid line) and
            the specific seed=42 subsample used on slide 254 (dashed line) are
            overlaid, with the central 95% band of the subsample estimates.

Structure-aware isTPS and foldseek alntmscore are intentionally omitted: no
n=500 structure/foldseek eval exists (only the n=50 subsample was folded).

Usage:
    python3 plot_n500_vs_n50_run41V.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUN = Path(
    "/Volumes/data/Users/Matous/terpene_synthases/output/dplm/training/"
    "TPS_dplm_150m_stage3_grid_run_41_lr1em3_wu2000_ts200000_ckpt10000_"
    "valee10000_lend0p0001_winit1em06_loratrue_ns50_r1_a2_ltmV"
)
N500_DIR = RUN / "post_train_evaluation" / "step_70000"
N50_DIR = RUN / "enzyme_explorer_structure_validation" / "step_70000"
OUT_DIR = Path(
    "/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
    "n500_vs_n50_run41V"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 5000          # number of random n=50 subsamples for the spread
SUB_N = 50
SEED = 0          # for the spread draws (NOT the slide-254 seed=42 subsample)

DDXXD = "DD..D"
NSEDTE = "(N|D)D(L|I|V).(S|T)...E"

C500 = "#4C72B0"   # full n=500
CSUB = "#DD8452"   # n=50 subsample spread
CSEED = "#C44E52"  # the specific seed=42 subsample


def _read(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def main():
    # --- load full n=500 ---
    ee500 = _read(N500_DIR / "generated_sequences_enzyme_explorer_sequence_only.csv")
    id500 = _read(N500_DIR / "generated_sequences_max_sequence_identity.csv")
    mo500 = _read(N500_DIR / "generated_sequences_motifs.csv")

    # --- the specific seed=42 n=50 subsample = the 50 IDs that were folded ---
    seed42_ids = set(_read(N50_DIR / "sequences_enzyme_explorer_sequence_only.csv")["ID"])
    assert seed42_ids.issubset(set(ee500["ID"])), "n=50 set is not a subset of n=500"

    isTPS = ee500.set_index("ID")["isTPS"].astype(float)
    ident = id500.set_index("ID")["sequence_identity"].astype(float)
    dd = mo500.set_index("ID")[DDXXD].astype(bool)
    nse = mo500.set_index("ID")[NSEDTE].astype(bool)
    ids = list(isTPS.index)
    n = len(ids)
    print(f"n={n} sequences; seed=42 subsample n={len(seed42_ids)}")

    rng = np.random.default_rng(SEED)
    sub_idx = np.array([rng.choice(n, size=SUB_N, replace=False) for _ in range(K)])

    def spread_stat(values, fn):
        v = values.loc[ids].to_numpy()
        return np.array([fn(v[ix]) for ix in sub_idx])

    seed42_mask = np.array([i in seed42_ids for i in ids])

    # continuous metrics use median; binary motifs use fraction (mean of bool)
    metrics = [
        dict(name="isTPS  (sequence-only EE)", values=isTPS, stat=np.median,
             statname="median", ylim=(0.5, 1.0), full_truth=float(np.median(isTPS)),
             seed42=float(np.median(isTPS[seed42_mask])), color=C500),
        dict(name="Max seq identity to MARTS-DB train", values=ident, stat=np.median,
             statname="median", ylim=(0.15, 0.55), full_truth=float(np.median(ident)),
             seed42=float(np.median(ident[seed42_mask])), color="#55A868"),
    ]
    motif_metrics = [
        dict(name="DDxxD motif", series=dd, color="#8172B3"),
        dict(name="NSE/DTE motif", series=nse, color="#937860"),
    ]

    nrows = len(metrics) + 1  # +1 row for motifs
    fig, axes = plt.subplots(nrows, 2, figsize=(13.33, 7.0),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.4]})

    # ----- continuous metric rows -----
    for r, m in enumerate(metrics):
        axL, axR = axes[r]
        v = m["values"].loc[ids].to_numpy()
        # LEFT: n=500 distribution
        parts = axL.violinplot([v], positions=[1], showextrema=False, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor(m["color"]); pc.set_alpha(0.35); pc.set_edgecolor("black")
        axL.boxplot([v], positions=[1], widths=0.25, showfliers=False,
                    medianprops=dict(color="black"))
        axL.scatter(np.full_like(v, 1) + rng.uniform(-0.07, 0.07, len(v)), v,
                    s=5, color="black", alpha=0.18, zorder=3)
        axL.set_xlim(0.4, 1.6); axL.set_xticks([])
        axL.set_ylim(*m["ylim"]); axL.set_ylabel(m["statname"].capitalize() + " scale")
        axL.set_title(f"{m['name']}\nfull n=500 distribution", fontsize=10)
        axL.text(0.97, 0.03, f"n=500 {m['statname']} = {m['full_truth']:.3f}",
                 transform=axL.transAxes, ha="right", va="bottom", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))

        # RIGHT: spread of n=50 estimate
        spread = spread_stat(m["values"], m["stat"])
        lo, hi = np.percentile(spread, [2.5, 97.5])
        axR.hist(spread, bins=40, color=CSUB, alpha=0.7, edgecolor="white")
        axR.axvspan(lo, hi, color=CSUB, alpha=0.15)
        axR.axvline(m["full_truth"], color=C500, lw=2.2,
                    label=f"n=500 truth = {m['full_truth']:.3f}")
        axR.axvline(m["seed42"], color=CSEED, lw=2.0, ls="--",
                    label=f"seed=42 n=50 = {m['seed42']:.3f}")
        axR.set_yticks([])
        axR.set_xlabel(f"{m['statname']} of a random n=50 subsample")
        axR.set_title(f"n=50 sampling spread  ·  95% band width = {hi - lo:.3f}",
                      fontsize=10)
        axR.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)

    # ----- motif row -----
    axL, axR = axes[-1]
    # LEFT: n=500 fractions as bars + seed=42 marker
    fr_full = [float(mm["series"].loc[ids].mean()) for mm in motif_metrics]
    fr_seed = [float(mm["series"].loc[ids][seed42_mask].mean()) for mm in motif_metrics]
    xpos = np.arange(len(motif_metrics))
    bars = axL.bar(xpos, fr_full, width=0.55,
                   color=[mm["color"] for mm in motif_metrics], alpha=0.75)
    axL.scatter(xpos, fr_seed, marker="D", color=CSEED, zorder=5, s=45,
                label="seed=42 n=50")
    for x, f in zip(xpos, fr_full):
        axL.text(x, f + 0.02, f"{f:.0%}", ha="center", va="bottom", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
    axL.set_xticks(xpos); axL.set_xticklabels([mm["name"] for mm in motif_metrics],
                                              fontsize=9)
    axL.set_ylim(0, 1.05); axL.set_ylabel("Fraction present")
    axL.set_title("Canonical class-I TPS motif presence\nfull n=500 fraction",
                  fontsize=10)
    axL.legend(loc="upper left", fontsize=8, frameon=True)

    # RIGHT: spread of n=50 fraction for each motif
    for mm in motif_metrics:
        spread = spread_stat(mm["series"].astype(float), np.mean)
        lo, hi = np.percentile(spread, [2.5, 97.5])
        full = float(mm["series"].loc[ids].mean())
        axR.hist(spread, bins=30, color=mm["color"], alpha=0.55, edgecolor="white",
                 label=f"{mm['name']}: n=500={full:.0%}, 95% band ±{(hi-lo)/2:.0%}")
        axR.axvline(full, color=mm["color"], lw=2.0)
    axR.set_yticks([]); axR.set_xlim(0, 1)
    axR.set_xlabel("Fraction present in a random n=50 subsample")
    axR.set_title("n=50 sampling spread (motif presence rate)", fontsize=10)
    axR.legend(loc="upper center", fontsize=8, frameon=True, framealpha=0.9)

    fig.suptitle(
        "run_41_V step 70k — does an n=50 subsample reproduce the n=500 distribution?  "
        "·  seq-side metrics",
        fontsize=13, fontweight="bold")

    out = OUT_DIR / "n500_vs_n50_run41V.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")

    from PIL import Image
    print("image size:", Image.open(out).size)


if __name__ == "__main__":
    main()
