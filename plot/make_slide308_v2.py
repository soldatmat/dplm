#!/usr/bin/env python3
"""Slide-308 v2: isTPS comparison with the 4 post-merge ("fcmerge" run_3) retrains
REPLACING their old counterparts, using each retrain's BEST checkpoint among steps
<= 50000 (by median sequence-only isTPS). Old runs we did NOT retrain are kept but
GRAYED out. run_41 V baseline stays as-is.

Ported from plot_struct_vs_seq_crossarch_bootstrap.py (the original slide-308 source).

METRIC SHOWN: sequence-only EnzymeExplorer isTPS median + bootstrap 95% CI for ALL
bars. The retrains have ONLY sequence-only isTPS per checkpoint (no per-ckpt ESMFold
structure eval), so to keep one honest, comparable metric across every bar we drop the
"+structure" light cap that the original slide had. The bootstrap-CI methodology is
identical to the original: 10,000 resamples (with replacement) of the 50 per-seq isTPS
values, median recomputed each resample, 2.5/97.5 percentiles.

Inputs:
  - old runs' seq-only per-seq isTPS: TRAIN/<run_folder>/enzyme_explorer_validation/
        <step_name>/generated_sequences_enzyme_explorer_sequence_only.csv   (NAS)
  - 4 retrains' best-<=50k 50 per-seq isTPS: retrains_le50k.json (pulled from Karolina
        node-local scratch via srun --overlap).
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

HERE = Path(__file__).resolve().parent
TRAIN = Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/training")
OUTDIR = Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
              "structure_vs_sequence_3arch_2026-06-08")
MANIFEST = OUTDIR / "manifest.csv"
RETRAINS_JSON = HERE / "retrains_le50k.json"
OUT_PNG = HERE / "slide308_v2_isTPS_compare.png"

B = 10000     # bootstrap resamples (identical to original)
SEED = 0

# --- the original 10 runs from plot_struct_vs_seq_crossarch_bootstrap.py SEL ---
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

# retrain short-key -> old SEL label it REPLACES, and the new display label
RETRAIN_REPLACES = {
    "QVKO":       ("PRE_QVKO",        "QVKO\n(retrain r3)"),
    "V29":        ("PRE_V29",         "V29\n(retrain r3)"),
    "V":          ("PRE_V_default",   "V\n(retrain r3)"),
    "ALLadapter": ("CA_ALLadap_orig", "ALLadapter\n(retrain r3)"),
}

ARCH = {
    "CA":   ("#1f4e79", "Cross-attention"),
    "PRE":  ("#1e6b2e", "Prepend"),
    "MINI": ("#c55a11", "Mini cross-attn"),
    "BASE": ("#5b2d8e", "run_41 V baseline (uncond.)"),
}
ORDER = ["CA", "PRE", "MINI", "BASE"]
GRAY = "#b0b0b0"

# trainable-param annotations (kept from original; retrains share their old run's arch
# variant so param count is unchanged)
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


def isTPS_csv(fp):
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
    retr = json.load(open(RETRAINS_JSON))
    # which old labels got replaced by a retrain
    replaced = {v[0]: k for k, v in RETRAIN_REPLACES.items()}  # old_label -> retrain key

    rng = np.random.default_rng(SEED)
    rows = []
    for lab, (disp, arch) in SEL.items():
        if lab in replaced:
            # REPLACED by a retrain: use retrain's best-<=50k seq-only 50 values
            rk = replaced[lab]
            so = np.asarray(retr[rk]["values"], dtype=float)
            disp = RETRAIN_REPLACES[rk][1]
            kind = "retrain"
            note = f"step_{retr[rk]['best_step']}"
        else:
            # kept old run: seq-only per-seq values from training dir
            m = man[lab]
            so = isTPS_csv(TRAIN / m["run_folder"] / "enzyme_explorer_validation"
                           / m["step_name"]
                           / "generated_sequences_enzyme_explorer_sequence_only.csv")
            # baseline stays colored; all other old (non-retrained) runs -> grayed
            kind = "baseline" if arch == "BASE" else "grayed"
            note = m["step_name"]
        so = np.asarray(so, dtype=float)
        med = float(np.median(so))
        lo, hi = boot_ci(rng, so)
        rows.append(dict(lab=lab, disp=disp, arch=arch, kind=kind,
                         med=med, lo=lo, hi=hi, note=note))

    # sort by sequence-only median desc (consistent ranking)
    rows.sort(key=lambda d: d["med"], reverse=True)

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.33, 6.8), constrained_layout=True)
    width = 0.72
    for xi, r in zip(x, rows):
        if r["kind"] == "grayed":
            color = GRAY
        else:
            color = ARCH[r["arch"]][0]
        hatch = "//" if r["kind"] == "retrain" else None
        ax.bar(xi, r["med"], width, color=color, edgecolor="white", linewidth=0.6,
               hatch=hatch, zorder=2)
        ax.errorbar(xi, r["med"], yerr=[[r["med"] - r["lo"]], [r["hi"] - r["med"]]],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=4,
                    capthick=1.5, zorder=6)
        ax.text(xi, r["hi"] + 0.006, f"{r['med']:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        ax.text(xi, -0.30, fmtp(PARAMS[r["lab"]]), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11.5, color="#333333", fontweight="bold")

    ax.text(-0.012, -0.30, "trainable\nparams:", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color="#333333", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([r["disp"] for r in rows], rotation=40, ha="right", fontsize=12)
    ax.set_ylim(0.80, 1.0)
    ax.set_yticks(np.arange(0.80, 1.001, 0.05))
    ax.set_ylabel("median sequence-only isTPS", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    handles = [Patch(facecolor=ARCH[a][0], edgecolor="white", label=ARCH[a][1]) for a in ORDER]
    handles.append(Patch(facecolor=GRAY, edgecolor="white", label="old run, not retrained (grayed)"))
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//",
                         label="post-merge retrain (run_3 fcmerge), best ckpt ≤ 50k"))
    handles.append(Line2D([0], [0], color="black", lw=1.5, marker="_",
                          label=f"bootstrap 95% CI ({B:,} resamples, with replacement)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.40),
              ncol=2, frameon=True, framealpha=0.9, fontsize=10.5,
              title="all bars = sequence-only EnzymeExplorer isTPS median (no +structure cap)",
              title_fontsize=10.5)

    fig.suptitle("isTPS comparison (v2): 4 post-merge retrains (best ckpt ≤ 50k) "
                 "vs grayed non-retrained runs + run_41V baseline  ·  "
                 "sequence-only median, bootstrap 95% CI",
                 fontsize=13, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{OUT_PNG.name}: {Image.open(OUT_PNG).size}  ({len(rows)} runs, B={B})")

    print(f"\n{'run':18s} {'kind':9s} {'ckpt':20s} {'seq-only med [95% CI]'}")
    for r in rows:
        print(f"{r['lab']:18s} {r['kind']:9s} {r['note']:20s} "
              f"{r['med']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")


if __name__ == "__main__":
    main()
