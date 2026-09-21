#!/usr/bin/env python3
"""Slide-309 v3: isTPS comparison restoring the +ESMFold STRUCTURE cap (original
slide-308 style) for all 10 bars, with the 4 post-merge "fcmerge" run_3 retrains
(best ckpt <= 50k) replacing their old prepend/CA counterparts.

Per bar (ported from plot_struct_vs_seq_crossarch_bootstrap.py):
  - dark lower segment = sequence-only EnzymeExplorer isTPS median
  - light cap          = gain from ESMFold structure (bar top = structure median)
  - bootstrap 95% CI (B=10,000, with replacement, 50 per-seq values) on BOTH the
    seq-only median (dark-segment top) and the structure median (bar top)
Sorted by structure median desc.

Colours:
  - FULL architecture colour for: baseline + 3 minis + 4 retrains.
  - 4 retrains additionally get a "//" hatch + "(retrain, <=50k)" hint.
  - CA_FT_rand / CA_FT_orig (old run_1 full-finetune CA, NOT retrained, pre-fcmerge)
    are rendered as a FAINT TINT of the CA colour (~75% toward white, muted thin
    edge, slightly lower alpha) so the CA hue is still perceptible but the bars
    read as de-emphasized/secondary.

Data sources:
  - reused 6 bars (baseline, CA_FT_rand, CA_FT_orig, 3 minis):
      seq-only  : NAS TRAIN/<run_folder>/enzyme_explorer_validation/<step>/
                  generated_sequences_enzyme_explorer_sequence_only.csv
      structure : NAS OUTDIR/inputs/<label>/sequences_enzyme_explorer.csv
  - 4 retrains:
      seq-only  : retrains_le50k.json  (best-<=50k 50 per-seq values)
      structure : retrains_structure.json  (50 per-seq isTPS from the
                  structure_eval_retrains_2026-06-12 staging EE output)
"""
import csv
import json
from pathlib import Path

import numpy as np
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
RETRAINS_STRUCT_JSON = HERE / "retrains_structure.json"
OUT_PNG = HERE / "slide309_v3_isTPS_compare.png"

B = 10000
SEED = 0

# --- the 6 REUSED bars (NAS data). label -> (display, arch, de-emphasized?) ---
# de-emphasized = old run_1 full-FT CA, pre-fcmerge, NOT retrained.
REUSE = {
    "BASELINE_run41_V": ("run_41 V\n(baseline)", "BASE", False),
    "CA_FT_rand":       ("FT, rand",             "CA",   True),
    "CA_FT_orig":       ("FT, orig",             "CA",   True),
    "MINI_QVK":         ("QVK",                  "MINI", False),
    "MINI_V15to28":     ("V15-28",               "MINI", False),
    "MINI_ltm0":        ("ltm0",                 "MINI", False),
}

# --- the 4 RETRAINS. retrain-key -> (display, arch) ---
RETRAIN = {
    "QVKO":       ("QVKO\n(retrain, ≤50k)",       "PRE"),
    "V29":        ("V29\n(retrain, ≤50k)",        "PRE"),
    "V":          ("V\n(retrain, ≤50k)",          "PRE"),
    "ALLadapter": ("ALLadapter\n(retrain, ≤50k)", "CA"),
}

# (dark, light, legend label) per architecture — from the original slide-308 source
ARCH = {
    "CA":   ("#1f4e79", "#9dc3e6", "Cross-attention"),
    "PRE":  ("#1e6b2e", "#a5d6a7", "Prepend"),
    "MINI": ("#c55a11", "#f4b183", "Mini cross-attn"),
    "BASE": ("#5b2d8e", "#c3a6e0", "run_41 V baseline (uncond.)"),
}
ORDER = ["CA", "PRE", "MINI", "BASE"]

PARAMS = {
    "CA_FT_rand": 4923520, "CA_FT_orig": 4923520,
    "MINI_QVK": 453280, "MINI_V15to28": 350880, "MINI_ltm0": 361840,
    "BASELINE_run41_V": 38400,
    # retrains share their old run's arch variant param count
    "QVKO": 153600, "V29": 1280, "V": 38400, "ALLadapter": 14080,
}


def fade(hexcolor, toward_white=0.75):
    """Blend a colour ~75% toward white (faint tint, hue still perceptible)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hexcolor)
    return tuple(c + (1.0 - c) * toward_white for c in (r, g, b))


def fmtp(n):
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    return f"{n / 1e3:.1f}k"


def isTPS_csv(fp):
    df_cols = None
    with open(fp) as f:
        r = csv.reader(f)
        header = next(r)
        cols = [c.strip() for c in header]
        idx = [i for i, c in enumerate(cols) if c.lower() == "istps"][0]
        vals = [float(row[idx]) for row in r if row and row[idx] not in ("", None)]
    return np.asarray(vals, dtype=float)


def boot_ci(rng, x, b=B, q=(2.5, 97.5)):
    boots = rng.choice(x, size=(b, len(x)), replace=True)
    med = np.median(boots, axis=1)
    return np.percentile(med, q)


def main():
    man = {r["label"]: r for r in csv.DictReader(open(MANIFEST))}
    retr = json.load(open(RETRAINS_JSON))
    if not RETRAINS_STRUCT_JSON.exists():
        raise SystemExit(f"missing {RETRAINS_STRUCT_JSON} — run the structure pull first")
    retr_st = json.load(open(RETRAINS_STRUCT_JSON))

    rng = np.random.default_rng(SEED)
    rows = []

    # reused 6 bars
    for lab, (disp, arch, faded) in REUSE.items():
        m = man[lab]
        so = isTPS_csv(TRAIN / m["run_folder"] / "enzyme_explorer_validation"
                       / m["step_name"]
                       / "generated_sequences_enzyme_explorer_sequence_only.csv")
        st = isTPS_csv(OUTDIR / "inputs" / lab / "sequences_enzyme_explorer.csv")
        rows.append(_row(rng, lab, disp, arch, so, st,
                         kind=("faded" if faded else "kept"),
                         pkey=lab))

    # 4 retrains
    for rk, (disp, arch) in RETRAIN.items():
        so = np.asarray(retr[rk]["values"], dtype=float)
        st = np.asarray(retr_st[rk]["values"], dtype=float)
        rows.append(_row(rng, rk, disp, arch, so, st, kind="retrain", pkey=rk))

    rows.sort(key=lambda d: d["st"], reverse=True)  # by structure median desc

    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(13.33, 6.8), constrained_layout=True)
    width = 0.72
    for xi, r in zip(x, rows):
        dark, light, _ = ARCH[r["arch"]]
        if r["kind"] == "faded":
            dark_c, light_c = fade(dark, 0.75), fade(light, 0.80)
            edge, alpha, lw = fade(dark, 0.45), 0.85, 0.8
        else:
            dark_c, light_c = dark, light
            edge, alpha, lw = "white", 1.0, 0.6
        hatch = "//" if r["kind"] == "retrain" else None
        ax.bar(xi, r["so"], width, color=dark_c, edgecolor=edge, linewidth=lw,
               alpha=alpha, hatch=hatch, zorder=2)
        ax.bar(xi, r["st"] - r["so"], width, bottom=r["so"], color=light_c,
               edgecolor=edge, linewidth=lw, alpha=alpha, hatch=hatch, zorder=2)
        ax.errorbar(xi, r["st"], yerr=[[r["st"] - r["st_lo"]], [r["st_hi"] - r["st"]]],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=4,
                    capthick=1.5, zorder=6)
        ax.errorbar(xi, r["so"], yerr=[[r["so"] - r["so_lo"]], [r["so_hi"] - r["so"]]],
                    fmt="none", ecolor="black", elinewidth=1.5, capsize=4,
                    capthick=1.5, zorder=6)
        ax.text(xi, r["st_hi"] + 0.006, f"{r['st']:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))
        so_color = "#555555" if r["kind"] == "faded" else "white"
        ax.text(xi, r["so_lo"] - 0.006, f"{r['so']:.3f}", ha="center", va="top",
                fontsize=10, color=so_color)
        ax.text(xi, -0.30, fmtp(PARAMS[r["pkey"]]), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11.5, color="#333333", fontweight="bold")

    ax.text(-0.012, -0.30, "trainable\nparams:", transform=ax.transAxes,
            ha="right", va="top", fontsize=10.5, color="#333333", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels([r["disp"] for r in rows], rotation=40, ha="right", fontsize=12)
    ax.set_ylim(0.80, 1.0)
    ax.set_yticks(np.arange(0.80, 1.001, 0.05))
    ax.set_ylabel("median isTPS", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    handles = [Patch(facecolor=ARCH[a][0], edgecolor="white", label=ARCH[a][2]) for a in ORDER]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="//",
                         label="post-merge retrain (run_3 fcmerge), best ckpt ≤ 50k"))
    handles.append(Patch(facecolor=fade(ARCH["CA"][0], 0.75),
                         edgecolor=fade(ARCH["CA"][0], 0.45),
                         label="faded = old run, pre-fcmerge (not retrained)"))
    handles.append(Line2D([0], [0], color="black", lw=1.5, marker="_",
                          label=f"bootstrap 95% CI ({B:,} resamples, with replacement)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.40),
              ncol=3, frameon=True, framealpha=0.9, fontsize=10.0,
              title="per bar:  dark = sequence-only    ·    light cap = + ESMFold structure",
              title_fontsize=10.5)

    fig.suptitle("isTPS comparison (v3): 4 post-merge retrains (best ≤50k) + minis + baseline  ·  "
                 "seq-only + ESMFold structure  ·  bootstrap 95% CI  ·  faded = old pre-fcmerge CA",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{OUT_PNG.name}: {Image.open(OUT_PNG).size}  ({len(rows)} runs, B={B})")

    print(f"\n{'run':20s} {'kind':8s} {'seq-only med [95% CI]':30s} {'struct med [95% CI]'}")
    for r in rows:
        print(f"{r['lab']:20s} {r['kind']:8s} "
              f"{r['so']:.3f} [{r['so_lo']:.3f},{r['so_hi']:.3f}]"
              f"      {r['st']:.3f} [{r['st_lo']:.3f},{r['st_hi']:.3f}]")


def _row(rng, lab, disp, arch, so, st, kind, pkey):
    so = np.asarray(so, dtype=float)
    st = np.asarray(st, dtype=float)
    so_med, st_med = float(np.median(so)), float(np.median(st))
    so_lo, so_hi = boot_ci(rng, so)
    st_lo, st_hi = boot_ci(rng, st)
    return dict(lab=lab, disp=disp, arch=arch, kind=kind, pkey=pkey,
                so=so_med, st=st_med, so_lo=so_lo, so_hi=so_hi,
                st_lo=st_lo, st_hi=st_hi)


if __name__ == "__main__":
    main()
