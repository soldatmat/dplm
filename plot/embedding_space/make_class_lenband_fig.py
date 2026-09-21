#!/usr/bin/env python3
"""How many known MARTS-DB TPSs per first-cyclization class fall in the
280-420 AA single-alpha-domain band? Horizontal bar chart, grouped by
substrate type (carbon order), coloured with the class palette. Bar length =
n_in_band; label = n_in_band / n_total_class (frac%)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_pca_tsne_fig import make_palette, TYPE_ORDER, N_CLASSES  # noqa

CSV = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
           "slide306_eval/class_lenband_280_420.csv")
OUT = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/class_predictor/"
           "slide306_eval/class_lenband_280_420.png")
BAND = (280, 420)


def main():
    df = pd.read_csv(CSV)
    # order: substrate type (carbon order) then class id, top-to-bottom
    df["torder"] = df["substrate_type"].map({t: i for i, t in enumerate(TYPE_ORDER)})
    df = df.sort_values(["torder", "class_id"]).reset_index(drop=True)
    cmap = make_palette(N_CLASSES)

    n = len(df)
    ypos = np.arange(n)[::-1]            # first row at top
    colors = [cmap(int(c)) for c in df["class_id"]]

    fig, ax = plt.subplots(figsize=(11.5, 7.4), constrained_layout=True)
    ax.barh(ypos, df["n_in_band"], color=colors, edgecolor="black", linewidth=0.5, height=0.74, zorder=3)

    xmax = df["n_in_band"].max()
    for y, (_, r) in zip(ypos, df.iterrows()):
        ax.text(r["n_in_band"] + xmax * 0.012, y,
                f"{int(r['n_in_band'])}/{int(r['n_total_class'])}  ({r['frac_in_band']*100:.0f}%)",
                va="center", ha="left", fontsize=8.5, zorder=4)
    ax.set_xlim(0, xmax * 1.22)

    ax.set_yticks(ypos)
    ax.set_yticklabels([f"class {int(c)}" for c in df["class_id"]], fontsize=9)

    # substrate-type group brackets on the far left (axes-y in data coords)
    trans = ax.get_yaxis_transform()  # x = axes fraction, y = data
    for t in TYPE_ORDER:
        idx = df.index[df["substrate_type"] == t].to_numpy()
        if not len(idx):
            continue
        ys = ypos[idx]
        top, bot = ys.max() + 0.45, ys.min() - 0.45
        ax.plot([-0.16, -0.16], [bot, top], transform=trans, color="0.3", lw=1.4,
                clip_on=False, zorder=2)
        ax.text(-0.175, (top + bot) / 2, t, transform=trans, rotation=90,
                ha="center", va="center", fontsize=10, fontweight="bold", clip_on=False)

    ax.set_xlabel(f"# known MARTS-DB TPSs with sequence length in [{BAND[0]}, {BAND[1]}] AA", fontsize=10.5)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="0.88", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    total = int(df["n_total_class"].sum()); inband = int(df["n_in_band"].sum())
    fig.suptitle(f"Known MARTS-DB TPSs in the 280–420 AA single-α-domain band, per first-cyclization class\n"
                 f"{inband} of {total} TPSs ({inband/total*100:.1f}%) fall in the band  ·  bar = count in band, "
                 f"label = in-band / class-total (% of class)",
                 fontsize=12, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}  (total={total}, in_band={inband}, {inband/total*100:.1f}%)")


if __name__ == "__main__":
    main()
