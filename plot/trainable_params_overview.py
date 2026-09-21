"""Plot trainable-param overview for DPLM conditioning finetuning runs.

Produces two figures:
  - params_only.png       : horizontal bar chart of trainable param counts,
                            grouped by architecture, log-scale x.
  - params_vs_isTPS.png   : same horizontal bar chart on the left and a
                            paired bar chart of best median isTPS on the right,
                            sharing the y-order so each run lines up.

Also writes a tidy data.csv next to the figures.

Usage:
    /opt/miniconda3/bin/python3 trainable_params_overview.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(
    "/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
    "trainable_params_overview"
)

# (architecture, run_label, trainable_params, best_median_isTPS, best_mean_isTPS)
#
# trainable_params taken from each run's train.log "Trainable params" line
# (DPLMClass model-build report; see Runs.md 2026-05-05 / 2026-05-06 / 2026-05-21).
# best_{median,mean}_isTPS taken from each comparison's best_scores_summary.csv:
#   cross_attention_isTPS/, prepend_isTPS/,
#   mini_first_cyclization_ESM_LoRA_target_sweep_isTPS/.
ROWS = [
    # Cross-attention (class_first_cyclization, 2026-05-06 grid)
    ("cross-attention", "FT, rand init",        4_923_520, 0.9138, 0.9034),
    ("cross-attention", "FT, orig init",        4_923_520, 0.8958, 0.8950),
    ("cross-attention", "Vca + 2 LNs only",       412_800, 0.8994, 0.8810),
    ("cross-attention", "ALLadap + allV",          52_480, 0.8946, 0.8829),
    ("cross-attention", "ALLadap, rand",           14_080, 0.8917, 0.8313),
    ("cross-attention", "ALLadap, orig",           14_080, 0.9060, 0.9005),
    ("cross-attention", "Vca + LN, orig",           3_840, 0.8980, 0.8649),

    # Prepend (class_prepend_first_cyclization, 2026-05-05 grid)
    ("prepend", "QVKO",                           153_600, 0.9163, 0.8927),
    ("prepend", "QVK + lm_head",                  116_480, 0.9104, 0.8673),
    ("prepend", "QVK",                            115_200, 0.9161, 0.8565),
    ("prepend", "QV",                              76_800, 0.9050, 0.8879),
    ("prepend", "V, r=2 (a=4)",                    76_800, 0.9139, 0.8567),
    ("prepend", "V (lr=1e-3, r=1)",                38_400, 0.8962, 0.8792),
    ("prepend", "V, lr=1e-4",                      38_400, 0.8959, 0.8925),
    ("prepend", "V, lr=1e-2",                      38_400, 0.8943, 0.8781),
    ("prepend", "Q",                               38_400, 0.9016, 0.8009),
    ("prepend", "K",                               38_400, 0.9159, 0.8913),
    ("prepend", "V15-29",                          19_200, 0.8891, 0.8611),
    ("prepend", "V29",                              1_280, 0.8920, 0.8328),

    # Mini cross-attention (class_mini_first_cyclization, 2026-05-21 grid)
    ("mini cross-attention", "mini QVK",          453_280, 0.9143, 0.9101),
    ("mini cross-attention", "mini QV",           414_880, 0.8887, 0.8687),
    ("mini cross-attention", "mini V",            376_480, 0.9130, 0.8654),
    ("mini cross-attention", "mini Q",            376_480, 0.9009, 0.8987),
    ("mini cross-attention", "mini K",            376_480, 0.8964, 0.8821),
    ("mini cross-attention", "mini ltm0 (no LoRA)", 361_840, 0.8880, 0.8040),
    ("mini cross-attention", "mini V15-28",       350_880, 0.9015, 0.8764),
]

ARCH_COLOR = {
    "cross-attention":      "#d62728",  # red
    "prepend":              "#1f77b4",  # blue
    "mini cross-attention": "#2ca02c",  # green
}
ARCH_ORDER = ["cross-attention", "mini cross-attention", "prepend"]


def build_df(sort_by: str = "arch_then_params"):
    df = pd.DataFrame(
        ROWS,
        columns=["arch", "run", "trainable_params", "best_median_isTPS", "best_mean_isTPS"],
    )
    df["arch_order"] = df["arch"].map({a: i for i, a in enumerate(ARCH_ORDER)})
    if sort_by == "arch_then_params":
        df = df.sort_values(["arch_order", "trainable_params"], ascending=[True, False])
    elif sort_by == "median":
        df = df.sort_values("best_median_isTPS", ascending=False)
    elif sort_by == "mean":
        df = df.sort_values("best_mean_isTPS", ascending=False)
    else:
        raise ValueError(sort_by)
    df = df.reset_index(drop=True)
    df["color"] = df["arch"].map(ARCH_COLOR)
    return df


def fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def draw_params_bars(ax, df, draw_arch_separators=True):
    y = np.arange(len(df))[::-1]  # top-down: first row at top
    ax.barh(y, df["trainable_params"], color=df["color"], edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["run"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)", fontsize=11)
    ax.set_xlim(left=800)  # leave room for the smallest bar (1280) to be visible
    ax.grid(axis="x", which="both", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    for yi, n in zip(y, df["trainable_params"]):
        ax.text(n * 1.08, yi, fmt_params(int(n)), va="center", ha="left", fontsize=8)

    if draw_arch_separators:
        # Architecture separators between contiguous groups.
        arch_runs = df.groupby("arch_order", sort=True).size()
        cum = 0
        for ao, count in arch_runs.items():
            if cum > 0:
                sep = y[cum - 1] - 0.5
                ax.axhline(sep, color="0.7", linewidth=0.7)
            cum += count


def draw_istps_bars(
    ax,
    df,
    metric: str = "median",
    draw_arch_separators=True,
    show_yticks: bool = False,
):
    """Draw best-{median,mean} isTPS bars on `ax`, in the row order of `df`."""
    if metric == "median":
        col = "best_median_isTPS"
        label = "Best median isTPS"
        vline = 0.9163  # overall max (QVKO)
        xlim = (0.80, 0.94)
    elif metric == "mean":
        col = "best_mean_isTPS"
        label = "Best mean isTPS"
        vline = float(df[col].max())
        xlim = (0.78, 0.94)
    else:
        raise ValueError(metric)

    y = np.arange(len(df))[::-1]
    ax.barh(y, df[col], color=df["color"], edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    if show_yticks:
        ax.set_yticklabels(df["run"], fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(*xlim)
    ax.set_xlabel(label, fontsize=11)
    ax.axvline(vline, color="0.3", linestyle="--", linewidth=0.7)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    for yi, v in zip(y, df[col]):
        ax.text(v + 0.0015, yi, f"{v:.3f}", va="center", ha="left", fontsize=8)

    if draw_arch_separators:
        arch_runs = df.groupby("arch_order", sort=True).size()
        cum = 0
        for ao, count in arch_runs.items():
            if cum > 0:
                sep = y[cum - 1] - 0.5
                ax.axhline(sep, color="0.7", linewidth=0.7)
            cum += count


def arch_legend(ax):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ARCH_COLOR[a], edgecolor="black", label=a) for a in ARCH_ORDER]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_df("arch_then_params")
    df.drop(columns=["arch_order", "color"]).to_csv(OUT_DIR / "data.csv", index=False)

    # ---------- Figure 1: params only ----------
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_params_bars(ax, df)
    arch_legend(ax)
    ax.set_title(
        "Trainable parameters per DPLM conditioning finetuning run\n"
        "(3 architectures × hyperparameter sweep, DPLM-150m backbone)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "params_only.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 2: params + best median isTPS, grouped by architecture ----------
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [2.2, 1.4], "wspace": 0.03}
    )
    draw_params_bars(axL, df)
    draw_istps_bars(axR, df)
    arch_legend(axL)
    fig.suptitle(
        "Trainable parameters vs. best median isTPS per DPLM conditioning finetuning run\n"
        "(grouped by architecture; DPLM-150m backbone)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "params_vs_isTPS.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 3: params + best median isTPS, sorted by best median ----------
    df_med = build_df("median")
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [2.2, 1.4], "wspace": 0.03}
    )
    draw_params_bars(axL, df_med, draw_arch_separators=False)
    draw_istps_bars(axR, df_med, metric="median", draw_arch_separators=False)
    arch_legend(axL)
    fig.suptitle(
        "Trainable parameters vs. best median isTPS per DPLM conditioning finetuning run\n"
        "(sorted by best median isTPS; DPLM-150m backbone)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "params_vs_isTPS_sorted_by_isTPS.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 4: params + best mean isTPS, grouped by architecture ----------
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [2.2, 1.4], "wspace": 0.03}
    )
    draw_params_bars(axL, df)
    draw_istps_bars(axR, df, metric="mean")
    arch_legend(axL)
    fig.suptitle(
        "Trainable parameters vs. best mean isTPS per DPLM conditioning finetuning run\n"
        "(grouped by architecture; DPLM-150m backbone)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "params_vs_mean_isTPS.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 5: params + best mean isTPS, sorted by best mean ----------
    df_mean = build_df("mean")
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [2.2, 1.4], "wspace": 0.03}
    )
    draw_params_bars(axL, df_mean, draw_arch_separators=False)
    draw_istps_bars(axR, df_mean, metric="mean", draw_arch_separators=False)
    arch_legend(axL)
    fig.suptitle(
        "Trainable parameters vs. best mean isTPS per DPLM conditioning finetuning run\n"
        "(sorted by best mean isTPS; DPLM-150m backbone)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "params_vs_mean_isTPS_sorted_by_mean.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 6/7: 3-panel (params + mean + median), sorted by mean / by median ----------
    def draw_three_panel(df_local, sort_metric_label, out_name):
        fig, (axL, axM, axR) = plt.subplots(
            1, 3, figsize=(18, 9),
            gridspec_kw={"width_ratios": [2.2, 1.2, 1.2], "wspace": 0.04},
        )
        draw_params_bars(axL, df_local, draw_arch_separators=False)
        draw_istps_bars(axM, df_local, metric="mean", draw_arch_separators=False)
        draw_istps_bars(axR, df_local, metric="median", draw_arch_separators=False)
        arch_legend(axL)
        fig.suptitle(
            "Trainable parameters, best mean isTPS, and best median isTPS\n"
            f"per DPLM conditioning finetuning run (sorted by {sort_metric_label}; DPLM-150m)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT_DIR / out_name, dpi=170, bbox_inches="tight")
        plt.close(fig)

    draw_three_panel(df_mean, "best mean isTPS",   "params_mean_median_sorted_by_mean.png")
    draw_three_panel(df_med,  "best median isTPS", "params_mean_median_sorted_by_median.png")

    print(f"wrote: {OUT_DIR/'params_only.png'}")
    print(f"wrote: {OUT_DIR/'params_vs_isTPS.png'}")
    print(f"wrote: {OUT_DIR/'params_vs_isTPS_sorted_by_isTPS.png'}")
    print(f"wrote: {OUT_DIR/'params_vs_mean_isTPS.png'}")
    print(f"wrote: {OUT_DIR/'params_vs_mean_isTPS_sorted_by_mean.png'}")
    print(f"wrote: {OUT_DIR/'params_mean_median_sorted_by_mean.png'}")
    print(f"wrote: {OUT_DIR/'params_mean_median_sorted_by_median.png'}")
    print(f"wrote: {OUT_DIR/'data.csv'}")


if __name__ == "__main__":
    main()
