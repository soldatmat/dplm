#!/usr/bin/env python3
"""Three diagram variants of the cross-attention conditioning architecture vs original DPLM.

A: side-by-side stacks (original DPLM  |  cross-attention conditioned)
B: zoom into the modified last layer (the adapter block internals)
C: data-flow with frozen/trained colour coding + delta callout

Cross-attention runs (slide 247/300): encoder_conditioning_mode = cross_attention.
The last ESM layer (layer 30, idx 29) is replaced by a GlobalAdapterLayer: original
self-attn + FFN, then a NEW cross-attention block where sequence tokens (Q) attend to
the class embedding (K/V) from a frozen ClassEncoder, then an adapter FFN + residual.
Base ESM frozen with LoRA r=1; adapter trained.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

OUTDIR = Path("/Volumes/data/Users/Matous/terpene_synthases/output/dplm/comparison/"
              "ca_architecture_2026-06-09")
OUTDIR.mkdir(parents=True, exist_ok=True)

# colour roles
FROZEN_F, FROZEN_E = "#dedede", "#7f7f7f"      # frozen original DPLM
ADAPT_F, ADAPT_E = "#f6b26b", "#b45f06"        # new adapter (trained)
CLS_F, CLS_E = "#9fc5e8", "#1f4e79"            # class encoder (frozen, new path)
LORA_F, LORA_E = "#b6d7a8", "#38761d"          # LoRA (trained)
IO_F, IO_E = "#ffffff", "#404040"              # i/o nodes
INK = "#202020"


def box(ax, cx, cy, w, h, text, fc, ec, fs=10, bold=False, ls="-", lw=1.4, round=True):
    style = "round,pad=0.02,rounding_size=1.2" if round else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=style,
                                linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.8, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                                shrinkA=1, shrinkB=1), zorder=1)


def newcanvas():
    fig, ax = plt.subplots(figsize=(13.33, 7.5), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def legend(ax, items, y=1.5, fs=11):
    handles = [Patch(facecolor=f, edgecolor=e, label=l) for l, f, e in items]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=len(items), frameon=False, fontsize=fs)


# ----------------------------------------------------------------------------- A
def variant_A():
    fig, ax = newcanvas()
    fig.suptitle("Cross-attention class conditioning vs. original DPLM",
                 fontsize=16, fontweight="bold")

    def stack(cx, title, conditioned):
        ax.text(cx, 53.5, title, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
        ys = [5, 12.5, 20, 30.5, 38.5, 45.5]
        box(ax, cx, ys[0], 26, 4.2, "noised sequence tokens", IO_F, IO_E, 10)
        box(ax, cx, ys[1], 26, 4.2, "token + position embedding", FROZEN_F, FROZEN_E, 10)
        if not conditioned:
            box(ax, cx, 25, 22, 9, "ESM encoder\n× 30 layers\n(self-attn + FFN)", FROZEN_F, FROZEN_E, 11)
        else:
            box(ax, cx, 22, 24, 6.6, "ESM layers 1–29\n(frozen, + LoRA r=1)", FROZEN_F, FROZEN_E, 10)
            box(ax, cx, 30.5, 24, 6.6, "layer 30 = cross-attention\nADAPTER  (trained)", ADAPT_F, ADAPT_E, 10, bold=True)
        box(ax, cx, ys[4], 26, 4.2, "final LayerNorm", FROZEN_F, FROZEN_E, 10)
        box(ax, cx, ys[5], 26, 4.2, "LM head  →  token logits", FROZEN_F, FROZEN_E, 10)
        # arrows
        chain = [ys[0], ys[1]]
        chain += ([25 - 4.5, 25 + 4.5] if not conditioned else [18.7, 33.8])
        chain += [ys[4], ys[5]]
        seq = [ys[0], ys[1]] + ([20.5, 29.5] if not conditioned else [18.7, 33.8]) + [ys[4], ys[5]]
        pts = [ys[0], ys[1], (20.4 if not conditioned else 18.7), (29.6 if not conditioned else 33.8), ys[4], ys[5]]
        for a, b in zip(pts, pts[1:]):
            arrow(ax, cx, a + 2.1, cx, b - 2.1)

    stack(24, "Original DPLM (150M)", False)
    stack(70, "+ class conditioning", True)

    # class encoder feeding the adapter on the right
    box(ax, 93, 30.5, 13, 9.5, "class id\n↓\nClassEncoder\n(FROZEN,\nBOS-mean init)\n↓\nclass embedding\n(640-d)", CLS_F, CLS_E, 9)
    arrow(ax, 86.3, 30.5, 82.2, 30.5, color=CLS_E, lw=2.2)
    ax.text(84.2, 32.6, "K / V", ha="center", fontsize=9, color=CLS_E, fontstyle="italic")

    # divider
    ax.plot([47, 47], [3, 51], color="#cccccc", lw=1.2, ls="--", zorder=0)

    legend(ax, [("frozen original DPLM", FROZEN_F, FROZEN_E),
                ("LoRA r=1 (trained)", LORA_F, LORA_E),
                ("new adapter (trained)", ADAPT_F, ADAPT_E),
                ("class encoder (frozen)", CLS_F, CLS_E)])
    out = OUTDIR / "ca_arch_A_sidebyside.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# ----------------------------------------------------------------------------- B
def variant_B():
    fig, ax = newcanvas()
    fig.suptitle("Cross-attention conditioning — inside the modified last layer (layer 30)",
                 fontsize=15, fontweight="bold")

    # trunk at bottom
    box(ax, 33, 5, 46, 4.6, "DPLM trunk: embedding + ESM layers 1–29   (frozen, + LoRA r=1)",
        FROZEN_F, FROZEN_E, 10)
    arrow(ax, 33, 7.3, 33, 11.3)

    # original part of layer 30 (grey)
    box(ax, 33, 14, 30, 4.4, "self-attention  (original)", FROZEN_F, FROZEN_E, 10)
    arrow(ax, 33, 16.2, 33, 19.3)
    box(ax, 33, 21.5, 30, 4.4, "feed-forward  (original)", FROZEN_F, FROZEN_E, 10)
    ax.text(50.5, 17.8, "original ESM layer\n(self-attn + FFN)", ha="left", va="center",
            fontsize=9, color=FROZEN_E, fontstyle="italic")

    # adapter block region
    ax.add_patch(FancyBboxPatch((11.5, 26.4), 43, 21.5, boxstyle="round,pad=0.2,rounding_size=1.5",
                 linewidth=1.6, edgecolor=ADAPT_E, facecolor="#fff4e8", linestyle="--", zorder=0))
    ax.text(33, 46.4, "ADAPTER  (newly added, trained)", ha="center", fontsize=10.5,
            fontweight="bold", color=ADAPT_E)

    arrow(ax, 33, 23.7, 33, 29.0)
    box(ax, 33, 31.2, 31, 4.6, "cross-attention\nQ = sequence tokens,   K/V = class embedding", ADAPT_F, ADAPT_E, 9.5, bold=True)
    arrow(ax, 33, 33.5, 33, 36.7)
    box(ax, 33, 38.8, 26, 4.4, "adapter LayerNorm + FFN", ADAPT_F, ADAPT_E, 9.5)
    # residual
    arrow(ax, 33, 41.0, 33, 49.2)
    ax.annotate("", xy=(46.5, 44), xytext=(46.5, 23.9),
                arrowprops=dict(arrowstyle="-|>", color=ADAPT_E, lw=1.6,
                                connectionstyle="arc3,rad=0.0"), zorder=1)
    ax.plot([46.5, 33], [23.9, 23.9], color=ADAPT_E, lw=1.6, zorder=1)
    ax.plot([46.5, 33], [44, 44], color=ADAPT_E, lw=1.6, zorder=1)
    ax.text(48.2, 34, "residual", ha="left", va="center", fontsize=8.5, color=ADAPT_E, fontstyle="italic")

    # class encoder on the left feeding K/V
    box(ax, 84, 31.2, 22, 13, "class id\n↓\nClassEncoder  (FROZEN)\nEmbedding(24, 640)\nBOS-mean per class\n↓\nclass embedding (640-d)\ncopied to every token", CLS_F, CLS_E, 9)
    arrow(ax, 72.8, 31.2, 48.7, 31.2, color=CLS_E, lw=2.2)

    # top: out
    box(ax, 33, 51.3, 30, 4.4, "final LayerNorm  →  LM head  →  logits", FROZEN_F, FROZEN_E, 10)

    legend(ax, [("frozen original DPLM", FROZEN_F, FROZEN_E),
                ("new adapter (trained)", ADAPT_F, ADAPT_E),
                ("class encoder (frozen)", CLS_F, CLS_E)])
    out = OUTDIR / "ca_arch_B_lastlayer.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


# ----------------------------------------------------------------------------- C
def variant_C():
    fig, ax = newcanvas()
    fig.suptitle("Cross-attention class conditioning — data flow & what is trained",
                 fontsize=15, fontweight="bold")

    Y = 33  # main left-to-right sequence flow
    box(ax, 10, Y, 15, 5.4, "noised\nsequence\ntokens", IO_F, IO_E, 9.5)
    box(ax, 30, Y, 20, 7, "DPLM trunk\nESM layers 1–29\n(frozen + LoRA r=1)", FROZEN_F, FROZEN_E, 9.5)
    box(ax, 55, Y, 22, 9, "layer 30:\nCROSS-ATTENTION\nADAPTER  (trained)\nseq = Q,  class = K/V", ADAPT_F, ADAPT_E, 9.5, bold=True)
    box(ax, 80, Y, 16, 6, "final LayerNorm\n→ LM head", FROZEN_F, FROZEN_E, 9.5)
    box(ax, 95, Y, 11, 5.4, "denoised\nsequence", IO_F, IO_E, 9.5)
    arrow(ax, 17.6, Y, 19.8, Y)
    arrow(ax, 40.2, Y, 43.8, Y)
    arrow(ax, 66.2, Y, 71.8, Y)
    arrow(ax, 88.2, Y, 89.3, Y)

    # class branch entering the adapter from the top
    box(ax, 55, 51, 24, 4.6, "class id  (first-cyclization)", IO_F, IO_E, 9.5)
    box(ax, 55, 43, 28, 5, "ClassEncoder  (FROZEN)\nBOS-mean per class  →  640-d", CLS_F, CLS_E, 9.5)
    arrow(ax, 55, 48.6, 55, 45.6, color=CLS_E)
    arrow(ax, 55, 40.4, 55, 37.7, color=CLS_E, lw=2.2)
    ax.text(57, 39, "K / V", ha="left", va="center", fontsize=9, color=CLS_E, fontstyle="italic")

    # delta callout
    ax.add_patch(FancyBboxPatch((6, 4), 88, 13.5, boxstyle="round,pad=0.3,rounding_size=1.5",
                 linewidth=1.3, edgecolor="#999999", facecolor="#fbfbfb", zorder=0))
    ax.text(9, 15.3, "Changes vs. original DPLM:", ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.text(9, 12.2,
            "•  add a frozen ClassEncoder  (class id → 640-d embedding, BOS-mean per class)\n"
            "•  replace the last ESM layer with a cross-attention adapter  (trained from ESM-init)\n"
            "•  every sequence token cross-attends to the class embedding  (sequence = Q,  class = K/V)\n"
            "•  base ESM frozen, adapted with LoRA r=1;  embedding / earlier layers / LM head unchanged",
            ha="left", va="top", fontsize=9.6, color=INK)

    legend(ax, [("frozen (ESM base, ClassEncoder)", FROZEN_F, FROZEN_E),
                ("LoRA r=1 (trained)", LORA_F, LORA_E),
                ("new adapter (trained)", ADAPT_F, ADAPT_E)])
    out = OUTDIR / "ca_arch_C_dataflow.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    from PIL import Image
    for fn in (variant_A, variant_B, variant_C):
        out = fn()
        print(f"{out.name}: {Image.open(out).size}")


if __name__ == "__main__":
    main()
