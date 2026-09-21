#!/usr/bin/env python3
"""Side-by-side architecture diagrams (slide-307 style) for prepend and mini cross-attention.

Reuses the helpers/colours from plot_ca_architecture.py so the look matches slide 307.

Verified against code + training logs:
- prepend (encoder_conditioning_mode=prepend): NO adapter layer; the frozen 640-d class
  embedding is prepended as an extra token at position 0 of the embedded sequence
  (dplm_modeling_esm.py:289-303), the whole ESM stack runs normally, and the class-token
  position is stripped from the output (:319). Only trained params = LoRA r=1.
- mini (encoder_conditioning_mode=mini_cross_attention): same as cross-attention but the
  layer-30 adapter is a bottleneck: cross-attn (K/V = 640-d class emb) -> downsize 640->80
  -> adapter FFN (hidden 80) -> upsize 80->640 -> residual.
"""
from plot_ca_architecture import (box, arrow, newcanvas, legend, OUTDIR,
                                  FROZEN_F, FROZEN_E, ADAPT_F, ADAPT_E,
                                  CLS_F, CLS_E, LORA_F, LORA_E, IO_F, IO_E, INK)


def original_stack(ax, cx):
    ax.text(cx, 53.5, "Original DPLM (150M)", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    box(ax, cx, 5, 26, 4.2, "noised sequence tokens", IO_F, IO_E, 10)
    box(ax, cx, 12.5, 26, 4.2, "token + position embedding", FROZEN_F, FROZEN_E, 10)
    box(ax, cx, 25, 22, 9, "ESM encoder\n× 30 layers\n(self-attn + FFN)", FROZEN_F, FROZEN_E, 11)
    box(ax, cx, 38.5, 26, 4.2, "final LayerNorm", FROZEN_F, FROZEN_E, 10)
    box(ax, cx, 45.5, 26, 4.2, "LM head  →  token logits", FROZEN_F, FROZEN_E, 10)
    for a, b in zip([5, 12.5, 20.4, 29.6, 38.5], [12.5, 20.4, 29.6, 38.5, 45.5]):
        arrow(ax, cx, a + 2.1, cx, b - 2.1)


def variant_prepend():
    fig, ax = newcanvas()
    fig.suptitle("Prepend class conditioning vs. original DPLM", fontsize=16, fontweight="bold")
    original_stack(ax, 24)
    cx = 70
    ax.text(cx, 53.5, "+ class conditioning (prepend)", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    box(ax, cx, 5, 26, 4.2, "noised sequence tokens", IO_F, IO_E, 10)
    box(ax, cx, 12, 26, 4.2, "token + position embedding", FROZEN_F, FROZEN_E, 10)
    box(ax, cx, 19, 27, 4.6, "PREPEND class-embedding token\nat position 0", CLS_F, CLS_E, 9.5, bold=True)
    box(ax, cx, 28, 26, 6.6, "ESM encoder × 30 layers\n(frozen, + LoRA r=1)\nall tokens attend to the class token", FROZEN_F, FROZEN_E, 9.5)
    box(ax, cx, 38.5, 26, 4.2, "final LayerNorm", FROZEN_F, FROZEN_E, 10)
    box(ax, cx, 45.5, 26, 4.2, "LM head  →  token logits", FROZEN_F, FROZEN_E, 10)
    for a, b in zip([5, 12, 19, 28, 38.5], [12, 19, 28, 38.5, 45.5]):
        arrow(ax, cx, a + 2.2, cx, b - 2.2)
    ax.text(cx + 14.3, 31.3, "class-token position\nstripped from the output", ha="left", va="center",
            fontsize=8.3, color=FROZEN_E, fontstyle="italic")

    # class encoder feeding the prepended token
    box(ax, 93, 19, 13, 9, "class id\n↓\nClassEncoder\n(FROZEN,\nBOS-mean init)\n↓\nclass embedding\nd = 640", CLS_F, CLS_E, 9)
    arrow(ax, 86.3, 19, 83.7, 19, color=CLS_E, lw=2.2)

    ax.plot([47, 47], [3, 51], color="#cccccc", lw=1.2, ls="--", zorder=0)
    legend(ax, [("frozen original DPLM", FROZEN_F, FROZEN_E),
                ("LoRA r=1 (trained)", LORA_F, LORA_E),
                ("class encoder + prepended token (frozen)", CLS_F, CLS_E)])
    out = OUTDIR / "arch_prepend_sidebyside.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    from PIL import Image; print(f"{out.name}: {Image.open(out).size}")
    return out


def variant_mini():
    fig, ax = newcanvas()
    fig.suptitle("Mini cross-attention class conditioning vs. original DPLM", fontsize=16, fontweight="bold")
    original_stack(ax, 24)
    cx = 70
    ax.text(cx, 53.5, "+ class conditioning (mini cross-attn)", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    box(ax, cx, 5, 26, 4.2, "noised sequence tokens", IO_F, IO_E, 10)
    box(ax, cx, 12.5, 26, 4.2, "token + position embedding", FROZEN_F, FROZEN_E, 10)
    box(ax, cx, 22, 24, 6.6, "ESM layers 1–29\n(frozen, + LoRA r=1)", FROZEN_F, FROZEN_E, 10)
    box(ax, cx, 30.5, 24, 6.6, "layer 30 = MINI cross-attn ADAPTER\n(trained, 640→80→640 bottleneck)", ADAPT_F, ADAPT_E, 9, bold=True)
    box(ax, cx, 38.5, 26, 4.2, "final LayerNorm", FROZEN_F, FROZEN_E, 10)
    box(ax, cx, 45.5, 26, 4.2, "LM head  →  token logits", FROZEN_F, FROZEN_E, 10)
    for a, b in zip([5, 12.5, 18.7, 33.8, 38.5], [12.5, 18.7, 33.8, 38.5, 45.5]):
        arrow(ax, cx, a + 2.1, cx, b - 2.1)

    box(ax, 93, 30.5, 13, 9.5, "class id\n↓\nClassEncoder\n(FROZEN,\nBOS-mean init)\n↓\nclass embedding\nd = 640", CLS_F, CLS_E, 9)
    arrow(ax, 86.3, 30.5, 82.2, 30.5, color=CLS_E, lw=2.2)
    ax.text(84.2, 32.6, "K / V", ha="center", fontsize=9, color=CLS_E, fontstyle="italic")

    ax.plot([47, 47], [3, 51], color="#cccccc", lw=1.2, ls="--", zorder=0)
    legend(ax, [("frozen original DPLM", FROZEN_F, FROZEN_E),
                ("LoRA r=1 (trained)", LORA_F, LORA_E),
                ("new adapter (trained)", ADAPT_F, ADAPT_E),
                ("class encoder (frozen)", CLS_F, CLS_E)])
    out = OUTDIR / "arch_mini_sidebyside.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    from PIL import Image; print(f"{out.name}: {Image.open(out).size}")
    return out


if __name__ == "__main__":
    variant_prepend()
    variant_mini()
