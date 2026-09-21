#!/usr/bin/env python3
"""Draw the adaLN-single conditioning schematic for DPLM (full-bleed diagram for
the deck). Output: dplm/_adaln_early_readout/adaln_architecture_diagram.png"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = "/Users/soldatmat/Documents/terpene_synthases/dplm/_adaln_early_readout/adaln_architecture_diagram.png"
COND='#2e86de'; MOD='#e67e22'; ORG='#a85b16'; BLK='#34495e'; FRZ='#eef2f4'


def box(ax,x,y,w,h,text,fc='white',ec=BLK,fs=10.5,bold=False,tc='black',lw=1.6,rs=0.04):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle=f"round,pad=0.02,rounding_size={rs}",fc=fc,ec=ec,lw=lw))
    ax.text(x,y,text,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',color=tc,zorder=5)


def arr(ax,x1,y1,x2,y2,c=BLK,lw=1.8,style='-|>',ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=15,lw=lw,color=c,ls=ls,shrinkA=2,shrinkB=2,zorder=4))


def main():
    fig,ax=plt.subplots(figsize=(13.33,7.0)); ax.set_xlim(0,13.33); ax.set_ylim(0,7.0); ax.axis('off')
    ax.text(6.66,6.72,"How adaLN-single conditioning steers DPLM",ha='center',va='center',fontsize=18,fontweight='bold')
    # LEFT conditioning path
    box(ax,2.25,5.85,3.7,0.72,"First-cyclization product class\n(1 of 22 classes)",fc='#eaf2fb',ec=COND,bold=True,tc=COND)
    arr(ax,2.25,5.49,2.25,5.13,c=COND)
    box(ax,2.25,4.78,3.7,0.6,"Class embedding   c ∈ ℝ⁶⁴⁰",fc='#eaf2fb',ec=COND,fs=11)
    arr(ax,2.25,4.48,2.25,4.06,c=COND)
    box(ax,2.25,3.46,3.95,1.02,"adaLN-single modulation MLP\nLinear(640→64) → SiLU → Linear(64→3840)\n+ per-layer offset table (zero-init)",fc='#fdf0e2',ec=MOD,fs=9.5,bold=True,tc=ORG)
    arr(ax,2.25,2.95,2.25,2.58,c=MOD)
    box(ax,2.25,2.2,3.95,0.74,"for every layer ℓ →\nshift, scale, gate  ×2  (attn + FFN)",fc='#fdf0e2',ec=MOD,fs=9.6,tc=ORG)
    arr(ax,4.27,2.3,6.15,3.35,c=MOD,lw=2.6)
    ax.text(5.25,3.12,"6 params\nper layer",ha='center',va='center',fontsize=8.6,color=ORG,style='italic')
    # RIGHT transformer block
    bx0,by0,bx1,by1=6.4,0.9,12.95,5.95
    ax.add_patch(FancyBboxPatch((bx0,by0),bx1-bx0,by1-by0,boxstyle="round,pad=0.02,rounding_size=0.04",fc=FRZ,ec=BLK,lw=1.8,ls='--'))
    ax.text((bx0+bx1)/2,by1-0.27,"ESM-2 / DPLM transformer block   × 30   (backbone FROZEN)",ha='center',va='center',fontsize=11,fontweight='bold',color=BLK)
    cx=8.75
    ax.text(cx,5.2,"hidden states  x",ha='center',va='center',fontsize=9.5)
    arr(ax,cx,5.05,cx,4.92)
    box(ax,cx,4.68,2.6,0.46,"LayerNorm",ec=MOD,lw=2.0,fs=10); ax.text(cx+2.05,4.68,"× (1+scale₁) + shift₁",ha='center',va='center',fontsize=8.4,color=ORG,style='italic')
    arr(ax,cx,4.45,cx,4.30)
    box(ax,cx,4.06,3.0,0.5,"Multi-Head Self-Attention",fc='#e3eaef',fs=10)
    arr(ax,cx,3.81,cx,3.66)
    ax.add_patch(Circle((cx,3.5),0.15,fc='white',ec=MOD,lw=1.8,zorder=6)); ax.text(cx,3.5,"⊙",ha='center',va='center',fontsize=12,color=MOD,zorder=7); ax.text(cx+1.55,3.5,"× gate₁",ha='center',va='center',fontsize=8.4,color=ORG,style='italic')
    arr(ax,cx,3.34,cx,3.18)
    ax.add_patch(Circle((cx,3.0),0.16,fc='white',ec=BLK,lw=1.5,zorder=6)); ax.text(cx,3.0,"+",ha='center',va='center',fontsize=14,zorder=7)
    arr(ax,cx-2.0,4.92,cx-2.0,3.0,c=BLK,lw=1.1); arr(ax,cx-2.0,3.0,cx-0.16,3.0,c=BLK,lw=1.1)
    ax.text(cx-2.18,3.95,"residual",ha='center',va='center',fontsize=7.8,color=BLK,rotation=90)
    arr(ax,cx,2.84,cx,2.66)
    box(ax,cx,2.18,4.7,0.82,"FFN sub-block  —  same pattern:\nLayerNorm → ×(1+scale₂)+shift₂  →  FFN  →  × gate₂  →  ⊕ residual",fc='#f6ede2',ec=MOD,fs=8.8,tc=ORG)
    arr(ax,cx,1.77,cx,1.55); ax.text(cx,1.33,"→ next layer",ha='center',va='center',fontsize=9,style='italic',color=BLK)
    # footer
    ax.add_patch(FancyBboxPatch((0.3,0.1),12.73,0.5,boxstyle="round,pad=0.02,rounding_size=0.03",fc='#f7f9fa',ec='#bbb',lw=1))
    ax.text(6.66,0.35,"zero-init gates ⇒ no-op at init (stable finetune)   •   only adaLN MLP + class embedding train ≈ 0.42M params   •   "
            "no timestep input (DPLM is time-agnostic)   •   CFG: class dropped 15% → learned null ⇒ guidance dial w",
            ha='center',va='center',fontsize=8.5,color='#333')
    fig.savefig(OUT,dpi=200,bbox_inches='tight'); print("saved",OUT)


if __name__ == "__main__":
    main()
