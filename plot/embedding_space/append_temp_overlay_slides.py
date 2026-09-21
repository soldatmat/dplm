#!/usr/bin/env python3
"""Append TWO slides (run_41V T=2.0 and T=5.0 embedding-space overlays) to
presentation/dplm.pptx (template slide 232). Slide-342 twins.
Run with PowerPoint CLOSED. Only APPENDS."""
from __future__ import annotations
import datetime, shutil
from copy import deepcopy
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

ROOT = "/Users/soldatmat/Documents/terpene_synthases"
PPTX = f"{ROOT}/presentation/dplm.pptx"
EMB = f"{ROOT}/dplm/plot/embedding_space"
TEMPLATE_SLIDE_INDEX = 232; LAYOUT_INDEX = 1
TITLE_SHAPE = "TextovéPole 11"; NUM_SHAPE = "TextovéPole 12"; ANNOT_SHAPE = "TextBox 5"
PIC_SLOT = (0.12, 0.96, 8.48, 6.35)

SLIDES = [
    dict(
        fig=f"{EMB}/run41V_t2.0_overlay_pca_tsne_umap.png",
        title=("run_41V (step-200000) at temperature T=2.0 — 50 generated seqs in the "
                "MARTS-DB TPS embedding space (PCA / t-SNE / UMAP; space fit on train, "
                "generated projected in; cf. slide 342 = T=1.0)"),
        annot=("Grey = 1349 MARTS-DB\nTPSs (mean-pooled,\nrun_41V step-200000).\n\n"
               "Green = 50 run_41V seqs\nsampled at T=2.0\n(same recipe as the\n"
               "slide-342 T=1.0 baseline,\nL=350, gumbel_argmax,\n500 iter).\n\n"
               "Same train-only basis\n(PC1 41.4%, PC2 17.3%).\n\n"
               "Effect of temperature:\nat T=1.0 (slide 342) gen sat\nON the train cloud;\n"
               "at T=2.0 it is pushed FAR\nOFF — median PC1-PC2\nradius 33.3 vs train 15.8\n"
               "(2.1x) — and COLLAPSES to\none tight off-manifold\ncluster (isolated blob\n"
               "in t-SNE & UMAP).\n\n"
               "More gumbel noise ->\nnear-degenerate seqs the\nencoder maps to a single\n"
               "corner, not more on-manifold\ndiversity.")),
    dict(
        fig=f"{EMB}/run41V_t5.0_overlay_pca_tsne_umap.png",
        title=("run_41V (step-200000) at temperature T=5.0 — 50 generated seqs in the "
                "MARTS-DB TPS embedding space (PCA / t-SNE / UMAP; space fit on train, "
                "generated projected in; cf. slide 342 = T=1.0)"),
        annot=("Grey = 1349 MARTS-DB\nTPSs (mean-pooled,\nrun_41V step-200000).\n\n"
               "Purple = 50 run_41V seqs\nsampled at T=5.0.\n\n"
               "Same train-only basis\n(PC1 41.4%, PC2 17.3%).\n\n"
               "T=5.0 is nearly identical\nto T=2.0: same far-off\ntight cluster (radius 34.0\n"
               "vs train 15.8, 2.1x).\n\n"
               "=> Temperature SATURATES:\nbeyond T~2 the sampler is\nalready near-random, so\n"
               "raising it further does NOT\nspread generations across\nthe manifold — it just\n"
               "parks them in the same\noff-manifold corner.\n\n"
               "Diversity within the TPS\nmanifold is NOT achievable\nby cranking temperature;\n"
               "it needs a better\nconditioning/sampler.")),
]


def fit(img, slot):
    sl, st, sw, sh = slot
    with Image.open(img) as im: iw, ih = im.size
    r, sr = iw/ih, sw/sh
    w, h = (sw, sw/r) if r > sr else (sh*r, sh)
    return sl+(sw-w)/2, st+(sh-h)/2, w, h


def set_text(tf, text):
    lines = text.split("\n"); p0 = tf.paragraphs[0]
    if p0.runs:
        p0.runs[0].text = lines[0]
        for e in p0.runs[1:]: e._r.getparent().remove(e._r)
    else: p0.text = lines[0]
    for ep in list(tf.paragraphs[1:]): ep._p.getparent().remove(ep._p)
    for ln in lines[1:]: tf.add_paragraph().text = ln


def add_slide(prs, spec):
    tmpl = prs.slides[TEMPLATE_SLIDE_INDEX]
    t = prs.slides.add_slide(prs.slide_layouts[LAYOUT_INDEX])
    for ph in list(t.placeholders):
        ph.element.getparent().remove(ph.element)
    for sh in tmpl.shapes:
        if sh.shape_type == 13: continue
        t.shapes._spTree.insert_element_before(deepcopy(sh.element), "p:extLst")
    l, tp, w, h = fit(spec["fig"], PIC_SLOT)
    t.shapes.add_picture(spec["fig"], Inches(l), Inches(tp), Inches(w), Inches(h))
    for sh in t.shapes:
        if sh.shape_type == 13 or not sh.has_text_frame: continue
        if sh.name == TITLE_SHAPE: set_text(sh.text_frame, spec["title"])
        elif sh.name == NUM_SHAPE: set_text(sh.text_frame, "")
        elif sh.name == ANNOT_SHAPE: set_text(sh.text_frame, spec["annot"])


def main():
    if list(Path(PPTX).parent.glob("~$*.pptx")):
        raise SystemExit("ABORT: PowerPoint lock present — close PowerPoint.")
    for s in SLIDES:
        if not Path(s["fig"]).exists():
            raise SystemExit(f"missing {s['fig']}")
    bak = f"/tmp/dplm_pptx_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.pptx"
    shutil.copy2(PPTX, bak); print("backup ->", bak)
    prs = Presentation(PPTX)
    for s in SLIDES:
        add_slide(prs, s)
    prs.save(PPTX)
    print(f"appended {len(SLIDES)} temp-overlay slides; total {len(prs.slides)}")


if __name__ == "__main__":
    main()
