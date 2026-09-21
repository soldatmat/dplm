#!/usr/bin/env python3
"""Append the 3 multiclass-steering-sweep overlay slides (one per mini model) to
presentation/dplm.pptx (template slide 232 layout). Run with PowerPoint CLOSED.

Modeled exactly on append_bos_conditioning_slide.py."""
from __future__ import annotations
import datetime, shutil
from copy import deepcopy
from pathlib import Path
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

ROOT = "/Users/soldatmat/Documents/terpene_synthases"
PPTX = f"{ROOT}/presentation/dplm.pptx"
FIG_DIR = (f"{ROOT}/dplm/run/class_predictor/slide306_eval/"
           "multiclass_sweep_2026-06-12")
TEMPLATE_SLIDE_INDEX = 232; LAYOUT_INDEX = 1
TITLE_SHAPE = "TextovéPole 11"; NUM_SHAPE = "TextovéPole 12"; ANNOT_SHAPE = "TextBox 5"
PIC_SLOT = (0.12, 0.96, 8.48, 6.35)

# one (model, fig, title, annot) per slide
SLIDES = [
    ("MINI_QVK", "gen_multiclass_overlay_MINI_QVK.png"),
    ("MINI_V15to28", "gen_multiclass_overlay_MINI_V15to28.png"),
    ("MINI_ltm0", "gen_multiclass_overlay_MINI_ltm0.png"),
]

TITLE_FMT = ("Multiclass steering sweep — {model}: generated seqs vs MARTS-DB "
             "TPS embedding space (PCA fit on train only, t-SNE co-embedded)")
ANNOT = ("Grey = all 1349 MARTS-DB\nTPSs (mean-pooled,\nrun_41V step-200000).\n\n"
         "Coloured points = this\nmodel's 300 generated\nseqs (6 conditioning\n"
         "classes {0,1,5,9,12,17}\nx 50), coloured by their\nTARGET class.\n\n"
         "PCA is fit on the training\nTPSs ONLY; gen seqs are\nprojected into that fixed\n"
         "basis (PC1 41.5%, PC2\n17.3% var, = slide 315).\nt-SNE is recomputed on\n"
         "train + gen together.\n\n"
         "Generated points do NOT\nseparate by target class\n(~17% land nearest their\n"
         "target-class centroid,\n~chance for 6 classes):\nconsistent with the\n"
         "weak/no-steering finding.")


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


def main():
    if list(Path(PPTX).parent.glob("~$*.pptx")):
        raise SystemExit("ABORT: PowerPoint lock present — close it.")
    for _, fn in SLIDES:
        if not Path(f"{FIG_DIR}/{fn}").exists():
            raise SystemExit(f"missing {FIG_DIR}/{fn}")
    bak = f"/tmp/dplm_pptx_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}.pptx"
    shutil.copy2(PPTX, bak); print("backup ->", bak)

    prs = Presentation(PPTX)
    tmpl = prs.slides[TEMPLATE_SLIDE_INDEX]
    appended = []
    for model, fn in SLIDES:
        fig = f"{FIG_DIR}/{fn}"
        t = prs.slides.add_slide(prs.slide_layouts[LAYOUT_INDEX])
        for ph in list(t.placeholders):
            ph.element.getparent().remove(ph.element)
        for sh in tmpl.shapes:
            if sh.shape_type == 13: continue
            t.shapes._spTree.insert_element_before(deepcopy(sh.element), "p:extLst")
        l, tp, w, h = fit(fig, PIC_SLOT)
        t.shapes.add_picture(fig, Inches(l), Inches(tp), Inches(w), Inches(h))
        for sh in t.shapes:
            if sh.shape_type == 13 or not sh.has_text_frame: continue
            if sh.name == TITLE_SHAPE: set_text(sh.text_frame, TITLE_FMT.format(model=model))
            elif sh.name == NUM_SHAPE: set_text(sh.text_frame, "")
            elif sh.name == ANNOT_SHAPE: set_text(sh.text_frame, ANNOT)
        appended.append(len(prs.slides))

    prs.save(PPTX)
    print(f"appended multiclass-sweep slides at #{appended}; total {len(prs.slides)}")


if __name__ == "__main__":
    main()
