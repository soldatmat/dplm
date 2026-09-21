#!/usr/bin/env python3
"""Per-first-cyclization-class count of known MARTS-DB TPSs whose amino-acid
sequence length is in the band [280, 420] (inclusive, ~one structural domain).

Reuses the EXACT data-loading + length + palette logic of the existing
length-band figure scripts in this dir (make_pca_tsne_fig_lenband.py /
make_train_class_highlight_lenband.py):

  * the reference set is the n=1349 known TPSs returned by the kNN classifier's
    ``load_reference`` (1:1 aligned with the bundled mean-embedding CSV);
  * each row's class label is ``First_cyclization_product_id`` (y), its group
    key is ``Enzyme_marts_ID`` (groups);
  * sequence length = ``len(Aminoacid_sequence)`` from the canonical
    TPS_first_cyclization.csv, mapped per row via Enzyme_marts_ID -> len;
  * class id -> substrate type via make_pca_tsne_fig.SUBSTRATE, ordered by
    TYPE_ORDER (mono, sesqui, di, sester, sterol) then class id.

Prints a per-class table + overall totals and writes a CSV.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# canonical kNN reference loader (n=1349 known TPSs, 1:1 with embeddings)
sys.path.insert(0, "/Users/soldatmat/Documents/terpene_synthases/tps-first-cyclization-knn")
from knn_first_cyclization import load_reference, N_CLASSES  # noqa: E402

# palette / substrate-type machinery from the sibling all-class fig script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pca_tsne_fig import SUBSTRATE, TYPE_ORDER, build_class_colors  # noqa: E402

CSV_PATH = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/data-bin/"
                "MARTS-DB/2026-04-12/TPS_first_cyclization.csv")

LEN_LO, LEN_HI = 280, 420  # inclusive band, ~one structural domain

OUT_CSV = Path("/Users/soldatmat/Documents/terpene_synthases/projects/dplm/run/"
               "class_predictor/slide306_eval/class_lenband_280_420.csv")


def build_id2len() -> dict[str, int]:
    """Map Enzyme_marts_ID -> len(Aminoacid_sequence). Duplicate rows in the CSV
    share the same enzyme sequence, so the last write is idempotent."""
    id2len: dict[str, int] = {}
    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            id2len[row["Enzyme_marts_ID"]] = len(row["Aminoacid_sequence"])
    return id2len


def main() -> None:
    X, y, groups = load_reference()
    n_total = len(y)
    assert n_total == 1349, f"expected 1349 known TPSs, got {n_total}"

    id2len = build_id2len()
    lengths = np.array([id2len[g] for g in groups])
    assert len(lengths) == n_total

    in_band = (lengths >= LEN_LO) & (lengths <= LEN_HI)

    # class-id ordering: substrate type (TYPE_ORDER) then class id
    _, by_type = build_class_colors()
    ordered_ids: list[int] = []
    for t in TYPE_ORDER:
        ordered_ids.extend(by_type[t])
    # safety: include any class id present in y but not in SUBSTRATE (shouldn't happen)
    for c in sorted(set(int(v) for v in y)):
        if c not in ordered_ids:
            ordered_ids.append(c)

    rows = []
    for cid in ordered_ids:
        cls_mask = y == cid
        n_cls = int(cls_mask.sum())
        n_band = int((cls_mask & in_band).sum())
        frac = (n_band / n_cls) if n_cls else 0.0
        rows.append({
            "class_id": cid,
            "substrate_type": SUBSTRATE.get(cid, "?"),
            "n_in_band": n_band,
            "n_total_class": n_cls,
            "frac_in_band": round(frac, 4),
        })

    total_band = int(in_band.sum())
    overall_frac = total_band / n_total

    # ---- print ----
    print(f"reference: {N_CLASSES} classes, {n_total} known MARTS-DB TPSs "
          f"(source: {CSV_PATH.name} + bundled embeddings via load_reference)")
    print(f"length = len(Aminoacid_sequence); band = [{LEN_LO}, {LEN_HI}] inclusive\n")
    hdr = f"{'class_id':>8}  {'substrate_type':<14}  {'n_in_band':>9}  {'n_total':>7}  {'frac':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['class_id']:>8}  {r['substrate_type']:<14}  "
              f"{r['n_in_band']:>9}  {r['n_total_class']:>7}  {r['frac_in_band']:>6.3f}")
    print("-" * len(hdr))
    print(f"{'TOTAL':>8}  {'':<14}  {total_band:>9}  {n_total:>7}  {overall_frac:>6.3f}")
    print(f"\noverall: {total_band} / {n_total} known TPSs in [{LEN_LO},{LEN_HI}] AA "
          f"({100 * overall_frac:.1f}%)")

    # ---- write CSV ----
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["class_id", "substrate_type",
                                           "n_in_band", "n_total_class", "frac_in_band"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {OUT_CSV}")


if __name__ == "__main__":
    main()
