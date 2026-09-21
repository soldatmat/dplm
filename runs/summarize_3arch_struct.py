#!/usr/bin/env python3
"""Struct-vs-seq isTPS median per run for the 3-arch structure eval."""
import csv
import os
import statistics
from pathlib import Path

SE = Path(os.environ.get("SE_STAGE", "/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08"))


def med(p):
    r = csv.DictReader(open(p))
    col = next((c for c in r.fieldnames if c.strip() == "isTPS"), None)
    v = [float(x[col]) for x in r if x[col] not in ("", None)]
    return statistics.median(v) if v else None


GROUPS = {"CA": "slide 247  cross-attention", "PRE": "slide 248  prepend",
          "MINI": "slide 266  mini cross-attn", "BAS": "slide 266  baseline (run_41 V)"}
ORDER = {"CA": 0, "PRE": 1, "MINI": 2, "BAS": 3}


def main():
    man = list(csv.DictReader(open(SE / "manifest.csv")))
    rows = []
    for m in man:
        lab = m["label"]
        seq = float(m["recomputed_median"] or m["expected_median"])
        s = med(SE / "inputs" / lab / "sequences_enzyme_explorer.csv")
        g = "BAS" if lab.startswith("BASELINE") else lab.split("_")[0]
        rows.append((g, lab, seq, s))
    rows.sort(key=lambda r: (ORDER[r[0]], -r[3]))

    out = SE / "struct_vs_seq_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "label", "seq_median", "struct_median", "delta"])
        for g, lab, seq, s in rows:
            w.writerow([GROUPS[g], lab, round(seq, 4), round(s, 4), round(s - seq, 4)])

    cg = None
    print(f"{'run':20s} {'seq':>6s} {'struct':>7s} {'delta':>7s}")
    for g, lab, seq, s in rows:
        if g != cg:
            print(f"--- {GROUPS[g]} ---")
            cg = g
        print(f"{lab:20s} {seq:6.3f} {s:7.3f} {s-seq:+7.3f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
