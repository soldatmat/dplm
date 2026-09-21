#!/usr/bin/env python3
"""Stage FASTAs for the 3-architecture structure-vs-sequence isTPS eval.

Covers every run on slides 247 (cross-attention, 7), 248 (prepend, 12) and
266 (mini cross-attention, 7 + run_41 V baseline) of dplm.pptx, each at the
best-median-isTPS checkpoint taken verbatim from the comparison
`best_steps_summary.csv` files (so the structure eval lines up 1:1 with the
seq-only bars on those slides).

Run with --verify to only check FASTAs exist (n=50, recompute median vs the
expected score); run without it to also write per-run input dirs + manifest.
"""
import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path("/mnt/proj2/fta-26-15/documents/dplm/logs")
STAGE = Path("/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08")
STAGE_INPUTS = STAGE / "inputs"

CA = "TPS_dplm_150m_class_first_cyclization_grid_run_1_lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_"
PRE1 = "TPS_dplm_150m_class_prepend_first_cyclization_grid_run_1_"
PRE_DEFAULT = "TPS_dplm_150m_class_prepend_first_cyclization_grid_lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmV"
MINI = "TPS_dplm_150m_class_mini_first_cyclization_grid_run_01_lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend0p0001_winit1em06_"
BASELINE = "TPS_dplm_150m_stage3_grid_run_41_lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend0p0001_winit1em06_loratrue_ns50_r1_a2_ltmV"

# (label, run_folder, step_dir_name, expected_best_median)
PICKS = [
    # ---- slide 247: cross-attention (7) ----
    ("CA_FT_rand",        CA + "lorafalse_ns50_r1_a2_ltm0_adapter_random",              "step_10000",  0.9138),
    ("CA_FT_orig",        CA + "lorafalse_ns50_r1_a2_ltm0_adapter_orig",                "step_20000",  0.8958),
    ("CA_Vca_LN_orig",    CA + "loratrue_ns50_r1_a2_ltmVca_adapter_orig",               "step_40000",  0.8980),
    ("CA_ALLadap_rand",   CA + "loratrue_ns50_r1_a2_ltmALLadapter_adapter_random",      "step_80000",  0.8917),
    ("CA_ALLadap_orig",   CA + "loratrue_ns50_r1_a2_ltmALLadapter_adapter_orig",        "step_20000",  0.9060),
    ("CA_Vca_2LN_only",   CA + "lorafalse_ns50_r1_a2_ltm0_adapter_orig_Vca_LN_only",    "step_70000",  0.8994),
    ("CA_ALLadap_allV",   CA + "loratrue_ns50_r1_a2_ltmALLadapter+allV_adapter_orig",   "step_100000", 0.8946),
    # ---- slide 248: prepend (12) ----
    ("PRE_QV",            PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmQV",      "step_50000", 0.9050),
    ("PRE_V29",           PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmV29",     "step_90000", 0.8920),
    ("PRE_V15to29",       PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmV15to29", "step_70000", 0.8891),
    ("PRE_V_default",     PRE_DEFAULT,                                                                                            "step_50000", 0.8962),
    ("PRE_V_lr1e4",       PRE1 + "lr1em4_wu2000_ts200000_ckpt10000_valee10000_lend1em5_winit1em7_loratrue_ns50_r1_a2_ltmV",       "step_50000", 0.8959),
    ("PRE_V_lr1e2",       PRE1 + "lr1em2_wu2000_ts200000_ckpt10000_valee10000_lend1em3_winit1em5_loratrue_ns50_r1_a2_ltmV",       "step_50000", 0.8943),
    ("PRE_V_r2a4",        PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r2_a4_ltmV",       "step_10000", 0.9139),
    ("PRE_QVK",           PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmQVK",     "step_10000", 0.9161),
    ("PRE_QVKO",          PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmQVKO",    "step_10000", 0.9163),
    ("PRE_QVK_lmhead",    PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmQVKLH",   "step_20000", 0.9104),
    ("PRE_K",             PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmK",       "step_60000", 0.9159),
    ("PRE_Q",             PRE1 + "lr1em3_wu2000_ts200000_ckpt10000_valee10000_lend1em4_winit1em6_loratrue_ns50_r1_a2_ltmQ",       "step_10000", 0.9016),
    # ---- slide 266: mini cross-attention (7) + baseline ----
    ("MINI_ltm0",         MINI + "lorafalse_ns50_r1_a2_ltm0",  "sanity_check_step_0", 0.8880),
    ("MINI_V",            MINI + "loratrue_ns50_r1_a2_ltmV",   "step_20000",          0.9130),
    ("MINI_Q",            MINI + "loratrue_ns50_r1_a2_ltmQ",   "step_10000",          0.9009),
    ("MINI_K",            MINI + "loratrue_ns50_r1_a2_ltmK",   "step_10000",          0.8964),
    ("MINI_QV",           MINI + "loratrue_ns50_r1_a2_ltmQV",  "step_10000",          0.8887),
    ("MINI_QVK",          MINI + "loratrue_ns50_r1_a2_ltmQVK", "step_30000",          0.9143),
    ("MINI_V15to28",      MINI + "loratrue_ns50_r1_a2_ltmV15to29", "step_30000",      0.9015),
    ("BASELINE_run41_V",  BASELINE,                            "step_120000",         0.8311),
]


def read_fasta(path):
    entries, header, seq = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq)))
                header, seq = line[1:].strip(), []
            elif line:
                seq.append(line)
        if header is not None:
            entries.append((header, "".join(seq)))
    return entries


def median_istps(csv_path):
    """Recompute median of the trailing-space 'isTPS ' column."""
    with open(csv_path) as f:
        r = csv.DictReader(f)
        col = next((c for c in r.fieldnames if c.strip() == "isTPS"), None)
        if col is None:
            return None
        vals = [float(row[col]) for row in r if row[col] not in ("", None)]
    return statistics.median(vals) if vals else None


def write_fasta(entries, path):
    with open(path, "w") as f:
        for h, s in entries:
            f.write(f">{h}\n{s}\n")


def write_csv(entries, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "sequence"])
        for h, s in entries:
            w.writerow([h, s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check only, do not stage")
    args = ap.parse_args()

    if not args.verify:
        STAGE_INPUTS.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    ok = True
    print(f"{'label':22s} {'n':>3s} {'len':>4s} {'med':>6s} {'exp':>6s} {'Δ':>6s}  status")
    for label, run_folder, step_name, exp_med in PICKS:
        step_sub = f"enzyme_explorer_validation/{step_name}"
        src = ROOT / run_folder / step_sub / "generated_sequences.fasta"
        csv_src = ROOT / run_folder / step_sub / "generated_sequences_enzyme_explorer_sequence_only.csv"
        if not src.exists():
            ok = False
            print(f"{label:22s} {'--':>3s} {'--':>4s} {'--':>6s} {exp_med:6.3f} {'--':>6s}  MISSING FASTA: {src}")
            continue
        entries = read_fasta(src)
        n = len(entries)
        lens = {len(s) for _, s in entries}
        lstr = str(next(iter(lens))) if len(lens) == 1 else f"{min(lens)}-{max(lens)}"
        med = median_istps(csv_src) if csv_src.exists() else None
        if med is None:
            status = "no seq-only csv (median uncheckable)"
            dstr = "  n/a"
        else:
            d = med - exp_med
            dstr = f"{d:+.3f}"
            status = "OK" if abs(d) < 0.01 else "!! MEDIAN MISMATCH"
            if abs(d) >= 0.01:
                ok = False
        if n != 50:
            status += f" !! n={n}"
            ok = False
        medstr = f"{med:6.3f}" if med is not None else "  n/a"
        print(f"{label:22s} {n:3d} {lstr:>4s} {medstr} {exp_med:6.3f} {dstr:>6s}  {status}")

        if not args.verify:
            out_dir = STAGE_INPUTS / label
            out_dir.mkdir(parents=True, exist_ok=True)
            write_fasta(entries, out_dir / "sequences.fasta")
            write_csv(entries, out_dir / "sequences.csv")
            manifest_rows.append({
                "label": label,
                "run_folder": run_folder,
                "step_subpath": step_sub,
                "step_name": step_name,
                "n_seqs": n,
                "expected_median": exp_med,
                "recomputed_median": "" if med is None else round(med, 4),
                "fasta_path": str(out_dir / "sequences.fasta"),
                "csv_path": str(out_dir / "sequences.csv"),
                "source_fasta": str(src),
            })

    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED — fix before staging'} "
          f"({len(PICKS)} runs)")
    if not args.verify and manifest_rows:
        STAGE.mkdir(parents=True, exist_ok=True)
        mpath = STAGE / "manifest.csv"
        with open(mpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
        print(f"Staged {len(manifest_rows)} runs. Manifest: {mpath}")


if __name__ == "__main__":
    main()
