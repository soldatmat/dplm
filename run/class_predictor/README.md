# First-cyclization-class predictor (kNN)

Independent predictor that maps a TPS sequence to one of the 22 first-cyclization
classes, so we can measure **conditioning fidelity** of the class-conditional DPLM
models (does conditioning on class *k* actually produce class-*k* sequences?) —
the question isTPS cannot answer. See the 2026-06-08 History.md entry for why this
exists.

## How it works
- **Embedder:** mean per-sequence embedding from the unconditional DPLM-150m base
  (stage3 `run_41 V`, step 200000) — independent of the class-conditional models
  under comparison, so it doesn't leak the conditioning signal.
- **Classifier:** distance-weighted kNN over the 1349 labelled MARTS-DB rows
  (`TPS_first_cyclization.csv`, `First_cyclization_product_id` ∈ 0..21).
- Reference embeddings already live at
  `dplm/data-bin/MARTS-DB/2026-04-12/embeddings/dplm_150m_stage3_grid_run_41V_step200000/`.

## Usage
```bash
# Validate on real TPSs (grouped leave-one-enzyme-out CV; sweeps metric/k):
python knn_first_cyclization.py --validate          # -> validation/

# Classify a query embedding CSV (id + 640 dims, same format as reference):
python knn_first_cyclization.py --predict QUERY_EMB.csv --k 3 --metric euclidean --out preds.csv
```
To embed generated sequences for `--predict`, run `dplm/run/extract_embeddings.py`
with the **same** run_41V checkpoint and `--embedding_type mean`.

## Validation verdict (held-out real TPSs)
Best config **euclidean, k=3**: strict acc **0.58**, lenient acc **0.79**
(predicted ∈ enzyme's true class set — fair for multi-product enzymes),
macro-recall **0.45**, vs majority-class baseline **0.17**. See
`validation/confusion_euclidean_k3.png`, `best_per_class_recall.csv`, `cv_grid.json`.

The predictor is **strong on well-populated classes and weak on rare ones** (a
class with 1–5 training members can't be recalled under leave-one-enzyme-out).
This bounds how much the fidelity numbers can be trusted **per class**.

### Trustworthy classes for the fidelity sweep
Pick conditioning classes with strict recall ≥ 0.6 and support ≥ 37 — there a
"predicted == conditioned" readout is meaningful:

| class | support | strict recall | notes |
|---|---|---|---|
| 0  | 227 | 0.63 | germacradienyl cation — current anchor |
| 9  | 127 | 0.98 | cleanest separation |
| 12 | 90  | 0.99 | cleanest separation |
| 1  | 116 | 0.70 | |
| 5  | 90  | 0.72 | |
| 17 | 37  | 0.78 | |

**Avoid** for strict fidelity claims: rare classes (3, 7, 13, 16, 19, 21; support ≤ 5,
recall ≈ 0) and the multi-label-confused **11** and **20** (strict recall 0.28 / 0.19
but lenient 0.80 / 0.84 — they co-occur with other classes and the predictor leaks
to those neighbours).

## Caveats / future improvements
- The embedder is DPLM-150m, not raw ESM-2; a raw-ESM-2 baseline would test whether
  the conditioning-base embedding biases the predictor (worth a quick check before trusting).
- kNN over an imbalanced set favours majority classes; class-balanced voting or a
  trained head could lift macro-recall.
- Multi-product enzymes make strict per-row accuracy pessimistic — the lenient number
  is the honest upper bound for single-label fidelity interpretation.

## Where the eval output went (2026-09-21)

`slide306_eval/` and `validation/` — 624 M / 339 files of steering / CFG / adaLN
eval output backing deck slides #306 and #384–388 — are **git-ignored** and archived
to **pluskal.nas** at `terpene_synthases/projects/dplm/class_predictor_2026-06-11/`
(byte- and count-verified 2026-09-21, 339 files / 653,936,291 B both sides).

They had no durable home at all until then: unignored, un-archived, and the single
largest piece of bulk output in the project. This README is deliberately kept tracked
so the archive stays discoverable from the repo.
