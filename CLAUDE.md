# CLAUDE.md — dplm (terpene-synthase fork)

Orientation for agent sessions working in this repo. The root `README.md` is the
**upstream** DPLM documentation (generic ByProt / Lightning-Hydra). This file
covers what is **project-specific**: how this fork is used for the terpene
synthase (TPS) generation sub-project, where things live, and the conventions
that aren't obvious from the code.

## What this repo is

A fork of **DPLM** (Diffusion Protein Language Model; ByProt codebase, PyTorch
Lightning + Hydra) adapted to **generate terpene-synthase enzyme sequences**,
optionally **conditioned on the first-cyclization product class**. The active
research is the class-conditioning architectures (cross-attention "CA",
"mini"-CA, and "prepend") vs. the unconditional `run_41 V` baseline.

## Where things run — IMPORTANT

- **Training / generation / embedding extraction run on Karolina (GPU)**, not
  locally. The git remote `karolina` *is* the cluster checkout:
  `karolina:/mnt/proj2/fta-26-15/documents/dplm`. SSH alias: `karolina.it4i.cz`.
  SLURM account/project: `fta-26-15`. Conda env: `/mnt/proj2/fta-26-15/.conda/envs/dplm`
  (also `enzyme_explorer`, `tps_eval`). See the **dplm-run**, **karolina-connect**,
  and **slurm-hpc** skills.
- **Plotting / deck / analysis run locally on the Mac** using the base
  `/opt/miniconda3` python3 (already has `pandas`, `sklearn`, `matplotlib`,
  `python-pptx`). The upstream `env.yml` (name `ByProt`, py37) is NOT what you use.

## Layout

- `src/byprot/` — model/task/datamodule code. Conditional-DPLM classes live in
  `src/byprot/tasks/lm/dplm_class.py` (`DPLMWithConditionalGlobalAdapter`,
  `ConditionalDPLMTrainingTask`).
- `runs/` — SLURM submit-grid + eval orchestration (sbatch templates, the
  generate/eval queue scripts, `restart_dplm_run.sh`). Submit/monitor via the
  **dplm-run** skill.
- `run/` — `generate_dplm_fixed.py` (the generation entry used for eval),
  `extract_embeddings.py`, and `run/class_predictor/` (first-cyclization-class
  fidelity eval scripts + `slide306_eval/` data & analysis).
- `plot/` — **figure code.** Per-slide metric figures follow a `plot_X.py`
  (build PNG) + `append_X.py` (add slide to the deck) pairing. Notebooks
  (`dplm_plot.ipynb`, `compare_runs_isTPS.ipynb`) are the older analysis path.
  - `plot/embedding_space/` — **space-exploration figures**: the ESM mean-embedding
    PCA/t-SNE map of training TPSs, the generated-sequence overlays, and the
    sequence-/structure-similarity maps. New embedding/landscape figure code goes
    here. See its `README.md` for the figure→slide map.
- `logs/` — training-run output folders, named
  `TPS_dplm_150m_<arch>_first_cyclization_grid_..._ltm<lora-target>` (encoding
  documented in the **dplm-run** skill).
- `data-bin/MARTS-DB/2026-04-12/` — datasets (`TPS_first_cyclization.csv` etc.)
  and `embeddings/` (extracted with the `run_41V` step-200000 encoder).
- `../presentation/dplm.pptx` — the results deck (sibling dir; DPLM is the legacy
  exception that lives at the project root, see the **tps-presentations** skill).

## External repos this depends on (do NOT re-vendor)

- **`../tps-first-cyclization-knn/`** — the **canonical** kNN first-cyclization-class
  predictor (standalone repo; GitHub `soldatmat/tps-first-cyclization-knn`,
  self-contained `data/`). Figure/eval code imports its `load_reference`,
  `KNNClassPredictor`, `N_CLASSES` via
  `sys.path.insert(0, ".../tps-first-cyclization-knn")`. A duplicate that used to
  live in `run/class_predictor/` was **retired** — do not recreate it. The repo's
  bundled embeddings are md5-identical to `data-bin/.../run_41V_step200000/`.
- **`../EnzymeExplorer/`** — isTPS scoring + ESMFold structure eval.
- **`../TPS-PRODUCT-PREDICTOR/`** — mechanism/product prediction (separate `knn.py`,
  unrelated to the first-cyclization kNN above).

## Conventions

- **Project history & run logs live in the Obsidian vault**
  (`../../notes/terpene_generation/History.md` and `Runs.md`), NOT in-repo. Record
  decisions/analysis via the **obsidian-history-notes** / **obsidian-run-notes**
  skills.
- **Deck writes:** PowerPoint must be **closed** (a `~$dplm.pptx` lock file means
  it's open). Append/replace scripts back up to `/tmp` first and title-match
  slides (robust to index shifts). Never run two pptx-writing steps concurrently.
- **Class merge:** current data has **22** first-cyclization classes (post-merge,
  2026-05-20). Pre-merge `run_1` models were 24-class and fail to load against the
  current config (`size mismatch encoder 24 vs 22`).
- **NAS mirror** (`/Volumes/data/Users/Matous/...`, pluskal.nas) is often
  unmounted off-VPN; figure scripts treat it as best-effort.

## Relevant skills

`dplm-run`, `import-dplm-runs`, `compare-dplm-runs-isTPS`, `tps-presentations`,
`karolina-connect`, `slurm-hpc`, `matplotlib-layout`, `python-pptx`.
