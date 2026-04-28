#!/bin/bash
#SBATCH -A fta-26-15
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=dplm_from_ckpt

# Example: launch a DPLM training job that initializes the model weights from a
# local checkpoint, with a FRESH optimizer / LR scheduler / global_step / epoch.
#
# How this differs from runs/restart_dplm_run.sh:
#   - restart_dplm_run.sh uses train.ckpt_path -> Lightning's full resume (model
#     arch must match exactly, optimizer + scheduler + global_step are restored
#     from the ckpt).
#   - this script uses train.init_weights_from -> only the model state_dict is
#     loaded (see src/byprot/training_pipeline.py). The loader normalizes PEFT
#     decorations ('base_model.model.' prefix and '.base_layer.' for wrapped
#     Linears), so source and target may differ in LoRA topology and the base
#     weights still transfer. Target keys with no source value (e.g. fresh
#     lora_A / lora_B) stay at their default init.
#   - LoRA -> no-LoRA: by default (train.init_weights_merge_lora=auto) the source
#     LoRA delta is folded into base_layer.weight at load time using
#     alpha/rank read from the ckpt's hyper_parameters. Set merge_lora=never
#     in EXTRA_OVERRIDES to get the unmerged base weights only (deltas dropped).
#   - Vanilla DPLM ckpt -> altered DPLMClass arch (class_first / class_mini /
#     class_prepend): set train.init_weights_source_arch=dplm. That rewrites
#     source 'model.net.*' keys to 'model.decoder.net.*' so the ESM stack
#     transfers; the new ClassEncoder, conditional adapter layers (e.g.
#     adapter_crossattention) and any fresh LoRA slots stay at default init.
#     For an arbitrary remap, pass train.init_weights_key_remap='{src.: dst.}'
#     (Hydra dict syntax) instead.
#
# Karolina (Slurm):
#     sbatch runs/start_dplm_run_from_ckpt.sh
#
# Override defaults via env vars + --export=ALL, e.g.:
#     EXP=tps/TPS_dplm_150m_stage3 \
#       RUN_NAME=tps_from_run31 \
#       MODEL_CHECKPOINT=/abs/path/to/some.ckpt \
#       EXTRA_OVERRIDES="train.lr=1e-5 trainer.max_steps=40000" \
#       sbatch --export=ALL runs/start_dplm_run_from_ckpt.sh
#
# MetaCentrum (PBS): replace the #SBATCH lines above with:
#     #PBS -N dplm_from_ckpt
#     #PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=46gb:mem=64gb:scratch_local=40gb
#     #PBS -l walltime=24:00:00
# Don't keep both header sets in one file -- Karolina's Slurm submit plugin
# parses #PBS lines and chokes on PBS-style memory units like '64gb'.

set -euo pipefail

# ---- User-Configurable Variables ----

EXP="${EXP:-tps/TPS_dplm_150m_stage3}"
RUN_NAME="${RUN_NAME:-TPS_dplm_150m_stage3_from_ckpt_example}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-/path/to/some.ckpt}"
DATADIR="${DATADIR:-/mnt/proj2/fta-26-15/documents/dplm}"
TRAIN_ENV="${TRAIN_ENV:-/mnt/proj2/fta-26-15/.conda/envs/dplm}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,}"

# ---- Logging & Scratch Setup ----

run_log_dir="$DATADIR/logs/${RUN_NAME}"
mkdir -p "$run_log_dir"

job_id="${PBS_JOBID:-${SLURM_JOB_ID:-unknown}}"

# Use $SCRATCHDIR if set (MetaCentrum); else mktemp under /tmp (Karolina, etc.).
if [[ -n "${SCRATCHDIR:-}" ]]; then
    work_scratch="$SCRATCHDIR"
else
    work_scratch="$(mktemp -d /tmp/dplm_${RUN_NAME}_XXXXXX)"
    trap 'rm -rf "$work_scratch"' EXIT
fi

{
    echo "$job_id is running on node $(hostname -f) in work scratch dir $work_scratch"
    echo "Init weights checkpoint: $MODEL_CHECKPOINT"
} >> "$run_log_dir/job_info.txt"

mkdir -p "$work_scratch/tmp"
export TMPDIR="$work_scratch/tmp"

# Pre-create dest and copy contents to avoid nesting (data-bin/data-bin) on retries.
mkdir -p "$work_scratch/dplm/data-bin"
cp -r "$DATADIR/data-bin/." "$work_scratch/dplm/data-bin/" || { echo >&2 "Error while copying data-bin!"; exit 2; }

# Stage the source checkpoint OUTSIDE the run log dir so it isn't copied back at
# the end. Use a distinct filename so Lightning doesn't mistake it for a resume target.
init_weights_path="$work_scratch/dplm/init_weights.ckpt"
cp "$MODEL_CHECKPOINT" "$init_weights_path" || { echo >&2 "Error while staging init weights!"; exit 2; }

scratch_log_dir="$work_scratch/dplm/logs/${RUN_NAME}"
mkdir -p "$scratch_log_dir"

# ---- Activate Environment ----

module add mambaforge >/dev/null 2>&1 || true
ml Anaconda3 >/dev/null 2>&1 || true

if command -v conda >/dev/null 2>&1; then
    if [[ "$TRAIN_ENV" == /* ]]; then
        PY_RUNNER=(conda run --no-capture-output -p "$TRAIN_ENV")
    else
        PY_RUNNER=(conda run --no-capture-output -n "$TRAIN_ENV")
    fi
elif command -v mamba >/dev/null 2>&1; then
    if [[ "$TRAIN_ENV" == /* ]]; then
        PY_RUNNER=(mamba run --no-capture-output -p "$TRAIN_ENV")
    else
        PY_RUNNER=(mamba run --no-capture-output -n "$TRAIN_ENV")
    fi
else
    echo >&2 "Neither conda nor mamba is available after module setup."
    exit 3
fi

# ---- Run Training ----

cd "$DATADIR"

# Re-tokenize EXTRA_OVERRIDES through bash so quoted values like "'(...)'"
# (e.g. lora_target_module regex) survive. Env-var expansion alone does
# word-splitting only, not quote interpretation.
eval "extra_args=( ${EXTRA_OVERRIDES} )"

# train.ckpt_path=null disables Lightning's resume path so optimizer / LR
# scheduler / global_step start fresh. train.init_weights_from is consumed in
# byprot/training_pipeline.py to load only the model state_dict (strict=False).
"${PY_RUNNER[@]}" python train.py \
    experiment="$EXP" \
    name="$RUN_NAME" \
    paths.data_dir="$work_scratch/dplm/data-bin" \
    paths.log_dir="$scratch_log_dir" \
    train.ckpt_path=null \
    train.init_weights_from="$init_weights_path" \
    "${extra_args[@]}"

# ---- Copy Results Back ----

cp -r "$scratch_log_dir" "$DATADIR/logs/" || { echo >&2 "Result file(s) copying failed (with a code $?)!"; exit 4; }

if command -v clean_scratch >/dev/null 2>&1; then
    clean_scratch
fi
