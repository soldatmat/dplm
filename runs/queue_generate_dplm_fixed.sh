#!/usr/bin/env bash
#SBATCH --job-name=dplmgen8k20
#SBATCH --nodes=1
#SBATCH --partition=qgpu
#SBATCH --account=fta-26-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/proj2/fta-26-15/documents/dplm/logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/proj2/fta-26-15/documents/dplm}"
CHECKPOINT="${CHECKPOINT:-/mnt/proj2/fta-26-15/documents/dplm/logs/TPS_dplm_150m_stage3_run_8/checkpoints/N-Step-Checkpoint_epoch=172_step=20000.ckpt}"
CSV_PATH="${CSV_PATH:-/mnt/proj2/fta-26-15/documents/output/dplm/sampled_lengths_1000.csv}"
SAVE_DIR="${SAVE_DIR:-/mnt/proj2/fta-26-15/documents/output/dplm/TPS_dplm_150m_stage3_run_8_step20000/sl1000_t1.0-generate_dplm_fixed}"
CONDA_ENV="${CONDA_ENV:-/mnt/proj2/fta-26-15/.conda/envs/dplm}"
# Persistent data path used by load_yaml_config to replace any stale
# ${paths.data_dir} interpolation baked into the saved training cfg.
export DPLM_DATA_DIR="${DPLM_DATA_DIR:-$PROJECT_ROOT/data-bin}"
PYTHON_BIN="$CONDA_ENV/bin/python"
INLINE_SEQ_LENS="${INLINE_SEQ_LENS:-}"
INLINE_NUM_SEQS="${INLINE_NUM_SEQS:-}"

# Optional generate_dplm_fixed.py flags. Empty => use the script's defaults.
GEN_SEED="${GEN_SEED:-}"
GEN_ARCHITECTURE="${GEN_ARCHITECTURE:-}"
# Space-separated list of class ids (DPLMClass only). Example: "0" or "0 1 2".
GEN_CLASS_IDS="${GEN_CLASS_IDS:-}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"
GEN_SAMPLING_STRATEGY="${GEN_SAMPLING_STRATEGY:-}"
GEN_MAX_ITER="${GEN_MAX_ITER:-}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-}"
GEN_BATCH_LENS_TOGETHER="${GEN_BATCH_LENS_TOGETHER:-}"  # set to "true" or "false" to override; empty => script default

# Cache locations (shared across runs to avoid re-downloads and home-dir issues).
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/triton}"
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
    export HF_HOME="$HF_CACHE_DIR"
    export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
    export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
fi
if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then
    export TRITON_CACHE_DIR
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found in env: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -n "$INLINE_SEQ_LENS" || -n "$INLINE_NUM_SEQS" ]]; then
    if [[ -z "$INLINE_SEQ_LENS" || -z "$INLINE_NUM_SEQS" ]]; then
        echo "Both INLINE_SEQ_LENS and INLINE_NUM_SEQS must be set when using inline values." >&2
        exit 1
    fi
    read -r -a SEQ_LENS <<< "$INLINE_SEQ_LENS"
    read -r -a NUM_SEQS <<< "$INLINE_NUM_SEQS"
else
    if [[ ! -f "$CSV_PATH" ]]; then
        echo "CSV not found: $CSV_PATH" >&2
        exit 1
    fi

readarray -t parsed_csv < <("$PYTHON_BIN" - <<'PY'
import csv
import os

csv_path = os.environ["CSV_PATH"]
lengths = []
counts = []

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        lengths.append(str(int(row["length"])))
        counts.append(str(int(row["count"])))

print(" ".join(lengths))
print(" ".join(counts))
PY
)

    if [[ ${#parsed_csv[@]} -ne 2 ]]; then
        echo "Failed to parse CSV into seq_lens and num_seqs." >&2
        exit 1
    fi

    read -r -a SEQ_LENS <<< "${parsed_csv[0]}"
    read -r -a NUM_SEQS <<< "${parsed_csv[1]}"
fi

if [[ ${#SEQ_LENS[@]} -eq 0 || ${#SEQ_LENS[@]} -ne ${#NUM_SEQS[@]} ]]; then
    echo "Parsed seq_lens/num_seqs are invalid: ${#SEQ_LENS[@]} vs ${#NUM_SEQS[@]}" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

GEN_ARGS=(
    --model_name "$CHECKPOINT"
    --no-from_huggingface
    --saveto "$SAVE_DIR"
    --temperature "$GEN_TEMPERATURE"
    --seq_lens "${SEQ_LENS[@]}"
    --num_seqs "${NUM_SEQS[@]}"
)
[[ -n "$GEN_SEED"              ]] && GEN_ARGS+=( --seed "$GEN_SEED" )
[[ -n "$GEN_ARCHITECTURE"      ]] && GEN_ARGS+=( --architecture "$GEN_ARCHITECTURE" )
[[ -n "$GEN_SAMPLING_STRATEGY" ]] && GEN_ARGS+=( --sampling_strategy "$GEN_SAMPLING_STRATEGY" )
[[ -n "$GEN_MAX_ITER"          ]] && GEN_ARGS+=( --max_iter "$GEN_MAX_ITER" )
[[ -n "$GEN_BATCH_SIZE"        ]] && GEN_ARGS+=( --batch_size "$GEN_BATCH_SIZE" )
if [[ -n "$GEN_CLASS_IDS" ]]; then
    read -r -a _CLASS_IDS_ARR <<< "$GEN_CLASS_IDS"
    GEN_ARGS+=( --class_ids "${_CLASS_IDS_ARR[@]}" )
fi
case "$GEN_BATCH_LENS_TOGETHER" in
    true)  GEN_ARGS+=( --batch_lens_together );;
    false) GEN_ARGS+=( --no-batch_lens_together );;
esac

"$PYTHON_BIN" generate_dplm_fixed.py "${GEN_ARGS[@]}"
