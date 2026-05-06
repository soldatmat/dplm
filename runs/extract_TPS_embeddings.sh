#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=qgpu
#SBATCH --account=fta-26-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/proj2/fta-26-15/documents/dplm/logs/embeddings/%x-%j.out

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/proj2/fta-26-15/documents/dplm}"
CONDA_ENV="${CONDA_ENV:-/mnt/proj2/fta-26-15/.conda/envs/dplm}"
PYTHON_BIN="$CONDA_ENV/bin/python"

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

CHECKPOINT="${CHECKPOINT:?must set CHECKPOINT}"
INPUT_FILE="${INPUT_FILE:?must set INPUT_FILE}"
EMBEDDING_TYPE="${EMBEDDING_TYPE:?must set EMBEDDING_TYPE (mean|bos)}"
OUTPUT_DIR="${OUTPUT_DIR:?must set OUTPUT_DIR}"
ID_COLUMN="${ID_COLUMN:-Enzyme_marts_ID}"
SEQUENCE_COLUMN="${SEQUENCE_COLUMN:-Aminoacid_sequence}"
ARCHITECTURE="${ARCHITECTURE:-DiffusionProteinLanguageModel}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Input file not found: $INPUT_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname -f)"
echo "  CHECKPOINT     = $CHECKPOINT"
echo "  INPUT_FILE     = $INPUT_FILE"
echo "  EMBEDDING_TYPE = $EMBEDDING_TYPE"
echo "  OUTPUT_DIR     = $OUTPUT_DIR"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" run/extract_embeddings.py \
    --input_file "$INPUT_FILE" \
    --id_column "$ID_COLUMN" \
    --sequence_column "$SEQUENCE_COLUMN" \
    --output_dir "$OUTPUT_DIR" \
    --embedding_type "$EMBEDDING_TYPE" \
    --save_as torch csv \
    --model_name "$CHECKPOINT" \
    --no-from_huggingface \
    --architecture "$ARCHITECTURE"
