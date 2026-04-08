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

PROJECT_ROOT="/mnt/proj2/fta-26-15/documents/dplm"
CHECKPOINT="/mnt/proj2/fta-26-15/documents/dplm/logs/TPS_dplm_150m_stage3_run_8/checkpoints/N-Step-Checkpoint_epoch=172_step=20000.ckpt"
CSV_PATH="/mnt/proj2/fta-26-15/documents/output/dplm/sampled_lengths_1000.csv"
SAVE_DIR="/mnt/proj2/fta-26-15/documents/output/dplm/TPS_dplm_150m_stage3_run_8_step20000/sl1000_t1.0-generate_dplm_fixed"
CONDA_ENV="/mnt/proj2/fta-26-15/.conda/envs/dplm"
PYTHON_BIN="$CONDA_ENV/bin/python"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV not found: $CSV_PATH" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

readarray -t parsed_csv < <(python - <<'PY'
import csv

csv_path = "/mnt/proj2/fta-26-15/documents/output/dplm/sampled_lengths_1000.csv"
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

if [[ ${#SEQ_LENS[@]} -eq 0 || ${#SEQ_LENS[@]} -ne ${#NUM_SEQS[@]} ]]; then
    echo "Parsed seq_lens/num_seqs are invalid: ${#SEQ_LENS[@]} vs ${#NUM_SEQS[@]}" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found in env: $PYTHON_BIN" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" generate_dplm_fixed.py \
    --model_name "$CHECKPOINT" \
    --no-from_huggingface \
    --saveto "$SAVE_DIR" \
    --temperature 1.0 \
    --seq_lens "${SEQ_LENS[@]}" \
    --num_seqs "${NUM_SEQS[@]}"
