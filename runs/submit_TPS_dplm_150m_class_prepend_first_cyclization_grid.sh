#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash runs/dplm/submit_TPS_dplm_150m_class_prepend_first_cyclization_grid.sh
#
# Optional environment overrides:
#   DATADIR=<path> RUN_PREFIX=<prefix> DRY_RUN=1 bash runs/dplm/submit_TPS_dplm_150m_class_prepend_first_cyclization_grid.sh

DATADIR="${DATADIR:-/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm}"
EXP="tps/TPS_dplm_150m_class_prepend_first_cyclization"
RUN_PREFIX="${RUN_PREFIX:-TPS_dplm_150m_class_prepend_first_cyclization_grid}"

# Grid values
TRAIN_LR_VALUES=(1e-2 1e-4)
WARMUP_STEPS_VALUES=(4000)
WARMUP_INIT_LR_VALUES=(1e-4*tlr)
LORA_ENABLE_VALUES=(true)

DRY_RUN="${DRY_RUN:-0}"

sanitize_float() {
    # Convert values like 4e-4 to filesystem-safe tokens like 4em4.
    echo "$1" | sed 's/+//g; s/-/m/g; s/\./p/g'
}

value_code() {
    # Compact scientific notation: 4e-4 -> 44, 1e-6 -> 16.
    if [[ "$1" =~ ^([0-9]+)e-([0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    else
        echo "$(sanitize_float "$1")"
    fi
}

resolve_lr_value() {
    # Resolve values like "1e-1*tlr" or "tlr*1e-1" against the current train lr.
    local value="$1"
    local train_lr="$2"
    local factor

    if [[ "$value" =~ ^([0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?)\*tlr$ ]]; then
        factor="${BASH_REMATCH[1]}"
    elif [[ "$value" =~ ^tlr\*([0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?)$ ]]; then
        factor="${BASH_REMATCH[1]}"
    else
        echo "$value"
        return 0
    fi

    awk -v a="$train_lr" -v b="$factor" 'BEGIN { printf "%.12g", a * b }'
}

submit_job() {
    local train_lr="$1"
    local warmup_steps="$2"
    local warmup_init_lr_raw="$3"
    local lora_enable="$4"
    local warmup_init_lr

    warmup_init_lr="$(resolve_lr_value "$warmup_init_lr_raw" "$train_lr")"

    local run_name="${RUN_PREFIX}_lr$(sanitize_float "$train_lr")_wu${warmup_steps}_winit$(sanitize_float "$warmup_init_lr")_lora${lora_enable}"
    local lr_code
    local warmup_code
    local warmup_init_code
    local lora_code
    local job_name

    lr_code=$(value_code "$train_lr")
    warmup_init_code=$(value_code "$warmup_init_lr")
    lora_code=$([ "$lora_enable" = "true" ] && echo "t" || echo "f")

    if (( warmup_steps % 1000 == 0 )); then
        warmup_code="$((warmup_steps / 1000))k"
    else
        warmup_code="$warmup_steps"
    fi

    job_name="pcl${lr_code}w${warmup_code}i${warmup_init_code}${lora_code}"
    job_name=$(echo "$job_name" | cut -c1-15)

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY_RUN] qsub -N ${job_name} for ${run_name} with lora.enable=${lora_enable}"
        return 0
    fi

    qsub -N "$job_name" <<EOF
#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=46gb:mem=64gb:scratch_local=40gb
#PBS -l walltime=72:00:00

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,
exp="${EXP}"
run_name="${run_name}"

DATADIR="${DATADIR}"

mkdir -p "\$DATADIR/logs/\${run_name}"
echo "\$PBS_JOBID is running on node \$(hostname -f) in a scratch directory \$SCRATCHDIR" >> "\$DATADIR/logs/\${run_name}/job_info.txt"

test -n "\$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }
mkdir -p "\$SCRATCHDIR/tmp"
export TMPDIR="\$SCRATCHDIR/tmp"

mkdir -p "\$SCRATCHDIR/dplm"
cp -r "\$DATADIR/data-bin" "\$SCRATCHDIR/dplm/data-bin/" || { echo >&2 "Error while copying input file(s)!"; exit 2; }

module add mambaforge
mamba activate /storage/brno2/home/soldatmat/.conda/envs/dplm

cd "\$DATADIR"

python train.py \
    experiment=\${exp} \
    name=\${run_name} \
    paths.data_dir=\$SCRATCHDIR/dplm/data-bin \
    paths.log_dir=\$SCRATCHDIR/dplm/logs/\${run_name} \
    train.lr=${train_lr} \
    task.lr_scheduler.warmup_steps=${warmup_steps} \
    task.lr_scheduler.warmup_init_lr=${warmup_init_lr} \
    model.decoder.lora.enable=${lora_enable}

cp -r "\$SCRATCHDIR/dplm/logs/\${run_name}" "\$DATADIR/logs/" || { echo >&2 "Result file(s) copying failed (with a code \$?)!"; exit 4; }

clean_scratch
EOF
}

total=0
for train_lr in "${TRAIN_LR_VALUES[@]}"; do
    for warmup_steps in "${WARMUP_STEPS_VALUES[@]}"; do
        for warmup_init_lr in "${WARMUP_INIT_LR_VALUES[@]}"; do
            for lora_enable in "${LORA_ENABLE_VALUES[@]}"; do
                submit_job "$train_lr" "$warmup_steps" "$warmup_init_lr" "$lora_enable"
                total=$((total + 1))
            done
        done
    done
done

echo "Submitted ${total} jobs."
