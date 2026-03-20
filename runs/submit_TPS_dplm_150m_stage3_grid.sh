#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash runs/dplm/submit_TPS_dplm_150m_stage3_grid.sh
#
# Optional environment overrides:
#   DATADIR=<path> RUN_PREFIX=<prefix> DRY_RUN=1 bash runs/dplm/submit_TPS_dplm_150m_stage3_grid.sh

DATADIR="${DATADIR:-/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm}"
EXP="tps/TPS_dplm_150m_stage3"
RUN_PREFIX="${RUN_PREFIX:-TPS_dplm_150m_stage3_grid}"

# Grid values
TRAIN_LR_VALUES=(1e-4 1e-4 1e-4)
WARMUP_STEPS_VALUES=(1000 2000 4000)
LR_END_VALUES=(1e-6 1e-5 5e-5)
WARMUP_INIT_LR_VALUES=(1e-8 1e-7 1e-6)

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

submit_job() {
    local train_lr="$1"
    local warmup_steps="$2"
    local lr_end="$3"
    local warmup_init_lr="$4"

    local run_name="${RUN_PREFIX}_lr$(sanitize_float "$train_lr")_wu${warmup_steps}_lend$(sanitize_float "$lr_end")_winit$(sanitize_float "$warmup_init_lr")"
    local lr_code
    local warmup_code
    local lr_end_code
    local warmup_init_code
    local job_name

    lr_code=$(value_code "$train_lr")
    lr_end_code=$(value_code "$lr_end")
    warmup_init_code=$(value_code "$warmup_init_lr")
    if (( warmup_steps % 1000 == 0 )); then
        warmup_code="$((warmup_steps / 1000))k"
    else
        warmup_code="$warmup_steps"
    fi

    job_name="d3l${lr_code}w${warmup_code}e${lr_end_code}i${warmup_init_code}"
    job_name=$(echo "$job_name" | cut -c1-15)

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY_RUN] qsub -N ${job_name} for ${run_name}"
        return 0
    fi

    qsub -N "$job_name" <<EOF
#!/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=46gb:mem=64gb:scratch_local=40gb
#PBS -l walltime=24:00:00

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
    task.lr_scheduler.lr_end=${lr_end} \
    task.lr_scheduler.warmup_init_lr=${warmup_init_lr}

cp -r "\$SCRATCHDIR/dplm/logs/\${run_name}" "\$DATADIR/logs/" || { echo >&2 "Result file(s) copying failed (with a code \$?)!"; exit 4; }

clean_scratch
EOF
}

total=0
for train_lr in "${TRAIN_LR_VALUES[@]}"; do
    for warmup_steps in "${WARMUP_STEPS_VALUES[@]}"; do
        for lr_end in "${LR_END_VALUES[@]}"; do
            for warmup_init_lr in "${WARMUP_INIT_LR_VALUES[@]}"; do
                submit_job "$train_lr" "$warmup_steps" "$lr_end" "$warmup_init_lr"
                total=$((total + 1))
            done
        done
    done
done

echo "Submitted ${total} jobs."