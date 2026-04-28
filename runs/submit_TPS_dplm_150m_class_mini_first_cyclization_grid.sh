#!/usr/bin/env bash
set -euo pipefail

# Submit TPS dPLM grid jobs for PBS (MetaCentrum) or Slurm (Karolina).
#
# Syntax:
#   [VAR=value ...] bash runs/dplm/submit_TPS_dplm_150m_class_mini_first_cyclization_grid.sh
#
# Examples:
#   DRY_RUN=1 bash runs/dplm/submit_TPS_dplm_150m_class_mini_first_cyclization_grid.sh
#   CLUSTER=karolina DRY_RUN=1 bash runs/dplm/submit_TPS_dplm_150m_class_mini_first_cyclization_grid.sh
#   CLUSTER=karolina SCHEDULER_TYPE=slurm bash runs/dplm/submit_TPS_dplm_150m_class_mini_first_cyclization_grid.sh
#
# Common overrides: DATADIR, RUN_PREFIX, DRY_RUN, CLUSTER, SCHEDULER_TYPE, SCHEDULER_RESOURCE_SPEC, JOB_*.

# -------------------------
# User-Configurable Variables
# -------------------------

DATADIR="${DATADIR:-/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm}"
EXP="${EXP:-tps/TPS_dplm_150m_class_mini_first_cyclization}"
RUN_PREFIX="${RUN_PREFIX:-TPS_dplm_150m_class_mini_first_cyclization_grid}"
DRY_RUN="${DRY_RUN:-0}"

CLUSTER="${CLUSTER:-metacentrum}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-}"
SCHEDULER_RESOURCE_SPEC="${SCHEDULER_RESOURCE_SPEC:-}"

JOB_NODES="${JOB_NODES:-1}"
JOB_NCPUS="${JOB_NCPUS:-1}"
JOB_NGPUS="${JOB_NGPUS:-1}"
JOB_ACCOUNT="${JOB_ACCOUNT:-fta-26-15}"
JOB_PARTITION="${JOB_PARTITION:-qgpu}"
JOB_GPU_MEM="${JOB_GPU_MEM:-46gb}"
JOB_MEM_PBS="${JOB_MEM_PBS:-64gb}"
JOB_MEM_SLURM="${JOB_MEM_SLURM:-64G}"
JOB_SCRATCH_LOCAL="${JOB_SCRATCH_LOCAL:-40gb}"
JOB_WALLTIME="${JOB_WALLTIME:-72:00:00}"
if [[ "$CLUSTER" == "karolina" ]]; then
    TRAIN_ENV="${TRAIN_ENV:-/mnt/proj2/fta-26-15/.conda/envs/dplm}"
else
    TRAIN_ENV="${TRAIN_ENV:-/storage/brno2/home/soldatmat/.conda/envs/dplm}"
fi

TRAIN_LR_LIST="${TRAIN_LR_LIST:-1e-2 1e-4}"
WARMUP_STEPS_LIST="${WARMUP_STEPS_LIST:-4000}"
WARMUP_INIT_LR_LIST="${WARMUP_INIT_LR_LIST:-1e-4*tlr}"
LORA_ENABLE_LIST="${LORA_ENABLE_LIST:-false}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

DEFAULT_RESOURCE_SPEC=""

# -------------------------
# Internal Setup And Validation
# -------------------------

if [[ "$CLUSTER" == "metacentrum" ]]; then
    SCHEDULER_TYPE="${SCHEDULER_TYPE:-pbs}"
    printf -v DEFAULT_RESOURCE_SPEC '%s\n%s' \
        "#PBS -l select=${JOB_NODES}:ncpus=${JOB_NCPUS}:ngpus=${JOB_NGPUS}:gpu_mem=${JOB_GPU_MEM}:mem=${JOB_MEM_PBS}:scratch_local=${JOB_SCRATCH_LOCAL}" \
        "#PBS -l walltime=${JOB_WALLTIME}"
    if [[ -z "$SCHEDULER_RESOURCE_SPEC" ]]; then
        SCHEDULER_RESOURCE_SPEC="$DEFAULT_RESOURCE_SPEC"
    fi
elif [[ "$CLUSTER" == "karolina" ]]; then
    SCHEDULER_TYPE="${SCHEDULER_TYPE:-slurm}"
    # Karolina typically uses Slurm. Override as needed for your partition/QoS.
    printf -v DEFAULT_RESOURCE_SPEC '%s\n%s\n%s\n%s\n%s\n%s\n%s' \
        "#SBATCH -A ${JOB_ACCOUNT}" \
        "#SBATCH --partition=${JOB_PARTITION}" \
        "#SBATCH --nodes=${JOB_NODES}" \
        "#SBATCH --gres=gpu:${JOB_NGPUS}" \
        "#SBATCH --cpus-per-task=${JOB_NCPUS}" \
        "#SBATCH --mem=${JOB_MEM_SLURM}" \
        "#SBATCH --time=${JOB_WALLTIME}"
    if [[ -z "$SCHEDULER_RESOURCE_SPEC" ]]; then
        SCHEDULER_RESOURCE_SPEC="$DEFAULT_RESOURCE_SPEC"
    fi
else
    echo >&2 "Unsupported CLUSTER='$CLUSTER'. Use 'metacentrum' or 'karolina', or set SCHEDULER_RESOURCE_SPEC manually."
    exit 1
fi

if [[ "$SCHEDULER_TYPE" != "pbs" && "$SCHEDULER_TYPE" != "slurm" ]]; then
    echo >&2 "Unsupported SCHEDULER_TYPE='$SCHEDULER_TYPE'. Use 'pbs' or 'slurm'."
    exit 1
fi

read -r -a TRAIN_LR_VALUES <<< "$TRAIN_LR_LIST"
read -r -a WARMUP_STEPS_VALUES <<< "$WARMUP_STEPS_LIST"
read -r -a WARMUP_INIT_LR_VALUES <<< "$WARMUP_INIT_LR_LIST"
read -r -a LORA_ENABLE_VALUES <<< "$LORA_ENABLE_LIST"

# -------------------------
# Script Logic
# -------------------------

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

    job_name="mcl${lr_code}w${warmup_code}i${warmup_init_code}${lora_code}"
    job_name=$(echo "$job_name" | cut -c1-15)

    if [[ "$DRY_RUN" == "1" ]]; then
        if [[ "$SCHEDULER_TYPE" == "pbs" ]]; then
            echo "[DRY_RUN] qsub -N ${job_name} for ${run_name} with lora.enable=${lora_enable}"
        else
            echo "[DRY_RUN] sbatch --job-name=${job_name} for ${run_name} with lora.enable=${lora_enable}"
        fi
        return 0
    fi

    if [[ "$SCHEDULER_TYPE" == "pbs" ]]; then
        qsub -N "$job_name" <<EOF
#!/bin/bash
${SCHEDULER_RESOURCE_SPEC}

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,
exp="${EXP}"
run_name="${run_name}"

DATADIR="${DATADIR}"

mkdir -p "\$DATADIR/logs/\${run_name}"
job_id="\${PBS_JOBID:-\${SLURM_JOB_ID:-unknown}}"
echo "\$job_id is running on node \$(hostname -f)" >> "\$DATADIR/logs/\${run_name}/job_info.txt"

if [[ -n "\${SCRATCHDIR:-}" ]]; then
    work_scratch="\$SCRATCHDIR"
else
    work_scratch="\$(mktemp -d /tmp/dplm_\${run_name}_XXXXXX)"
    trap 'rm -rf "\$work_scratch"' EXIT
fi

mkdir -p "\$work_scratch/tmp"
export TMPDIR="\$work_scratch/tmp"

mkdir -p "\$work_scratch/dplm"
cp -r "\$DATADIR/data-bin" "\$work_scratch/dplm/data-bin/" || { echo >&2 "Error while copying input file(s)!"; exit 2; }

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

cd "\$DATADIR"

"\${PY_RUNNER[@]}" python train.py \
    experiment=\${exp} \
    name=\${run_name} \
    paths.data_dir=\$work_scratch/dplm/data-bin \
    paths.log_dir=\$work_scratch/dplm/logs/\${run_name} \
    train.lr=${train_lr} \
    task.lr_scheduler.warmup_steps=${warmup_steps} \
    task.lr_scheduler.warmup_init_lr=${warmup_init_lr} \
    model.decoder.lora.enable=${lora_enable} \
    ${EXTRA_OVERRIDES}

cp -r "\$work_scratch/dplm/logs/\${run_name}" "\$DATADIR/logs/" || { echo >&2 "Result file(s) copying failed (with a code \$?)!"; exit 4; }

if command -v clean_scratch >/dev/null 2>&1; then
    clean_scratch
fi
EOF
    else
        mkdir -p "$DATADIR/logs/${run_name}"
        sbatch \
            --job-name="$job_name" \
            --output="$DATADIR/logs/${run_name}/slurm-%j.out" \
            --error="$DATADIR/logs/${run_name}/slurm-%j.err" <<EOF
#!/bin/bash
${SCHEDULER_RESOURCE_SPEC}

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,
exp="${EXP}"
run_name="${run_name}"

DATADIR="${DATADIR}"

mkdir -p "\$DATADIR/logs/\${run_name}"
job_id="\${PBS_JOBID:-\${SLURM_JOB_ID:-unknown}}"
echo "\$job_id is running on node \$(hostname -f)" >> "\$DATADIR/logs/\${run_name}/job_info.txt"

if [[ -n "\${SCRATCHDIR:-}" ]]; then
    work_scratch="\$SCRATCHDIR"
else
    work_scratch="\$(mktemp -d /tmp/dplm_\${run_name}_XXXXXX)"
    trap 'rm -rf "\$work_scratch"' EXIT
fi

mkdir -p "\$work_scratch/tmp"
export TMPDIR="\$work_scratch/tmp"

mkdir -p "\$work_scratch/dplm"
cp -r "\$DATADIR/data-bin" "\$work_scratch/dplm/data-bin/" || { echo >&2 "Error while copying input file(s)!"; exit 2; }

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

cd "\$DATADIR"

"\${PY_RUNNER[@]}" python train.py \
    experiment=\${exp} \
    name=\${run_name} \
    paths.data_dir=\$work_scratch/dplm/data-bin \
    paths.log_dir=\$work_scratch/dplm/logs/\${run_name} \
    train.lr=${train_lr} \
    task.lr_scheduler.warmup_steps=${warmup_steps} \
    task.lr_scheduler.warmup_init_lr=${warmup_init_lr} \
    model.decoder.lora.enable=${lora_enable} \
    ${EXTRA_OVERRIDES}

cp -r "\$work_scratch/dplm/logs/\${run_name}" "\$DATADIR/logs/" || { echo >&2 "Result file(s) copying failed (with a code \$?)!"; exit 4; }

if command -v clean_scratch >/dev/null 2>&1; then
    clean_scratch
fi
EOF
    fi
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
