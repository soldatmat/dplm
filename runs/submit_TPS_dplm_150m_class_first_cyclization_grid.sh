#!/usr/bin/env bash
set -euo pipefail

# Submit TPS dPLM grid jobs for PBS (MetaCentrum) or Slurm (Karolina).
#
# Syntax:
#   [VAR=value ...] bash runs/submit_TPS_dplm_150m_class_first_cyclization_grid.sh
#
# Examples:
#   DRY_RUN=1 bash runs/submit_TPS_dplm_150m_class_first_cyclization_grid.sh
#   CLUSTER=karolina DRY_RUN=1 bash runs/submit_TPS_dplm_150m_class_first_cyclization_grid.sh
#   CLUSTER=karolina SCHEDULER_TYPE=slurm bash runs/submit_TPS_dplm_150m_class_first_cyclization_grid.sh
#
# Common overrides: DATADIR, RUN_PREFIX, RUN_SUFFIX, DRY_RUN, CLUSTER,
#   SCHEDULER_TYPE, SCHEDULER_RESOURCE_SPEC, JOB_*, TRAIN_LR_LIST,
#   WARMUP_STEPS_LIST, TOTAL_STEPS_LIST, SAVE_STEP_FREQ_LIST, VAL_EE_FREQ_LIST,
#   NUM_SEQS_LIST, LR_END_LIST, WARMUP_INIT_LR_LIST, LORA_ENABLE_LIST,
#   LORA_RANK_LIST, LORA_ALPHA_LIST, LORA_TARGET_MODULE_LIST,
#   LORA_TARGET_MODULE_LABELS_LIST, EXTRA_OVERRIDES,
#   INIT_WEIGHTS_FROM, INIT_WEIGHTS_SOURCE_ARCH, INIT_WEIGHTS_MERGE_LORA.
#
# Init from a prior checkpoint (state_dict only; fresh optimizer/scheduler/step):
#   INIT_WEIGHTS_FROM=/abs/path/to/some.ckpt
#     Mirrors runs/start_dplm_run_from_ckpt.sh: stages the ckpt to scratch and
#     passes train.ckpt_path=null + train.init_weights_from=<staged>.
#   INIT_WEIGHTS_SOURCE_ARCH=dplm
#     Set when the source ckpt is a vanilla DPLM run; remaps 'model.net.*' ->
#     'model.decoder.net.*' so the ESM stack transfers into the DPLMClass arch.
#     Leave unset when initialising from another DPLMClass run.
#   INIT_WEIGHTS_MERGE_LORA=auto|never
#     Forwarded to train.init_weights_merge_lora when set.

# -------------------------
# User-Configurable Variables
# -------------------------

DATADIR="${DATADIR:-/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm}"
EXP="${EXP:-tps/TPS_dplm_150m_class_first_cyclization}"
RUN_PREFIX="${RUN_PREFIX:-TPS_dplm_150m_class_first_cyclization_grid}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
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

# Cache locations (shared across runs to avoid re-downloads and home-dir issues).
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/triton}"

TRAIN_LR_LIST="${TRAIN_LR_LIST:-1e-3 1e-4 1e-5}"
WARMUP_STEPS_LIST="${WARMUP_STEPS_LIST:-4000}"
TOTAL_STEPS_LIST="${TOTAL_STEPS_LIST:-200000}" # controls trainer.max_steps and (++)task.lr_scheduler.total_steps
SAVE_STEP_FREQ_LIST="${SAVE_STEP_FREQ_LIST:-10000}"
VAL_EE_FREQ_LIST="${VAL_EE_FREQ_LIST:-10000}"
NUM_SEQS_LIST="${NUM_SEQS_LIST:-[50]}"
LR_END_LIST="${LR_END_LIST:-1e-1*tlr}"
WARMUP_INIT_LR_LIST="${WARMUP_INIT_LR_LIST:-1e-4*tlr}"

LORA_ENABLE_LIST="${LORA_ENABLE_LIST:-false}"
LORA_RANK_LIST="${LORA_RANK_LIST:-1}"
LORA_ALPHA_LIST="${LORA_ALPHA_LIST:-2}"
# Patterns themselves contain '|', so this list is ';'-separated.
LORA_TARGET_MODULE_LIST="${LORA_TARGET_MODULE_LIST:-(esm.encoder.layer.[0-9]*.attention.self.key)}"
# Optional ';'-separated labels matched 1:1 with LORA_TARGET_MODULE_LIST entries.
# Used in run_name/job_name in place of the numeric index. Empty -> use indices.
LORA_TARGET_MODULE_LABELS_LIST="${LORA_TARGET_MODULE_LABELS_LIST:-}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

INIT_WEIGHTS_FROM="${INIT_WEIGHTS_FROM:-}"
INIT_WEIGHTS_SOURCE_ARCH="${INIT_WEIGHTS_SOURCE_ARCH:-}"
INIT_WEIGHTS_MERGE_LORA="${INIT_WEIGHTS_MERGE_LORA:-}"

if [[ -n "$INIT_WEIGHTS_FROM" && ! -f "$INIT_WEIGHTS_FROM" ]]; then
    echo >&2 "INIT_WEIGHTS_FROM does not point to an existing file: $INIT_WEIGHTS_FROM"
    exit 1
fi

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
read -r -a TOTAL_STEPS_VALUES <<< "$TOTAL_STEPS_LIST"
read -r -a SAVE_STEP_FREQ_VALUES <<< "$SAVE_STEP_FREQ_LIST"
read -r -a VAL_EE_FREQ_VALUES <<< "$VAL_EE_FREQ_LIST"
# Supports absolute values (e.g. 1e-6) and relative values (e.g. 1e-1*tlr).
read -r -a LR_END_VALUES <<< "$LR_END_LIST"
read -r -a WARMUP_INIT_LR_VALUES <<< "$WARMUP_INIT_LR_LIST"
read -r -a LORA_ENABLE_VALUES <<< "$LORA_ENABLE_LIST"
# Read NUM_SEQS_LIST as space-separated values (e.g., "[50] [150]")
read -r -a NUM_SEQS_VALUES <<< "$NUM_SEQS_LIST"
read -r -a LORA_RANK_VALUES <<< "$LORA_RANK_LIST"
read -r -a LORA_ALPHA_VALUES <<< "$LORA_ALPHA_LIST"
# Use ';' as separator since the patterns themselves contain '|' (and may contain spaces).
IFS=';' read -r -a LORA_TARGET_MODULE_VALUES <<< "$LORA_TARGET_MODULE_LIST"
LORA_TARGET_MODULE_LABELS_VALUES=()
if [[ -n "$LORA_TARGET_MODULE_LABELS_LIST" ]]; then
    IFS=';' read -r -a LORA_TARGET_MODULE_LABELS_VALUES <<< "$LORA_TARGET_MODULE_LABELS_LIST"
    if [[ "${#LORA_TARGET_MODULE_LABELS_VALUES[@]}" -ne "${#LORA_TARGET_MODULE_VALUES[@]}" ]]; then
        echo >&2 "LORA_TARGET_MODULE_LABELS_LIST has ${#LORA_TARGET_MODULE_LABELS_VALUES[@]} entries but LORA_TARGET_MODULE_LIST has ${#LORA_TARGET_MODULE_VALUES[@]}; counts must match."
        exit 1
    fi
fi

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
    local total_steps="$3"
    local save_step_freq="$4"
    local val_ee_freq="$5"
    local lr_end_raw="$6"
    local warmup_init_lr_raw="$7"
    local lora_enable="$8"
    local num_seqs="$9"
    local lora_rank="${10}"
    local lora_alpha="${11}"
    local lora_target_module="${12}"
    local ltm_idx="${13}"
    local lr_end
    local warmup_init_lr

    lr_end="$(resolve_lr_value "$lr_end_raw" "$train_lr")"
    warmup_init_lr="$(resolve_lr_value "$warmup_init_lr_raw" "$train_lr")"

    local run_name="${RUN_PREFIX}_lr$(sanitize_float "$train_lr")_wu${warmup_steps}_ts${total_steps}_ckpt${save_step_freq}_valee${val_ee_freq}_lend$(sanitize_float "$lr_end")_winit$(sanitize_float "$warmup_init_lr")_lora${lora_enable}_ns$(echo $num_seqs | tr -d '[]')_r${lora_rank}_a${lora_alpha}_ltm${ltm_idx}"
    if [[ -n "$RUN_SUFFIX" ]]; then
        run_name="${run_name}_${RUN_SUFFIX}"
    fi
    local lr_code
    local warmup_code
    local total_code
    local save_code
    local val_code
    local lr_end_code
    local warmup_init_code
    local lora_code
    local job_name

    lr_code=$(value_code "$train_lr")
    lr_end_code=$(value_code "$lr_end")
    warmup_init_code=$(value_code "$warmup_init_lr")
    lora_code=$([ "$lora_enable" = "true" ] && echo "t" || echo "f")
    if (( warmup_steps % 1000 == 0 )); then
        warmup_code="$((warmup_steps / 1000))k"
    else
        warmup_code="$warmup_steps"
    fi
    if (( total_steps % 1000 == 0 )); then
        total_code="$((total_steps / 1000))k"
    else
        total_code="$total_steps"
    fi
    if (( save_step_freq % 1000 == 0 )); then
        save_code="$((save_step_freq / 1000))k"
    else
        save_code="$save_step_freq"
    fi
    if (( val_ee_freq % 1000 == 0 )); then
        val_code="$((val_ee_freq / 1000))k"
    else
        val_code="$val_ee_freq"
    fi

    job_name="cfl${lr_code}w${warmup_code}t${total_code}s${save_code}v${val_code}e${lr_end_code}i${warmup_init_code}${lora_code}m${ltm_idx}"
    job_name=$(echo "$job_name" | cut -c1-15)

    if [[ "$DRY_RUN" == "1" ]]; then
        if [[ "$SCHEDULER_TYPE" == "pbs" ]]; then
            echo "[DRY_RUN] qsub -N ${job_name} for ${run_name} with lora.enable=${lora_enable} num_seqs=${num_seqs} lora_rank=${lora_rank} lora_alpha=${lora_alpha} lora_target_module=${lora_target_module}"
        else
            echo "[DRY_RUN] sbatch --job-name=${job_name} for ${run_name} with lora.enable=${lora_enable} num_seqs=${num_seqs} lora_rank=${lora_rank} lora_alpha=${lora_alpha} lora_target_module=${lora_target_module}"
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
    trap 'cp -r "\$work_scratch/dplm/logs/\${run_name}" "\$DATADIR/logs/" 2>/dev/null; rm -rf "\$work_scratch"' EXIT
fi
echo "Using work scratch directory: \$work_scratch"

mkdir -p "\$work_scratch/tmp"
export TMPDIR="\$work_scratch/tmp"
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
    export HF_HOME="${HF_CACHE_DIR}"
    export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
    export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
fi
if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
fi

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

init_weights_args=()
if [[ -n "${INIT_WEIGHTS_FROM}" ]]; then
    staged_init_weights="\$work_scratch/dplm/init_weights.ckpt"
    cp "${INIT_WEIGHTS_FROM}" "\$staged_init_weights" || { echo >&2 "Error staging init weights from ${INIT_WEIGHTS_FROM}!"; exit 2; }
    init_weights_args+=( train.ckpt_path=null "train.init_weights_from=\$staged_init_weights" )
    if [[ -n "${INIT_WEIGHTS_SOURCE_ARCH}" ]]; then
        init_weights_args+=( "train.init_weights_source_arch=${INIT_WEIGHTS_SOURCE_ARCH}" )
    fi
    if [[ -n "${INIT_WEIGHTS_MERGE_LORA}" ]]; then
        init_weights_args+=( "train.init_weights_merge_lora=${INIT_WEIGHTS_MERGE_LORA}" )
    fi
fi

cd "\$DATADIR"

"\${PY_RUNNER[@]}" python train.py \
    experiment=\${exp} \
    name=\${run_name} \
    paths.data_dir=\$work_scratch/dplm/data-bin \
    paths.log_dir=\$work_scratch/dplm/logs/\${run_name} \
    train.lr=${train_lr} \
    task.lr_scheduler.warmup_steps=${warmup_steps} \
    ++task.lr_scheduler.total_steps=${total_steps} \
    trainer.max_steps=${total_steps} \
    callbacks.checkpoint_every_n_steps.save_step_frequency=${save_step_freq} \
    callbacks.validate_with_enzyme_explorer.every_n_train_steps=${val_ee_freq} \
    callbacks.validate_with_enzyme_explorer.num_seqs=${num_seqs} \
    ++task.lr_scheduler.lr_end=${lr_end} \
    task.lr_scheduler.warmup_init_lr=${warmup_init_lr} \
    model.decoder.lora.enable=${lora_enable} \
    ++model.decoder.lora.lora_rank=${lora_rank} \
    ++model.decoder.lora.lora_alpha=${lora_alpha} \
    ++model.decoder.lora.lora_target_module="'${lora_target_module}'" \
    "\${init_weights_args[@]}" \
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
    trap 'cp -r "\$work_scratch/dplm/logs/\${run_name}" "\$DATADIR/logs/" 2>/dev/null; rm -rf "\$work_scratch"' EXIT
fi
echo "Using work scratch directory: \$work_scratch"

mkdir -p "\$work_scratch/tmp"
export TMPDIR="\$work_scratch/tmp"
if [[ -n "${HF_CACHE_DIR:-}" ]]; then
    export HF_HOME="${HF_CACHE_DIR}"
    export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
    export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
fi
if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
fi

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

init_weights_args=()
if [[ -n "${INIT_WEIGHTS_FROM}" ]]; then
    staged_init_weights="\$work_scratch/dplm/init_weights.ckpt"
    cp "${INIT_WEIGHTS_FROM}" "\$staged_init_weights" || { echo >&2 "Error staging init weights from ${INIT_WEIGHTS_FROM}!"; exit 2; }
    init_weights_args+=( train.ckpt_path=null "train.init_weights_from=\$staged_init_weights" )
    if [[ -n "${INIT_WEIGHTS_SOURCE_ARCH}" ]]; then
        init_weights_args+=( "train.init_weights_source_arch=${INIT_WEIGHTS_SOURCE_ARCH}" )
    fi
    if [[ -n "${INIT_WEIGHTS_MERGE_LORA}" ]]; then
        init_weights_args+=( "train.init_weights_merge_lora=${INIT_WEIGHTS_MERGE_LORA}" )
    fi
fi

cd "\$DATADIR"

"\${PY_RUNNER[@]}" python train.py \
    experiment=\${exp} \
    name=\${run_name} \
    paths.data_dir=\$work_scratch/dplm/data-bin \
    paths.log_dir=\$work_scratch/dplm/logs/\${run_name} \
    train.lr=${train_lr} \
    task.lr_scheduler.warmup_steps=${warmup_steps} \
    ++task.lr_scheduler.total_steps=${total_steps} \
    trainer.max_steps=${total_steps} \
    callbacks.checkpoint_every_n_steps.save_step_frequency=${save_step_freq} \
    callbacks.validate_with_enzyme_explorer.every_n_train_steps=${val_ee_freq} \
    callbacks.validate_with_enzyme_explorer.num_seqs=${num_seqs} \
    ++task.lr_scheduler.lr_end=${lr_end} \
    task.lr_scheduler.warmup_init_lr=${warmup_init_lr} \
    model.decoder.lora.enable=${lora_enable} \
    ++model.decoder.lora.lora_rank=${lora_rank} \
    ++model.decoder.lora.lora_alpha=${lora_alpha} \
    ++model.decoder.lora.lora_target_module="'${lora_target_module}'" \
    "\${init_weights_args[@]}" \
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
        for total_steps in "${TOTAL_STEPS_VALUES[@]}"; do
            for save_step_freq in "${SAVE_STEP_FREQ_VALUES[@]}"; do
                for val_ee_freq in "${VAL_EE_FREQ_VALUES[@]}"; do
                    for lr_end in "${LR_END_VALUES[@]}"; do
                        for warmup_init_lr in "${WARMUP_INIT_LR_VALUES[@]}"; do
                            for lora_enable in "${LORA_ENABLE_VALUES[@]}"; do
                                for num_seqs in "${NUM_SEQS_VALUES[@]}"; do
                                    for lora_rank in "${LORA_RANK_VALUES[@]}"; do
                                        for lora_alpha in "${LORA_ALPHA_VALUES[@]}"; do
                                            for ltm_idx in "${!LORA_TARGET_MODULE_VALUES[@]}"; do
                                                lora_target_module="${LORA_TARGET_MODULE_VALUES[$ltm_idx]}"
                                                if [[ "${#LORA_TARGET_MODULE_LABELS_VALUES[@]}" -gt 0 ]]; then
                                                    ltm_token="${LORA_TARGET_MODULE_LABELS_VALUES[$ltm_idx]}"
                                                else
                                                    ltm_token="$ltm_idx"
                                                fi
                                                submit_job "$train_lr" "$warmup_steps" "$total_steps" "$save_step_freq" "$val_ee_freq" "$lr_end" "$warmup_init_lr" "$lora_enable" "$num_seqs" "$lora_rank" "$lora_alpha" "$lora_target_module" "$ltm_token"
                                                total=$((total + 1))
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Submitted ${total} jobs."
