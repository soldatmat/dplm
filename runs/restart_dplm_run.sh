#!/usr/bin/env bash
set -euo pipefail

# Restart exactly one DPLM training run from its last checkpoint.
#
# Usage:
#   SOURCE_RUN_DIR=/path/to/dplm/logs/<run_name> bash runs/restart_dplm_run.sh
#   SOURCE_RUN_DIR=/path/to/dplm/logs/<run_name> DRY_RUN=1 bash runs/restart_dplm_run.sh
#
# Examples:
#   SOURCE_RUN_DIR=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm/logs/TPS_dplm_150m_stage3_run_8 \
#   bash runs/restart_dplm_run.sh
#
#   SOURCE_RUN_DIR=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm/logs/TPS_dplm_150m_class_first_cyclization_grid_lr1em4_wu4k_winit1em8_loraf \
#   RUN_NAME=TPS_dplm_150m_class_first_cyclization_grid_lr1em4_wu4k_winit1em8_loraf_01 \
#   bash runs/restart_dplm_run.sh
#
# Optional overrides:
#   DATADIR, EXP, SOURCE_RUN_DIR, SOURCE_CKPT,
#   CLUSTER, SCHEDULER_TYPE, SCHEDULER_RESOURCE_SPEC,
#   JOB_NODES, JOB_NCPUS, JOB_NGPUS, JOB_GPU_MEM, JOB_MEM_PBS,
#   JOB_MEM_SLURM, JOB_SCRATCH_LOCAL, JOB_WALLTIME, JOB_ACCOUNT,
#   JOB_PARTITION, TRAIN_ENV, DRY_RUN, EXTRA_OVERRIDES,
#   STRICT_MAX_STEPS_GUARD, ALLOW_NOOP_RESUME
#
# EXTRA_OVERRIDES: extra Hydra overrides appended to the train.py command.
#   Use this to extend training, e.g. to run for more steps:
#     EXTRA_OVERRIDES="trainer.max_steps=400000 task.lr_scheduler.total_steps=400000"
#   The strict-max-steps guard reads trainer.max_steps from EXTRA_OVERRIDES if
#   present, so increasing it lets the guard pass without ALLOW_NOOP_RESUME.

# -------------------------
# User Configuration
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATADIR="${DATADIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
EXP="${EXP:-}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-}"
SOURCE_RUN_NAME=""
RUN_NAME="${RUN_NAME:-}"
TARGET_RUN_DIR=""
SOURCE_CKPT="${SOURCE_CKPT:-$SOURCE_RUN_DIR/checkpoints/last.ckpt}"
DRY_RUN="${DRY_RUN:-0}"
STRICT_MAX_STEPS_GUARD="${STRICT_MAX_STEPS_GUARD:-0}"
ALLOW_NOOP_RESUME="${ALLOW_NOOP_RESUME:-0}"
DPLM_PYTHON="${DPLM_PYTHON:-/storage/brno2/home/soldatmat/.conda/envs/dplm/bin/python}"

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
JOB_WALLTIME="${JOB_WALLTIME:-24:00:00}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

if [[ "$CLUSTER" == "karolina" ]]; then
    TRAIN_ENV="${TRAIN_ENV:-/mnt/proj2/fta-26-15/.conda/envs/dplm}"
else
    TRAIN_ENV="${TRAIN_ENV:-/storage/brno2/home/soldatmat/.conda/envs/dplm}"
fi

# Cache locations (shared across runs to avoid re-downloads and home-dir issues).
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/triton}"

DEFAULT_RESOURCE_SPEC=""

infer_exp_from_source_run() {
    local overrides_file="$SOURCE_RUN_DIR/.hydra/overrides.yaml"
    if [[ ! -f "$overrides_file" ]]; then
        return 1
    fi

    awk -F= '/^- experiment=/{print $2; exit}' "$overrides_file"
}

read_source_overrides_args() {
    # Replay all Hydra overrides from the source run except experiment/name/paths.*,
    # which are set explicitly by this script. Required so model arch matches the
    # checkpoint's state_dict (e.g. lora.enable, lora_rank).
    local overrides_file="$SOURCE_RUN_DIR/.hydra/overrides.yaml"
    if [[ ! -f "$overrides_file" ]]; then
        return 0
    fi
    awk '
        /^- / {
            sub(/^- /, "")
            if ($0 ~ /^experiment=/) next
            if ($0 ~ /^name=/) next
            if ($0 ~ /^paths\./) next
            # model.lora.* (and ++model.lora.*) are emitted separately by
            # read_source_model_overrides with proper Hydra quoting for values
            # containing parens/special chars. Replaying them from overrides.yaml
            # as bare lines loses the single-quote wrapping and crashes Hydra at
            # parse time (e.g. lora_target_module=(esm.encoder.layer...)).
            if ($0 ~ /^\+*model\.lora\./) next
            print
        }
    ' "$overrides_file"
}

read_source_model_overrides() {
    # Replay every model.lora.* setting from the source's resolved config.yaml.
    # overrides.yaml is not enough: when LoRA params (e.g. lora_target_module) come
    # from the experiment yaml's defaults at the source's submission time, they are
    # not recorded in overrides.yaml, but they affect which modules get wrapped and
    # therefore the ckpt's state_dict keys. Without this, resume fails with
    # "Missing/Unexpected keys" when defaults drift between source and resume.
    local config_file="$SOURCE_RUN_DIR/.hydra/config.yaml"
    [[ -f "$config_file" ]] || return 0
    python3 - "$config_file" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

def render(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    s = str(v)
    # Hydra needs values containing parens/pipes/etc. quoted in its grammar
    # ('value'). The output is embedded into a bash heredoc and then re-parsed
    # by bash on the compute node, which would strip plain singles. Wrap in
    # double-then-single quotes so bash strips the doubles, leaving 'value'
    # for Hydra to unquote.
    if s == "":
        return "\"''\""
    if any(c in s for c in " ()|*[]?+,'\""):
        return "\"'" + s.replace("'", "'\"'\"'") + "'\""
    return s

lora = cfg.get("model", {}).get("lora", {}) or {}
out = [f"model.lora.{k}={render(v)}" for k, v in lora.items()]
print(" ".join(out))
PY
}

# -------------------------
# Scheduler Resource Setup
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

if [[ -z "$SOURCE_RUN_DIR" ]]; then
    echo >&2 "SOURCE_RUN_DIR is required."
    echo >&2 "Example: SOURCE_RUN_DIR=$DATADIR/logs/<run_name> bash runs/restart_dplm_run.sh"
    exit 1
fi

if [[ ! -d "$SOURCE_RUN_DIR" ]]; then
    echo >&2 "Source run directory not found: $SOURCE_RUN_DIR"
    exit 1
fi

SOURCE_RUN_NAME="$(basename "$SOURCE_RUN_DIR")"

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="${SOURCE_RUN_NAME}_01"
fi

TARGET_RUN_DIR="$DATADIR/logs/$RUN_NAME"

if [[ -z "$EXP" ]]; then
    EXP="$(infer_exp_from_source_run)" || {
        echo >&2 "Could not infer EXP from $SOURCE_RUN_DIR/.hydra/overrides.yaml"
        echo >&2 "Set EXP explicitly, e.g. EXP=tps/TPS_dplm_150m_stage3"
        exit 1
    }
fi

if [[ ! -f "$SOURCE_CKPT" ]]; then
    echo >&2 "Source checkpoint not found: $SOURCE_CKPT"
    exit 1
fi

SOURCE_CKPT_BASENAME="$(basename "$SOURCE_CKPT")"
SOURCE_OVERRIDES_ARGS="$(read_source_overrides_args | tr '\n' ' ')"
SOURCE_MODEL_LORA_ARGS="$(read_source_model_overrides)"

# -------------------------
# Strict Max-Steps Guard
# -------------------------
# Blocks submission when checkpoint global_step already reached trainer.max_steps.

read_trainer_max_steps() {
    local source_config="$SOURCE_RUN_DIR/.hydra/config.yaml"
    if [[ ! -f "$source_config" ]]; then
        echo >&2 "Guard check failed: source config not found at $source_config"
        return 1
    fi

    awk '
        /^trainer:/ { in_trainer=1; next }
        in_trainer && /^[^[:space:]]/ { in_trainer=0 }
        in_trainer && $1 == "max_steps:" { print $2; exit }
    ' "$source_config"
}

read_checkpoint_global_step() {
    local ckpt_path="$1"
    local guard_python=""

    if command -v python3 >/dev/null 2>&1 && python3 - <<'PY' >/dev/null 2>&1
import torch
PY
    then
        guard_python="python3"
    elif [[ -x "$DPLM_PYTHON" ]] && "$DPLM_PYTHON" - <<'PY' >/dev/null 2>&1
import torch
PY
    then
        guard_python="$DPLM_PYTHON"
    else
        echo >&2 "ERROR: Could not find a Python interpreter with torch for strict guard."
        echo >&2 "Tried: python3 and DPLM_PYTHON=$DPLM_PYTHON"
        return 2
    fi

    "$guard_python" - "$ckpt_path" <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"ERROR: Unable to import torch for guard check: {exc}", file=sys.stderr)
    raise SystemExit(2)

ckpt_path = sys.argv[1]
try:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
except Exception as exc:
    print(f"ERROR: Unable to read checkpoint '{ckpt_path}': {exc}", file=sys.stderr)
    raise SystemExit(3)

global_step = ckpt.get("global_step")
if global_step is None:
    print("ERROR: Checkpoint does not contain 'global_step'.", file=sys.stderr)
    raise SystemExit(4)

print(int(global_step))
PY
}

parse_max_steps_override() {
    # If EXTRA_OVERRIDES contains trainer.max_steps=<N>, return N (digits and underscores only).
    local pattern='trainer\.max_steps=([0-9_]+)'
    if [[ "$EXTRA_OVERRIDES" =~ $pattern ]]; then
        echo "${BASH_REMATCH[1]}" | tr -d '_'
    fi
}

enforce_max_steps_guard() {
    [[ "$STRICT_MAX_STEPS_GUARD" == "1" ]] || return 0

    local trainer_max_steps
    local checkpoint_global_step
    local override_max_steps

    override_max_steps="$(parse_max_steps_override)"
    if [[ -n "$override_max_steps" ]]; then
        trainer_max_steps="$override_max_steps"
        echo "Guard: using trainer.max_steps=$trainer_max_steps from EXTRA_OVERRIDES."
    else
        trainer_max_steps="$(read_trainer_max_steps)" || {
            echo >&2 "Guard check failed while reading trainer.max_steps."
            echo >&2 "Set STRICT_MAX_STEPS_GUARD=0 to bypass this check."
            exit 1
        }
    fi

    if [[ -z "$trainer_max_steps" ]]; then
        echo >&2 "Guard check failed: could not parse trainer.max_steps from source config."
        echo >&2 "Set STRICT_MAX_STEPS_GUARD=0 to bypass this check."
        exit 1
    fi

    checkpoint_global_step="$(read_checkpoint_global_step "$SOURCE_CKPT")" || {
        echo >&2 "Guard check failed while reading checkpoint global_step."
        echo >&2 "Set STRICT_MAX_STEPS_GUARD=0 to bypass this check."
        exit 1
    }

    if (( checkpoint_global_step >= trainer_max_steps )); then
        echo >&2 "Refusing to submit: checkpoint global_step=$checkpoint_global_step already reached trainer.max_steps=$trainer_max_steps."
        echo >&2 "This would be a no-op resume."
        if [[ "$ALLOW_NOOP_RESUME" == "1" ]]; then
            echo >&2 "ALLOW_NOOP_RESUME=1 set, continuing despite no-op guard."
        else
            echo >&2 "Set ALLOW_NOOP_RESUME=1 to force submission, or STRICT_MAX_STEPS_GUARD=0 to disable the check."
            exit 1
        fi
    fi
}

# -------------------------
# Job Submission
# -------------------------

emit_job_body() {
    cat <<EOF
#!/bin/bash
${SCHEDULER_RESOURCE_SPEC}

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,
exp="${EXP}"
source_run_dir="${SOURCE_RUN_DIR}"
run_name="${RUN_NAME}"
DATADIR="${DATADIR}"
run_log_dir="\$DATADIR/logs/\${run_name}"
source_ckpt="${SOURCE_CKPT}"

mkdir -p "\$run_log_dir"
job_id="\${PBS_JOBID:-\${SLURM_JOB_ID:-unknown}}"
echo "\$job_id is running on node \$(hostname -f)" >> "\$run_log_dir/job_info.txt"
echo "source_run_dir=\$source_run_dir" >> "\$run_log_dir/job_info.txt"
echo "source_ckpt=\$source_ckpt" >> "\$run_log_dir/job_info.txt"

# Keep a copy of source Hydra config and checkpoints in the new run folder.
mkdir -p "\$run_log_dir/checkpoints"
if [[ -d "\$source_run_dir/.hydra" ]]; then
    rm -rf "\$run_log_dir/.hydra_source"
    cp -r "\$source_run_dir/.hydra" "\$run_log_dir/.hydra_source"
fi
cp -r "\$source_run_dir/checkpoints/." "\$run_log_dir/checkpoints/"

if [[ -n "\${SCRATCHDIR:-}" ]]; then
    work_scratch="\$SCRATCHDIR"
else
    work_scratch="\$(mktemp -d /tmp/dplm_\${run_name}_XXXXXX)"
    trap 'rm -rf "\$work_scratch"' EXIT
fi

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

# Pre-create target dir and copy contents to avoid nesting (data-bin/data-bin) on retries.
mkdir -p "\$work_scratch/dplm/data-bin"
cp -r "\$DATADIR/data-bin/." "\$work_scratch/dplm/data-bin/" || { echo >&2 "Error while copying input file(s)!"; exit 2; }

scratch_log_dir="\$work_scratch/dplm/logs/\${run_name}"
mkdir -p "\$scratch_log_dir/checkpoints"
cp -r "\$source_run_dir/checkpoints/." "\$scratch_log_dir/checkpoints/"
if [[ -d "\$source_run_dir/.hydra" ]]; then
    cp -r "\$source_run_dir/.hydra" "\$scratch_log_dir/.hydra_source"
fi
resume_ckpt="\$scratch_log_dir/checkpoints/${SOURCE_CKPT_BASENAME}"

module add mambaforge >/dev/null 2>&1 || true
ml Anaconda3 >/dev/null 2>&1 || true

if command -v conda >/dev/null 2>&1; then
    if [[ "${TRAIN_ENV}" == /* ]]; then
        PY_RUNNER=(conda run --no-capture-output -p "${TRAIN_ENV}")
    else
        PY_RUNNER=(conda run --no-capture-output -n "${TRAIN_ENV}")
    fi
elif command -v mamba >/dev/null 2>&1; then
    if [[ "${TRAIN_ENV}" == /* ]]; then
        PY_RUNNER=(mamba run --no-capture-output -p "${TRAIN_ENV}")
    else
        PY_RUNNER=(mamba run --no-capture-output -n "${TRAIN_ENV}")
    fi
else
    echo >&2 "Neither conda nor mamba is available after module setup."
    exit 3
fi

cd "\$DATADIR"

# Pass train.ckpt_path so Lightning resumes optimizer/LR scheduler/global_step exactly.
# Source-run overrides (e.g. model.lora.enable) are replayed so model arch matches the ckpt.
# Use EXTRA_OVERRIDES to extend training (e.g. trainer.max_steps=400000).
"\${PY_RUNNER[@]}" python train.py \\
    experiment=\${exp} \\
    ${SOURCE_OVERRIDES_ARGS} \\
    ${SOURCE_MODEL_LORA_ARGS} \\
    name=\${run_name} \\
    paths.data_dir=\$work_scratch/dplm/data-bin \\
    paths.log_dir=\$scratch_log_dir \\
    train.ckpt_path=\$resume_ckpt \\
    ${EXTRA_OVERRIDES}

cp -r "\$scratch_log_dir" "\$DATADIR/logs/" || { echo >&2 "Result file(s) copying failed (with a code \$?)!"; exit 4; }

if command -v clean_scratch >/dev/null 2>&1; then
    clean_scratch
fi
EOF
}

submit_restart_job() {
    local job_name="d3restart01"

    if [[ "$DRY_RUN" == "1" ]]; then
        if [[ "$SCHEDULER_TYPE" == "pbs" ]]; then
            echo "[DRY_RUN] qsub -N ${job_name} for ${RUN_NAME} (resume from ${SOURCE_CKPT})"
        else
            echo "[DRY_RUN] sbatch --job-name=${job_name} for ${RUN_NAME} (resume from ${SOURCE_CKPT})"
        fi
        return 0
    fi

    if [[ "$SCHEDULER_TYPE" == "pbs" ]]; then
        emit_job_body | qsub -N "$job_name"
    else
        mkdir -p "$TARGET_RUN_DIR"
        emit_job_body | sbatch \
            --job-name="$job_name" \
            --output="$TARGET_RUN_DIR/slurm-%j.out" \
            --error="$TARGET_RUN_DIR/slurm-%j.err"
    fi
}

# -------------------------
# Main
# -------------------------

enforce_max_steps_guard
submit_restart_job
