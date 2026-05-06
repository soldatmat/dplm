#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Generate and evaluate sequences from a trained checkpoint using EnzymeExplorer.
#
# Syntax:
#   [VAR=value ...] bash runs/run_checkpoint_evaluation.sh
#
# Examples:
#   DRY_RUN=1 bash runs/run_checkpoint_evaluation.sh
#   CLUSTER=karolina CHECKPOINT=path/to/checkpoint.ckpt DRY_RUN=1 bash runs/run_checkpoint_evaluation.sh
#   CLUSTER=karolina CHECKPOINT=path/to/checkpoint.ckpt bash runs/run_checkpoint_evaluation.sh
#   EXISTING_FASTA_PATH=logs/my_eval/generated_sequences/sampled.fasta DRY_RUN=1 bash runs/run_checkpoint_evaluation.sh
#
# Common overrides: DATADIR, CHECKPOINT, EXISTING_FASTA_PATH, CLUSTER, SCHEDULER_TYPE, SCHEDULER_RESOURCE_SPEC, JOB_*, GEN_*, METACENTRUM_MAMBA_ENV, KAROLINA_CONDA_ENV.

# -------------------------
# User-Configurable Variables
# -------------------------

DATADIR="${DATADIR:-$REPO_ROOT}"
CHECKPOINT="${CHECKPOINT:-logs/TPS_dplm_150m_stage3_run_8/checkpoints/N-Step-Checkpoint_epoch=172_step=20000.ckpt}"
EXISTING_FASTA_PATH="${EXISTING_FASTA_PATH:-}"
EVAL_NAME="${EVAL_NAME:-}"
EVAL_EXPERIMENT="${EVAL_EXPERIMENT:-tps/TPS_dplm_150m_stage3}"
DRY_RUN="${DRY_RUN:-0}"

CLUSTER="${CLUSTER:-metacentrum}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-}"
SCHEDULER_RESOURCE_SPEC="${SCHEDULER_RESOURCE_SPEC:-}"

JOB_NODES="${JOB_NODES:-1}"
JOB_NCPUS="${JOB_NCPUS:-1}"
JOB_NGPUS="${JOB_NGPUS:-1}"
JOB_PARTITION="${JOB_PARTITION:-qgpu}"
JOB_GPU_MEM="${JOB_GPU_MEM:-46gb}"
JOB_MEM_PBS="${JOB_MEM_PBS:-64gb}"
JOB_MEM_SLURM="${JOB_MEM_SLURM:-64G}"
JOB_SCRATCH_LOCAL="${JOB_SCRATCH_LOCAL:-40gb}"
JOB_WALLTIME="${JOB_WALLTIME:-48:00:00}"
PROJECT_ID="${PROJECT_ID:-}"

GEN_NUM_SEQS="${GEN_NUM_SEQS:-10}"
GEN_SEQ_LENS="${GEN_SEQ_LENS:-330}"
GEN_USE_TEMPLATE="${GEN_USE_TEMPLATE:-true}"
GEN_SEQUENCE_COLUMN_NAME="${GEN_SEQUENCE_COLUMN_NAME:-Aminoacid_sequence}"
GEN_TEMPLATE_LENGTH_COLUMN_NAME="${GEN_TEMPLATE_LENGTH_COLUMN_NAME:-length}"
GEN_TEMPLATE_COUNT_COLUMN_NAME="${GEN_TEMPLATE_COUNT_COLUMN_NAME:-count}"
GEN_MAX_ITER="${GEN_MAX_ITER:-500}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-256}"
GEN_BATCH_LENS_TOGETHER="${GEN_BATCH_LENS_TOGETHER:-true}"
GEN_SAMPLING_STRATEGY="${GEN_SAMPLING_STRATEGY:-gumbel_argmax}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-1.0}"

ENZYME_EXPLORER_TEMPLATE_SEQS="${ENZYME_EXPLORER_TEMPLATE_SEQS:-/mnt/proj2/fta-26-15/documents/output/dplm/sampled_lengths_1000.csv}"
ENZYME_EXPLORER_CHECKPOINT_DIR="${ENZYME_EXPLORER_CHECKPOINT_DIR:-data-bin/checkpoints/enzymeexplorer}"
ENZYME_EXPLORER_DETECTION_THRESHOLD="${ENZYME_EXPLORER_DETECTION_THRESHOLD:-0.0}"
ENZYME_EXPLORER_DETECT_PRECURSOR_SYNTHASES="${ENZYME_EXPLORER_DETECT_PRECURSOR_SYNTHASES:-true}"

METACENTRUM_MAMBA_ENV="${METACENTRUM_MAMBA_ENV:-/storage/brno2/home/soldatmat/.conda/envs/dplm}"
KAROLINA_CONDA_ENV="${KAROLINA_CONDA_ENV:-/mnt/proj2/fta-26-15/.conda/envs/dplm}"

# Cache locations (shared across runs to avoid re-downloads and home-dir issues).
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/proj2/fta-26-15/.cache/triton}"

DEFAULT_RESOURCE_SPEC=""

resolve_path_under_datadir() {
    local path_value="$1"
    local datadir_basename
    datadir_basename="$(basename "$DATADIR")"

    if [[ "$path_value" == /* ]]; then
        printf '%s' "$path_value"
        return
    fi

    path_value="${path_value#./}"
    if [[ "$path_value" == "$datadir_basename/"* ]]; then
        path_value="${path_value#"$datadir_basename/"}"
    fi

    printf '%s/%s' "$DATADIR" "$path_value"
}

sanitize_name_component() {
    local value="$1"
    value="${value// /_}"
    value="${value//[^a-zA-Z0-9._-]/-}"
    value="${value##[-_]}"
    value="${value%%[-_]}"
    if [[ -z "$value" ]]; then
        value="unknown"
    fi
    printf '%s' "$value"
}

derive_eval_name_from_checkpoint() {
    local checkpoint_path="$1"
    local run_name=""
    local checkpoint_name

    checkpoint_name="$(basename "$checkpoint_path")"
    checkpoint_name="${checkpoint_name%.ckpt}"

    if [[ "$checkpoint_path" =~ /logs/([^/]+)/checkpoints/ ]]; then
        run_name="${BASH_REMATCH[1]}"
    fi

    run_name="$(sanitize_name_component "$run_name")"
    checkpoint_name="$(sanitize_name_component "$checkpoint_name")"

    printf 'checkpoint_eval_%s_%s' "$run_name" "$checkpoint_name"
}

derive_eval_name_from_fasta() {
    local fasta_path="$1"
    local fasta_name

    fasta_name="$(basename "$fasta_path")"
    fasta_name="${fasta_name%.fasta}"
    fasta_name="${fasta_name%.fa}"
    fasta_name="${fasta_name%.faa}"
    fasta_name="$(sanitize_name_component "$fasta_name")"

    printf 'checkpoint_eval_existingfasta_%s' "$fasta_name"
}

derive_seqlen_suffix() {
    local seq_lens_value="$1"
    local first_len

    first_len="${seq_lens_value%%,*}"
    first_len="${first_len%%:*}"
    first_len="${first_len// /}"
    first_len="$(sanitize_name_component "$first_len")"

    printf 'seqlen%s' "$first_len"
}

derive_template_suffix() {
    local template_path="$1"
    local template_name

    template_name="$(basename "$template_path")"
    template_name="${template_name%.csv}"
    template_name="$(sanitize_name_component "$template_name")"

    printf 'tpl%s' "$template_name"
}

# Normalize path-like inputs so the script is robust regardless of caller CWD.
DATADIR="$(cd "$DATADIR" && pwd)"
if [[ -n "$CHECKPOINT" ]]; then
    CHECKPOINT="$(resolve_path_under_datadir "$CHECKPOINT")"
fi
if [[ -n "$EXISTING_FASTA_PATH" ]]; then
    EXISTING_FASTA_PATH="$(resolve_path_under_datadir "$EXISTING_FASTA_PATH")"
fi
ENZYME_EXPLORER_TEMPLATE_SEQS="$(resolve_path_under_datadir "$ENZYME_EXPLORER_TEMPLATE_SEQS")"
ENZYME_EXPLORER_CHECKPOINT_DIR="$(resolve_path_under_datadir "$ENZYME_EXPLORER_CHECKPOINT_DIR")"
if [[ -z "$EVAL_NAME" ]]; then
    if [[ -n "$EXISTING_FASTA_PATH" ]]; then
        EVAL_NAME="$(derive_eval_name_from_fasta "$EXISTING_FASTA_PATH")"
    else
        EVAL_NAME="$(derive_eval_name_from_checkpoint "$CHECKPOINT")"
        case "${GEN_USE_TEMPLATE,,}" in
            false|0|no)
                EVAL_NAME+="_$(derive_seqlen_suffix "$GEN_SEQ_LENS")"
                ;;
            *)
                EVAL_NAME+="_$(derive_template_suffix "$ENZYME_EXPLORER_TEMPLATE_SEQS")"
                ;;
        esac
    fi
    EVAL_NAME="checkpoint_eval/$EVAL_NAME"
fi

if [[ -n "$EXISTING_FASTA_PATH" && ! -f "$EXISTING_FASTA_PATH" ]]; then
    echo >&2 "EXISTING_FASTA_PATH does not exist: $EXISTING_FASTA_PATH"
    exit 1
fi

if [[ -n "$EXISTING_FASTA_PATH" && -n "$CHECKPOINT" ]]; then
    echo >&2 "Warning: EXISTING_FASTA_PATH is set, so generation is skipped and CHECKPOINT is ignored."
fi

# Ensure output directory exists before scheduler submission so scheduler-managed logs can land there.
mkdir -p "$DATADIR/logs/$EVAL_NAME"

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
    if [[ -z "$PROJECT_ID" ]]; then
        echo >&2 "For CLUSTER='karolina', PROJECT_ID must be set (e.g., PROJECT_ID=fta-26-15)."
        exit 1
    fi
    # Karolina typically uses Slurm. Override as needed for your partition/QoS.
    printf -v DEFAULT_RESOURCE_SPEC '%s\n%s\n%s\n%s\n%s\n%s' \
        "#SBATCH --nodes=${JOB_NODES}" \
        "#SBATCH --partition=${JOB_PARTITION}" \
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

# -------------------------
# Script Logic
# -------------------------

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$SCHEDULER_TYPE" == "pbs" ]]; then
        echo "[DRY_RUN] qsub for checkpoint evaluation: $CHECKPOINT"
    else
        echo "[DRY_RUN] sbatch for checkpoint evaluation: $CHECKPOINT"
    fi
    if [[ -n "$EXISTING_FASTA_PATH" ]]; then
        echo "[DRY_RUN] existing FASTA mode: $EXISTING_FASTA_PATH"
    fi
    echo "[DRY_RUN] eval logs folder: $DATADIR/logs/$EVAL_NAME"
    exit 0
fi

if [[ "$SCHEDULER_TYPE" == "pbs" ]]; then
    qsub -N "chkpt_eval" <<PBES_SCRIPT
#!/bin/bash
${SCHEDULER_RESOURCE_SPEC}

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,

DATADIR="${DATADIR}"
CHECKPOINT="${CHECKPOINT}"
EXISTING_FASTA_PATH="${EXISTING_FASTA_PATH}"
EVAL_NAME="${EVAL_NAME}"
HYDRA_RUN_DIR="$DATADIR/logs/$EVAL_NAME/hydra"

mkdir -p "$DATADIR/logs/$EVAL_NAME"
job_id="\${PBS_JOBID:-\${SLURM_JOB_ID:-unknown}}"
echo "\$job_id is running on node \$(hostname -f)" >> "$DATADIR/logs/$EVAL_NAME/job_info.txt"

if [[ -n "\${SCRATCHDIR:-}" ]]; then
    work_scratch="\$SCRATCHDIR"
else
    work_scratch="\$(mktemp -d /tmp/dplm_eval_XXXXXX)"
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

mkdir -p "\$work_scratch/dplm"
cp -r "$DATADIR/data-bin" "\$work_scratch/dplm/data-bin/" || { echo >&2 "Error while copying input file(s)!"; exit 2; }

module add mambaforge
mamba activate "$METACENTRUM_MAMBA_ENV"

cd "$DATADIR"

cd "$DATADIR/runs"

python evaluate_checkpoint.py --config-path ../configs \
    experiment='"$EVAL_EXPERIMENT"' \
    +checkpoint_path='"$CHECKPOINT"' \
    +input_fasta_path='"$EXISTING_FASTA_PATH"' \
    +eval_name='"$EVAL_NAME"' \
    +datadir='"$DATADIR"' \
    hydra.run.dir='"$DATADIR/logs/$EVAL_NAME/hydra"' \
    +gen_use_template="${GEN_USE_TEMPLATE}" \
    +gen_num_seqs="${GEN_NUM_SEQS}" \
    +gen_seq_lens="${GEN_SEQ_LENS}" \
    +gen_sequence_column_name='"${GEN_SEQUENCE_COLUMN_NAME}"' \
    +gen_template_length_column_name='"${GEN_TEMPLATE_LENGTH_COLUMN_NAME}"' \
    +gen_template_count_column_name='"${GEN_TEMPLATE_COUNT_COLUMN_NAME}"' \
    +gen_max_iter="${GEN_MAX_ITER}" \
    +gen_batch_size="${GEN_BATCH_SIZE}" \
    +gen_batch_lens_together="${GEN_BATCH_LENS_TOGETHER}" \
    +gen_sampling_strategy="${GEN_SAMPLING_STRATEGY}" \
    +gen_temperature="${GEN_TEMPERATURE}" \
    +enzyme_explorer_template_seqs="${ENZYME_EXPLORER_TEMPLATE_SEQS}" \
    +enzyme_explorer_checkpoint_dir="${ENZYME_EXPLORER_CHECKPOINT_DIR}" \
    +enzyme_explorer_detection_threshold="${ENZYME_EXPLORER_DETECTION_THRESHOLD}" \
    +enzyme_explorer_detect_precursor_synthases="${ENZYME_EXPLORER_DETECT_PRECURSOR_SYNTHASES}"

if command -v clean_scratch >/dev/null 2>&1; then
    clean_scratch
fi
PBES_SCRIPT
else
    sbatch --job-name="chkpt_eval" -A "$PROJECT_ID" \
        --output="$DATADIR/logs/$EVAL_NAME/slurm-%j.out" \
        --error="$DATADIR/logs/$EVAL_NAME/slurm-%j.out" <<SLURM_SCRIPT
#!/bin/bash
${SCHEDULER_RESOURCE_SPEC}

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,

DATADIR="${DATADIR}"
CHECKPOINT="${CHECKPOINT}"
EXISTING_FASTA_PATH="${EXISTING_FASTA_PATH}"
EVAL_NAME="${EVAL_NAME}"
HYDRA_RUN_DIR="$DATADIR/logs/$EVAL_NAME/hydra"

mkdir -p "$DATADIR/logs/$EVAL_NAME"
job_id="\${PBS_JOBID:-\${SLURM_JOB_ID:-unknown}}"
echo "\$job_id is running on node \$(hostname -f)" >> "$DATADIR/logs/$EVAL_NAME/job_info.txt"

if [[ -n "\${SCRATCHDIR:-}" ]]; then
    work_scratch="\$SCRATCHDIR"
else
    work_scratch="\$(mktemp -d /tmp/dplm_eval_XXXXXX)"
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

mkdir -p "\$work_scratch/dplm"
cp -r "$DATADIR/data-bin" "\$work_scratch/dplm/data-bin/" || { echo >&2 "Error while copying input file(s)!"; exit 2; }

ml Anaconda3

# Prefer conda shell hook on Karolina and activate only the target env.
set +u
if ! command -v conda >/dev/null 2>&1; then
    echo >&2 "conda command not found after loading Anaconda3 module"
    exit 3
fi
CONDA_BASE="\$(conda info --base 2>/dev/null || true)"
if [[ -n "\$CONDA_BASE" && -f "\$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "\$CONDA_BASE/etc/profile.d/conda.sh"
else
    eval "\$(conda shell.bash hook)"
fi
if [[ "$KAROLINA_CONDA_ENV" == /* ]]; then
    CONDA_RUN_ARGS=( -p "$KAROLINA_CONDA_ENV" )
else
    CONDA_RUN_ARGS=( -n "$KAROLINA_CONDA_ENV" )
fi
set -u

cd "$DATADIR"

cd "$DATADIR/runs"

conda run "\${CONDA_RUN_ARGS[@]}" python evaluate_checkpoint.py --config-path ../configs \
    experiment='"$EVAL_EXPERIMENT"' \
    +checkpoint_path='"$CHECKPOINT"' \
    +input_fasta_path='"$EXISTING_FASTA_PATH"' \
    +eval_name='"$EVAL_NAME"' \
    +datadir='"$DATADIR"' \
    hydra.run.dir='"$DATADIR/logs/$EVAL_NAME/hydra"' \
    +gen_use_template="${GEN_USE_TEMPLATE}" \
    +gen_num_seqs="${GEN_NUM_SEQS}" \
    +gen_seq_lens="${GEN_SEQ_LENS}" \
    +gen_sequence_column_name='"${GEN_SEQUENCE_COLUMN_NAME}"' \
    +gen_template_length_column_name='"${GEN_TEMPLATE_LENGTH_COLUMN_NAME}"' \
    +gen_template_count_column_name='"${GEN_TEMPLATE_COUNT_COLUMN_NAME}"' \
    +gen_max_iter="${GEN_MAX_ITER}" \
    +gen_batch_size="${GEN_BATCH_SIZE}" \
    +gen_batch_lens_together="${GEN_BATCH_LENS_TOGETHER}" \
    +gen_sampling_strategy="${GEN_SAMPLING_STRATEGY}" \
    +gen_temperature="${GEN_TEMPERATURE}" \
    +enzyme_explorer_template_seqs="${ENZYME_EXPLORER_TEMPLATE_SEQS}" \
    +enzyme_explorer_checkpoint_dir="${ENZYME_EXPLORER_CHECKPOINT_DIR}" \
    +enzyme_explorer_detection_threshold="${ENZYME_EXPLORER_DETECTION_THRESHOLD}" \
    +enzyme_explorer_detect_precursor_synthases="${ENZYME_EXPLORER_DETECT_PRECURSOR_SYNTHASES}"

if command -v clean_scratch >/dev/null 2>&1; then
    clean_scratch
fi
SLURM_SCRIPT
fi
