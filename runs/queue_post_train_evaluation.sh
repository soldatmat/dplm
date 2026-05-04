#!/usr/bin/env bash
# Post-train evaluation driver for a DPLM training run.
#
# For each N-Step checkpoint listed in STEPS:
#   1. Submit a generation job (NUM_SEQS seqs, length SEQ_LEN) into
#      <RUN_DIR>/post_train_evaluation/step_<N>/.
#   2. Submit four evaluation jobs that depend on the generation job:
#        - EnzymeExplorer sequence-only
#        - max_sequence_identity vs training fasta
#        - max_sequence_identity self
#        - motif_search
#
# Required:
#   RUN_DIR         Path to the training run directory (contains checkpoints/ and
#                   will receive post_train_evaluation/<step>/ subdirs).
#
# Optional (env vars):
#   JOB_PREFIX      Prefix for all submitted job names (default: pte). Pass a
#                   run-specific prefix (e.g. JOB_PREFIX=pte41) to disambiguate
#                   when evaluating multiple runs concurrently.
#   STEPS_OVERRIDE  Space-separated list of steps to evaluate
#                   (default: 10000..200000 every 10000).
#   TRAIN_FASTA     Reference fasta for max_sequence_identity vs train.
#   NUM_SEQS, SEQ_LEN, ACCOUNT, QGEN, JOBS_DIR, PROJECT_ROOT, SKIP_EVAL
#                   Standard knobs (see defaults below).
#   DRY_RUN=1       Print sbatch commands without submitting.
#
# Usage examples:
#   RUN_DIR=/path/to/run bash queue_post_train_evaluation.sh
#   DRY_RUN=1 RUN_DIR=/path/to/run JOB_PREFIX=pte42 bash queue_post_train_evaluation.sh

set -euo pipefail

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "RUN_DIR is required (path to the training run directory)." >&2
    echo "Usage: RUN_DIR=/path/to/run bash $(basename "$0")" >&2
    exit 1
fi

CKPT_DIR="$RUN_DIR/checkpoints"
EVAL_ROOT="$RUN_DIR/post_train_evaluation"

TRAIN_FASTA="${TRAIN_FASTA:-/mnt/proj2/fta-26-15/documents/dplm/data-bin/MARTS-DB/2026-04-12/TPS_sequences.fasta}"

QGEN="${QGEN:-/mnt/proj2/fta-26-15/documents/dplm/runs/queue_generate_dplm_fixed.sh}"
JOBS_DIR="${JOBS_DIR:-/mnt/proj2/fta-26-15/documents/tps_eval/scripts/karolina/jobs}"
PROJECT_ROOT_EXPORT="${PROJECT_ROOT:-/mnt/proj2/fta-26-15/documents/dplm}"

NUM_SEQS="${NUM_SEQS:-500}"
SEQ_LEN="${SEQ_LEN:-350}"
ACCOUNT="${ACCOUNT:-fta-26-15}"
JOB_PREFIX="${JOB_PREFIX:-pte}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

if [[ -n "${STEPS_OVERRIDE:-}" ]]; then
    read -r -a STEPS <<< "$STEPS_OVERRIDE"
else
    STEPS=(10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
           110000 120000 130000 140000 150000 160000 170000 180000 190000 200000)
fi

# -------------------------
# Sanity checks
# -------------------------

[[ -d "$CKPT_DIR"   ]] || { echo "Checkpoint dir not found: $CKPT_DIR"   >&2; exit 1; }
[[ -f "$TRAIN_FASTA" ]] || { echo "Train fasta not found: $TRAIN_FASTA" >&2; exit 1; }
[[ -x "$QGEN"       ]] || [[ -f "$QGEN" ]] || { echo "Generation wrapper not found: $QGEN" >&2; exit 1; }
for s in enzyme_explorer_sequence_only.sh max_sequence_identity.sh motif_search.sh; do
    [[ -f "$JOBS_DIR/$s" ]] || { echo "Missing eval job script: $JOBS_DIR/$s" >&2; exit 1; }
done

mkdir -p "$EVAL_ROOT"

# -------------------------
# Helpers
# -------------------------

# Submit a generation job for step N. Echos the job id (or "DRY" in dry-run).
submit_generation() {
    local step="$1" ckpt="$2" save_dir="$3"
    local job_name="${JOB_PREFIX}_gen_s${step}"

    if [[ "$DRY_RUN" == "1" ]]; then
        cat >&2 <<EOF
[DRY_RUN][gen step=$step]
  PROJECT_ROOT=$PROJECT_ROOT_EXPORT CHECKPOINT=$ckpt SAVE_DIR=$save_dir INLINE_SEQ_LENS=$SEQ_LEN INLINE_NUM_SEQS=$NUM_SEQS \\
  sbatch --parsable -J $job_name \\
         --export=ALL,PROJECT_ROOT=$PROJECT_ROOT_EXPORT,CHECKPOINT=$ckpt,SAVE_DIR=$save_dir,INLINE_SEQ_LENS=$SEQ_LEN,INLINE_NUM_SEQS=$NUM_SEQS \\
         -o $save_dir/slurm-gen-%j.out -e $save_dir/slurm-gen-%j.err \\
         $QGEN
EOF
        echo "DRY${step}"
        return 0
    fi

    sbatch --parsable \
           -J "$job_name" \
           --export="ALL,PROJECT_ROOT=$PROJECT_ROOT_EXPORT,CHECKPOINT=$ckpt,SAVE_DIR=$save_dir,INLINE_SEQ_LENS=$SEQ_LEN,INLINE_NUM_SEQS=$NUM_SEQS" \
           -o "$save_dir/slurm-gen-%j.out" \
           -e "$save_dir/slurm-gen-%j.err" \
           "$QGEN"
}

# Submit one dependent eval job. $1=script, $2=job_name, $3=log_tag, $4=gen_id, $5=save_dir, rest=script args.
submit_eval() {
    local script="$1" job_name="$2" log_tag="$3" gen_id="$4" save_dir="$5"
    shift 5

    if [[ "$DRY_RUN" == "1" ]]; then
        cat <<EOF
[DRY_RUN][eval $log_tag] sbatch --parsable --dependency=afterok:$gen_id \\
    -J $job_name --account=$ACCOUNT \\
    -o $save_dir/slurm-${log_tag}-%j.out -e $save_dir/slurm-${log_tag}-%j.err \\
    $JOBS_DIR/$script $*
EOF
        return 0
    fi

    sbatch --parsable \
           --dependency=afterok:"$gen_id" \
           -J "$job_name" \
           --account="$ACCOUNT" \
           -o "$save_dir/slurm-${log_tag}-%j.out" \
           -e "$save_dir/slurm-${log_tag}-%j.err" \
           "$JOBS_DIR/$script" "$@"
}

# -------------------------
# Main loop
# -------------------------

total_gen=0
total_eval=0

for step in "${STEPS[@]}"; do
    matches=( "$CKPT_DIR"/N-Step-Checkpoint_epoch=*_step=${step}.ckpt )
    if (( ${#matches[@]} != 1 )) || [[ ! -f "${matches[0]}" ]]; then
        echo "Could not uniquely resolve checkpoint for step=$step (found ${#matches[@]} matches): ${matches[*]}" >&2
        exit 1
    fi
    ckpt="${matches[0]}"
    save_dir="$EVAL_ROOT/step_${step}"
    mkdir -p "$save_dir"

    fasta="$save_dir/generated_sequences.fasta"

    gen_id=$(submit_generation "$step" "$ckpt" "$save_dir")
    gen_id="${gen_id%%;*}"  # strip optional ";cluster" suffix from --parsable
    total_gen=$((total_gen + 1))
    echo "[step=$step] generation job id: $gen_id"

    if [[ "$SKIP_EVAL" == "1" ]]; then
        continue
    fi
    submit_eval enzyme_explorer_sequence_only.sh "${JOB_PREFIX}_ee_s${step}"        "ee"        "$gen_id" "$save_dir" --fasta_path "$fasta"
    submit_eval max_sequence_identity.sh         "${JOB_PREFIX}_msi_train_s${step}" "msi-train" "$gen_id" "$save_dir" --fasta_path "$fasta" --train_path "$TRAIN_FASTA"
    submit_eval max_sequence_identity.sh         "${JOB_PREFIX}_msi_self_s${step}"  "msi-self"  "$gen_id" "$save_dir" --fasta_path "$fasta"
    submit_eval motif_search.sh                  "${JOB_PREFIX}_motif_s${step}"     "motif"     "$gen_id" "$save_dir" --fasta_path "$fasta"
    total_eval=$((total_eval + 4))
done

echo
echo "Submitted $total_gen generation jobs and $total_eval evaluation jobs (DRY_RUN=$DRY_RUN)."
