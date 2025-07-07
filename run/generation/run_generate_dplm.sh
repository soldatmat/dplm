#!/bin/bash
#PBS -N TPS_dplm_generate
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=40gb:mem=32gb:scratch_local=16gb
#PBS -l walltime=00:30:00

# :gpu_mem=40gb set empirically to filter out nodes where jobs don't finish in 30min
# RAM: 150m model uses 3gb, 650m model uses up to 18gb

# -----------------------------------------

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,

DATADIR=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm

module add mambaforge
mamba activate /storage/brno2/home/soldatmat/.conda/envs/dplm

cd $DATADIR

# Vars MODEL, SAVETO, SEQ_LENS and NUM_SEQS have to be initialized before running this script.
#
# If `strategy=gumbel_argmax` is used, `temperature=0.0` is used instead of the provided `temperature`.
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

MODEL_DIR=$(dirname "$MODEL")
mkdir -p "$SCRATCHDIR/checkpoints"
cp "$MODEL" "$SCRATCHDIR/checkpoints" || { echo >&2 "Error while copying model weights file!"; exit 2; }
cp -r "$MODEL_DIR/../.hydra" "$SCRATCHDIR" || { echo >&2 "Error while copying model config files!"; exit 3; }

MODEL_BASENAME=$(basename "$MODEL")
MODEL_SCRATCH="$SCRATCHDIR/checkpoints/$MODEL_BASENAME"

python generate_dplm_fixed.py \
  --model_name "$MODEL_SCRATCH" \
  --saveto  "$SAVETO" \
  --seq_lens "$SEQ_LENS" \
  --num_seqs "$NUM_SEQS" \
  --temperature "$TEMPERATURE"

clean_scratch
