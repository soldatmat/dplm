#!/bin/bash
#PBS -N TPS_dplm_generate
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=40gb:mem=32gb
#PBS -l walltime=00:30:00

# :gpu_mem=40gb set empirically to filter out nodes where jobs don't finish in 30min
# RAM: 150m model uses 3gb, 650m model uses up to 18gb

# -----------------------------------------

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,

DATADIR=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm
CACHE_DIR=/storage/brno2/home/soldatmat/.cache/huggingface/hub

module add mambaforge
mamba activate /storage/brno2/home/soldatmat/.conda/envs/dplm

cd $DATADIR

python generate_dplm_fixed.py \
  --model_name "$MODEL" \
  --from_huggingface "$FROM_HUGGINGFACE" \
  --architecture "$ARCHITECTURE" \
  --saveto  "$SAVETO" \
  --seq_lens "$SEQ_LENS" \
  --num_seqs "$NUM_SEQS" \
  --temperature "$TEMPERATURE" \
  --cache_dir "$CACHE_DIR"

clean_scratch
