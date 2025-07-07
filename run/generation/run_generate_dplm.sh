#!/bin/bash
#PBS -N TPS_dplm_generate
#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb
#PBS -l walltime=00:30:00

# removed as unnecessary: :gpu_mem=46gb

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,

DATADIR=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/logs/${run_name}/job_info.txt

module add mambaforge
mamba activate /storage/brno2/home/soldatmat/.conda/envs/dplm

cd $DATADIR

# Vars MODEL, SAVETO, SEQ_LENS and NUM_SEQS have to be initialized before running this script.
#
# If `strategy=gumbel_argmax` is used, `temperature=0.0` is used instead of the provided `temperature`.
python generate_dplm_fixed.py \
  --model_name "$MODEL" \
  --saveto  "$SAVETO" \
  --seq_lens "$SEQ_LENS" \
  --num_seqs "$NUM_SEQS" \
  --temperature "$TEMPERATURE"
