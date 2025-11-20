#!/bin/bash
#PBS -N TPS_dplm_generate
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=40gb:mem=32gb
#PBS -l walltime=00:30:00

# :gpu_mem=40gb set empirically to filter out nodes where jobs don't finish in 30min
# RAM: 150m model uses 3gb, 650m model uses up to 18gb

# usage: qsub -N <job_name> -l walltime=<walltime> -v args="--seq_lens <seq_lens> [--num_seqs <num_seqs> --model_name <model> --architecture <architecture> --saveto <saveto> --temperature <temperature> --max_iter <max_iter> --seed <seed> --cond_position <cond_position> --cond_seq <cond_seq> --cache_dir <cache_dir>] [--from_huggingface | --no-from_huggingface] [--batch_lens_together | --no-batch_lens_together]" run_generate_dplm.sh

# -----------------------------------------

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,

DATADIR=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm
CACHE_DIR=/storage/brno2/home/soldatmat/.cache/huggingface/hub

module add mambaforge
mamba activate /storage/brno2/home/soldatmat/.conda/envs/dplm

cd $DATADIR

python generate_dplm_fixed.py $args

clean_scratch
