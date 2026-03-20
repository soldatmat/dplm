#!/bin/bash
#PBS -N TPS_dplm_150m_stage3
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=46gb:mem=64gb:scratch_local=40gb
#PBS -l walltime=24:00:00

# 14871 global steps in 24 hours with 1 GPU, max_tokens: 4096

# 200_000 global steps, 17.5 hours, 1 GPU H100 NVL, max_tokens: 8192
# select=1:ncpus=1:ngpus=1:gpu_mem=46gb:mem=16gb:scratch_local=40gb

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,
exp=tps/TPS_dplm_150m_stage3
run_name=TPS_dplm_150m_stage3_run_debug
model_checkpoint=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm/logs/TPS_dplm_150m_stage3_run_8/checkpoints/best.ckpt
# model_checkpoint=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm/logs/TPS_dplm_150m_stage3_run_debug/checkpoints/init.ckpt

DATADIR=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm
run_log_dir=$DATADIR/logs/${run_name}

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually
mkdir -p "$run_log_dir"
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> "$run_log_dir"/job_info.txt

# test if the scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# Stage the source checkpoint directly in scratch for this run.
scratch_ckpt_dir=$SCRATCHDIR/dplm/logs/${run_name}/checkpoints
initial_checkpoint_source=$model_checkpoint
mkdir -p "$scratch_ckpt_dir"
cp "$model_checkpoint" "$scratch_ckpt_dir"/init.ckpt || { echo >&2 "Error while copying initial checkpoint to scratch!"; exit 2; }
cp "$scratch_ckpt_dir"/init.ckpt "$scratch_ckpt_dir"/last.ckpt || { echo >&2 "Error while creating last.ckpt from init.ckpt in scratch!"; exit 2; }
echo "Checkpoint source: $initial_checkpoint_source" >> "$run_log_dir"/job_info.txt
echo "Checkpoint copied to: $scratch_ckpt_dir/init.ckpt" >> "$run_log_dir"/job_info.txt
echo "Checkpoint copied to: $scratch_ckpt_dir/last.ckpt" >> "$run_log_dir"/job_info.txt
model_checkpoint="$scratch_ckpt_dir"/last.ckpt
mkdir -p $SCRATCHDIR/tmp
export TMPDIR=$SCRATCHDIR/tmp

# if the copy operation fails, issue an error message and exit
mkdir -p $SCRATCHDIR/dplm
cp -r $DATADIR/data-bin $SCRATCHDIR/dplm/data-bin/ || { echo >&2 "Error while copying input file(s)!"; exit 2; }

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"
# queue job to retrieve the scratch directory if the job is terminated
trap "qsub retrieve_scrath.sh $PBS_JOBID $SCRATCHDIR/dplm/logs/${run_name} $DATADIR/logs/" TERM



module add mambaforge
mamba activate /storage/brno2/home/soldatmat/.conda/envs/dplm

cd $DATADIR

HYDRA_FULL_ERROR=1 # TODO delete
python train.py \
    experiment=${exp} \
    name=${run_name} \
    paths.data_dir=$SCRATCHDIR/dplm/data-bin \
    paths.log_dir=$SCRATCHDIR/dplm/logs/${run_name} \
    train.ckpt_path=$model_checkpoint
    # model.net.name=${model_checkpoint} \

    # root_dir=$DATADIR # is already true by default
    # ckpt_dir=$SCRATCHDIR/dplm/logs/${run_name}/checkpoints # is set by log_dir by default



# move the output to user's DATADIR or exit in case of failure
cp -r $SCRATCHDIR/dplm/logs/${run_name} $DATADIR/logs/ || { echo >&2 "Result file(s) copying failed (with a code $?)!"; exit 4; }

# clean the SCRATCH directory
clean_scratch
