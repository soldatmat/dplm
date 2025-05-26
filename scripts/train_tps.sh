# conda activate dplm
# srun --nodes 1 --ntasks-per-node=1 --mem=128G --time=03:59:00 -p b32_128_gpu --gres=gpu:geforce_rtx_3090:1 -J DPLM --pty bash

cd ~/documents/dplm # to dplm parent folder

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,

# The effective batch size is GPU_number(8) * max_tokens(8192) * accumulate_grad_batcher(16), resulting in approximately 1 million.

max_tokens=4096
accumulate_grad_batches=256 # 16 * 8 (1 CPU instead of 8 GPUs) * 2 (4096 tokens instead of 8192 tokens)

exp=tps/$1
run_name=$2

python train.py \
    experiment=${exp} \
    name=${run_name} \
    datamodule.max_tokens=${max_tokens} \
    trainer.accumulate_grad_batches=${accumulate_grad_batches}
