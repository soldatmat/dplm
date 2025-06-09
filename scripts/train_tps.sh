cd ~/documents/dplm # to dplm parent folder

#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,

exp=tps/$1
run_name=$2

python train.py \
    experiment=${exp} \
    name=${run_name}
