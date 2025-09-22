#!/bin/bash

CSV_FILE="/storage/brno2/home/soldatmat/documents/terpene_synthases/output/dplm/sampled_lengths_1000.csv"

MODEL_NAME="TPS_dplm_650m_stage3_run_3" # "airkingbd/dplm_650m"
FROM_HUGGINGFACE=False
ARCHITECTURE="DiffusionProteinLanguageModel" # DiffusionProteinLanguageModel, DPLMClass
# CHECKPOINT="N-Step-Checkpoint_epoch=86_step=20000.ckpt"

TEMPERATURE=0.0 # 0.0, 1.0, 4.0, 8.0

safe_model_name=$(echo "$MODEL_NAME" | tr '/' '-')
SAVETO="/storage/brno2/home/soldatmat/documents/terpene_synthases/output/dplm/${safe_model_name}/sl1000_t${TEMPERATURE}"

if [[ "$FROM_HUGGINGFACE" == "True" ]]; then
    MODEL="$MODEL_NAME"
else
    MODEL="/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm/logs/${MODEL_NAME}/checkpoints/${CHECKPOINT}"
fi



if [[ ! -f "$CSV_FILE" ]]; then
    echo "File not found: $CSV_FILE"
    exit 1
fi

mkdir -p "$SAVETO"



# read -r header < "$CSV_FILE"
# if [[ "$header" != "length, count" ]] && [[ "$header" != "length,count" ]]; then
#     echo "Error: First line must be 'length,count'"
#     echo "Found: $header"
#     exit 1
# fi

tail -n +2 "$CSV_FILE" | while IFS=, read -r length count; do
    # Trim whitespace, newlines, and carriage returns
    length=$(echo "$length" | tr -d '\r' | xargs)
    count=$(echo "$count" | tr -d '\r' | xargs)

    total_tokens=$((length * count))
    TOKENS_PER_SECOND=40
    WALLTIME_RESERVE=900 # [seconds]
    WALLTIME_MINIMUM=300 # [seconds]
    walltime_seconds=$(( (total_tokens / TOKENS_PER_SECOND) + WALLTIME_RESERVE ))
    if (( walltime_seconds < WALLTIME_MINIMUM )); then
        walltime_seconds=$WALLTIME_MINIMUM
    fi
    walltime=$(printf '%02d:%02d:%02d' $((walltime_seconds/3600)) $(( (walltime_seconds%3600)/60 )) $((walltime_seconds%60)))

    # echo "length: $length, count: $count"
    job_name=generate_"$safe_model_name"_sl_"$length"_ns_"$count"_t_"$TEMPERATURE"
    echo $job_name

    qsub -v MODEL="$MODEL",FROM_HUGGINGFACE="$FROM_HUGGINGFACE",ARCHITECTURE="$ARCHITECTURE",SAVETO="$SAVETO",SEQ_LENS="$length",NUM_SEQS="$count",TEMPERATURE="$TEMPERATURE" \
        -N $job_name \
        -l walltime="$walltime" \
        run_generate_dplm.sh
done
