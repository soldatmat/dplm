#!/bin/bash

CSV_FILE=/storage/brno2/home/soldatmat/documents/terpene_synthases/output/dplm/sampled_lengths.csv
MODEL=/storage/brno2/home/soldatmat/documents/terpene_synthases/dplm/logs/TPS_dplm_650m_stage3_run_1/checkpoints/best.ckpt
SAVETO=/storage/brno2/home/soldatmat/documents/terpene_synthases/output/dplm/TPS_dplm_650m_stage3_run_1
TEMPERATURE=4.0



if [[ ! -f "$CSV_FILE" ]]; then
    echo "File not found: $CSV_FILE"
    exit 1
fi

mkdir -p $SAVETO



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
    job_name=TPS_dplm_generate_sl_"$length"_ns_"$count"_t_"$TEMPERATURE"
    echo $job_name

    qsub -v MODEL="$MODEL",SAVETO="$SAVETO",SEQ_LENS="$length",NUM_SEQS="$count",TEMPERATURE="$TEMPERATURE" \
        -N $job_name \
        -l walltime="$walltime" \
        run_generate_dplm.sh
done
