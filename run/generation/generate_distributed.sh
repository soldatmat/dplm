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

    # echo "length: $length, count: $count"
    job_name=TPS_dplm_generate_sl_"$length"_ns_"$count"_t_"$TEMPERATURE"
    echo $job_name

    qsub -v MODEL="$MODEL",SAVETO="$SAVETO",SEQ_LENS="$length",NUM_SEQS="$count",TEMPERATURE="$TEMPERATURE" \
        -N $job_name \
        run_generate_dplm.sh
done



