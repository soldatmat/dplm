#!/bin/bash

# Usage: ./generate_distributed.sh data.csv

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
    # echo "length: $length, count: $count"
    qsub -v MODEL="$MODEL",SAVETO="$SAVETO",SEQ_LENS="$length",NUM_SEQS="$count" run_generate_dplm.sh
done



