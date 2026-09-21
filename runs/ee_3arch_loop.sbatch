#!/bin/bash
#SBATCH --job-name=ee_3arch
#SBATCH --partition=qgpu_free
#SBATCH --account=fta-26-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --output=/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08/logs/ee-%j.out
#SBATCH --error=/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08/logs/ee-%j.err

set -euo pipefail
export SE_STAGE=/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08
mkdir -p "$SE_STAGE/logs"
cd "$SE_STAGE"

# Karolina compute nodes ship an older system libstdc++ lacking GLIBCXX_3.4.29
# (needed by the enzyme_explorer env's pandas). Prepend the env's libstdc++.
export LD_LIBRARY_PATH="/mnt/proj2/fta-26-15/.conda/envs/enzyme_explorer/lib:${LD_LIBRARY_PATH:-}"

PY_DPLM=/mnt/proj2/fta-26-15/.conda/envs/dplm/bin/python
EE_SH=/mnt/proj2/fta-26-15/documents/tps_eval/scripts/run_enzyme_explorer.sh

echo "== EE-full (structure-aware) over all staged runs =="
nvidia-smi --query-gpu=name --format=csv,noheader
n=0
for d in "$SE_STAGE"/inputs/*/; do
    label=$(basename "$d")
    out="$d/sequences_enzyme_explorer.csv"
    if [[ -s "$out" ]]; then
        echo "-- $label: already done ($out), skipping"
        continue
    fi
    n=$((n+1))
    echo "===== [$n] EE for $label ====="
    bash "$EE_SH" \
        --sequences_csv_path "$d/sequences.csv" \
        --structs_dir "$SE_STAGE/structures/$label"
done
echo "== ran EE for $n run(s) =="

echo "== land per-run into <run>/enzyme_explorer_structure_validation/<step>/ =="
$PY_DPLM /mnt/proj2/fta-26-15/documents/dplm/runs/land_structure_eval_results.py

echo "== done =="
