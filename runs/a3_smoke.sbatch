#!/usr/bin/env bash
#SBATCH -A fta-26-15
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=a3_smoke
#SBATCH --output=/mnt/proj2/fta-26-15/documents/dplm/logs/a3_smoke/slurm-%j.out
#SBATCH --error=/mnt/proj2/fta-26-15/documents/dplm/logs/a3_smoke/slurm-%j.err

set -euo pipefail
mkdir -p /mnt/proj2/fta-26-15/documents/dplm/logs/a3_smoke
cd /mnt/proj2/fta-26-15/documents/dplm

export CUDA_VISIBLE_DEVICES=0,
export HF_HOME=/mnt/proj2/fta-26-15/.cache/huggingface
export HF_HUB_CACHE=/mnt/proj2/fta-26-15/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/mnt/proj2/fta-26-15/.cache/huggingface
export TRITON_CACHE_DIR=/mnt/proj2/fta-26-15/.cache/triton

module add mambaforge >/dev/null 2>&1 || true
ml Anaconda3 >/dev/null 2>&1 || true
ENV=/mnt/proj2/fta-26-15/.conda/envs/dplm
PY="conda run --no-capture-output -p $ENV python"

CFGS=(
  "tps/TPS_dplm_150m_class_prepend_first_cyclization_A3cfg"
  "tps/TPS_dplm_150m_class_first_cyclization_A3neigh"
  "tps/TPS_dplm_150m_class_prepend_first_cyclization_A3neigh"
)

rc=0
for cfg in "${CFGS[@]}"; do
  echo "==================== SMOKE: $cfg ===================="
  $PY runs/a3_smoke_test.py \
      experiment="$cfg" \
      name=a3_smoke \
      ++datamodule.num_workers=0 \
      ++datamodule.mini_run=true \
      ++callbacks.validate_with_enzyme_explorer.class_ids=[0] \
      hydra.run.dir=/mnt/proj2/fta-26-15/documents/dplm/logs/a3_smoke/${cfg//\//_} \
    && echo "SMOKE-RESULT $cfg PASS" \
    || { echo "SMOKE-RESULT $cfg FAIL"; rc=1; }
done
echo "==================== DONE rc=$rc ===================="
exit $rc
