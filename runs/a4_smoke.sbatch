#!/usr/bin/env bash
#SBATCH -A fta-26-15
#SBATCH --partition=qgpu_exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=a4_smoke
#SBATCH --output=/mnt/proj2/fta-26-15/documents/dplm/logs/a4_smoke/slurm-%j.out
#SBATCH --error=/mnt/proj2/fta-26-15/documents/dplm/logs/a4_smoke/slurm-%j.err

set -euo pipefail
mkdir -p /mnt/proj2/fta-26-15/documents/dplm/logs/a4_smoke
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

# Each entry: "<label>|<experiment>|<extra hydra overrides>"
# Reuses runs/a3_smoke_test.py (verifies conditioning, trainable params, null
# fraction + force_null when cfg_dropout>0, 3 train steps, gen at w=0 and w=2).
RUNS=(
  "run1_pre_neighcfg|tps/TPS_dplm_150m_class_prepend_first_cyclization_A4neighcfg|"
  "run2_ca_neighcfg|tps/TPS_dplm_150m_class_first_cyclization_A4neighcfg|"
  "run3_ca_neigh|tps/TPS_dplm_150m_class_first_cyclization_A3neigh|"
  "run4_pre_neigh_r8|tps/TPS_dplm_150m_class_prepend_first_cyclization_A3neigh|++model.decoder.lora.lora_rank=8 ++model.decoder.lora.lora_alpha=16"
  "run5_ca_neigh_Vca|tps/TPS_dplm_150m_class_first_cyclization_A4neighVca|"
)

rc=0
for entry in "${RUNS[@]}"; do
  IFS='|' read -r label cfg extra <<< "$entry"
  echo "==================== SMOKE: $label ($cfg) ===================="
  $PY runs/a3_smoke_test.py \
      experiment="$cfg" \
      name=a4_smoke \
      ++datamodule.num_workers=0 \
      ++datamodule.mini_run=true \
      ++callbacks.validate_with_enzyme_explorer.class_ids=[0] \
      $extra \
      hydra.run.dir=/mnt/proj2/fta-26-15/documents/dplm/logs/a4_smoke/${label} \
    && echo "SMOKE-RESULT $label PASS" \
    || { echo "SMOKE-RESULT $label FAIL"; rc=1; }
done
echo "==================== DONE rc=$rc ===================="
exit $rc
