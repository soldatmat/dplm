#!/bin/bash
#SBATCH --job-name=fold_3arch
#SBATCH --partition=qgpu_free
#SBATCH --account=fta-26-15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --output=/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08/logs/fold-%j.out
#SBATCH --error=/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08/logs/fold-%j.err

set -euo pipefail
export SE_STAGE=/mnt/proj2/fta-26-15/documents/dplm/structure_eval_3arch_2026-06-08
mkdir -p "$SE_STAGE/logs"
cd "$SE_STAGE"

PY=/mnt/proj2/fta-26-15/.conda/envs/dplm/bin/python
export TORCH_HOME=/mnt/proj2/fta-26-15/.cache/torch
export HF_HOME=/mnt/proj2/fta-26-15/.cache/huggingface
export TRITON_CACHE_DIR=/mnt/proj2/fta-26-15/.cache/triton
export TORCH_EXTENSIONS_DIR=/mnt/proj2/fta-26-15/.cache/torch_extensions
export PYTHONUNBUFFERED=1

echo "== env =="
$PY -c 'import torch; print("torch", torch.__version__, "cuda_avail=", torch.cuda.is_available())'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== fold (idempotent — skips existing PDBs) =="
$PY /mnt/proj2/fta-26-15/documents/dplm/runs/fold_structure_eval.py
echo "== done =="
