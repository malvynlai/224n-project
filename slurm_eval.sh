#!/bin/bash
#SBATCH --job-name=oss-eval
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --gres=gpu:a4000:4
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
mkdir -p logs

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi

# ── Environment ──────────────────────────────────────────────────
source /deep/u/malvlai/miniconda3/etc/profile.d/conda.sh
conda activate /deep2/u/malvlai/envs/cs224n

export HF_HOME=/deep2/u/malvlai/.cache/huggingface
export TRANSFORMERS_CACHE=/deep2/u/malvlai/.cache/huggingface
export HF_DATASETS_CACHE=/deep2/u/malvlai/.cache/huggingface/datasets
export PYTHONUNBUFFERED=1

cd /deep/u/malvlai/224n-project

# ── Sanity check: verify GPU is visible to PyTorch ────────────
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
else:
    print('ERROR: No GPU! Will be extremely slow on CPU.')
    exit(1)
"

# ── Full evaluation matrix ───────────────────────────────────────
# 5 models × 2 approaches × 7 datasets = 70 runs
#
# Models:
#   Qwen/Qwen2.5-0.5B-Instruct         (0.5B, ~1GB  VRAM)
#   Qwen/Qwen2.5-3B-Instruct           (3B,   ~6GB  VRAM)
#   microsoft/Phi-3-mini-4k-instruct    (3.8B, ~8GB  VRAM)
#   Qwen/Qwen2.5-7B-Instruct           (7B,   ~14GB VRAM)
#   mistralai/Mistral-7B-Instruct-v0.2  (7B,   ~14GB VRAM)
#
# Approaches:
#   default                       — single-shot baseline
#   DynamicCheatsheet_Cumulative  — self-improving cheatsheet
#
# Datasets:
#   AIME_2024, AIME_2025, AIME_2020_2024,
#   GPQA_Diamond, MMLU_Pro_Physics, MMLU_Pro_Engineering,
#   MathEquationBalancer

python run_all_evaluations.py \
    --quantization 4bit \
    --max_tokens 2048 \
    --temperature 0.0 \
    --max_num_rounds 1 \
    --no_code_execution \
    --save_dir results_oss \
    --resume

echo "=== Job $SLURM_JOB_ID finished at $(date) ==="
