#!/bin/bash
#SBATCH --job-name=eval_batch
#SBATCH --output=eval_batch_%j.out
#SBATCH --error=eval_batch_%j.err
#SBATCH --chdir=/home/users/malvlai/224n-project
#SBATCH --partition=interactive
#SBATCH --qos=interactive
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
################################################################################
#  Batch: 4 remaining Baseline + 2 Self-Consistency + 2 MultiGen-Cu (non-shared)
#  NOTE: MultiGen-Cu shared (M2, M5) is running separately — do not run here.
#  Submit: sbatch run_batch_remaining.sh  (from 224n-project dir)
################################################################################

set -e
# Working dir set by #SBATCH --chdir; when run directly, ensure we're in script dir
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"

echo "=============================================="
echo "  Batch 1: 4 remaining Baseline (B2,B4,B6,B8)"
echo "  DC-Cumulative: 7B+14B on MMLU+GPQA"
echo "=============================================="
python run_all_evaluations.py \
    --backend vllm \
    --no_parallel \
    --approaches DynamicCheatsheet_Cumulative \
    --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct \
    --datasets MMLU_Pro_Engineering GPQA_Diamond \
    --max_samples -1 \
    --quantization 4bit \
    --resume

echo ""
echo "=============================================="
echo "  Batch 2: 2 remaining Self-Consistency (S1, S3)"
echo "  7B + 14B on MMLU_Pro_Engineering, default_sc5"
echo "=============================================="
python run_all_evaluations.py \
    --backend vllm \
    --no_parallel \
    --approaches default \
    --models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct \
    --datasets MMLU_Pro_Engineering \
    --max_samples -1 \
    --quantization 4bit \
    --self_consistency \
    --self_consistency_k 5 \
    --temperature 0.7 \
    --resume

echo ""
echo "=============================================="
echo "  Batch 3: 2 MultiGen-Cu non-shared (M3, M6)"
echo "  MMLU + GPQA, --no_shared_memory"
echo "=============================================="
python run_multi_agent_eval.py \
    --backend vllm \
    --approaches MultiGenerator_Cumulative \
    --datasets MMLU_Pro_Engineering GPQA_Diamond \
    --max_samples -1 \
    --quantization 4bit \
    --no_shared_memory \
    --resume

echo ""
echo "=============================================="
echo "  Batch complete."
echo "=============================================="
