#!/bin/bash
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --gres=gpu:a4000:2
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=dc-medium-ensemble
#SBATCH --output=logs/medium_ensemble_%j.log

# --- EDIT THESE PATHS ---
CONDA_ENV="224n"
PROJECT_DIR="$HOME/224n/224n-project"
# If your HF cache is on scratch, uncomment:
# export HF_HOME=/scratch/users/$USER/hf_cache
# -------------------------

set -e
mkdir -p "$PROJECT_DIR/logs"

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

echo "=== Medium Ensemble: 3 generators (32B each) → 32B curator ==="
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

python run_benchmark.py \
    --task "MathEquationBalancer" \
    --approach "MultiGenerator_Cumulative" \
    --model_name "Qwen/Qwen2.5-32B-Instruct" \
    --generator_model_names \
        "Qwen/Qwen2.5-32B-Instruct" \
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
        "Qwen/QwQ-32B" \
    --curator_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --use_local_models \
    --quantization "4bit" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatshet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --save_directory "results" \
    --additional_flag_for_save_path "MediumEnsemble_Local" \
    --max_n_samples 3

echo "=== Done: $(date) ==="
