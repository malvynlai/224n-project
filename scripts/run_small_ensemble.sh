#!/bin/bash
#SBATCH --account=deep
#SBATCH --partition=deep
#SBATCH --gres=gpu:a4000:2
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=dc-small-ensemble
#SBATCH --output=logs/small_ensemble_%j.log

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

echo "=== Small Ensemble: 3 generators (7B, 8B, 14B) → 14B curator ==="
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start: $(date)"

python run_benchmark.py \
    --task "MathEquationBalancer" \
    --approach "MultiGenerator_Cumulative" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --generator_model_names \
        "Qwen/Qwen2.5-7B-Instruct" \
        "meta-llama/Llama-3.1-8B-Instruct" \
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    --curator_model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    --use_local_models \
    --quantization "4bit" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatshet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --save_directory "results" \
    --additional_flag_for_save_path "SmallEnsemble_Local" \
    --max_n_samples 3

echo "=== Done: $(date) ==="
