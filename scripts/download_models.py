#!/usr/bin/env python3
"""
Pre-download models to the HuggingFace cache before running experiments.

Run this ONCE on a login/CPU node (no GPU needed) before submitting GPU jobs.
Models are cached at $HF_HOME (default: ~/.cache/huggingface).

To cache to a shared/scratch directory instead:
    export HF_HOME=/scratch/users/$USER/hf_cache
    python scripts/download_models.py

NOTE: Llama models require you to:
  1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  2. Run: huggingface-cli login
"""

import argparse
from huggingface_hub import snapshot_download

SMALL_ENSEMBLE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]

MEDIUM_ENSEMBLE_MODELS = [
    "Qwen/Qwen2.5-32B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/QwQ-32B",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ensemble", choices=["small", "medium", "all"], default="all",
        help="Which ensemble's models to download"
    )
    args = parser.parse_args()

    models = []
    if args.ensemble in ("small", "all"):
        models += SMALL_ENSEMBLE_MODELS
    if args.ensemble in ("medium", "all"):
        models += MEDIUM_ENSEMBLE_MODELS

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Downloading: {model_name}")
        print(f"{'='*60}")
        try:
            snapshot_download(model_name)
            print(f"Done: {model_name}")
        except Exception as e:
            print(f"FAILED: {model_name} — {e}")
            print("If this is a gated model (e.g. Llama), make sure you've accepted")
            print("the license on HuggingFace and run: huggingface-cli login")

    print(f"\n{'='*60}")
    print("All downloads complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
