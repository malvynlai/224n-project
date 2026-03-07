#!/usr/bin/env python3
"""
Run evaluation across ALL datasets in data/ with small OSS models.

Supports multi-GPU parallelism: small models are pinned to individual GPUs
and run concurrently, while large models that need sharding use all GPUs.

Usage examples:
  # Default (auto-detects GPUs, runs in parallel):
  python run_all_evaluations.py --max_samples 15

  # Quick sanity check:
  python run_all_evaluations.py --max_samples 5 --models Qwen/Qwen2.5-0.5B-Instruct

  # Force sequential mode (single GPU):
  python run_all_evaluations.py --no_parallel

  # 4-bit quantization (default):
  python run_all_evaluations.py --quantization 4bit

  # Worker mode (used internally by parallel orchestrator):
  python run_all_evaluations.py --_worker --_gpu 0 --models Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from dynamic_cheatsheet.language_model import LanguageModel
from dynamic_cheatsheet.utils.evaluation import (
    eval_equation_balancer,
    eval_for_exact_matching_with_no_punctuation,
    eval_for_multiple_choice,
)
from dynamic_cheatsheet.utils.extractor import extract_answer

# ─────────────────────────────────────────────────────────────────────
# MODEL CATALOGUE
# ─────────────────────────────────────────────────────────────────────

ALL_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

# bf16 VRAM in GB (approximate)
_MODEL_VRAM_BF16: Dict[str, float] = {
    "0.5B": 1.5, "1.5B": 3.5, "3B": 7, "mini-4k": 8,
    "7B": 15, "8B": 17, "Nemo": 25, "12B": 25, "14B": 29,
}

ALL_APPROACHES = [
    "default",
    "DynamicCheatsheet_Cumulative",
]

ALL_DATASETS = [
    "AIME_2024",
    "AIME_2025",
    "AIME_2020_2024",
    "GPQA_Diamond",
    "MMLU_Pro_Physics",
    "MMLU_Pro_Engineering",
    "MathEquationBalancer",
    "GSM8K",
]

EVAL_TYPE: Dict[str, str] = {
    "AIME_2024":            "exact",
    "AIME_2025":            "exact",
    "AIME_2020_2024":       "exact",
    "GPQA_Diamond":         "mcq",
    "MMLU_Pro_Physics":     "mcq",
    "MMLU_Pro_Engineering": "mcq",
    "MathEquationBalancer": "equation",
    "GSM8K":                "exact",
}

TASK_INPUT_SUFFIX: Dict[str, str] = {
    "AIME_2024": (
        " (Please provide your answer in the form of an integer, e.g., 1234, "
        "with no Markdown formatting or additional text; make sure to pay "
        "attention to the desired format of the final answer though.)"
    ),
    "AIME_2025": (
        " (Please provide your answer in the form of an integer, e.g., 1234, "
        "with no Markdown formatting or additional text; make sure to pay "
        "attention to the desired format of the final answer though.)"
    ),
    "AIME_2020_2024": (
        " (Please provide your answer in the form of an integer, e.g., 1234, "
        "with no Markdown formatting or additional text; make sure to pay "
        "attention to the desired format of the final answer though.)"
    ),
    "GSM8K": (
        " (Please provide your final numerical answer as a single "
        "number with no units, commas, or additional text.)"
    ),
}

TASK_INPUT_PREFIX: Dict[str, str] = {
    "MathEquationBalancer": (
        "Below is an equation with missing operators. Your task is to fill in "
        "the blanks with the correct mathematical operators: +, -, *, or /. "
        "Ensure that the equation is correct once the operators are added. "
        "The operators should be placed in the sequence they appear from left "
        "to right. Include the full equation with the operators filled in. "
        "For instance, for the equation 1 ? 2 ? 3 = 6, the correct answer is "
        "1 + 2 + 3 = 6.\n\nEquation: "
    ),
}

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

def setup_logging(save_dir: str, suffix: str = "") -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(save_dir, f"eval_{ts}{suffix}.log")

    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logging to {log_path}")
    return logger

log = logging.getLogger("eval")

# ─────────────────────────────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

GENERATOR_PROMPT   = read_file("prompts/generator_prompt.txt")
CURATOR_CUMULATIVE = read_file("prompts/curator_prompt_for_dc_cumulative.txt")


def format_input(task: str, raw_input: str, idx: int) -> str:
    text = f"Question #{idx+1}:\n{raw_input}"
    if task in TASK_INPUT_PREFIX:
        text = TASK_INPUT_PREFIX[task] + text
    if task in TASK_INPUT_SUFFIX:
        text = text + TASK_INPUT_SUFFIX[task]
    return text


def evaluate_one(task: str, raw_input: str, answer: str, target: str) -> bool:
    etype = EVAL_TYPE[task]
    if etype == "exact":
        return eval_for_exact_matching_with_no_punctuation(
            answer.lower(), target.lower()
        )
    elif etype == "mcq":
        return eval_for_multiple_choice(raw_input, answer, target)
    elif etype == "equation":
        return eval_equation_balancer(None, answer, target)
    return False


# ─────────────────────────────────────────────────────────────────────
# Resume support
# ─────────────────────────────────────────────────────────────────────

def run_key(model_name: str, approach: str, task: str) -> str:
    return f"{model_name}|{approach}|{task}"


def find_completed_runs(save_dir: str) -> set:
    completed = set()
    save_path = Path(save_dir)
    if not save_path.exists():
        return completed
    for task_dir in save_path.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for jsonl in task_dir.glob("*.jsonl"):
            stem = jsonl.stem
            parts = stem.rsplit("_", 2)
            if len(parts) >= 3:
                short_model = parts[0]
                approach = parts[1]
                for m in ALL_MODELS:
                    if m.split("/")[-1] == short_model:
                        completed.add(run_key(m, approach, task))
                        break
    return completed


# ─────────────────────────────────────────────────────────────────────
# GPU helpers
# ─────────────────────────────────────────────────────────────────────

def free_gpu_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def estimate_vram_gb(model_name: str, quantize: str) -> float:
    """Rough VRAM estimate for a model in GB."""
    base = 15.0
    for key, gb in _MODEL_VRAM_BF16.items():
        if key in model_name:
            base = gb
            break
    if quantize == "4bit":
        return base / 4
    elif quantize == "8bit":
        return base / 2
    return base


def get_gpu_info():
    """Return (count, mem_gb_per_gpu) or (0, 0) if no GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0, 0
        count = torch.cuda.device_count()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return count, mem_gb
    except Exception:
        return 0, 0


def plan_gpu_assignment(
    models: List[str], quantize: str, gpu_count: int, gpu_mem_gb: float
) -> tuple:
    """
    Assign models to GPUs.

    Returns:
        single_gpu_plan: dict mapping gpu_id -> [model_names]
        multi_gpu_models: list of models that need sharding across all GPUs
    """
    single_gpu = []
    multi_gpu = []

    for m in models:
        vram = estimate_vram_gb(m, quantize)
        if vram <= gpu_mem_gb * 0.9:
            single_gpu.append(m)
        else:
            multi_gpu.append(m)

    assignment: Dict[int, List[str]] = {i: [] for i in range(gpu_count)}
    for i, model in enumerate(single_gpu):
        assignment[i % gpu_count].append(model)

    return assignment, multi_gpu


# ─────────────────────────────────────────────────────────────────────
# Core evaluation loop for one (model, approach, dataset)
# ─────────────────────────────────────────────────────────────────────

def run_single(
    model: LanguageModel,
    model_name: str,
    approach: str,
    task: str,
    *,
    max_samples: int = -1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_num_rounds: int = 1,
    execute_code: bool = True,
    save_dir: str = "results_oss",
    shuffle_seed: int = 10,
) -> Dict:
    dataset = load_from_disk(f"data/{task}")
    rng = np.random.RandomState(shuffle_seed)
    n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    dataset = dataset.select(indices)

    cheatsheet = "(empty)"
    cheatsheet_template = (
        CURATOR_CUMULATIVE if approach == "DynamicCheatsheet_Cumulative" else "(empty)"
    )

    outputs: List[dict] = []
    generator_outputs_so_far: List[str] = []
    correct = 0
    total = 0
    short_model_name = model_name.split("/")[-1]
    t0 = time.time()

    pbar = tqdm(range(n), desc=f"{short_model_name}|{approach[:12]}|{task}", unit="q", leave=True)
    for idx in pbar:
        example = dataset[idx]
        raw_input = example["input"]
        target = example["target"]
        input_txt = format_input(task, raw_input, idx)

        try:
            output_dict = model.advanced_generate(
                approach_name=approach,
                input_txt=input_txt,
                cheatsheet=cheatsheet,
                generator_template=GENERATOR_PROMPT,
                cheatsheet_template=cheatsheet_template,
                temperature=temperature,
                max_tokens=max_tokens,
                max_num_rounds=max_num_rounds,
                allow_code_execution=execute_code,
                code_execution_flag="EXECUTE CODE!",
                original_input_corpus=None,
                original_input_embeddings=None,
                generator_outputs_so_far=generator_outputs_so_far,
                retrieve_top_k=3,
            )
        except Exception as exc:
            log.error(f"  [ERROR] idx={idx}: {exc}\n{traceback.format_exc()}")
            output_dict = {
                "final_answer": "",
                "final_output": f"ERROR: {exc}",
                "final_cheatsheet": cheatsheet,
                "steps": [],
            }
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass

        final_answer = output_dict.get("final_answer", "")
        generator_outputs_so_far.append(output_dict.get("final_output", ""))

        if approach == "DynamicCheatsheet_Cumulative" and output_dict.get("final_cheatsheet"):
            cheatsheet = output_dict["final_cheatsheet"]

        is_correct = evaluate_one(task, raw_input, final_answer, target)
        if is_correct:
            correct += 1
        total += 1

        outputs.append({
            "idx": idx,
            "input": input_txt,
            "raw_input": raw_input,
            "target": target,
            "final_answer": final_answer,
            "correct": is_correct,
            **output_dict,
        })

        acc = correct / total
        pbar.set_postfix(acc=f"{acc:.1%}", correct=f"{correct}/{total}")

    elapsed = time.time() - t0
    accuracy = correct / total if total > 0 else 0.0

    short_model = model_name.split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(save_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{short_model}_{approach}_{ts}.jsonl")
    with open(out_path, "w") as f:
        for row in outputs:
            f.write(json.dumps(row, default=str) + "\n")

    summary = {
        "model": model_name,
        "approach": approach,
        "task": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "output_file": out_path,
    }
    log.info(f"  >> {task} done: {correct}/{total} = {accuracy:.1%} in {elapsed:.0f}s -> {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# Batched evaluation (vLLM) — for approaches without sequential deps
# ─────────────────────────────────────────────────────────────────────

def run_single_batched(
    model: LanguageModel,
    model_name: str,
    task: str,
    *,
    max_samples: int = -1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    save_dir: str = "results_oss",
    shuffle_seed: int = 10,
) -> Dict:
    """
    Batched evaluation for the 'default' approach (no cheatsheet dependency).

    All questions are formatted up-front and processed in a single vLLM
    batch_generate() call, giving massive throughput gains via continuous
    batching and prefix caching.
    """
    dataset = load_from_disk(f"data/{task}")
    rng = np.random.RandomState(shuffle_seed)
    n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    dataset = dataset.select(indices)

    short_model_name = model_name.split("/")[-1]
    t0 = time.time()

    # Phase 1: format all prompts
    histories = []
    raw_inputs = []
    input_txts = []
    targets = []
    for idx in range(n):
        example = dataset[idx]
        raw_input = example["input"]
        target = example["target"]
        input_txt = format_input(task, raw_input, idx)
        prompt = GENERATOR_PROMPT.replace(
            "[[QUESTION]]", input_txt
        ).replace("[[CHEATSHEET]]", "(empty)")
        histories.append([{"role": "user", "content": prompt}])
        raw_inputs.append(raw_input)
        input_txts.append(input_txt)
        targets.append(target)

    # Phase 2: batch generate
    log.info(f"  Batch-generating {n} prompts for {short_model_name}|default|{task}...")
    all_outputs = model.batch_generate(
        histories, temperature=temperature, max_tokens=max_tokens,
    )
    batch_time = time.time() - t0
    log.info(f"  Batch generation done in {batch_time:.1f}s ({batch_time/n:.2f}s/prompt)")

    # Phase 3: extract answers and evaluate
    outputs: List[dict] = []
    correct = 0
    for idx in range(n):
        final_answer = extract_answer(all_outputs[idx])
        is_correct = evaluate_one(task, raw_inputs[idx], final_answer, targets[idx])
        if is_correct:
            correct += 1
        outputs.append({
            "idx": idx,
            "input": input_txts[idx],
            "raw_input": raw_inputs[idx],
            "target": targets[idx],
            "final_answer": final_answer,
            "correct": is_correct,
            "generator_output": all_outputs[idx],
            "final_output": all_outputs[idx],
        })

    elapsed = time.time() - t0
    accuracy = correct / n if n > 0 else 0.0

    short_model = model_name.split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(save_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{short_model}_default_{ts}.jsonl")
    with open(out_path, "w") as f:
        for row in outputs:
            f.write(json.dumps(row, default=str) + "\n")

    summary = {
        "model": model_name,
        "approach": "default",
        "task": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "elapsed_s": round(elapsed, 1),
        "output_file": out_path,
    }
    log.info(f"  >> {task} done: {correct}/{n} = {accuracy:.1%} in {elapsed:.0f}s -> {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# Sequential runner (1 model at a time, original behavior)
# ─────────────────────────────────────────────────────────────────────

def run_sequential(args, models, approaches, datasets, completed):
    all_summaries: List[Dict] = []
    total_runs = len(models) * len(approaches) * len(datasets)
    skip_count = sum(
        1 for m in models for a in approaches for d in datasets
        if run_key(m, a, d) in completed
    )
    runs_to_do = total_runs - skip_count
    run_pbar = tqdm(total=runs_to_do, desc="Overall runs", unit="run", position=0)

    summary_path = os.path.join(args.save_dir, "summary_latest.json")

    for model_name in models:
        model_runs_needed = any(
            run_key(model_name, a, d) not in completed
            for a in approaches for d in datasets
        )
        if not model_runs_needed:
            log.info(f"\n[SKIP] All runs for {model_name} already completed.")
            continue

        log.info(f"\n{'─'*72}")
        log.info(f"Loading model: {model_name}  (quantization={args.quantization})")
        log.info(f"{'─'*72}")

        backend = getattr(args, 'backend', 'hf')
        try:
            model = LanguageModel(
                model_name=model_name,
                use_local_models=True,
                quantization=args.quantization,
                backend=backend,
            )
        except Exception as exc:
            log.error(f"  [SKIP] Failed to load {model_name}: {exc}")
            log.error(traceback.format_exc())
            continue

        use_batch = (backend == "vllm")

        for approach in approaches:
            for task in datasets:
                key = run_key(model_name, approach, task)

                if key in completed:
                    log.info(f"\n  SKIP (already done): "
                             f"{model_name.split('/')[-1]} | {approach} | {task}")
                    continue

                log.info(f"\n  >>> {model_name.split('/')[-1]} | {approach} | {task}")

                try:
                    if use_batch and approach == "default":
                        summary = run_single_batched(
                            model=model,
                            model_name=model_name,
                            task=task,
                            max_samples=args.max_samples,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            save_dir=args.save_dir,
                        )
                    else:
                        summary = run_single(
                            model=model,
                            model_name=model_name,
                            approach=approach,
                            task=task,
                            max_samples=args.max_samples,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            max_num_rounds=args.max_num_rounds,
                            execute_code=not args.no_code_execution,
                            save_dir=args.save_dir,
                        )
                    all_summaries.append(summary)
                except Exception as exc:
                    log.error(f"  [FAIL] {key}: {exc}")
                    log.error(traceback.format_exc())
                    all_summaries.append({
                        "model": model_name, "approach": approach, "task": task,
                        "accuracy": None, "correct": 0, "total": 0,
                        "elapsed_s": 0, "error": str(exc),
                    })
                run_pbar.update(1)

                with open(summary_path, "w") as f:
                    json.dump(all_summaries, f, indent=2, default=str)

        del model
        free_gpu_memory()
        log.info(f"  [Unloaded {model_name.split('/')[-1]}, freed GPU memory]")

    run_pbar.close()
    return all_summaries


# ─────────────────────────────────────────────────────────────────────
# Parallel runner (multiple models on different GPUs simultaneously)
# ─────────────────────────────────────────────────────────────────────

def run_parallel(args, models, approaches, datasets, completed):
    """
    Orchestrate parallel GPU workers via subprocesses.

    Each subprocess gets CUDA_VISIBLE_DEVICES set to a single GPU and runs
    its assigned models sequentially. Models that don't fit on one GPU are
    run afterwards using all GPUs with device_map=auto.
    """
    import torch

    gpu_count = torch.cuda.device_count()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    log.info(f"\n  Parallel mode: {gpu_count} GPUs detected ({gpu_mem_gb:.1f} GB each)")

    assignment, multi_gpu_models = plan_gpu_assignment(
        models, args.quantization, gpu_count, gpu_mem_gb
    )

    for gpu_id, gpu_models in assignment.items():
        if gpu_models:
            log.info(f"  GPU {gpu_id}: {[m.split('/')[-1] for m in gpu_models]}")
    if multi_gpu_models:
        log.info(f"  Multi-GPU (sequential): {[m.split('/')[-1] for m in multi_gpu_models]}")

    # --- Phase 1: Launch single-GPU workers in parallel ---
    processes = []
    for gpu_id, gpu_models in assignment.items():
        if not gpu_models:
            continue

        cmd = [
            sys.executable, __file__,
            "--_worker",
            "--_gpu", str(gpu_id),
            "--models", *gpu_models,
            "--approaches", *approaches,
            "--datasets", *datasets,
            "--quantization", args.quantization,
            "--backend", getattr(args, 'backend', 'hf'),
            "--max_tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--max_num_rounds", str(args.max_num_rounds),
            "--max_samples", str(args.max_samples),
            "--save_dir", args.save_dir,
        ]
        if args.no_code_execution:
            cmd.append("--no_code_execution")
        if args.resume:
            cmd.append("--resume")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log.info(f"  Launching worker on GPU {gpu_id} for {len(gpu_models)} model(s)...")
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        processes.append((gpu_id, gpu_models, proc))

    active = {gpu_id: proc for gpu_id, _, proc in processes}

    while active:
        finished = []
        for gpu_id, proc in active.items():
            ret = proc.poll()
            if ret is not None:
                remaining = proc.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        log.info(f"  [GPU {gpu_id}] {line}")
                finished.append(gpu_id)
                if ret != 0:
                    log.error(f"  [GPU {gpu_id}] Worker exited with code {ret}")
                else:
                    log.info(f"  [GPU {gpu_id}] Worker finished successfully")
            else:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    log.info(f"  [GPU {gpu_id}] {line.rstrip()}")

        for gpu_id in finished:
            del active[gpu_id]

        if active:
            time.sleep(1)

    # --- Phase 2: Run multi-GPU models sequentially ---
    multi_gpu_summaries = []
    if multi_gpu_models:
        log.info(f"\n  Running {len(multi_gpu_models)} multi-GPU model(s) sequentially...")
        multi_gpu_summaries = run_sequential(
            args, multi_gpu_models, approaches, datasets, completed
        )

    # --- Collect summaries from worker JSONL outputs ---
    all_summaries = multi_gpu_summaries
    for gpu_id, gpu_models, _ in processes:
        for model_name in gpu_models:
            short = model_name.split("/")[-1]
            for approach in approaches:
                for task in datasets:
                    task_dir = Path(args.save_dir) / task
                    matches = sorted(task_dir.glob(f"{short}_{approach}_*.jsonl"))
                    if matches:
                        latest = matches[-1]
                        rows = []
                        with open(latest) as f:
                            for line in f:
                                rows.append(json.loads(line))
                        correct = sum(1 for r in rows if r.get("correct"))
                        total = len(rows)
                        all_summaries.append({
                            "model": model_name, "approach": approach, "task": task,
                            "accuracy": correct / total if total else 0,
                            "correct": correct, "total": total,
                            "output_file": str(latest),
                        })

    return all_summaries


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run evaluation: OSS models × approaches × datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help=f"HF model IDs. Default: all → {[m.split('/')[-1] for m in ALL_MODELS]}",
    )
    p.add_argument(
        "--approaches", nargs="+", default=None,
        help=f"Approach names. Default: {ALL_APPROACHES}",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Dataset names. Default: {ALL_DATASETS}",
    )
    p.add_argument("--quantization", default="4bit",
                    choices=["none", "4bit", "8bit"])
    p.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                    help="Inference backend: 'hf' (HuggingFace transformers) or "
                         "'vllm' (faster, supports batch generation)")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_num_rounds", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=15,
                    help="Cap samples per dataset (-1 = run all)")
    p.add_argument("--no_code_execution", action="store_true")
    p.add_argument("--save_dir", default="results_oss")
    p.add_argument("--resume", action="store_true",
                    help="Skip already-completed (model, approach, dataset) triples")
    p.add_argument("--no_parallel", action="store_true",
                    help="Disable multi-GPU parallelism (run sequentially)")
    # Internal flags for worker subprocesses
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_gpu", type=int, default=None, help=argparse.SUPPRESS)
    return p


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    suffix = f"_gpu{args._gpu}" if args._worker and args._gpu is not None else ""
    logger = setup_logging(args.save_dir, suffix=suffix)

    models     = args.models     or ALL_MODELS
    approaches = args.approaches or ALL_APPROACHES
    datasets   = args.datasets   or ALL_DATASETS

    completed = find_completed_runs(args.save_dir) if args.resume else set()

    # ── GPU availability check ─────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{gpu_count}x {gpu_name} ({gpu_mem:.1f} GB)"
        else:
            gpu_count = 0
            gpu_info = "NO GPU DETECTED — will run on CPU (very slow!)"
    except ImportError:
        gpu_count = 0
        gpu_info = "torch not installed"

    total_runs = len(models) * len(approaches) * len(datasets)
    skip_count = sum(
        1 for m in models for a in approaches for d in datasets
        if run_key(m, a, d) in completed
    )

    use_parallel = (
        not args.no_parallel
        and not args._worker
        and gpu_count > 1
    )

    log.info("=" * 72)
    log.info("  OSS Model Evaluation Run")
    log.info("=" * 72)
    log.info(f"  GPU:          {gpu_info}")
    log.info(f"  Backend:      {args.backend}")
    log.info(f"  Quantization: {args.quantization}")
    log.info(f"  Parallel:     {use_parallel} ({gpu_count} GPUs)")
    log.info(f"  Models ({len(models)}):")
    for m in models:
        vram = estimate_vram_gb(m, args.quantization)
        log.info(f"    - {m}  (~{vram:.1f} GB)")
    log.info(f"  Approaches ({len(approaches)}): {approaches}")
    log.info(f"  Datasets ({len(datasets)}): {datasets}")
    log.info(f"  Max tokens: {args.max_tokens}   Temperature: {args.temperature}")
    log.info(f"  Max rounds: {args.max_num_rounds}   Code exec: {not args.no_code_execution}")
    log.info(f"  Samples:    {'all' if args.max_samples <= 0 else args.max_samples}")
    log.info(f"  Total runs: {total_runs}  (skipping {skip_count} already completed)")
    if args._worker:
        log.info(f"  Worker mode: GPU {args._gpu}")
    log.info("=" * 72)

    # ── Dispatch ──────────────────────────────────────────────────
    if use_parallel:
        all_summaries = run_parallel(args, models, approaches, datasets, completed)
    else:
        all_summaries = run_sequential(args, models, approaches, datasets, completed)

    # ── Final summary table ──────────────────────────────────────────
    if not args._worker:
        log.info("\n" + "=" * 72)
        log.info("  RESULTS SUMMARY")
        log.info("=" * 72)
        header = (f"{'Model':<42} {'Approach':<30} {'Dataset':<22} "
                  f"{'Acc':>7} {'N':>5} {'Time':>7}")
        log.info(header)
        log.info("-" * len(header))
        for s in all_summaries:
            short = s['model'].split('/')[-1]
            acc = f"{s['accuracy']:.1%}" if s.get('accuracy') is not None else "ERROR"
            elapsed = s.get('elapsed_s', 0)
            log.info(f"{short:<42} {s['approach']:<30} {s['task']:<22} "
                     f"{acc:>7} {s['total']:>5} {elapsed:>6.0f}s")

        os.makedirs(args.save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        summary_path = os.path.join(args.save_dir, f"summary_{ts}.json")
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)
        log.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
