#!/usr/bin/env python3
"""
Multi-agent evaluation: 3 generator models + 1 curator model.

Generators independently solve each problem, then majority vote picks the answer.
In DC-cumulative mode, the curator also maintains an evolving cheatsheet.

Models (from baseline benchmark):
  Generators: Qwen/Qwen2.5-1.5B-Instruct,
              Qwen/Qwen2.5-3B-Instruct,
              mistralai/Mistral-7B-Instruct-v0.2
  Curator:    Qwen/Qwen2.5-7B-Instruct

Approaches:
  MultiGenerator              — majority vote only (no cheatsheet)
  MultiGenerator_Cumulative   — majority vote + evolving cheatsheet

Usage:
  python run_multi_agent_eval.py --max_samples 15
  python run_multi_agent_eval.py --max_samples 15 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List

import numpy as np

from datasets import load_from_disk
from tqdm import tqdm

from dynamic_cheatsheet.language_model import LanguageModel
from dynamic_cheatsheet.utils.evaluation import (
    eval_for_exact_matching_with_no_punctuation,
    eval_for_multiple_choice,
    eval_equation_balancer,
)
from dynamic_cheatsheet.utils.extractor import extract_answer

log = logging.getLogger("multi_agent_eval")

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

GENERATOR_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

CURATOR_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ALL_APPROACHES = [
    "MultiGenerator",
    "MultiGenerator_Cumulative",
]

ALL_DATASETS = [
    "AIME_2024",
    "AIME_2025",
    "AIME_2020_2024",
    "GPQA_Diamond",
    "MMLU_Pro_Physics",
    "MMLU_Pro_Engineering",
    "MathEquationBalancer",
]

GENERATOR_PROMPT_PATH = "prompts/generator_prompt.txt"
CURATOR_PROMPT_PATH = "prompts/curator_prompt_for_dc_cumulative.txt"


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


GENERATOR_PROMPT = read_file(GENERATOR_PROMPT_PATH)
CURATOR_PROMPT = read_file(CURATOR_PROMPT_PATH)


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────

def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(save_dir, f"eval_{ts}.log")

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(sh)
    log.info(f"Logging to {log_path}")
    return log


# ─────────────────────────────────────────────────────────────────────
# Input formatting (matches baseline)
# ─────────────────────────────────────────────────────────────────────

def format_input(task: str, raw_input: str, idx: int) -> str:
    txt = f"Question #{idx+1}:\n{raw_input}"
    if task in ("AIME_2020_2024", "AIME_2024", "AIME_2025"):
        txt += (" (Please provide your answer in the form of an integer, "
                "e.g., 1234, with no Markdown formatting or additional text; "
                "make sure to pay attention to the desired format of the "
                "final answer though.)")
    elif task == "MathEquationBalancer":
        txt = (
            "Below is an equation with missing operators. Your task is to "
            "fill in the blanks with the correct mathematical operators: "
            "+, -, *, or /. Ensure that the equation is correct once the "
            "operators are added. The operators should be placed in the "
            "sequence they appear from left to right. Include the full "
            "equation with the operators filled in. For instance, for the "
            "equation 1 ? 2 ? 3 = 6, the correct answer is 1 + 2 + 3 = 6."
            f"\n\nEquation: {txt}"
        )
    return txt


def evaluate_answer(task: str, input_txt: str, answer: str, target: str) -> bool:
    if task in ("AIME_2025", "AIME_2024", "AIME_2020_2024"):
        return eval_for_exact_matching_with_no_punctuation(
            answer.lower(), target.lower()
        )
    elif task in ("GPQA_Diamond", "MMLU_Pro_Engineering", "MMLU_Pro_Physics"):
        return eval_for_multiple_choice(input_txt, answer, target)
    elif task == "MathEquationBalancer":
        return eval_equation_balancer(None, answer, target)
    else:
        raise ValueError(f"Unknown task: {task}")


# ─────────────────────────────────────────────────────────────────────
# Resume support
# ─────────────────────────────────────────────────────────────────────

def run_key(approach: str, task: str) -> str:
    gen_short = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_short = CURATOR_MODEL.split("/")[-1]
    return f"{gen_short}__{cur_short}__{approach}__{task}"


def find_completed_runs(save_dir: str) -> set:
    completed = set()
    for root, _dirs, files in os.walk(save_dir):
        for f in files:
            if not f.endswith(".jsonl"):
                continue
            parts = f.replace(".jsonl", "").rsplit("_", 1)
            if len(parts) < 2:
                continue
            name_part = parts[0]
            for approach in ALL_APPROACHES:
                if f"_{approach}_" in f:
                    for dataset in ALL_DATASETS:
                        if f.startswith(dataset) or f"/{dataset}/" in os.path.join(root, f):
                            task = os.path.basename(root)
                            key = run_key(approach, task)
                            fp = os.path.join(root, f)
                            try:
                                with open(fp) as fh:
                                    lines = fh.readlines()
                                if len(lines) > 0:
                                    completed.add(key)
                            except Exception:
                                pass
    return completed


# ─────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────

def run_single(
    model: LanguageModel,
    approach: str,
    task: str,
    *,
    max_samples: int = -1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_num_rounds: int = 1,
    execute_code: bool = True,
    save_dir: str = "results_multi_agent",
    shuffle_seed: int = 10,
) -> Dict:
    dataset = load_from_disk(f"data/{task}")
    rng = np.random.RandomState(shuffle_seed)
    n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    dataset = dataset.select(indices)

    cheatsheet = "(empty)"
    cheatsheet_template = (
        CURATOR_PROMPT if approach == "MultiGenerator_Cumulative" else "(empty)"
    )

    outputs: List[dict] = []
    generator_outputs_so_far: List[str] = []
    correct = 0
    total = 0
    t0 = time.time()

    gen_short = "+".join(m.split("/")[-1].split("-Instruct")[0] for m in GENERATOR_MODELS)
    pbar = tqdm(range(n), desc=f"{gen_short}|{approach[:12]}|{task}", unit="q", leave=True)

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
            }

        final_answer = output_dict.get("final_answer", "")
        cheatsheet = output_dict.get("final_cheatsheet", cheatsheet) or cheatsheet
        generator_outputs_so_far.append(output_dict.get("final_output", ""))

        is_correct = evaluate_answer(task, input_txt, final_answer, target)
        if is_correct:
            correct += 1
        total += 1
        acc = correct / total if total else 0

        per_gen = output_dict.get("all_generator_answers")
        if per_gen:
            log.info(f"  Q{idx+1} per-generator answers: {per_gen}")
        log.info(f"  Q{idx+1} final_answer={final_answer!r}  target={target!r}  correct={is_correct}")

        pbar.set_postfix(acc=f"{acc:.1%}", correct=f"{correct}/{total}")

        outputs.append({
            "input": input_txt,
            "target": target,
            "raw_input": raw_input,
            **output_dict,
            "is_correct": is_correct,
        })

    elapsed = time.time() - t0
    accuracy = correct / total if total else 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    gen_tag = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_tag = CURATOR_MODEL.split("/")[-1]
    fname = f"{gen_tag}__{cur_tag}_{approach}_{ts}.jsonl"
    out_dir = os.path.join(save_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        for row in outputs:
            f.write(json.dumps(row) + "\n")

    log.info(f"  => {accuracy:.1%} ({correct}/{total}) in {elapsed:.0f}s  -> {out_path}")

    return {
        "generators": [m.split("/")[-1] for m in GENERATOR_MODELS],
        "curator": CURATOR_MODEL.split("/")[-1],
        "approach": approach,
        "dataset": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_s": round(elapsed, 1),
        "output_path": out_path,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-agent evaluation")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_num_rounds", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=15,
                    help="Cap samples per dataset (-1 = run all)")
    p.add_argument("--no_code_execution", action="store_true")
    p.add_argument("--save_dir", default="results_multi_agent")
    p.add_argument("--resume", action="store_true",
                    help="Skip already-completed (approach, dataset) triples")
    p.add_argument("--quantization", default="4bit",
                    choices=["none", "4bit", "8bit"])
    p.add_argument("--approaches", nargs="+", default=None,
                    help=f"Approaches to run. Default: {ALL_APPROACHES}")
    p.add_argument("--datasets", nargs="+", default=None,
                    help=f"Datasets to run. Default: {ALL_DATASETS}")
    return p


def main():
    args = build_parser().parse_args()
    setup_logging(args.save_dir)

    approaches = args.approaches or ALL_APPROACHES
    datasets = args.datasets or ALL_DATASETS

    completed = find_completed_runs(args.save_dir) if args.resume else set()

    # ── GPU info ────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{gpu_count}x {gpu_name} ({gpu_mem:.1f} GB)"
        else:
            gpu_info = "NO GPU — will run on CPU (very slow!)"
    except ImportError:
        gpu_info = "torch not installed"

    total_runs = len(approaches) * len(datasets)
    skip_count = sum(
        1 for a in approaches for d in datasets
        if run_key(a, d) in completed
    )

    log.info("=" * 72)
    log.info("  Multi-Agent Evaluation")
    log.info("=" * 72)
    log.info(f"  GPU:          {gpu_info}")
    log.info(f"  Quantization: {args.quantization}")
    log.info(f"  Generators ({len(GENERATOR_MODELS)}):")
    for m in GENERATOR_MODELS:
        log.info(f"    - {m}")
    log.info(f"  Curator:      {CURATOR_MODEL}")
    log.info(f"  Approaches:   {approaches}")
    log.info(f"  Datasets ({len(datasets)}): {datasets}")
    log.info(f"  Max tokens:   {args.max_tokens}   Temperature: {args.temperature}")
    log.info(f"  Max rounds:   {args.max_num_rounds}   Code exec: {not args.no_code_execution}")
    log.info(f"  Samples:      {'all' if args.max_samples <= 0 else args.max_samples}")
    log.info(f"  Total runs:   {total_runs}  (skipping {skip_count} already completed)")
    log.info("=" * 72)

    # ── Load model ──────────────────────────────────────────────────
    log.info("\nInitializing LanguageModel with multi-generator setup...")
    model = LanguageModel(
        model_name=CURATOR_MODEL,
        generator_model_names=GENERATOR_MODELS,
        curator_model_name=CURATOR_MODEL,
        use_local_models=True,
        quantization=args.quantization,
    )
    log.info("  LanguageModel ready (models loaded on-demand by LocalModelManager)")

    # ── Run evaluations ─────────────────────────────────────────────
    all_summaries = []
    runs_to_do = total_runs - skip_count
    summary_path = os.path.join(args.save_dir, "summary_latest.json")

    run_pbar = tqdm(total=runs_to_do, desc="Overall runs", unit="run", position=0)

    for task in datasets:
        for approach in approaches:
            key = run_key(approach, task)
            if key in completed:
                log.info(f"\n  SKIP (already done): {approach} | {task}")
                continue

            log.info(f"\n  >>> {approach} | {task}")

            try:
                summary = run_single(
                    model=model,
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

            run_pbar.update(1)

            # Save incremental summary after each (approach, dataset) run
            with open(summary_path, "w") as f:
                json.dump(all_summaries, f, indent=2)
            log.info(f"  [checkpoint] summary saved to {summary_path}")

    run_pbar.close()

    # ── Final summary table ──────────────────────────────────────────
    log.info("\n" + "=" * 72)
    log.info("  RESULTS SUMMARY")
    log.info("=" * 72)
    header = f"{'Approach':<30} {'Dataset':<22} {'Acc':>7} {'N':>5} {'Time':>7}"
    log.info(header)
    log.info("-" * len(header))
    for s in all_summaries:
        log.info(
            f"{s['approach']:<30} {s['dataset']:<22} "
            f"{s['accuracy']:>6.1%} {s['total']:>5} "
            f"{s['elapsed_s']:>6.0f}s"
        )
    log.info("=" * 72)

    log.info(f"\nGenerators: {[m.split('/')[-1] for m in GENERATOR_MODELS]}")
    log.info(f"Curator:    {CURATOR_MODEL.split('/')[-1]}")

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    final_summary_path = os.path.join(args.save_dir, f"summary_{ts}.json")
    with open(final_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"Final summary saved to {final_summary_path}")


if __name__ == "__main__":
    main()
