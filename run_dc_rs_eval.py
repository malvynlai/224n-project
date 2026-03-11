#!/usr/bin/env python3
"""
DC-RS (Dynamic Cheatsheet with Retrieval & Synthesis) evaluation.

Implements the DC-RS variant from the paper:
  "Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory"
  https://arxiv.org/abs/2504.07952

DC-RS flow (per question i):
  1. Retrieve: R_i = Retr(x_i, {(x_j, ỹ_j)}_{j<i}, k)  — top-k similar past (input, output) pairs
  2. Curate:   M_i = Cur(M_{i-1}, x_i, R_i)          — synthesize memory from retrieved pairs (before generation)
  3. Generate: ỹ_i = Gen(x_i, M_i)                   — generate answer with updated memory

Unlike DC-Cu (Cumulative), DC-RS refines memory BEFORE responding and uses retrieval to surface
relevant past examples. This helps on diverse benchmarks (e.g. GPQA-Diamond).

Uses pre-computed embeddings from embeddings/{task}.csv when available; falls back to
sentence-transformers (pip install sentence-transformers) for datasets without embeddings.

Usage:
  python run_dc_rs_eval.py --models Qwen/Qwen2.5-7B-Instruct --datasets GSM8K --max_samples 50
  python run_dc_rs_eval.py --backend vllm --retrieve_top_k 5
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dynamic_cheatsheet.language_model import LanguageModel
from dynamic_cheatsheet.utils.evaluation import (
    eval_equation_balancer,
    eval_for_exact_matching_with_no_punctuation,
    eval_for_multiple_choice,
)
from dynamic_cheatsheet.utils.extractor import extract_answer, extract_cheatsheet

# ─────────────────────────────────────────────────────────────────────
# Config (aligned with run_all_evaluations)
# ─────────────────────────────────────────────────────────────────────

ALL_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
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
    "AIME_2024": "exact",
    "AIME_2025": "exact",
    "AIME_2020_2024": "exact",
    "GPQA_Diamond": "mcq",
    "MMLU_Pro_Physics": "mcq",
    "MMLU_Pro_Engineering": "mcq",
    "MathEquationBalancer": "equation",
    "GSM8K": "exact",
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

MAX_CHEATSHEET_WORDS = 800

log = logging.getLogger("dc_rs")


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(save_dir, f"dc_rs_eval_{ts}.log")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.info(f"Logging to {log_path}")
    return log


def format_input(task: str, raw_input: str, idx: int) -> str:
    text = f"Question #{idx+1}:\n{raw_input}"
    if task in TASK_INPUT_PREFIX:
        text = TASK_INPUT_PREFIX[task] + text
    if task in TASK_INPUT_SUFFIX:
        text = text + TASK_INPUT_SUFFIX[task]
    return text


def cap_cheatsheet(cheatsheet: str) -> str:
    if cheatsheet == "(empty)":
        return cheatsheet
    words = cheatsheet.split()
    if len(words) <= MAX_CHEATSHEET_WORDS:
        return cheatsheet
    log.warning(f"  Cheatsheet exceeded {MAX_CHEATSHEET_WORDS} words ({len(words)}), truncating")
    return " ".join(words[:MAX_CHEATSHEET_WORDS]) + "\n\n[...cheatsheet truncated]"


def evaluate_one(task: str, raw_input: str, answer: str, target: str) -> bool:
    etype = EVAL_TYPE.get(task, "exact")
    if etype == "exact":
        return eval_for_exact_matching_with_no_punctuation(answer.lower(), target.lower())
    elif etype == "mcq":
        return eval_for_multiple_choice(raw_input, answer, target)
    elif etype == "equation":
        return eval_equation_balancer(None, answer, target)
    return False


# ─────────────────────────────────────────────────────────────────────
# Embedding & retrieval (DC-RS)
# ─────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = "embeddings"


def load_precomputed_embeddings(task: str, dataset_size: int) -> Optional[np.ndarray]:
    """
    Load pre-computed embeddings from embeddings/{task}.csv.
    Returns (dataset_size, dim) array or None if file missing/invalid.
    CSV columns: input, tokens, embedding (embedding is string repr of list of floats).
    """
    path = os.path.join(EMBEDDINGS_DIR, f"{task}.csv")
    if not os.path.exists(path):
        return None
    try:
        import csv
        rows = []
        with open(path) as f:
            for row in csv.DictReader(f):
                emb_str = row.get("embedding", "")
                if emb_str:
                    rows.append(np.array(ast.literal_eval(emb_str), dtype=np.float32))
        if len(rows) < dataset_size:
            log.warning(f"  embeddings/{task}.csv has {len(rows)} rows, dataset has {dataset_size}; using fallback")
            return None
        return np.array(rows[:dataset_size])
    except Exception as e:
        log.warning(f"  Failed to load embeddings/{task}.csv: {e}; using fallback")
        return None


def get_embedder():
    """Lazy-load sentence-transformers (fallback when pre-computed embeddings unavailable)."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        raise ImportError(
            "DC-RS fallback requires sentence-transformers when embeddings/{task}.csv "
            "is missing. Install with: pip install sentence-transformers"
        )


def retrieve_top_k(
    current_embedding: np.ndarray,
    past_embeddings: np.ndarray,
    past_inputs: List[str],
    past_outputs: List[str],
    k: int,
) -> List[Tuple[str, str]]:
    """
    Retrieve top-k most similar (input, output) pairs.
    Returns list of (input_txt, output_txt) ordered by similarity (highest first).
    """
    if len(past_embeddings) == 0 or k <= 0:
        return []
    sims = cosine_similarity([current_embedding], past_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    return [(past_inputs[i], past_outputs[i]) for i in top_indices]


def format_retrieved_pairs(pairs: List[Tuple[str, str]]) -> str:
    """Format retrieved (input, output) pairs for the curator prompt."""
    if not pairs:
        return "(No previous examples retrieved.)"
    lines = []
    for i, (inp, out) in enumerate(pairs, 1):
        lines.append(f"### Previous Input #{i}:\n\n{inp}\n\n### Model Solution #{i}:\n\n{out}\n\n---\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# DC-RS evaluation loop
# ─────────────────────────────────────────────────────────────────────

def run_dc_rs(
    model: LanguageModel,
    model_name: str,
    task: str,
    *,
    max_samples: int = -1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    save_dir: str = "results_dc_rs",
    shuffle_seed: int = 10,
    retrieve_top_k: int = 3,
    execute_code: bool = True,
) -> Dict:
    """
    Run DC-RS (Retrieval & Synthesis) evaluation.

    Per question i:
      1. Retrieve top-k similar (x_j, y_j) from j < i
      2. Curator: M_i = Cur(M_{i-1}, x_i, R_i)
      3. Generator: y_i = Gen(x_i, M_i)
    """
    full_dataset = load_from_disk(f"data/{task}")
    full_size = len(full_dataset)
    precomputed = load_precomputed_embeddings(task, full_size)
    embedder = None if precomputed is not None else get_embedder()
    if precomputed is not None:
        log.info(f"  Using pre-computed embeddings from embeddings/{task}.csv")

    dataset = full_dataset
    rng = np.random.RandomState(shuffle_seed)
    n = full_size if max_samples <= 0 else min(max_samples, full_size)
    indices = rng.choice(full_size, size=n, replace=False).tolist()
    dataset = dataset.select(indices)

    generator_prompt = read_file("prompts/generator_prompt.txt")
    curator_prompt_rs = read_file("prompts/curator_prompt_for_dc_retrieval_synthesis.txt")

    short_model_name = model_name.split("/")[-1]
    t0 = time.time()

    cheatsheet = "(empty)"
    past_inputs: List[str] = []
    past_outputs: List[str] = []
    past_embeddings: List[np.ndarray] = []

    outputs: List[dict] = []
    correct = 0

    pbar = tqdm(range(n), desc=f"{short_model_name}|DC-RS|{task}", unit="q", leave=True)

    for idx in pbar:
        example = dataset[idx]
        raw_input = example["input"]
        target = example["target"]
        input_txt = format_input(task, raw_input, idx)

        # 1. Get embedding (pre-computed or compute on the fly)
        if precomputed is not None:
            current_emb = precomputed[indices[idx]]
        else:
            current_emb = embedder.encode([input_txt], convert_to_numpy=True)[0]

        try:
            # 2. Retrieve top-k similar past (input, output) pairs
            if past_embeddings:
                past_emb_arr = np.array(past_embeddings)
                pairs = retrieve_top_k(
                    current_emb, past_emb_arr, past_inputs, past_outputs, retrieve_top_k
                )
                retrieved_section = format_retrieved_pairs(pairs)
            else:
                retrieved_section = "(No previous examples yet.)"

            # 3. Curator: synthesize M_i from M_{i-1}, x_i, R_i (BEFORE generation)
            curator_prompt = (
                curator_prompt_rs.replace("[[PREVIOUS_CHEATSHEET]]", cheatsheet)
                .replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", retrieved_section)
                .replace("[[NEXT_INPUT]]", input_txt)
            )
            curator_history = [{"role": "user", "content": curator_prompt}]
            curator_output = model.generate(
                history=curator_history,
                temperature=temperature,
                max_tokens=2 * max_tokens,
                allow_code_execution=False,
            )
            cheatsheet = cap_cheatsheet(extract_cheatsheet(curator_output, cheatsheet))

            # 4. Generator: y_i = Gen(x_i, M_i)
            gen_prompt = generator_prompt.replace("[[QUESTION]]", input_txt).replace(
                "[[CHEATSHEET]]", cheatsheet
            )
            gen_history = [{"role": "user", "content": gen_prompt}]
            gen_output = model.generate(
                history=gen_history,
                temperature=temperature,
                max_tokens=max_tokens,
                allow_code_execution=execute_code,
            )
            final_answer = extract_answer(gen_output)

            # 5. Update history for future retrieval
            past_inputs.append(input_txt)
            past_outputs.append(gen_output)
            past_embeddings.append(current_emb)

        except Exception as exc:
            log.error(f"  [ERROR] idx={idx}: {exc}\n{traceback.format_exc()}")
            final_answer = ""
            gen_output = f"ERROR: {exc}"
            past_inputs.append(input_txt)
            past_outputs.append(gen_output)
            past_embeddings.append(current_emb)

        is_correct = evaluate_one(task, raw_input, final_answer, target)
        if is_correct:
            correct += 1

        outputs.append({
            "idx": idx,
            "input": input_txt,
            "raw_input": raw_input,
            "target": target,
            "final_answer": final_answer,
            "correct": is_correct,
            "generator_output": gen_output,
            "final_output": gen_output,
            "final_cheatsheet": cheatsheet,
        })

        pbar.set_postfix(acc=f"{correct/(idx+1):.1%}", correct=f"{correct}/{idx+1}")

    elapsed = time.time() - t0
    accuracy = correct / n if n > 0 else 0.0

    # Save results
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(save_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{short_model_name}_DynamicCheatsheet_RetrievalSynthesis_{ts}.jsonl")

    header = {
        "_summary": True,
        "file": os.path.basename(out_path).replace(".jsonl", ""),
        "dataset": task,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy:.1%}",
        "correct": correct,
        "total": n,
    }
    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in outputs:
            f.write(json.dumps(row, default=str) + "\n")

    summary = {
        "model": model_name,
        "approach": "DynamicCheatsheet_RetrievalSynthesis",
        "task": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "elapsed_s": round(elapsed, 1),
        "output_file": out_path,
        "retrieve_top_k": retrieve_top_k,
    }
    log.info(f"  >> {task} done: {correct}/{n} = {accuracy:.1%} in {elapsed:.0f}s -> {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DC-RS (Dynamic Cheatsheet with Retrieval & Synthesis) evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help=f"Model IDs. Default: {[m.split('/')[-1] for m in ALL_MODELS]}",
    )
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Datasets. Default: GSM8K MMLU_Pro_Engineering",
    )
    p.add_argument("--quantization", default="4bit", choices=["none", "4bit", "8bit"])
    p.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=50, help="Samples per dataset (-1 = all)")
    p.add_argument("--no_code_execution", action="store_true")
    p.add_argument("--save_dir", default="results_dc_rs")
    p.add_argument(
        "--retrieve_top_k", type=int, default=3,
        help="Number of past (input, output) pairs to retrieve (default: 3)",
    )
    p.add_argument("--resume", action="store_true", help="Skip completed runs")
    return p


def find_completed(save_dir: str) -> set:
    completed = set()
    path = Path(save_dir)
    if not path.exists():
        return completed
    for task_dir in path.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for f in task_dir.glob("*_DynamicCheatsheet_RetrievalSynthesis_*.jsonl"):
            stem = f.stem
            for m in ALL_MODELS:
                short = m.split("/")[-1]
                if stem.startswith(short + "_"):
                    completed.add(f"{m}|DynamicCheatsheet_RetrievalSynthesis|{task}")
                    break
    return completed


def main():
    args = build_parser().parse_args()
    setup_logging(args.save_dir)

    models = args.models or ALL_MODELS
    datasets = args.datasets or ["GSM8K", "MMLU_Pro_Engineering"]
    completed = find_completed(args.save_dir) if args.resume else set()

    log.info("=" * 72)
    log.info("  DC-RS (Dynamic Cheatsheet with Retrieval & Synthesis)")
    log.info("  Paper: https://arxiv.org/abs/2504.07952")
    log.info("=" * 72)
    log.info(f"  Models:    {models}")
    log.info(f"  Datasets:  {datasets}")
    log.info(f"  Backend:   {args.backend}   Quantization: {args.quantization}")
    log.info(f"  Retrieve top-k: {args.retrieve_top_k}")
    log.info(f"  Max samples:   {args.max_samples}")
    log.info("=" * 72)

    all_summaries: List[Dict] = []

    for model_name in models:
        key = f"{model_name}|DynamicCheatsheet_RetrievalSynthesis|"
        runs_needed = any(
            f"{model_name}|DynamicCheatsheet_RetrievalSynthesis|{d}" not in completed
            for d in datasets
        )
        if not runs_needed:
            log.info(f"\n[SKIP] All DC-RS runs for {model_name} already completed.")
            continue

        log.info(f"\nLoading model: {model_name}")
        try:
            model = LanguageModel(
                model_name=model_name,
                use_local_models=True,
                quantization=args.quantization,
                backend=args.backend,
            )
        except Exception as exc:
            log.error(f"  [SKIP] Failed to load {model_name}: {exc}")
            log.error(traceback.format_exc())
            continue

        for task in datasets:
            run_key = f"{model_name}|DynamicCheatsheet_RetrievalSynthesis|{task}"
            if run_key in completed:
                log.info(f"\n  SKIP (done): {model_name.split('/')[-1]} | {task}")
                continue

            log.info(f"\n  >>> {model_name.split('/')[-1]} | {task}")
            try:
                summary = run_dc_rs(
                    model=model,
                    model_name=model_name,
                    task=task,
                    max_samples=args.max_samples,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    save_dir=args.save_dir,
                    retrieve_top_k=args.retrieve_top_k,
                    execute_code=not args.no_code_execution,
                )
                all_summaries.append(summary)
            except Exception as exc:
                log.error(f"  [FAIL] {run_key}: {exc}")
                log.error(traceback.format_exc())

        del model

    # Summary table
    log.info("\n" + "=" * 72)
    log.info("  DC-RS RESULTS")
    log.info("=" * 72)
    for s in all_summaries:
        short = s["model"].split("/")[-1]
        log.info(f"  {short:<35} {s['task']:<22} {s['accuracy']:.1%}  ({s['correct']}/{s['total']})")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
['correct']}/{s['total']})")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
