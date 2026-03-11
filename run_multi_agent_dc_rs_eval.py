#!/usr/bin/env python3
"""
Multi-agent DC-RS (Dynamic Cheatsheet with Retrieval & Synthesis) evaluation.

Combines:
  - Multi-agent setup: 3 generators + 1 curator, majority vote
  - DC-RS flow (from https://arxiv.org/abs/2504.07952):
      1. Retrieve: top-k similar past (input, output) pairs
      2. Curate: synthesize memory from retrieved pairs (BEFORE generation)
      3. Generate: all 3 generators answer with updated cheatsheet
      4. Majority vote for final answer

Uses pre-computed embeddings from embeddings/{task}.csv when available; falls back to
sentence-transformers (pip install sentence-transformers) for datasets without embeddings.

Usage:
  python run_multi_agent_dc_rs_eval.py --datasets GSM8K --max_samples 50
  python run_multi_agent_dc_rs_eval.py --backend vllm --no_shared_memory --resume
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import time
import traceback
from collections import Counter
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

log = logging.getLogger("multi_agent_dc_rs")

# ─────────────────────────────────────────────────────────────────────
# Configuration (aligned with run_multi_agent_eval)
# ─────────────────────────────────────────────────────────────────────

GENERATOR_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

CURATOR_MODEL = "Qwen/Qwen2.5-14B-Instruct"

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

GENERATOR_PROMPT_PATH = "prompts/generator_prompt.txt"
CURATOR_PROMPT_RS_PATH = "prompts/curator_prompt_for_dc_retrieval_synthesis.txt"

MAX_CHEATSHEET_WORDS = 3000

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


GENERATOR_PROMPT = read_file(GENERATOR_PROMPT_PATH)
CURATOR_PROMPT_RS = read_file(CURATOR_PROMPT_RS_PATH)


def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(save_dir, f"multi_agent_dc_rs_{ts}.log")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.info(f"Logging to {log_path}")
    return log


def format_input(task: str, raw_input: str, idx: int) -> str:
    txt = f"Question #{idx+1}:\n{raw_input}"
    if task in ("AIME_2020_2024", "AIME_2024", "AIME_2025"):
        txt += (
            " (Please provide your answer in the form of an integer, "
            "e.g., 1234, with no Markdown formatting or additional text; "
            "make sure to pay attention to the desired format of the "
            "final answer though.)"
        )
    elif task == "GSM8K":
        txt += (
            " (Please provide your final numerical answer as a single "
            "number with no units, commas, or additional text.)"
        )
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


def cap_cheatsheet(cheatsheet: str) -> str:
    if cheatsheet == "(empty)":
        return cheatsheet
    words = cheatsheet.split()
    if len(words) <= MAX_CHEATSHEET_WORDS:
        return cheatsheet
    log.warning(
        f"  Cheatsheet exceeded {MAX_CHEATSHEET_WORDS} words ({len(words)}), truncating"
    )
    return " ".join(words[:MAX_CHEATSHEET_WORDS]) + "\n\n[...cheatsheet truncated]"


def evaluate_answer(task: str, input_txt: str, answer: str, target: str) -> bool:
    if task in ("AIME_2025", "AIME_2024", "AIME_2020_2024", "GSM8K"):
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
# Embedding & retrieval (DC-RS)
# ─────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR = "embeddings"


def load_precomputed_embeddings(task: str, dataset_size: int) -> Optional[np.ndarray]:
    """
    Load pre-computed embeddings from embeddings/{task}.csv.
    Returns (dataset_size, dim) array or None if file missing/invalid.
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
    """Retrieve top-k most similar (input, output) pairs."""
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
        lines.append(
            f"### Previous Input #{i}:\n\n{inp}\n\n### Model Solution #{i}:\n\n{out}\n\n---\n"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Resume support
# ─────────────────────────────────────────────────────────────────────


def run_key(task: str, shared_memory: bool = True) -> str:
    gen_short = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_short = CURATOR_MODEL.split("/")[-1]
    suffix = "MultiGenerator_DCRS" if shared_memory else "MultiGenerator_DCRS_SepMem"
    return f"{gen_short}__{cur_short}__{suffix}__{task}"


def find_completed_runs(save_dir: str) -> set:
    completed = set()
    path = Path(save_dir)
    if not path.exists():
        return completed
    for task_dir in path.iterdir():
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for f in task_dir.glob("*_MultiGenerator_DCRS*.jsonl"):
            name = f.name
            shared = "_MultiGenerator_DCRS_SepMem_" not in name
            key = run_key(task, shared_memory=shared)
            try:
                with open(f) as fh:
                    lines = fh.readlines()
                if len(lines) > 0:
                    completed.add(key)
            except Exception:
                pass
    return completed


# ─────────────────────────────────────────────────────────────────────
# Core DC-RS evaluation (multi-agent)
# ─────────────────────────────────────────────────────────────────────


def run_multi_agent_dc_rs(
    model: LanguageModel,
    task: str,
    *,
    max_samples: int = -1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    save_dir: str = "results_multi_agent_dc_rs",
    shuffle_seed: int = 10,
    retrieve_top_k: int = 3,
    execute_code: bool = True,
    shared_memory: bool = True,
    cheatsheet_verbose: bool = False,
) -> Dict:
    """
    Run multi-agent DC-RS: 3 generators + 1 curator, retrieval + synthesis before generation.

    Per question i:
      1. Retrieve top-k similar (input, output) from past
      2. Curator: M_i = Cur(M_{i-1}, x_i, R_i) — before generation
      3. All 3 generators: y_i = Gen(x_i, M_i)
      4. Majority vote for final answer
      5. Update retrieval history (shared: majority output; separate: per-generator)

    shared_memory=True: one cheatsheet, one retrieval history (majority-voted outputs)
    shared_memory=False: 3 cheatsheets, 3 retrieval histories (one per generator)
    """
    full_dataset = load_from_disk(f"data/{task}")
    full_size = len(full_dataset)
    precomputed = load_precomputed_embeddings(task, full_size)
    embedder = None if precomputed is not None else get_embedder()
    if precomputed is not None:
        log.info(f"  Using pre-computed embeddings from embeddings/{task}.csv")

    rng = np.random.RandomState(shuffle_seed)
    n = full_size if max_samples <= 0 else min(max_samples, full_size)
    indices = rng.choice(full_size, size=n, replace=False).tolist()
    dataset = full_dataset.select(indices)

    auditor = None
    if cheatsheet_verbose:
        from dynamic_cheatsheet.utils.cheatsheet_auditor import CheatsheetAuditor
        auditor = CheatsheetAuditor(
            save_dir=save_dir,
            model_name=CURATOR_MODEL,
            task=task,
            approach="MultiGenerator_DCRS",
            generator_model=GENERATOR_MODELS[0],
            curator_model=CURATOR_MODEL,
        )

    gen_short = "+".join(m.split("/")[-1].split("-Instruct")[0] for m in GENERATOR_MODELS)
    t0 = time.time()

    # Shared vs separate memory
    cheatsheet = "(empty)" if shared_memory else None
    cheatsheets = None if shared_memory else ["(empty)"] * len(GENERATOR_MODELS)
    past_inputs: List[str] = []
    past_outputs: List[str] = []  # shared: majority-voted
    past_outputs_per_gen: List[List[str]] = []  # separate: per generator
    past_embeddings: List[np.ndarray] = []

    outputs: List[dict] = []
    correct = 0

    curator_client = model.curator_client if model.curator_client is not None else model.client
    curator_model = model.curator_model_name if model.curator_model_name is not None else model.model_name

    pbar = tqdm(range(n), desc=f"{gen_short}|DCRS|{task}", unit="q", leave=True)

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
            # Shared: use majority-voted representative output; separate: use per-gen below
            if past_embeddings:
                past_emb_arr = np.array(past_embeddings)
                pairs = retrieve_top_k(
                    current_emb, past_emb_arr, past_inputs, past_outputs, retrieve_top_k
                )
                retrieved_section = format_retrieved_pairs(pairs)
            else:
                retrieved_section = "(No previous examples yet.)"

            # 3. Curator: synthesize M_i from M_{i-1}, x_i, R_i (BEFORE generation)
            if shared_memory:
                curator_prompt = (
                    CURATOR_PROMPT_RS.replace("[[PREVIOUS_CHEATSHEET]]", cheatsheet)
                    .replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", retrieved_section)
                    .replace("[[NEXT_INPUT]]", input_txt)
                )
                curator_history = [{"role": "user", "content": curator_prompt}]
                curator_output = model.generate_with_client(
                    client=curator_client,
                    model_name=curator_model,
                    history=curator_history,
                    temperature=temperature,
                    max_tokens=2 * max_tokens,
                    allow_code_execution=False,
                )
                cheatsheet = cap_cheatsheet(extract_cheatsheet(curator_output, cheatsheet))
            else:
                # Separate: 3 curator runs, 3 cheatsheets
                for g in range(len(GENERATOR_MODELS)):
                    _past_out = [p[g] for p in past_outputs_per_gen] if past_outputs_per_gen else []
                    if _past_out:
                        _pairs = retrieve_top_k(
                            current_emb,
                            np.array(past_embeddings),
                            past_inputs,
                            _past_out,
                            retrieve_top_k,
                        )
                        _retrieved = format_retrieved_pairs(_pairs)
                    else:
                        _retrieved = "(No previous examples yet.)"
                    curator_prompt = (
                        CURATOR_PROMPT_RS.replace("[[PREVIOUS_CHEATSHEET]]", cheatsheets[g])
                        .replace("[[PREVIOUS_INPUT_OUTPUT_PAIRS]]", _retrieved)
                        .replace("[[NEXT_INPUT]]", input_txt)
                    )
                    curator_history = [{"role": "user", "content": curator_prompt}]
                    curator_output = model.generate_with_client(
                        client=curator_client,
                        model_name=curator_model,
                        history=curator_history,
                        temperature=temperature,
                        max_tokens=2 * max_tokens,
                        allow_code_execution=False,
                    )
                    cheatsheets[g] = cap_cheatsheet(
                        extract_cheatsheet(curator_output, cheatsheets[g])
                    )

            # 4. Generate: all 3 generators
            all_gen_outputs = []
            all_gen_answers = []
            gen_steps = []
            for gi, gen_name in enumerate(GENERATOR_MODELS):
                _gen_client = model.generator_clients[gi]
                _cs = cheatsheet if shared_memory else cheatsheets[gi]
                _prompt = GENERATOR_PROMPT.replace("[[QUESTION]]", input_txt).replace(
                    "[[CHEATSHEET]]", _cs
                )
                gen_history = [{"role": "user", "content": _prompt}]
                gen_output = model.generate_with_client(
                    client=_gen_client,
                    model_name=gen_name,
                    history=gen_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    allow_code_execution=execute_code,
                )
                gen_answer = extract_answer(gen_output)
                all_gen_outputs.append(gen_output)
                all_gen_answers.append(gen_answer)
                gen_steps.append({
                    "generator_index": gi,
                    "generator_model": gen_name,
                    "generator_output": gen_output,
                    "generator_answer": gen_answer,
                })

            # 5. Majority vote
            final_answer = Counter(all_gen_answers).most_common(1)[0][0]
            combined_outputs = "".join(
                f"### Generator {gi+1} ({gen_name}) Output:\n{gen_output}\n---\n\n"
                for gi, (gen_name, gen_output) in enumerate(
                    zip(GENERATOR_MODELS, all_gen_outputs)
                )
            )
            # Representative output for retrieval: full output from generator that produced majority answer
            winner_idx = next(i for i, a in enumerate(all_gen_answers) if a == final_answer)
            rep_output = all_gen_outputs[winner_idx]

            # 6. Update retrieval history
            past_inputs.append(input_txt)
            past_outputs.append(rep_output)
            past_outputs_per_gen.append(all_gen_outputs)
            past_embeddings.append(current_emb)

        except Exception as exc:
            log.error(f"  [ERROR] idx={idx}: {exc}\n{traceback.format_exc()}")
            final_answer = ""
            combined_outputs = f"ERROR: {exc}"
            all_gen_answers = []
            gen_steps = []
            past_inputs.append(input_txt)
            past_outputs.append("")
            past_outputs_per_gen.append([""] * len(GENERATOR_MODELS))
            past_embeddings.append(current_emb)

        is_correct = evaluate_answer(task, input_txt, final_answer, target)
        if is_correct:
            correct += 1

        _cs = cheatsheet if shared_memory else (cheatsheets[0] if cheatsheets else "(empty)")
        outputs.append({
            "idx": idx,
            "input": input_txt,
            "raw_input": raw_input,
            "target": target,
            "final_answer": final_answer,
            "correct": is_correct,
            "steps": gen_steps,
            "all_generator_answers": all_gen_answers,
            "final_output": combined_outputs,
            "final_cheatsheet": _cs,
        })

        if auditor:
            auditor.record(
                question_idx=idx,
                question_text=input_txt,
                cheatsheet=_cs,
                generator_output=combined_outputs,
                final_answer=final_answer,
                target=target,
                is_correct=is_correct,
            )

        pbar.set_postfix(acc=f"{correct/(idx+1):.1%}", correct=f"{correct}/{idx+1}")

    elapsed = time.time() - t0
    accuracy = correct / n if n > 0 else 0.0

    # Save results
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(save_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    gen_tag = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_tag = CURATOR_MODEL.split("/")[-1]
    suffix = "MultiGenerator_DCRS" if shared_memory else "MultiGenerator_DCRS_SepMem"
    out_path = os.path.join(out_dir, f"{gen_tag}__{cur_tag}_{suffix}_{ts}.jsonl")

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

    if auditor:
        audit_report = auditor.finalize()
        log.info(f"  >> Cheatsheet audit saved to {audit_report.get('audit_dir', '')}")

    summary = {
        "generators": [m.split("/")[-1] for m in GENERATOR_MODELS],
        "curator": CURATOR_MODEL.split("/")[-1],
        "approach": "MultiGenerator_DCRS",
        "dataset": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "elapsed_s": round(elapsed, 1),
        "output_path": out_path,
        "retrieve_top_k": retrieve_top_k,
        "shared_memory": shared_memory,
    }
    log.info(f"  >> {task} done: {correct}/{n} = {accuracy:.1%} in {elapsed:.0f}s -> {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-agent DC-RS (Dynamic Cheatsheet with Retrieval & Synthesis)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_samples", type=int, default=50,
                   help="Samples per dataset (-1 = all)")
    p.add_argument("--no_code_execution", action="store_true")
    p.add_argument("--save_dir", default="results_multi_agent_dc_rs")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed (task, shared_memory) runs")
    p.add_argument("--quantization", default="4bit",
                   choices=["none", "4bit", "8bit"])
    p.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    p.add_argument("--datasets", nargs="+", default=None,
                   help=f"Datasets. Default: {ALL_DATASETS[:2]}")
    p.add_argument("--retrieve_top_k", type=int, default=3,
                   help="Number of past (input, output) pairs to retrieve")
    p.add_argument("--no_shared_memory", action="store_true",
                   help="Use 3 separate cheatsheets (one per generator) instead of "
                        "one shared cheatsheet. Each generator curates its own memory.")
    p.add_argument("--cheatsheet_verbose", action="store_true",
                   help="Enable cheatsheet auditing")
    p.add_argument("--shuffle_seed", type=int, default=10)
    p.add_argument("--dc_batch_size", type=int, default=1,
                   help="Not used in DC-RS (per-question); kept for CLI compatibility")
    p.add_argument("--max_num_rounds", type=int, default=1,
                   help="Not used in DC-RS; kept for CLI compatibility")
    return p


def main():
    args = build_parser().parse_args()
    setup_logging(args.save_dir)

    datasets = args.datasets or ["GSM8K", "MMLU_Pro_Engineering"]
    shared_mem = not getattr(args, "no_shared_memory", False)
    completed = find_completed_runs(args.save_dir) if args.resume else set()

    # GPU info
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

    total_runs = len(datasets)
    skip_count = sum(1 for d in datasets if run_key(d, shared_memory=shared_mem) in completed)

    log.info("=" * 72)
    log.info("  Multi-Agent DC-RS (Retrieval & Synthesis)")
    log.info("  Paper: https://arxiv.org/abs/2504.07952")
    log.info("=" * 72)
    log.info(f"  GPU:          {gpu_info}")
    log.info(f"  Backend:      {args.backend}")
    log.info(f"  Quantization: {args.quantization}")
    log.info(f"  Generators ({len(GENERATOR_MODELS)}):")
    for m in GENERATOR_MODELS:
        log.info(f"    - {m}")
    log.info(f"  Curator:      {CURATOR_MODEL}")
    log.info(f"  Shared mem:   {shared_mem} (--no_shared_memory={not shared_mem})")
    log.info(f"  Datasets:     {datasets}")
    log.info(f"  Retrieve k:   {args.retrieve_top_k}")
    log.info(f"  Max samples:  {'all' if args.max_samples <= 0 else args.max_samples}")
    log.info(f"  Total runs:   {total_runs}  (skipping {skip_count} already completed)")
    log.info("=" * 72)

    # Load model
    log.info("\nInitializing LanguageModel with multi-generator setup...")
    model = LanguageModel(
        model_name=CURATOR_MODEL,
        generator_model_names=GENERATOR_MODELS,
        curator_model_name=CURATOR_MODEL,
        use_local_models=True,
        quantization=args.quantization,
        backend=args.backend,
    )
    log.info("  LanguageModel ready")

    # Run evaluations
    all_summaries = []
    summary_path = os.path.join(args.save_dir, "summary_latest.json")

    for task in datasets:
        key = run_key(task, shared_memory=shared_mem)
        if key in completed:
            log.info(f"\n  SKIP (already done): {task}")
            continue

        log.info(f"\n  >>> {task}")
        try:
            summary = run_multi_agent_dc_rs(
                model=model,
                task=task,
                max_samples=args.max_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                save_dir=args.save_dir,
                retrieve_top_k=args.retrieve_top_k,
                execute_code=not args.no_code_execution,
                shared_memory=shared_mem,
                cheatsheet_verbose=args.cheatsheet_verbose,
                shuffle_seed=args.shuffle_seed,
            )
            all_summaries.append(summary)
        except Exception as exc:
            log.error(f"  [FAIL] {key}: {exc}")
            log.error(traceback.format_exc())

        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        log.info(f"  [checkpoint] summary saved to {summary_path}")

    # Summary table
    log.info("\n" + "=" * 72)
    log.info("  RESULTS SUMMARY")
    log.info("=" * 72)
    header = f"{'Dataset':<22} {'Acc':>7} {'N':>5} {'Time':>7}"
    log.info(header)
    log.info("-" * len(header))
    for s in all_summaries:
        log.info(
            f"{s['dataset']:<22} {s['accuracy']:>6.1%} {s['total']:>5} "
            f"{s['elapsed_s']:>6.0f}s"
        )
    log.info("=" * 72)

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    final_summary_path = os.path.join(args.save_dir, f"summary_{ts}.json")
    with open(final_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    log.info(f"Final summary saved to {final_summary_path}")


if __name__ == "__main__":
    main()
