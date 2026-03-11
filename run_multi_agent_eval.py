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
from dynamic_cheatsheet.utils.extractor import extract_answer, extract_cheatsheet

log = logging.getLogger("multi_agent_eval")

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

GENERATOR_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

CURATOR_MODEL = "Qwen/Qwen2.5-14B-Instruct"

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
    elif task == "GSM8K":
        txt += (" (Please provide your final numerical answer as a single "
                "number with no units, commas, or additional text.)")
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


MAX_CHEATSHEET_WORDS = 800

def cap_cheatsheet(cheatsheet: str) -> str:
    """Enforce a hard word cap on the cheatsheet to prevent context overflow."""
    if cheatsheet == "(empty)":
        return cheatsheet
    words = cheatsheet.split()
    if len(words) <= MAX_CHEATSHEET_WORDS:
        return cheatsheet
    log.warning(f"  Cheatsheet exceeded {MAX_CHEATSHEET_WORDS} words ({len(words)}), truncating")
    return " ".join(words[:MAX_CHEATSHEET_WORDS]) + "\n\n[...cheatsheet truncated to fit context window]"


def shrink_cheatsheet_to_words(cheatsheet: str, max_words: int) -> str:
    """Truncate cheatsheet to at most max_words."""
    if cheatsheet == "(empty)":
        return cheatsheet
    words = cheatsheet.split()
    if len(words) <= max_words:
        return cheatsheet
    return " ".join(words[:max_words]) + "\n\n[...truncated to fit context]"


def _truncate_qa_section(qa_entries: list, max_total_words: int) -> str:
    """Truncate Q&A entries to fit within ~max_total_words (split evenly)."""
    if not qa_entries:
        return ""
    n = len(qa_entries)
    words_per_entry = max(30, max_total_words // n)
    out = []
    for entry in qa_entries:
        words = entry.split()
        if len(words) > words_per_entry:
            out.append(" ".join(words[:words_per_entry]) + "\n[...truncated]\n\n---\n\n")
        else:
            out.append(entry)
    return "".join(out)


MAX_CONTEXT_TOKENS = 8192
MAX_CURATOR_INPUT_TOKENS = 6000  # ~800-word output leaves room for larger input


def build_curator_prompt_within_limit(
    manager,
    curator_model_name: str,
    template: str,
    cheatsheet: str,
    qa_entries: list,
    batch_size: int = 32,
) -> str:
    """
    Build curator prompt, shrinking Q&A section first (main culprit), then cheatsheet if needed.
    Token count must be <= MAX_CURATOR_INPUT_TOKENS. Uses actual tokenizer for accurate count.
    """
    qa_placeholder = "[[MODEL_ANSWER]]"
    question_text = f"(Batch of {batch_size} questions — see below)"
    cheatsheet_words = cheatsheet.split() if cheatsheet != "(empty)" else []
    max_cs_words = len(cheatsheet_words)
    max_qa_words = 5000

    for attempt in range(25):
        cs = shrink_cheatsheet_to_words(cheatsheet, max_cs_words) if cheatsheet != "(empty)" else "(empty)"
        qa_section = _truncate_qa_section(qa_entries, max_qa_words)
        shell = template.replace("[[PREVIOUS_CHEATSHEET]]", cs).replace(
            "[[QUESTION]]", question_text
        )
        curator_prompt = shell.replace(qa_placeholder, qa_section)
        history = [{"role": "user", "content": curator_prompt}]
        if hasattr(manager, "count_prompt_tokens"):
            count = manager.count_prompt_tokens(history, curator_model_name)
        else:
            count = int(len(curator_prompt.split()) * 1.4)
        if count <= MAX_CURATOR_INPUT_TOKENS:
            return curator_prompt
        if max_qa_words > 800:
            max_qa_words = max(400, int(max_qa_words * 0.7))
            log.warning(f"  Curator prompt {count} tokens > {MAX_CURATOR_INPUT_TOKENS}; shrinking Q&A to ~{max_qa_words} words")
        elif max_cs_words > 100:
            max_cs_words = max(100, int(max_cs_words * 0.8))
            log.warning(f"  Curator prompt {count} tokens > {MAX_CURATOR_INPUT_TOKENS}; shrinking cheatsheet to {max_cs_words} words")
        else:
            log.warning(f"  Curator prompt still {count} tokens; using best-effort truncation")
            return curator_prompt
    return curator_prompt


def save_results_jsonl(out_path: str, rows: list, accuracy: float,
                       correct: int, total: int):
    """Write results with a summary header line."""
    fname = os.path.basename(out_path).replace(".jsonl", "")
    dataset = os.path.basename(os.path.dirname(out_path))
    header = {
        "_summary": True,
        "file": fname,
        "dataset": dataset,
        "accuracy": round(accuracy, 4),
        "accuracy_pct": f"{accuracy:.1%}",
        "correct": correct,
        "total": total,
    }
    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


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
# Resume support
# ─────────────────────────────────────────────────────────────────────

def run_key(approach: str, task: str, shared_memory: bool = True) -> str:
    gen_short = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_short = CURATOR_MODEL.split("/")[-1]
    suffix = approach if (shared_memory or approach != "MultiGenerator_Cumulative") else "MultiGenerator_Cumulative_SepMem"
    return f"{gen_short}__{cur_short}__{suffix}__{task}"


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
                if f"_{approach}_" in f or (approach == "MultiGenerator_Cumulative" and "_MultiGenerator_Cumulative_SepMem_" in f):
                    for dataset in ALL_DATASETS:
                        if f.startswith(dataset) or f"/{dataset}/" in os.path.join(root, f):
                            task = os.path.basename(root)
                            shared = approach != "MultiGenerator_Cumulative" or "_MultiGenerator_Cumulative_SepMem_" not in f
                            key = run_key(approach, task, shared_memory=shared)
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
    cheatsheet_verbose: bool = False,
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

    auditor = None
    if cheatsheet_verbose and "Cumulative" in approach:
        from dynamic_cheatsheet.utils.cheatsheet_auditor import CheatsheetAuditor
        auditor = CheatsheetAuditor(
            save_dir=save_dir,
            model_name=CURATOR_MODEL,
            task=task,
            approach=approach,
            generator_model=GENERATOR_MODELS[0],
            curator_model=CURATOR_MODEL,
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
        cheatsheet = cap_cheatsheet(output_dict.get("final_cheatsheet", cheatsheet) or cheatsheet)
        generator_outputs_so_far.append(output_dict.get("final_output", ""))

        is_correct = evaluate_answer(task, input_txt, final_answer, target)
        if is_correct:
            correct += 1
        total += 1
        acc = correct / total if total else 0

        if auditor:
            auditor.record(
                question_idx=idx,
                question_text=input_txt,
                cheatsheet=cheatsheet,
                generator_output=output_dict.get("final_output", ""),
                final_answer=final_answer,
                target=target,
                is_correct=is_correct,
            )

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
    save_results_jsonl(out_path, outputs, accuracy, correct, total)

    log.info(f"  => {accuracy:.1%} ({correct}/{total}) in {elapsed:.0f}s  -> {out_path}")

    if auditor:
        audit_report = auditor.finalize()
        log.info(f"  >> Cheatsheet audit saved to {audit_report.get('audit_dir', '')}")

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
# Batched evaluation for MultiGenerator (vLLM) — no sequential deps
# ─────────────────────────────────────────────────────────────────────

def run_multi_generator_batched(
    model: LanguageModel,
    task: str,
    *,
    max_samples: int = -1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    save_dir: str = "results_multi_agent",
    shuffle_seed: int = 10,
) -> Dict:
    """
    Batched MultiGenerator evaluation: process all questions per-generator
    in a single vLLM batch call, then majority-vote across generators.

    Instead of switching models for every question, we load each generator
    once and batch all questions through it. This means only len(GENERATOR_MODELS)
    model loads instead of len(GENERATOR_MODELS) * n.
    """
    from collections import Counter as _Counter

    dataset = load_from_disk(f"data/{task}")
    rng = np.random.RandomState(shuffle_seed)
    n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    dataset = dataset.select(indices)

    t0 = time.time()

    # Phase 1: format all prompts (shared across generators — no cheatsheet)
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

    # Phase 2: batch generate per-generator (one model load + one batch per generator)
    per_gen_outputs: Dict[str, List[str]] = {}
    for gen_name in GENERATOR_MODELS:
        log.info(f"  Batch-generating {n} prompts for {gen_name.split('/')[-1]}...")
        gen_t0 = time.time()
        outputs = model.batch_generate_with_model(
            histories, gen_name, temperature=temperature, max_tokens=max_tokens,
        )
        gen_elapsed = time.time() - gen_t0
        log.info(f"  {gen_name.split('/')[-1]} done in {gen_elapsed:.1f}s")
        per_gen_outputs[gen_name] = outputs

    # Phase 3: majority vote + evaluation
    outputs_list: List[dict] = []
    correct = 0
    for idx in range(n):
        gen_answers = []
        gen_steps = []
        combined_outputs = ""
        for gi, gen_name in enumerate(GENERATOR_MODELS):
            gen_output = per_gen_outputs[gen_name][idx]
            gen_answer = extract_answer(gen_output)
            gen_answers.append(gen_answer)
            gen_steps.append({
                "generator_index": gi,
                "generator_model": gen_name,
                "generator_output": gen_output,
                "generator_answer": gen_answer,
            })
            combined_outputs += (
                f"### Generator {gi+1} ({gen_name}) Output:\n{gen_output}\n---\n\n"
            )

        final_answer = _Counter(gen_answers).most_common(1)[0][0]
        is_correct = evaluate_answer(task, input_txts[idx], final_answer, targets[idx])
        if is_correct:
            correct += 1

        log.info(f"  Q{idx+1} per-gen={gen_answers}  final={final_answer!r}  "
                 f"target={targets[idx]!r}  correct={is_correct}")

        outputs_list.append({
            "input": input_txts[idx],
            "target": targets[idx],
            "raw_input": raw_inputs[idx],
            "steps": gen_steps,
            "all_generator_answers": gen_answers,
            "final_answer": final_answer,
            "final_cheatsheet": None,
            "final_output": combined_outputs,
            "is_correct": is_correct,
        })

    elapsed = time.time() - t0
    accuracy = correct / n if n else 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    gen_tag = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_tag = CURATOR_MODEL.split("/")[-1]
    fname = f"{gen_tag}__{cur_tag}_MultiGenerator_{ts}.jsonl"
    out_dir = os.path.join(save_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    save_results_jsonl(out_path, outputs_list, accuracy, correct, n)

    log.info(f"  => {accuracy:.1%} ({correct}/{n}) in {elapsed:.0f}s  -> {out_path}")

    return {
        "generators": [m.split("/")[-1] for m in GENERATOR_MODELS],
        "curator": CURATOR_MODEL.split("/")[-1],
        "approach": "MultiGenerator",
        "dataset": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
        "elapsed_s": round(elapsed, 1),
        "output_path": out_path,
    }


# ─────────────────────────────────────────────────────────────────────
# Batched MultiGenerator_Cumulative (vLLM)
# ─────────────────────────────────────────────────────────────────────

def _ckpt_path_mg_cu(save_dir: str, task: str, shared_memory: bool = True) -> str:
    gen_tag = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_tag = CURATOR_MODEL.split("/")[-1]
    suffix = "MultiGenerator_Cumulative" if shared_memory else "MultiGenerator_Cumulative_SepMem"
    return os.path.join(save_dir, task, f".ckpt_{gen_tag}__{cur_tag}_{suffix}.json")


def run_multi_generator_cumulative_batched(
    model: LanguageModel,
    task: str,
    *,
    max_samples: int = -1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    save_dir: str = "results_multi_agent",
    shuffle_seed: int = 10,
    batch_size: int = 32,
    cheatsheet_verbose: bool = False,
    shared_memory: bool = True,
) -> Dict:
    """
    Batched MultiGenerator_Cumulative for vLLM.

    shared_memory=True (default): One cheatsheet shared by all 3 generators; curator
        uses majority-voted answers.
    shared_memory=False: 3 separate cheatsheets; each generator curates from its own
        outputs; then majority vote. Effectively runs DC-Cu 3 times in parallel.

    Per mini-batch of `batch_size` questions:
      1. For each generator: batch all questions with (shared or per-gen) cheatsheet
      2. Majority-vote across generators per question
      3. Curator(s): 1 call if shared, len(GENERATOR_MODELS) calls if separate
    """
    from collections import Counter as _Counter

    dataset = load_from_disk(f"data/{task}")
    rng = np.random.RandomState(shuffle_seed)
    n = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    dataset = dataset.select(indices)

    auditor = None
    if cheatsheet_verbose:
        from dynamic_cheatsheet.utils.cheatsheet_auditor import CheatsheetAuditor
        auditor = CheatsheetAuditor(
            save_dir=save_dir, model_name=CURATOR_MODEL,
            task=task, approach="MultiGenerator_Cumulative",
            generator_model=GENERATOR_MODELS[0],
            curator_model=CURATOR_MODEL,
        )

    gen_short = "+".join(m.split("/")[-1].split("-Instruct")[0] for m in GENERATOR_MODELS)
    t0 = time.time()

    cheatsheet = "(empty)" if shared_memory else None
    cheatsheets = None if shared_memory else ["(empty)"] * len(GENERATOR_MODELS)
    outputs_list: List[dict] = []
    correct = 0
    start_batch = 0

    # Resume from checkpoint if available
    ckpt_file = _ckpt_path_mg_cu(save_dir, task, shared_memory)
    os.makedirs(os.path.join(save_dir, task), exist_ok=True)
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            if (ckpt.get("n") == n and ckpt.get("batch_size") == batch_size
                    and ckpt.get("shuffle_seed") == shuffle_seed
                    and ckpt.get("shared_memory") == shared_memory):
                start_batch = ckpt["next_batch"]
                if shared_memory:
                    cheatsheet = ckpt["cheatsheet"]
                else:
                    cheatsheets = ckpt["cheatsheets"]
                outputs_list = ckpt["outputs"]
                correct = ckpt["correct"]
                log.info(f"  Resuming from checkpoint: batch {start_batch+1} "
                         f"({len(outputs_list)} questions done, {correct} correct)")
        except Exception as e:
            log.warning(f"  Could not load checkpoint: {e}; starting fresh")

    num_batches = (n + batch_size - 1) // batch_size
    pbar = tqdm(range(n), desc=f"{gen_short}|MGCu_batch|{task}", unit="q",
                leave=True, initial=len(outputs_list))

    for b in range(start_batch, num_batches):
        batch_start = b * batch_size
        batch_end = min(batch_start + batch_size, n)
        k = batch_end - batch_start

        # Phase 1: format prompts for this batch
        batch_raw_inputs = []
        batch_input_txts = []
        batch_targets = []
        if shared_memory:
            histories = []
            for idx in range(batch_start, batch_end):
                example = dataset[idx]
                raw_input = example["input"]
                target = example["target"]
                input_txt = format_input(task, raw_input, idx)
                prompt = GENERATOR_PROMPT.replace(
                    "[[QUESTION]]", input_txt
                ).replace("[[CHEATSHEET]]", cheatsheet)
                histories.append([{"role": "user", "content": prompt}])
                batch_raw_inputs.append(raw_input)
                batch_input_txts.append(input_txt)
                batch_targets.append(target)
            histories_per_gen = [histories] * len(GENERATOR_MODELS)
        else:
            histories_per_gen = []
            for g in range(len(GENERATOR_MODELS)):
                h = []
                for idx in range(batch_start, batch_end):
                    example = dataset[idx]
                    raw_input = example["input"]
                    target = example["target"]
                    input_txt = format_input(task, raw_input, idx)
                    prompt = GENERATOR_PROMPT.replace(
                        "[[QUESTION]]", input_txt
                    ).replace("[[CHEATSHEET]]", cheatsheets[g])
                    h.append([{"role": "user", "content": prompt}])
                    if g == 0:
                        batch_raw_inputs.append(raw_input)
                        batch_input_txts.append(input_txt)
                        batch_targets.append(target)
                histories_per_gen.append(h)

        # Phase 2: batch-generate per generator (1 vLLM call per generator)
        per_gen_outputs: Dict[str, List[str]] = {}
        for gi, gen_name in enumerate(GENERATOR_MODELS):
            short_gen = gen_name.split("/")[-1]
            log.info(f"  Batch {b+1}/{num_batches}: {short_gen} generating {k} answers...")
            gen_out = model.batch_generate_with_model(
                histories_per_gen[gi], gen_name, temperature=temperature, max_tokens=max_tokens,
            )
            per_gen_outputs[gen_name] = gen_out

        # Phase 3: majority vote + evaluate each question
        for i in range(k):
            gen_answers = []
            gen_steps = []
            combined_outputs = ""
            for gi, gen_name in enumerate(GENERATOR_MODELS):
                gen_output = per_gen_outputs[gen_name][i]
                gen_answer = extract_answer(gen_output)
                gen_answers.append(gen_answer)
                gen_steps.append({
                    "generator_index": gi,
                    "generator_model": gen_name,
                    "generator_output": gen_output,
                    "generator_answer": gen_answer,
                })
                combined_outputs += (
                    f"### Generator {gi+1} ({gen_name}) Output:\n{gen_output}\n---\n\n"
                )

            final_answer = _Counter(gen_answers).most_common(1)[0][0]
            is_correct = evaluate_answer(
                task, batch_input_txts[i], final_answer, batch_targets[i],
            )
            if is_correct:
                correct += 1

            _cs = cheatsheet if shared_memory else cheatsheets[0]
            outputs_list.append({
                "input": batch_input_txts[i],
                "target": batch_targets[i],
                "raw_input": batch_raw_inputs[i],
                "steps": gen_steps,
                "all_generator_answers": gen_answers,
                "final_answer": final_answer,
                "final_cheatsheet": _cs,
                "final_output": combined_outputs,
                "is_correct": is_correct,
            })

            if auditor:
                auditor.record(
                    question_idx=batch_start + i,
                    question_text=batch_input_txts[i],
                    cheatsheet=_cs,
                    generator_output=combined_outputs,
                    final_answer=final_answer,
                    target=batch_targets[i],
                    is_correct=is_correct,
                )

            pbar.update(1)
            pbar.set_postfix(
                acc=f"{correct/(batch_start+i+1):.1%}",
                correct=f"{correct}/{batch_start+i+1}",
            )

        # Phase 4: run curator in sub-batches of 8
        CURATOR_SUB_BATCH_SIZE = 8
        if shared_memory:
            curator_cheatsheet = cheatsheet
            for sub_start in range(0, k, CURATOR_SUB_BATCH_SIZE):
                sub_end = min(sub_start + CURATOR_SUB_BATCH_SIZE, k)
                sub_k = sub_end - sub_start
                qa_entries = [
                    (
                        f"### Question {batch_start + sub_start + i + 1}\n{batch_input_txts[sub_start + i]}\n\n"
                        f"### Majority-Voted Answer {batch_start + sub_start + i + 1}\n{outputs_list[-(k - (sub_start + i))]['final_answer']}\n\n---\n\n"
                    )
                    for i in range(sub_k)
                ]
                manager = getattr(model, "local_manager", None)
                if manager is not None:
                    curator_prompt = build_curator_prompt_within_limit(
                        manager, CURATOR_MODEL, CURATOR_PROMPT, curator_cheatsheet, qa_entries, batch_size=sub_k,
                    )
                else:
                    curator_shell = CURATOR_PROMPT.replace(
                        "[[PREVIOUS_CHEATSHEET]]", curator_cheatsheet
                    ).replace("[[QUESTION]]", f"(Batch of {sub_k} questions — see below)")
                    qa_section = "".join(qa_entries)
                    curator_prompt = curator_shell.replace("[[MODEL_ANSWER]]", qa_section)
                curator_history = [{"role": "user", "content": curator_prompt}]
                log.info(f"  Batch {b+1}/{num_batches}: curating shared cheatsheet from {sub_k} Q&A pairs (sub-batch {sub_start//CURATOR_SUB_BATCH_SIZE + 1}/{(k + CURATOR_SUB_BATCH_SIZE - 1)//CURATOR_SUB_BATCH_SIZE})...")
                curator_output = model.generate(
                    history=curator_history,
                    temperature=temperature,
                    max_tokens=2 * max_tokens,
                    allow_code_execution=False,
                )
                curator_cheatsheet = cap_cheatsheet(extract_cheatsheet(curator_output, curator_cheatsheet))
            cheatsheet = curator_cheatsheet
        else:
            # Separate memory: per-generator curator runs
            for gi in range(len(GENERATOR_MODELS)):
                gen_name = GENERATOR_MODELS[gi]
                curator_cs = cheatsheets[gi]
                for sub_start in range(0, k, CURATOR_SUB_BATCH_SIZE):
                    sub_end = min(sub_start + CURATOR_SUB_BATCH_SIZE, k)
                    sub_k = sub_end - sub_start
                    qa_entries = [
                        (
                            f"### Question {batch_start + sub_start + i + 1}\n{batch_input_txts[sub_start + i]}\n\n"
                            f"### Model Answer {batch_start + sub_start + i + 1}\n{per_gen_outputs[gen_name][sub_start + i]}\n\n---\n\n"
                        )
                        for i in range(sub_k)
                    ]
                    manager = getattr(model, "local_manager", None)
                    if manager is not None:
                        curator_prompt = build_curator_prompt_within_limit(
                            manager, CURATOR_MODEL, CURATOR_PROMPT, curator_cs, qa_entries, batch_size=sub_k,
                        )
                    else:
                        curator_shell = CURATOR_PROMPT.replace(
                            "[[PREVIOUS_CHEATSHEET]]", curator_cs
                        ).replace("[[QUESTION]]", f"(Batch of {sub_k} questions — see below)")
                        qa_section = "".join(qa_entries)
                        curator_prompt = curator_shell.replace("[[MODEL_ANSWER]]", qa_section)
                    curator_history = [{"role": "user", "content": curator_prompt}]
                    short_gen = gen_name.split("/")[-1]
                    log.info(f"  Batch {b+1}/{num_batches}: curating {short_gen} cheatsheet from {sub_k} Q&A pairs...")
                    curator_output = model.generate(
                        history=curator_history,
                        temperature=temperature,
                        max_tokens=2 * max_tokens,
                        allow_code_execution=False,
                    )
                    curator_cs = cap_cheatsheet(extract_cheatsheet(curator_output, curator_cs))
                cheatsheets[gi] = curator_cs
        outputs_list[-1]["final_cheatsheet"] = cheatsheet if shared_memory else cheatsheets[0]

        # Checkpoint after every batch
        ckpt_data = {
            "next_batch": b + 1,
            "cheatsheet": cheatsheet if shared_memory else None,
            "cheatsheets": cheatsheets if not shared_memory else None,
            "outputs": outputs_list,
            "correct": correct,
            "n": n,
            "batch_size": batch_size,
            "shuffle_seed": shuffle_seed,
            "task": task,
            "shared_memory": shared_memory,
        }
        with open(ckpt_file, "w") as f:
            json.dump(ckpt_data, f, default=str)
        log.info(f"  Checkpoint saved: batch {b+1}/{num_batches} ({len(outputs_list)}/{n} questions)")

    pbar.close()
    elapsed = time.time() - t0
    accuracy = correct / n if n else 0

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    gen_tag = "+".join(m.split("/")[-1] for m in GENERATOR_MODELS)
    cur_tag = CURATOR_MODEL.split("/")[-1]
    approach_suffix = "MultiGenerator_Cumulative" if shared_memory else "MultiGenerator_Cumulative_SepMem"
    fname = f"{gen_tag}__{cur_tag}_{approach_suffix}_{ts}.jsonl"
    out_dir = os.path.join(save_dir, task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    save_results_jsonl(out_path, outputs_list, accuracy, correct, n)

    # Remove checkpoint now that final results are saved
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
        log.info(f"  Checkpoint removed (run complete)")

    log.info(f"  => {accuracy:.1%} ({correct}/{n}) in {elapsed:.0f}s  -> {out_path}")

    if auditor:
        audit_report = auditor.finalize()
        log.info(f"  >> Cheatsheet audit saved to {audit_report.get('audit_dir', '')}")

    return {
        "generators": [m.split("/")[-1] for m in GENERATOR_MODELS],
        "curator": CURATOR_MODEL.split("/")[-1],
        "approach": "MultiGenerator_Cumulative",
        "dataset": task,
        "accuracy": accuracy,
        "correct": correct,
        "total": n,
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
    p.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                    help="Inference backend: 'hf' (HuggingFace transformers) or "
                         "'vllm' (faster, supports batch generation)")
    p.add_argument("--approaches", nargs="+", default=None,
                    help=f"Approaches to run. Default: {ALL_APPROACHES}")
    p.add_argument("--datasets", nargs="+", default=None,
                    help=f"Datasets to run. Default: {ALL_DATASETS}")
    p.add_argument("--cheatsheet_verbose", action="store_true",
                    help="Enable cheatsheet auditing: save snapshots after every "
                         "update, track token growth, reuse, failure patterns, "
                         "and abstraction mismatch. Output goes to "
                         "<save_dir>/cheatsheet_audit/")
    p.add_argument("--dc_batch_size", type=int, default=32,
                    help="DC-Cumulative batch size: questions per curator update. "
                         "Use 1 for per-question curation (good for cheatsheet "
                         "auditing with small sample counts). Default: 32")
    p.add_argument("--no_shared_memory", action="store_true",
                    help="Use 3 separate cheatsheets (one per generator) instead of "
                         "one shared cheatsheet. Each generator curates its own memory "
                         "from its own outputs; then majority vote.")
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
    shared_mem = not getattr(args, "no_shared_memory", False)
    def _key(a: str, d: str):
        return run_key(a, d, shared_memory=shared_mem if a == "MultiGenerator_Cumulative" else True)
    skip_count = sum(1 for a in approaches for d in datasets if _key(a, d) in completed)

    log.info("=" * 72)
    log.info("  Multi-Agent Evaluation")
    log.info("=" * 72)
    log.info(f"  GPU:          {gpu_info}")
    log.info(f"  Backend:      {args.backend}")
    log.info(f"  Quantization: {args.quantization}")
    log.info(f"  Generators ({len(GENERATOR_MODELS)}):")
    for m in GENERATOR_MODELS:
        log.info(f"    - {m}")
    log.info(f"  Curator:      {CURATOR_MODEL}")
    log.info(f"  Approaches:   {approaches}")
    log.info(f"  Shared mem:   {shared_mem} (--no_shared_memory={not shared_mem})")
    log.info(f"  Datasets ({len(datasets)}): {datasets}")
    log.info(f"  Max tokens:   {args.max_tokens}   Temperature: {args.temperature}")
    log.info(f"  Max rounds:   {args.max_num_rounds}   Code exec: {not args.no_code_execution}")
    log.info(f"  Samples:      {'all' if args.max_samples <= 0 else args.max_samples}")
    log.info(f"  Total runs:   {total_runs}  (skipping {skip_count} already completed)")
    log.info("=" * 72)

    # ── Load model ──────────────────────────────────────────────────
    log.info(f"\nInitializing LanguageModel with multi-generator setup (backend={args.backend})...")
    model = LanguageModel(
        model_name=CURATOR_MODEL,
        generator_model_names=GENERATOR_MODELS,
        curator_model_name=CURATOR_MODEL,
        use_local_models=True,
        quantization=args.quantization,
        backend=args.backend,
    )
    log.info("  LanguageModel ready")

    # ── Run evaluations ─────────────────────────────────────────────
    all_summaries = []
    runs_to_do = total_runs - skip_count
    summary_path = os.path.join(args.save_dir, "summary_latest.json")

    run_pbar = tqdm(total=runs_to_do, desc="Overall runs", unit="run", position=0)

    for task in datasets:
        for approach in approaches:
            key = _key(approach, task)
            if key in completed:
                log.info(f"\n  SKIP (already done): {approach} | {task}")
                continue

            log.info(f"\n  >>> {approach} | {task}")

            use_vllm = (args.backend == "vllm")
            try:
                if use_vllm and approach == "MultiGenerator":
                    summary = run_multi_generator_batched(
                        model=model,
                        task=task,
                        max_samples=args.max_samples,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        save_dir=args.save_dir,
                    )
                elif use_vllm and approach == "MultiGenerator_Cumulative":
                    summary = run_multi_generator_cumulative_batched(
                        model=model,
                        task=task,
                        max_samples=args.max_samples,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        save_dir=args.save_dir,
                        batch_size=getattr(args, 'dc_batch_size', 32),
                        cheatsheet_verbose=getattr(args, 'cheatsheet_verbose', False),
                        shared_memory=shared_mem,
                    )
                else:
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
                        cheatsheet_verbose=getattr(args, 'cheatsheet_verbose', False),
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
