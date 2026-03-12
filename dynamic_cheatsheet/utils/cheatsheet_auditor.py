"""
Cheatsheet auditing utilities for diagnosing Dynamic Cheatsheet performance.

Tracks per-question cheatsheet evolution, token counts, memory item reuse,
structural quality, and abstraction mismatch between curator and generator.
"""

import json
import os
import re
import textwrap
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional


def _token_count(text: str) -> int:
    """Approximate token count (words × 1.3 is a rough BPE estimate)."""
    return int(len(text.split()) * 1.3)


def _extract_memory_items(cheatsheet: str) -> List[str]:
    """Pull out individual <memory_item>...</memory_item> blocks."""
    pattern = re.compile(
        r"<memory_item>(.*?)</memory_item>", re.DOTALL
    )
    return [m.strip() for m in pattern.findall(cheatsheet)]


def _extract_descriptions(cheatsheet: str) -> List[str]:
    """Pull out <description>...</description> blocks from memory items."""
    pattern = re.compile(
        r"<description>(.*?)</description>", re.DOTALL
    )
    return [d.strip() for d in pattern.findall(cheatsheet)]


def _extract_usage_counts(cheatsheet: str) -> List[int]:
    """Parse ** Count: N patterns."""
    pattern = re.compile(r"\*\*\s*Count:\s*(\d+)", re.IGNORECASE)
    return [int(c) for c in pattern.findall(cheatsheet)]


class CheatsheetAuditor:
    """
    Records cheatsheet snapshots after every update and produces a
    diagnostic report at the end of the run.

    Usage:
        auditor = CheatsheetAuditor(save_dir, model_name, task)
        for idx, question in enumerate(questions):
            ...
            auditor.record(
                question_idx=idx,
                question_text=input_txt,
                cheatsheet=new_cheatsheet,
                generator_output=gen_output,
                final_answer=gen_answer,
                target=target,
                is_correct=is_correct,
            )
        auditor.finalize()
    """

    def __init__(
        self,
        save_dir: str,
        model_name: str,
        task: str,
        approach: str = "DynamicCheatsheet_Cumulative",
        generator_model: Optional[str] = None,
        curator_model: Optional[str] = None,
        run_index: Optional[int] = None,
        run_flags: Optional[Dict[str, Any]] = None,
    ):
        short_model = model_name.split("/")[-1]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_prefix = f"run{run_index}_" if run_index is not None else ""
        dir_name = f"{run_prefix}{short_model}_{approach}_{task}_{ts}"
        self.audit_dir = os.path.join(save_dir, "cheatsheet_audit", dir_name)
        os.makedirs(self.audit_dir, exist_ok=True)

        self.model_name = model_name
        self.task = task
        self.approach = approach
        self.generator_model = generator_model or model_name
        self.curator_model = curator_model or model_name
        self.run_index = run_index
        self.run_flags = run_flags or {}

        self.snapshots: List[Dict] = []
        self.item_history: List[List[str]] = []

        # Write RUN_INFO.txt for easy organization into labeled folders
        self._write_run_info()

    def _write_run_info(self) -> None:
        """Write RUN_INFO.txt with run metadata and flags for easy folder organization."""
        lines = [
            "=" * 72,
            "  CHEATSHEET AUDIT RUN INFO",
            "  Use this to organize into cheatsheet_audit_by_label/<label>/",
            "=" * 72,
            "",
            f"  Run index:    {self.run_index if self.run_index is not None else 'N/A'}",
            f"  Model:        {self.model_name}",
            f"  Task:         {self.task}",
            f"  Approach:     {self.approach}",
            f"  Generator:    {self.generator_model}",
            f"  Curator:      {self.curator_model}",
            "",
            "  Flags / config:",
        ]
        for k, v in sorted(self.run_flags.items()):
            lines.append(f"    {k}: {v}")
        lines.extend(["", "=" * 72])
        run_info_path = os.path.join(self.audit_dir, "RUN_INFO.txt")
        with open(run_info_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def record(
        self,
        question_idx: int,
        question_text: str,
        cheatsheet: str,
        generator_output: str,
        final_answer: str,
        target: str,
        is_correct: bool,
        curator_output: Optional[str] = None,
    ):
        """Record one question's cheatsheet state and save the snapshot to disk."""
        items = _extract_memory_items(cheatsheet)
        descriptions = _extract_descriptions(cheatsheet)
        usage_counts = _extract_usage_counts(cheatsheet)
        token_count = _token_count(cheatsheet)

        cheatsheet_changed = True
        if self.snapshots:
            cheatsheet_changed = cheatsheet != self.snapshots[-1]["cheatsheet"]

        # Check if the generator's output references cheatsheet content
        gen_lower = generator_output.lower()
        items_referenced = 0
        for desc in descriptions:
            keywords = [w for w in desc.lower().split() if len(w) > 5][:3]
            if keywords and any(kw in gen_lower for kw in keywords):
                items_referenced += 1

        snapshot = {
            "version": len(self.snapshots),
            "question_idx": question_idx,
            "question_text": question_text[:300],
            "final_answer": final_answer,
            "target": target,
            "is_correct": is_correct,
            "cheatsheet_changed": cheatsheet_changed,
            "token_count": token_count,
            "word_count": len(cheatsheet.split()),
            "num_memory_items": len(items),
            "usage_counts": usage_counts,
            "items_referenced_by_generator": items_referenced,
            "cheatsheet": cheatsheet,
        }
        self.snapshots.append(snapshot)
        self.item_history.append(descriptions)

        # Save cheatsheet snapshot to disk
        cs_path = os.path.join(
            self.audit_dir, f"cheatsheet_v{len(self.snapshots):03d}_q{question_idx}.txt"
        )
        with open(cs_path, "w") as f:
            f.write(f"# Cheatsheet after Q{question_idx+1}\n")
            f.write(f"# Correct: {is_correct}  |  Answer: {final_answer}  |  Target: {target}\n")
            f.write(f"# Tokens: ~{token_count}  |  Memory items: {len(items)}\n")
            f.write(f"# {'='*70}\n\n")
            f.write(cheatsheet)

        # Append to running audit log
        log_entry = {k: v for k, v in snapshot.items() if k != "cheatsheet"}
        log_path = os.path.join(self.audit_dir, "audit_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")

    def finalize(self) -> Dict:
        """Produce the aggregate audit report and save to disk."""
        if not self.snapshots:
            return {}

        n = len(self.snapshots)
        correct_count = sum(1 for s in self.snapshots if s["is_correct"])

        # Token growth over time
        token_counts = [s["token_count"] for s in self.snapshots]
        item_counts = [s["num_memory_items"] for s in self.snapshots]

        # Track which descriptions persist across versions (reuse)
        desc_lifespan: Counter = Counter()
        all_descs_seen: set = set()
        for descs in self.item_history:
            current = set(descs)
            all_descs_seen.update(current)
            for d in current:
                desc_lifespan[d] += 1

        long_lived = [(d, c) for d, c in desc_lifespan.most_common(20) if c >= 2]
        ephemeral = [d for d, c in desc_lifespan.items() if c == 1]

        # Cheatsheet-ignored rate: generator didn't reference any items
        ignored_count = sum(
            1 for s in self.snapshots
            if s["num_memory_items"] > 0 and s["items_referenced_by_generator"] == 0
        )
        questions_with_items = sum(1 for s in self.snapshots if s["num_memory_items"] > 0)

        # Failure pattern: correct accuracy after cheatsheet exists vs before
        first_nonempty = next(
            (i for i, s in enumerate(self.snapshots) if s["num_memory_items"] > 0),
            n
        )
        acc_before = (
            sum(1 for s in self.snapshots[:first_nonempty] if s["is_correct"]) / first_nonempty
            if first_nonempty > 0 else 0
        )
        acc_after = (
            sum(1 for s in self.snapshots[first_nonempty:] if s["is_correct"]) / (n - first_nonempty)
            if (n - first_nonempty) > 0 else 0
        )

        # Lost-in-the-middle: flag if cheatsheet exceeds thresholds
        max_tokens = max(token_counts) if token_counts else 0
        over_2k = sum(1 for t in token_counts if t > 2000)
        over_4k = sum(1 for t in token_counts if t > 4000)

        # Structural quality checks on final cheatsheet
        final_cs = self.snapshots[-1]["cheatsheet"]
        structural_issues = []
        if "<memory_item>" not in final_cs and final_cs != "(empty)":
            structural_issues.append("no_memory_item_tags")
        if "<description>" not in final_cs and final_cs != "(empty)":
            structural_issues.append("no_description_tags")
        if "Count:" not in final_cs and final_cs != "(empty)":
            structural_issues.append("no_usage_counters")
        if len(final_cs.split()) > 2500:
            structural_issues.append("exceeds_2500_word_guideline")

        # Abstraction mismatch: curator vs generator model sizes
        gen_size = self._parse_model_size(self.generator_model)
        cur_size = self._parse_model_size(self.curator_model)
        abstraction_mismatch = (
            cur_size is not None and gen_size is not None and cur_size > gen_size * 1.5
        )

        report = {
            "run_index": self.run_index,
            "run_flags": self.run_flags,
            "model": self.model_name,
            "task": self.task,
            "approach": self.approach,
            "generator_model": self.generator_model,
            "curator_model": self.curator_model,
            "total_questions": n,
            "accuracy": correct_count / n if n else 0,
            "correct": correct_count,

            "token_growth": {
                "initial": token_counts[0] if token_counts else 0,
                "final": token_counts[-1] if token_counts else 0,
                "max": max_tokens,
                "mean": sum(token_counts) / n if n else 0,
                "over_2k_tokens": over_2k,
                "over_4k_tokens": over_4k,
                "trajectory": token_counts,
            },

            "memory_items": {
                "count_trajectory": item_counts,
                "final_count": item_counts[-1] if item_counts else 0,
                "unique_items_ever_created": len(all_descs_seen),
                "long_lived_items": len(long_lived),
                "ephemeral_items": len(ephemeral),
                "top_reused_items": [
                    {"description": d[:120], "versions_present": c}
                    for d, c in long_lived[:10]
                ],
            },

            "reuse_analysis": {
                "questions_with_cheatsheet": questions_with_items,
                "generator_ignored_cheatsheet": ignored_count,
                "ignore_rate": (
                    ignored_count / questions_with_items
                    if questions_with_items > 0 else 0
                ),
            },

            "cheatsheet_effectiveness": {
                "acc_before_cheatsheet_exists": round(acc_before, 3),
                "acc_after_cheatsheet_exists": round(acc_after, 3),
                "delta": round(acc_after - acc_before, 3),
                "first_nonempty_question": first_nonempty,
            },

            "structural_quality": {
                "issues": structural_issues,
                "final_word_count": len(final_cs.split()),
                "final_token_count": token_counts[-1] if token_counts else 0,
            },

            "abstraction_mismatch": {
                "detected": abstraction_mismatch,
                "generator_size": gen_size,
                "curator_size": cur_size,
                "note": (
                    "Curator is significantly larger than generator; "
                    "cheatsheet may contain abstractions the generator cannot execute."
                    if abstraction_mismatch else "Models are similar size."
                ),
            },

            "failure_patterns": self._detect_failure_patterns(),

            "audit_dir": self.audit_dir,
        }

        # Save report
        report_path = os.path.join(self.audit_dir, "audit_summary.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save human-readable summary
        self._write_readable_summary(report)

        return report

    def _parse_model_size(self, model_name: str) -> Optional[float]:
        """Extract parameter count in billions from model name."""
        m = re.search(r"(\d+(?:\.\d+)?)[Bb]", model_name)
        if m:
            return float(m.group(1))
        return None

    def _detect_failure_patterns(self) -> Dict:
        """Classify common cheatsheet failure modes."""
        patterns = {
            "overly_specific_tricks": 0,
            "contradictions": 0,
            "shallow_summaries": 0,
            "stale_no_updates": 0,
        }

        for i, snap in enumerate(self.snapshots):
            cs = snap["cheatsheet"]
            if cs == "(empty)":
                continue

            items = _extract_memory_items(cs)

            # Overly specific: items referencing exact numbers/IDs from specific questions
            for item in items:
                if re.search(r"Q\d+.*(?:answer is|equals|result is)\s+\d+", item, re.IGNORECASE):
                    patterns["overly_specific_tricks"] += 1
                    break

            # Shallow: very short memory items (< 20 words)
            short_items = [it for it in items if len(it.split()) < 20]
            if len(short_items) > len(items) * 0.5 and len(items) >= 2:
                patterns["shallow_summaries"] += 1

            # Stale: cheatsheet didn't change despite new question
            if not snap["cheatsheet_changed"] and i > 0:
                patterns["stale_no_updates"] += 1

        return patterns

    def _write_readable_summary(self, report: Dict):
        """Write a human-readable audit summary."""
        lines = []
        lines.append("=" * 72)
        lines.append("  CHEATSHEET AUDIT SUMMARY")
        lines.append("=" * 72)
        lines.append(f"  Model:     {report['model']}")
        lines.append(f"  Task:      {report['task']}")
        lines.append(f"  Approach:  {report['approach']}")
        lines.append(f"  Generator: {report['generator_model']}")
        lines.append(f"  Curator:   {report['curator_model']}")
        lines.append(f"  Questions: {report['total_questions']}")
        lines.append(f"  Accuracy:  {report['accuracy']:.1%} ({report['correct']}/{report['total_questions']})")
        lines.append("")

        tg = report["token_growth"]
        lines.append("--- Token Growth ---")
        lines.append(f"  Initial: {tg['initial']}  Final: {tg['final']}  Max: {tg['max']}  Mean: {tg['mean']:.0f}")
        lines.append(f"  Exceeded 2K tokens: {tg['over_2k_tokens']} times")
        lines.append(f"  Exceeded 4K tokens: {tg['over_4k_tokens']} times")
        lines.append("")

        mi = report["memory_items"]
        lines.append("--- Memory Items ---")
        lines.append(f"  Final count: {mi['final_count']}")
        lines.append(f"  Unique items ever created: {mi['unique_items_ever_created']}")
        lines.append(f"  Long-lived (present in 2+ versions): {mi['long_lived_items']}")
        lines.append(f"  Ephemeral (appeared once): {mi['ephemeral_items']}")
        if mi["top_reused_items"]:
            lines.append("  Top reused items:")
            for item in mi["top_reused_items"][:5]:
                lines.append(f"    [{item['versions_present']}x] {item['description'][:80]}...")
        lines.append("")

        ru = report["reuse_analysis"]
        lines.append("--- Reuse Analysis ---")
        lines.append(f"  Questions where cheatsheet existed: {ru['questions_with_cheatsheet']}")
        lines.append(f"  Generator ignored cheatsheet: {ru['generator_ignored_cheatsheet']} "
                      f"({ru['ignore_rate']:.0%} ignore rate)")
        lines.append("")

        ce = report["cheatsheet_effectiveness"]
        lines.append("--- Effectiveness ---")
        lines.append(f"  Accuracy BEFORE cheatsheet exists: {ce['acc_before_cheatsheet_exists']:.1%}")
        lines.append(f"  Accuracy AFTER  cheatsheet exists: {ce['acc_after_cheatsheet_exists']:.1%}")
        lines.append(f"  Delta: {ce['delta']:+.1%}")
        lines.append("")

        sq = report["structural_quality"]
        lines.append("--- Structural Quality ---")
        lines.append(f"  Final word count: {sq['final_word_count']}")
        if sq["issues"]:
            lines.append(f"  Issues: {', '.join(sq['issues'])}")
        else:
            lines.append("  No structural issues detected.")
        lines.append("")

        am = report["abstraction_mismatch"]
        lines.append("--- Abstraction Mismatch ---")
        lines.append(f"  Generator: {am['generator_size']}B  Curator: {am['curator_size']}B")
        lines.append(f"  Mismatch detected: {am['detected']}")
        if am["detected"]:
            lines.append(f"  WARNING: {am['note']}")
        lines.append("")

        fp = report["failure_patterns"]
        lines.append("--- Failure Patterns ---")
        lines.append(f"  Overly specific tricks: {fp['overly_specific_tricks']}")
        lines.append(f"  Shallow summaries: {fp['shallow_summaries']}")
        lines.append(f"  Stale (no update): {fp['stale_no_updates']}")
        lines.append("")
        lines.append(f"Full audit data: {report['audit_dir']}")
        lines.append("=" * 72)

        summary_path = os.path.join(self.audit_dir, "audit_summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(lines) + "\n")
