#!/usr/bin/env python3
"""
Scan all result directories and update summary_all_latest.json with a table of completed runs.
Run: python scripts/update_summary.py
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIRS = [
    ("results_oss", "run_all_evaluations"),
    ("results_dc_rs", "run_dc_rs_eval"),
    ("results_multi_agent", "run_multi_agent_eval"),
    ("results_multi_agent_dc_rs", "run_multi_agent_dc_rs_eval"),
]


def parse_summary_line(path: Path) -> dict | None:
    """Parse first line of jsonl if it has _summary."""
    try:
        with open(path) as f:
            first = f.readline()
        if not first.strip():
            return None
        row = json.loads(first)
        if not row.get("_summary"):
            return None
        return {
            "file": row.get("file", path.stem),
            "dataset": row.get("dataset", path.parent.name),
            "accuracy": row.get("accuracy", 0),
            "accuracy_pct": row.get("accuracy_pct", "0%"),
            "correct": row.get("correct", 0),
            "total": row.get("total", 0),
            "output_file": str(path.relative_to(PROJECT_ROOT)),
        }
    except Exception:
        return None


def extract_model_and_approach(path: Path, source: str) -> tuple[str, str]:
    """Extract human-readable model/setup and approach from filename."""
    stem = path.stem
    dataset = path.parent.name

    if source == "results_oss":
        # Qwen2.5-7B-Instruct_default_20260311-1543
        # Qwen2.5-14B-Instruct_default_sc5_20260311-1656
        # Qwen2.5-7B-Instruct_DynamicCheatsheet_Cumulative_20260310-1726
        parts = stem.split("_")
        model = parts[0] if parts else "?"
        if "DynamicCheatsheet_Cumulative" in stem:
            approach = "DC-Cumulative"
        elif "default_sc5" in stem or "_sc5_" in stem:
            approach = "default_sc5"
        elif "default" in stem:
            approach = "default"
        elif "DynamicCheatsheet_RetrievalSynthesis" in stem:
            approach = "DC-RS"
        else:
            approach = stem.split("_")[-2] if len(parts) >= 2 else "?"
        return model, approach

    if source == "results_dc_rs":
        # Qwen2.5-7B-Instruct_DynamicCheatsheet_RetrievalSynthesis_20260312-0214
        model = stem.split("_")[0] if "_" in stem else stem
        approach = "DC-RS"
        return model, approach

    if source in ("results_multi_agent", "results_multi_agent_dc_rs"):
        # Qwen2.5-7B-Instruct+Qwen2.5-14B-Instruct+Mistral-7B-Instruct-v0.2__Qwen2.5-14B-Instruct_MultiGenerator_20260311-013112
        if "__" in stem:
            right = stem.split("__")[1]
            if "MultiGenerator_DCRS" in right or "MultiGenerator_DCRS_SepMem" in right:
                approach = "MultiGen_DCRS" + ("_nonshared" if "SepMem" in right else "")
            elif "MultiGenerator_Cumulative" in right:
                approach = "MultiGen_Cu"
            elif "MultiGenerator" in right:
                approach = "MultiGenerator"
            else:
                approach = right.split("_")[0] if "_" in right else "?"
        else:
            approach = "?"
        return "3-gen ensemble", approach

    return "?", "?"


def collect_all_runs() -> list[dict]:
    """Collect all runs from result directories, deduplicating by (source, dataset, model, approach)."""
    seen: dict[tuple, dict] = {}  # (source, dataset, model, approach) -> best run

    for result_dir, _script in RESULT_DIRS:
        base = PROJECT_ROOT / result_dir
        if not base.exists():
            continue
        for jsonl_path in base.rglob("*.jsonl"):
            if "cheatsheet_audit" in str(jsonl_path):
                continue
            summary = parse_summary_line(jsonl_path)
            if not summary or summary.get("total", 0) == 0:
                continue
            model, approach = extract_model_and_approach(jsonl_path, result_dir)
            key = (result_dir, summary["dataset"], model, approach)
            # Prefer run with higher total (more complete) or higher accuracy
            existing = seen.get(key)
            if existing is None or summary["total"] > existing["total"] or (
                summary["total"] == existing["total"] and summary["accuracy"] > existing["accuracy"]
            ):
                summary["source"] = result_dir
                summary["model_display"] = model
                summary["approach_display"] = approach
                seen[key] = summary

    return sorted(seen.values(), key=lambda r: (r["dataset"], r["model_display"], r["approach_display"]))


def main():
    runs = collect_all_runs()

    out = {
        "_meta": {
            "description": "Coalesced summary of all evaluation runs",
            "sources": [d[0] for d in RESULT_DIRS],
            "generated": datetime.now().strftime("%Y-%m-%d"),
        },
        "runs": runs,
    }

    out_path = PROJECT_ROOT / "summary_all_latest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Updated {out_path} with {len(runs)} runs")

    # Write table to EXPERIMENT_RUNS.md (replace the Results Summary section)
    table_path = PROJECT_ROOT / "EXPERIMENT_RUNS.md"
    if table_path.exists():
        with open(table_path) as f:
            content = f.read()
        table_lines = [
            "| Dataset | Model / Setup | Approach | Accuracy | Correct | Total | Source |",
            "|---------|----------------|----------|----------|---------|-------|--------|",
        ]
        for r in runs:
            src = r["source"].replace("results_", "")
            table_lines.append(
                f"| {r['dataset']} | {r['model_display']} | {r['approach_display']} | "
                f"{r['accuracy_pct']} | {r['correct']} | {r['total']} | {src} |"
            )
        new_table = "\n".join(table_lines)
        # Replace table between "## Results Summary" and next "---"
        start_marker = "## Results Summary (from summary_all_latest.json)"
        start = content.find(start_marker)
        if start >= 0:
            table_start = content.find("\n\n", start) + 2
            table_end = content.find("\n\n---", table_start)
            if table_end < 0:
                table_end = content.find("\n## ", table_start)
            if table_end < 0:
                table_end = len(content)
            content = content[:table_start] + new_table + "\n" + content[table_end:]
        with open(table_path, "w") as f:
            f.write(content)
        print(f"Updated table in {table_path}")

    # Print markdown table
    print("\n" + "=" * 100)
    print("  RESULTS TABLE (all completed runs)")
    print("=" * 100)
    print()
    print("| Dataset | Model / Setup | Approach | Accuracy | Correct | Total | Source |")
    print("|---------|----------------|----------|----------|---------|-------|--------|")
    for r in runs:
        src = r["source"].replace("results_", "")
        print(f"| {r['dataset']} | {r['model_display']} | {r['approach_display']} | {r['accuracy_pct']} | {r['correct']} | {r['total']} | {src} |")
    print("=" * 100)


if __name__ == "__main__":
    main()
