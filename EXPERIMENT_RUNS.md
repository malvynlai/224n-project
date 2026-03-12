# Experiment Runs (2 Models × 2 Datasets)

**Models:** Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct  
**Datasets:** MMLU_Pro_Engineering, GPQA_Diamond

**Completed → Label mapping (from summary_all_latest.json):**
- B1: Qwen2.5-7B-Instruct_default (MMLU)
- B3: Qwen2.5-7B-Instruct_default (GPQA)
- B5: Qwen2.5-14B-Instruct_default (MMLU)
- M1: MultiGenerator (MMLU)
- M2: MultiGenerator_Cumulative shared (MMLU)
- M4: MultiGenerator (GPQA)
- S2: Qwen2.5-7B-Instruct_default_sc5 (GPQA)
- S4: Qwen2.5-14B-Instruct_default_sc5 (GPQA)

---

## Results Summary (from summary_all_latest.json)

*Regenerate with: `python scripts/update_summary.py`*

| Dataset | Model / Setup | Approach | Accuracy | Correct | Total | Source |
|---------|----------------|----------|----------|---------|-------|--------|
| AIME_2025 | Qwen2.5-14B-Instruct | DC-Cumulative | 10.0% | 3 | 30 | oss |
| AIME_2025 | Qwen2.5-14B-Instruct | default | 6.7% | 2 | 30 | oss |
| AIME_2025 | Qwen2.5-7B-Instruct | DC-Cumulative | 3.3% | 1 | 30 | oss |
| AIME_2025 | Qwen2.5-7B-Instruct | default | 0.0% | 0 | 30 | oss |
| GPQA_Diamond | 3-gen ensemble | MultiGen_Cu | 28.8% | 57 | 198 | multi_agent |
| GPQA_Diamond | 3-gen ensemble | MultiGenerator | 30.8% | 61 | 198 | multi_agent |
| GPQA_Diamond | Qwen2.5-14B-Instruct | default_sc5 | 37.9% | 75 | 198 | oss |
| GPQA_Diamond | Qwen2.5-7B-Instruct | DC-RS | 0.0% | 0 | 198 | dc_rs |
| GPQA_Diamond | Qwen2.5-7B-Instruct | default | 29.8% | 59 | 198 | oss |
| GPQA_Diamond | Qwen2.5-7B-Instruct | default_sc5 | 35.9% | 71 | 198 | oss |
| GSM8K | 3-gen ensemble | MultiGen_Cu | 87.0% | 174 | 200 | multi_agent |
| GSM8K | 3-gen ensemble | MultiGenerator | 90.5% | 181 | 200 | multi_agent |
| GSM8K | Qwen2.5-14B-Instruct | DC-Cumulative | 74.5% | 149 | 200 | oss |
| GSM8K | Qwen2.5-14B-Instruct | default | 76.9% | 1014 | 1319 | oss |
| GSM8K | Qwen2.5-7B-Instruct | DC-Cumulative | 78.0% | 1029 | 1319 | oss |
| GSM8K | Qwen2.5-7B-Instruct | default | 87.0% | 1148 | 1319 | oss |
| MMLU_Pro_Engineering | 3-gen ensemble | MultiGen_Cu | 33.5% | 67 | 200 | multi_agent |
| MMLU_Pro_Engineering | 3-gen ensemble | MultiGenerator | 32.5% | 65 | 200 | multi_agent |
| MMLU_Pro_Engineering | Qwen2.5-14B-Instruct | default | 38.1% | 369 | 969 | oss |
| MMLU_Pro_Engineering | Qwen2.5-7B-Instruct | DC-RS | 0.0% | 0 | 200 | dc_rs |
| MMLU_Pro_Engineering | Qwen2.5-7B-Instruct | default | 33.5% | 325 | 969 | oss |


---

## 1. Baseline (8 runs) — `run_all_evaluations.py`

2 models × 2 datasets × 2 modes (default, DynamicCheatsheet_Cumulative)

| Label | Model | Dataset | Mode | Status |
|-------|-------|---------|------|--------|
| B1 | 7B | MMLU_Pro_Engineering | default | ✓ |
| B2 | 7B | MMLU_Pro_Engineering | DC-Cumulative | — |
| B3 | 7B | GPQA_Diamond | default | ✓ |
| B4 | 7B | GPQA_Diamond | DC-Cumulative | — |
| B5 | 14B | MMLU_Pro_Engineering | default | ✓ |
| B6 | 14B | MMLU_Pro_Engineering | DC-Cumulative | — |
| B7 | 14B | GPQA_Diamond | default | — |
| B8 | 14B | GPQA_Diamond | DC-Cumulative | — |

---

## 2. Multi-Agent Ensemble (8 runs) — `run_multi_agent_eval.py`

2 datasets × 2 modes (MultiGenerator, MultiGenerator_Cumulative) × 2 memory (shared, non-shared)

*Note: MultiGenerator has no cheatsheet, so "memory" = N/A; we run it once per dataset. MultiGenerator_Cumulative runs with shared and non-shared.*

| Label | Dataset | Mode | Memory | Status |
|-------|---------|------|--------|--------|
| M1 | MMLU_Pro_Engineering | MultiGenerator | — | ✓ |
| M2 | MMLU_Pro_Engineering | MultiGenerator_Cumulative | shared | ✓ |
| M3 | MMLU_Pro_Engineering | MultiGenerator_Cumulative | non-shared | — |
| M4 | GPQA_Diamond | MultiGenerator | — | ✓ |
| M5 | GPQA_Diamond | MultiGenerator_Cumulative | shared | — |
| M6 | GPQA_Diamond | MultiGenerator_Cumulative | non-shared | — |
| M7 | (reserved) | | | |
| M8 | (reserved) | | | |

*M7–M8: If 8 = 2×2×2, use M4–M6 + M7 (GPQA MultiGen), M8 (GPQA MultiGen_Cu both memory).*

---

## 3. Self-Consistency (4 runs) — `run_all_evaluations.py --self_consistency`

2 models × 2 datasets (default approach only)

| Label | Model | Dataset | Status |
|-------|-------|---------|--------|
| S1 | 7B | MMLU_Pro_Engineering | — |
| S2 | 7B | GPQA_Diamond | ✓ |
| S3 | 14B | MMLU_Pro_Engineering | — |
| S4 | 14B | GPQA_Diamond | ✓ |

---

## 4. Stronger Curator Multi-Agent (8 runs) — `run_multi_agent_eval.py` (curator = weakest Gemini)

2 datasets × 2 modes × 2 memory modes

| Label | Dataset | Mode | Memory | Status |
|-------|---------|------|--------|--------|
| C1 | MMLU_Pro_Engineering | MultiGenerator | — | — |
| C2 | MMLU_Pro_Engineering | MultiGenerator_Cumulative | shared | — |
| C3 | MMLU_Pro_Engineering | MultiGenerator_Cumulative | non-shared | — |
| C4 | GPQA_Diamond | MultiGenerator | — | — |
| C5 | GPQA_Diamond | MultiGenerator_Cumulative | shared | — |
| C6 | GPQA_Diamond | MultiGenerator_Cumulative | non-shared | — |
| C7 | (reserved) | | | |
| C8 | (reserved) | | | |

---

## 5. DC-RS Normal (4 runs) — `run_dc_rs_eval.py`

2 models × 2 datasets

| Label | Model | Dataset | Status |
|-------|-------|---------|--------|
| R1 | 7B | MMLU_Pro_Engineering | — |
| R2 | 7B | GPQA_Diamond | — |
| R3 | 14B | MMLU_Pro_Engineering | — |
| R4 | 14B | GPQA_Diamond | — |

---

## 6. DC-RS Multi-Agent (4 runs) — `run_multi_agent_dc_rs_eval.py`

2 datasets × 2 memory modes (shared, non-shared)

| Label | Dataset | Memory | Status |
|-------|---------|--------|--------|
| D1 | MMLU_Pro_Engineering | shared | — |
| D2 | MMLU_Pro_Engineering | non-shared | — |
| D3 | GPQA_Diamond | shared | — |
| D4 | GPQA_Diamond | non-shared | — |

---

## Cheatsheet-Verbose Folders

Labeled placeholder folders: `cheatsheet_audit_by_label/<label>_<description>/`

Each folder contains `LABEL.txt` describing which run it belongs to. When you run with `--cheatsheet_verbose`, output goes to `<save_dir>/cheatsheet_audit/` (with model/approach/task/timestamp). Copy results into the matching labeled folder for easy identification.

| Label | Folder |
|-------|--------|
| B2 | B2_7B_MMLU_Pro_Engineering_DC-Cumulative |
| B4 | B4_7B_GPQA_Diamond_DC-Cumulative |
| B6 | B6_14B_MMLU_Pro_Engineering_DC-Cumulative |
| B8 | B8_14B_GPQA_Diamond_DC-Cumulative |
| M2 | M2_MMLU_Pro_Engineering_MultiGen_Cu_shared |
| M3 | M3_MMLU_Pro_Engineering_MultiGen_Cu_nonshared |
| M5 | M5_GPQA_Diamond_MultiGen_Cu_shared |
| M6 | M6_GPQA_Diamond_MultiGen_Cu_nonshared |
| C2–C6 | Stronger curator MultiGen_Cu |
| R1–R4 | DC-RS normal |
| D1–D4 | DC-RS multi-agent |
