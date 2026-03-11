# Experiment Details

This document records every configurable detail for each experiment type. **Five experiment types** (Baseline, Multi-Agent, Self-Consistency, DC-RS, Multi-Agent DC-RS); more may be added.

---

## Shared Infrastructure (All Experiments)

| Parameter | Value |
|----------|-------|
| **Backend** | vLLM (or HuggingFace as fallback) |
| **Quantization** | 4bit (configurable: none, 4bit, 8bit) |
| **Max tokens** | 2048 |
| **Temperature** | 0.0 (default); 0.7 for self-consistency |
| **Shuffle seed** | 10 |
| **Max cheatsheet words** | 800 |
| **Max curator input tokens** | 6000 |
| **Retrieve top-k** (for retrieval-based approaches) | 3 |

### Evaluation Types (by dataset)
| Dataset | Eval type |
|---------|-----------|
| AIME_2024, AIME_2025, AIME_2020_2024, GSM8K | exact |
| GPQA_Diamond, MMLU_Pro_Physics, MMLU_Pro_Engineering | mcq |
| MathEquationBalancer | equation |

---

## Experiment 1: Baseline (OSS Single-Model)

**Script:** `run_all_evaluations.py`

### Models
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct

### Datasets
- **Run so far:** GSM8K, MMLU_Pro_Engineering
- **Available:** AIME_2024, AIME_2025, AIME_2020_2024, GPQA_Diamond, MMLU_Pro_Physics, MMLU_Pro_Engineering, MathEquationBalancer, GSM8K

### Approaches (2 versions)
1. **default** — Single model, no cheatsheet. Prompt: `[[QUESTION]]` + `[[CHEATSHEET]]` = "(empty)".
2. **DynamicCheatsheet_Cumulative (DC-Cu)** — Single model + evolving cheatsheet curated after each batch.

### DC-Cumulative Curation Details
- **Generator batch size** (`--dc_batch_size`): 32 (default); use 1 for per-question curation.
- **Curator sub-batch size**: 8 Q&A pairs per curator call (4 curator calls per batch when batch_size=32).
- **Curator prompt**: `prompts/curator_prompt_for_dc_cumulative.txt`.
- **Flow per batch**: (1) Batch-generate answers with current cheatsheet; (2) Run curator in sub-batches of 8; (3) Update cheatsheet; (4) Checkpoint.

### Output
- `results_oss/{dataset}/{model}_{approach}_{timestamp}.jsonl`

---

## Experiment 2: Multi-Agent (3 Generators + 1 Curator)

**Script:** `run_multi_agent_eval.py`

### Models
- **Generators:** Qwen2.5-7B, Qwen2.5-14B, Mistral-7B-Instruct-v0.2
- **Curator:** Qwen2.5-14B-Instruct

### Datasets
- AIME_2024, AIME_2025, AIME_2020_2024, GPQA_Diamond, MMLU_Pro_Physics, MMLU_Pro_Engineering, MathEquationBalancer
- (GSM8K not in multi-agent `ALL_DATASETS` by default)

### Approaches (2 versions)
1. **MultiGenerator** — 3 generators, majority vote, no cheatsheet.
2. **MultiGenerator_Cumulative** — 3 generators, majority vote, + evolving cheatsheet.

### Shared vs Separate Memory (MultiGenerator_Cumulative only)
- **Shared memory (default):** One cheatsheet shared by all 3 generators; curator uses majority-voted answers.
- **Separate memory (`--no_shared_memory`):** Three cheatsheets, one per generator; each curator updates from its own generator's outputs; then majority vote. Output: `*_MultiGenerator_Cumulative_SepMem_*.jsonl`.

### Prompting Procedure
1. **Per question:** Format as `Question #N:\n{raw_input}` + task-specific suffix (AIME integer format, GSM8K number format, etc.).
2. **Generator prompt:** `prompts/generator_prompt.txt` — `[[QUESTION]]` + `[[CHEATSHEET]]`.
3. **Per batch:** Each generator produces answers for all questions in batch (1 vLLM call per generator).
4. **Majority vote:** Per question, take most common answer across 3 generators.

### Curation Procedure (MultiGenerator_Cumulative)
1. **Batch size** (`--dc_batch_size`): 32 (default).
2. **Curator sub-batch size**: 8 Q&A pairs per curator call.
3. **Curator input:** Previous cheatsheet + batch of Q&A (question + majority-voted answer).
4. **Curator prompt:** `prompts/curator_prompt_for_dc_cumulative.txt`.
5. **Flow per batch:** (1) Each generator batch-generates for batch; (2) Majority vote per question; (3) Curator runs in sub-batches of 8; (4) Update cheatsheet; (5) Checkpoint.

### Output
- Shared memory: `results_multi_agent/{dataset}/{gen_tag}__{cur_tag}_MultiGenerator_Cumulative_{timestamp}.jsonl`
- Separate memory: `results_multi_agent/{dataset}/{gen_tag}__{cur_tag}_MultiGenerator_Cumulative_SepMem_{timestamp}.jsonl`

---

## Experiment 3: Self-Consistency

**Script:** `run_all_evaluations.py` (default approach only)

### Flags
- `--self_consistency` — Enable k-way resampling + majority vote.
- `--self_consistency_k` — Number of samples (default: 5).

### Behavior
- Resample each question k times; majority vote for final answer.
- **Only for:** `default` approach. **Ignored for:** DC-Cumulative.
- Requires **temperature > 0** for diversity (e.g. 0.7).

### Output
- `results_oss/{dataset}/{model}_default_sc{k}_{timestamp}.jsonl` (e.g. `_default_sc5_`)

---

## Experiment 4: DC-RS (Retrieval & Synthesis)

**Script:** `run_dc_rs_eval.py`  
**Paper:** [Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory](https://arxiv.org/abs/2504.07952)

### Flow (per question i)
1. **Retrieve:** R_i = Retr(x_i, {(x_j, ỹ_j)}_{j<i}, k) — top-k similar past (input, output) pairs via cosine similarity of embeddings.
2. **Curate (before generation):** M_i = Cur(M_{i-1}, x_i, R_i) — curator synthesizes memory from retrieved pairs.
3. **Generate:** ỹ_i = Gen(x_i, M_i).

### Key difference from DC-Cu
- DC-Cu curates *after* generation; DC-RS curates *before*.
- DC-RS uses retrieval to surface relevant past examples; DC-Cu is purely cumulative.
- DC-RS tends to help more on diverse benchmarks (e.g. GPQA-Diamond).

### Parameters
- **Retrieve top-k** (`--retrieve_top_k`): 3 (default).
- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2.
- **Curator prompt:** `prompts/curator_prompt_for_dc_retrieval_synthesis.txt`.

### Output
- `results_dc_rs/{dataset}/{model}_DynamicCheatsheet_RetrievalSynthesis_{timestamp}.jsonl`

---

## Experiment 5: Multi-Agent DC-RS

**Script:** `run_multi_agent_dc_rs_eval.py`  
**Paper:** [Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory](https://arxiv.org/abs/2504.07952)

Combines multi-agent (3 generators + 1 curator, majority vote) with DC-RS (retrieve → curate before generation → generate).

### Flow (per question i)
1. **Retrieve:** Top-k similar past (input, output) pairs via embeddings.
2. **Curate:** Curator synthesizes cheatsheet from retrieved pairs (before generation).
3. **Generate:** All 3 generators answer with updated cheatsheet.
4. **Majority vote** for final answer.
5. Update retrieval history (shared: majority-voted representative output; separate: per-generator outputs).

### Shared vs Separate Memory
- **Shared (default):** One cheatsheet, one retrieval history (representative = winning generator's full output).
- **Separate (`--no_shared_memory`):** Three cheatsheets, three retrieval histories; each curator updates from its own generator's outputs.

### Parameters
- **Retrieve top-k** (`--retrieve_top_k`): 3 (default).
- **Embedding model:** sentence-transformers/all-MiniLM-L6-v2.
- **Curator prompt:** `prompts/curator_prompt_for_dc_retrieval_synthesis.txt`.

### Output
- Shared: `results_multi_agent_dc_rs/{dataset}/{gen_tag}__{cur_tag}_MultiGenerator_DCRS_{timestamp}.jsonl`
- Separate: `*_MultiGenerator_DCRS_SepMem_*.jsonl`

---

## Future Experiments (Planned)

- [ ] *(Add more experiment types here)*

---

## Cheatsheet Audit Notes

*(Use this section to record observations from `--cheatsheet_verbose` runs. Audit output goes to `<save_dir>/cheatsheet_audit/<model>_<approach>_<task>_<timestamp>/`.)*

---

### Audit run: [date / run id]

**Setup:** model, approach, dataset, dc_batch_size, sample count

**Observations:**
- Token growth over questions:
- Memory item reuse patterns:
- Failure patterns:
- Abstraction mismatch (curator vs generator):

---
