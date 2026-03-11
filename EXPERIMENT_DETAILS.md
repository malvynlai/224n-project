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
| **Max cheatsheet words** | 3000 |
| **Max context (vLLM)** | 32768 tokens; no curator input truncation |
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
- **Empirical note:** On GSM8K, DC-Cu often *hurts* SLM accuracy (see [Cheatsheet Audit](#cheatsheet-audit-empirical-findings--failure-analysis)).
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
- **Embeddings:** Pre-computed from `embeddings/{task}.csv` when available; fallback to sentence-transformers/all-MiniLM-L6-v2.
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
- **Embeddings:** Pre-computed from `embeddings/{task}.csv` when available; fallback to sentence-transformers/all-MiniLM-L6-v2.
- **Curator prompt:** `prompts/curator_prompt_for_dc_retrieval_synthesis.txt`.

### Output
- Shared: `results_multi_agent_dc_rs/{dataset}/{gen_tag}__{cur_tag}_MultiGenerator_DCRS_{timestamp}.jsonl`
- Separate: `*_MultiGenerator_DCRS_SepMem_*.jsonl`

---

## Future Experiments (Planned)

- [ ] *(Add more experiment types here)*

---

## Cheatsheet Audit: Empirical Findings & Failure Analysis

**Key empirical finding:** For SLMs, DC-Cu (Dynamic Cheatsheet Cumulative) often **does not help and sometimes hurts** accuracy. This section documents concrete failure patterns, quantifies reuse, and provides audit guidance.

### How to Run Audits

Use `--cheatsheet_verbose` with DC-Cumulative approaches. Output goes to:
`<save_dir>/cheatsheet_audit/<model>_<approach>_<task>_<timestamp>/`

Example:
```bash
python run_all_evaluations.py --approaches DynamicCheatsheet_Cumulative \
  --datasets GSM8K --max_samples 200 --dc_batch_size 1 --cheatsheet_verbose
```

---

### Empirical Results (GSM8K, 200 samples)

| Model | default | DC-Cu | Delta |
|-------|---------|-------|-------|
| Qwen2.5-7B  | 89.5% | 79.5% | **-10.0%** |
| Qwen2.5-14B | 75.5% | 74.5% | **-1.0%** |

**Accuracy before vs after cheatsheet exists:**
- 7B:  90.6% (Q1–31) → 77.4% (Q32–200) = **-13.2%**
- 14B: 84.4% (Q1–31) → 72.6% (Q32–200) = **-11.8%**

DC-Cu consistently hurts once the cheatsheet is populated. The generator *uses* the cheatsheet (low ignore rate ~2%) but the content is not helpful.

---

### Failure Patterns (from audit)

#### 1. Overly specific tricks
Memory items often encode **problem-specific solutions** rather than reusable strategies. Example from 14B audit:

```
<description>Calculate the total distance biked by Alisa and Stanley based on their speeds and times. (Reference: Q177)</description>
<example>
alisa_speed = 12  # miles per hour
stanley_speed = 10  # miles per hour
total_distance = alisa_speed * 4.5 + stanley_speed * 2.5
</example>
```

These are near-copy-paste solutions with hardcoded numbers. They do not generalize to new problems.

#### 2. Shallow summaries
Some curators produce very short, generic items (“break into smaller parts”) that add little signal. The 7B curator tended toward slightly more abstract heuristics (e.g., “Use algebraic equations”, “Break down complex problems”) but still with limited reuse.

#### 3. Stale / no updates
With `dc_batch_size=32`, the cheatsheet updates only every 32 questions. Within a batch, it stays unchanged (**162 “stale” snapshots** in one 200-Q run). The curator sees 8 Q&A pairs at once and may overwrite rather than incrementally refine.

#### 4. Low effective reuse
- **Usage counts:** Most items have `Count: 1` — they are written once and never reused.
- **Long-lived vs ephemeral:** Many items persist across versions (20 long-lived) but their *content* is too specific to help on new questions.
- **Reuse metric:** “Top reused” tracks *presence* in versions, not whether the generator actually applied the strategy correctly.

---

### Practical SLM Issues

#### Token growth & “lost in the middle”
- **Current cap:** 3000 words (~3900 tokens) via `MAX_CHEATSHEET_WORDS`.
- **Observed:** 7B final ~806 words, 14B final ~521 words; neither exceeded 2K tokens in these runs.
- **Recommendation:** Enforce a strict token cap (e.g., 600–800 tokens) and monitor growth. Long cheatsheets push useful content to the middle, where SLMs attend less.

#### Structured formatting
The curator prompt asks for `<memory_item>`, `<description>`, `<example>`, and `** Count: N **`. Small models sometimes:
- Drop tags or produce malformed blocks
- Produce examples that are too long or too specific
- Fail to maintain usage counters

Audit flags: `no_memory_item_tags`, `no_description_tags`, `no_usage_counters`, `exceeds_2500_word_guideline`.

#### Abstraction mismatch (mixed-scale curator)
When the **curator is larger** than the generator (e.g., 14B curator, 7B generators):
- The curator may write compact, abstract rules the 7B cannot follow
- The 7B may ignore or misapply them
- **Audit check:** `abstraction_mismatch.detected` when `curator_size > 1.5 × generator_size`

In same-model runs (7B–7B, 14B–14B), mismatch is false. For multi-agent setups with a larger curator, this should be monitored.

---

### Quantified Reuse (from audit_summary.json)

| Metric | 7B | 14B |
|--------|-----|-----|
| Final memory items | 2 | 8 |
| Unique items ever created | 58 | 50 |
| Long-lived (2+ versions) | 20 | 20 |
| Generator ignored cheatsheet | 3 (2%) | 3 (2%) |
| Stale (no update) | 162 | 162 |

---

### Example Cheatsheet (14B, after Q200)

The final cheatsheet contained 8 items. Representative item:

```
<memory_item>
<description>Calculate the total cost of 6 erasers and 8 pencils. (Reference: Q181)</description>
<example>
cost_eraser = 2
cost_pencil = 3
total_cost = 6 * 2 + 8 * 3
</example>
**Count: 1**
</memory_item>
```

This is a restatement of a single problem with concrete numbers, not a reusable pattern like “For unit-cost problems: total = quantity × unit_price”.

---

### Recommendations

1. **Stricter caps:** Enforce a hard token limit (e.g., 600 tokens) and prioritize recent, high-value items.
2. **Better curation prompts:** Encourage *general* heuristics and patterns, not problem-specific solutions. Discourage hardcoded numbers and Q-refs in examples.
3. **Smaller batches:** Use `dc_batch_size=1` for per-question curation when auditing; this reduces staleness and may improve incremental updates.
4. **Abstraction alignment:** For mixed-scale setups, prefer a curator no larger than the generator, or prompt the curator to write at the generator’s level.
5. **Reuse validation:** Track whether items with Count > 1 actually correlate with correct answers on similar questions.

### Audit Output Structure

Each audit run produces:
- `audit_summary.json` — Full metrics (token growth, reuse, failure patterns, abstraction mismatch)
- `audit_summary.txt` — Human-readable summary
- `audit_log.jsonl` — Per-question log (token_count, num_memory_items, is_correct, etc.)
- `cheatsheet_v{N}_q{M}.txt` — Cheatsheet snapshot after question M

---
