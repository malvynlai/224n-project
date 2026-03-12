Cheatsheet audit folders by experiment label.

When running with --cheatsheet_verbose, the scripts save to:
  results_oss/cheatsheet_audit/           (run_all_evaluations)
  results_dc_rs/cheatsheet_audit/         (run_dc_rs_eval)
  results_multi_agent/cheatsheet_audit/   (run_multi_agent_eval)
  results_multi_agent_dc_rs/cheatsheet_audit/  (run_multi_agent_dc_rs_eval)

Each audit is saved as run1_..., run2_..., run3_... with RUN_INFO.txt inside
listing model, task, approach, and all flags. Use RUN_INFO.txt to match
audits to the correct labeled folder (e.g. B2_7B_MMLU_Pro_Engineering_DC-Cumulative).

Currently populated:
  M5_GPQA_Diamond_MultiGen_Cu_shared/  — full audit (198 Qs) + partial_164951/

Each folder contains LABEL.txt describing the run. See EXPERIMENT_RUNS.md
for the full run catalog.
