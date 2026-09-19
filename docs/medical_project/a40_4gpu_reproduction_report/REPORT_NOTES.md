# Report notes

## Audience and structure

- Audience: technical.
- Delivery: canonical portable HTML used as the source for a verified PDF.
- Required structure mapping:
  - Title: title block.
  - Technical summary: technical_summary.
  - Key findings and visual evidence: result_interpretation, ppl_stage, existing_metrics.
  - Scope/data/definitions: reading_map, positioning, data_lineage, evaluation.
  - Methodology and model specification: pipeline, data_method, scoring_deep_dive, sft_internals, grpo_algorithm, reward_path_audit.
  - Limitations/robustness: data_quality_protocol, eval_deep_dive, experiment_design, limitations.
  - Recommended next steps: recommendations, production_blueprint.
  - Further questions: further_questions.

## Revision scope

The detailed edition preserves all blocks, tables, chart data, sources, caveats, commands, and interview content from the first edition. It adds implementation-depth sections, four-GPU scheduling, reward-path auditing, statistical evaluation guidance, reproducibility manifests, additional interview Q&A, and a production architecture.

## Chart map

- Segment: stage results.
- Question: how held-out long-text PPL differs across Base/SFT/GRPO stages.
- Family/type: categorical comparison / bar.
- Fields: model, ppl; QA similarity and safety pass retained in tooltips and source rows.
- Supported claim: the main distribution-fit gain occurs from Base to SFT; later GRPO changes are small.
- Palette: single-root/no redundant legend.
- Caveat: PPL is not medical correctness.

No additional charts were added because the remaining quantitative sections are exact run budgets, artifact contracts, and low-row configuration comparisons where tables are more auditable than graphics.
