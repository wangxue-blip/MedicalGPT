# MedicalGPT v2 Hard-Sample GRPO Workspace

This directory documents the v2 hard-sample GRPO extension. The v2 workflow is
kept separate from the existing MedicalGPT medical project workflow so previous
data, scripts, and experiment outputs remain intact.

## Scope

The v2 workflow adds:

1. Hard prompt pool construction.
2. Base/SFT mining generation.
3. Reward-based hard sample selection.
4. 1500-prompt GRPO training for A100 80GB.
5. Additional evaluation metrics for keyword coverage, format compliance,
   safety violations, and pairwise preference.

## Directory Layout

```text
tools/medical_project/v2/          Data construction and mining utilities
eval/medical_project/v2/           v2 evaluation utilities
scripts/medical_project/v2/        Manual run scripts for each v2 stage
configs/medical_project/v2/        v2 experiment configs
data_processed/medical_project/v2/ Generated v2 datasets
outputs/medical_project/v2/        Generated v2 predictions, models, metrics
docs/medical_project/v2/           v2 task lists, notes, and reports
```

## Non-Overwrite Rule

Do not overwrite existing v1 scripts or outputs. New v2 implementation should
write to the v2 paths above. Existing files such as
`scripts/medical_project/run_08_grpo_train.sh`,
`scripts/medical_project/run_09_eval_all.sh`, and
`training/medical_grpo_rewards.py` are treated as references, not edit targets
for the v2 workflow.

## Data Leakage Rule

C-Eval test data must not be used for filtering, training, tuning, or report
selection. If C-Eval dev data is used to generate v2 training prompts, final
C-Eval reporting should use a disjoint dev/valid split such as
`data_raw/medical_project/ceval_val`.

## Metric Wording Rule

Reference similarity, keyword coverage, and automatic preference scores are
heuristic evaluation signals. They must not be described as true medical
accuracy or clinical correctness.
