# MedicalGPT Medical SFT + GRPO Project

This directory documents the isolated MedicalGPT medical-domain experiment.

## Scope

- Build filtered medical SFT data from local raw medical QA data.
- Fine-tune Qwen2.5-7B-Instruct with LoRA or QLoRA.
- Build small medical GRPO data and medical reward functions.
- Evaluate Base, SFT, and GRPO models with PPL, reference similarity, structure/safety heuristics, and C-eval medical dev subsets.

## Data Safety Rules

- C-eval test data is never used for filtering, tuning, or report claims.
- Embedding similarity is reported only as reference similarity.
- This project is a model training experiment, not a medical diagnosis system.

## Raw Data Conversion

Downloaded `shibing624/medical` finetune files are treated as source data:

```text
data_raw/medical_project/medical/train_zh_0.json
data_raw/medical_project/medical/valid_zh_0.json
```

Convert them into the project raw entry:

```bash
python tools/medical_project/convert_medical_json_to_raw.py \
  --inputs data_raw/medical_project/medical/train_zh_0.json data_raw/medical_project/medical/valid_zh_0.json \
  --output data_raw/medical_project/medical_raw.jsonl \
  --overwrite
```

Conversion logic:

- The source files are read as JSONL when they start with `{`, or as JSON arrays when they start with `[`.
- Original fields such as `instruction`, `input`, and `output` are preserved.
- Metadata fields `source`, `source_file`, and `source_split` are added for traceability.
- No filtering, deduplication, or test-split usage happens during conversion.
- Filtering and normalization are handled later by `prepare_raw_medical.py`.

## Stage Status

- Stage 0: directory, script, config, and document skeleton.
- Later stages will fill each script with executable logic.
