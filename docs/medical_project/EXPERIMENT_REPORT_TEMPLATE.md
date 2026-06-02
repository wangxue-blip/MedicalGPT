# MedicalGPT Medical SFT + GRPO Experiment Report

## 1. Project Background

TODO

## 2. MedicalGPT Baseline and Project Goal

TODO

## 3. Data Source and Leakage Avoidance

- C-eval test data was not used for filtering, tuning, or model selection.

## 4. Embedding Similarity Filtering

TODO

## 5. SFT Training Settings

TODO

## 6. GRPO Reward Design

```text
R = 0.35 * format
  + 0.30 * reference_similarity
  + 0.25 * safety
  - 0.10 * repetition_penalty
```

Reference similarity is not medical accuracy.

## 7. Evaluation Sets and Metrics

TODO

## 8. Results

TODO

## 9. Case Study: Base vs SFT vs GRPO

TODO

## 10. Error Analysis

TODO

## 11. Limitations

This project is not a medical diagnosis system.

## 12. Resume Project Description

TODO
