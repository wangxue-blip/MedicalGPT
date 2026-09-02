# FLOWGM GLM 核心 LoRA 消融实验

本目录保存 FLOWGM GLM-4-9B Chat 的核心消融实验代码、配置和训练日志。

## 实验设计

3 个任务（`white-stage2`、`malware-Cridex-stage2`、`apt`）分别测试 6 个配置：LoRA rank 8/16/32、QKV、QKV+dense、attention+MLP，以及是否训练 prompt，共 18 组。

固定设置：seed=42，batch size=1，gradient accumulation=4，max length=2048，验证集最多 2,000 条。模型权重和 checkpoint 未纳入 Git（本地约 11 GB），可按配置重新生成。

## 当前状态

截至 2026-09-02，13/18 组已生成 `final`；5 组需要继续补跑。原始并行日志见 `logs/ablation_run.log`。该日志包含训练 loss、验证 loss、ROUGE-1/2/L 和 BLEU-4。

## 运行

```bash
cd repro_glm
python -u run_ablation_core6.py \
  --config ablation_core6.json \
  --model /path/to/glm-4-9b-chat \
  --data-root /path/to/glm_dataset/by_adapter \
  --output outputs/ablation_core6 \
  --plan train_plan.json \
  --gpus 2,3,4,5,6,7
```

调度器会自动跳过已有 `final` 的实验，并继续未完成任务；单个任务失败不会中止其余任务。
