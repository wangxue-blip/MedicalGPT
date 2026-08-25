# 补充消融实验设计

所有新 SFT 对照固定为 Qwen2.5-7B-Instruct、10k 医疗样本、Q/V+FFN LoRA、`alpha=16`、`dropout=0.05`、500 条保留集验证。已有的 10k、`r=8`、`lr=2e-5`、1 epoch、seed 42 是共同基线。

| 因素 | 对照 | 新运行 |
|---|---|---|
| LoRA 秩 | r=8 | r=4、r=16 |
| 学习率 | 2e-5 | 1e-5、5e-5 |
| epoch | 1 | 2、3 |
| 随机种子 | 42 | 17、73 |
| GRPO 最大生成长度 | 256（已有） | 384、512 |
| GRPO 奖励组合 | 四个独立奖励直接相加（已有） | 384 token 下 0.35/0.30/0.25/-0.10 加权组合 |

这不是全排列网格：全排列会把参数交互与随机误差混在一起，并将成本扩大到数十次训练。上述单因素矩阵覆盖每个待验证因素，且每组都有明确定义的对照。

执行顺序：

1. `bash scripts/medical_project/run_11_sft_supplementary_ablations.sh all`
2. `bash scripts/medical_project/run_12_grpo_supplementary_ablations.sh all`
3. 对所有已完成 adapter 运行统一 500 条保留集评测，再更新报告。

默认单 GPU 的 SFT 新训练共 8 组，估计约 10–12 GPU 小时；GRPO 新训练共 3 组，因生成上限变长预计约 7–10 GPU 小时。所有输出、日志和时间元数据使用唯一名称，已完成的 adapter 会自动跳过。
