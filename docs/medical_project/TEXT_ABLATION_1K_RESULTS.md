# 1k 文本 LoRA 消融结果

本轮仅改变 LoRA 注入模块和秩；其余条件固定：Qwen2.5-7B-Instruct、同一 1,000 条医疗 SFT 训练集、1 epoch、seed 42、learning rate `2e-5`、`alpha=2r`、dropout `0.05`、fp16、梯度累积 8。

PPL 使用独立的 500 条医疗长文本集，问答参考相似度和规则安全检查使用独立的 100 条医疗问答集。参考相似度与规则指标不能替代医学正确性或临床安全性人工评测。

| 运行 | LoRA target | rank | PPL ↓ | Eval loss ↓ | Mean ref. sim ↑ | Safety pass | High-risk expr. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_1k_qv` | Q/V | 8 | 20.6077 | 3.0257 | 0.0504 | 100% | 0% |
| `text_1k_qvffn_r4` | Q/V + FFN | 4 | 19.5886 | 2.9749 | 0.0545 | 100% | 0% |
| `text_1k_qvffn_r8` | Q/V + FFN | 8 | 18.8985 | 2.9391 | 0.0568 | 100% | 0% |
| `text_1k_qvffn_r16` | Q/V + FFN | 16 | **18.5610** | **2.9211** | **0.0619** | 100% | 0% |

结论：在这一固定小样本设置中，加入 FFN 投影层带来一致的 PPL 改善；`r=16` 是本轮最优配置，相比只训练 Q/V 的 PPL 下降 9.93%。这只支持在该数据与单 epoch 设定下选择 `Q/V + FFN, r=16` 作为下一轮文本候选，仍需多 seed 和更大训练集验证。

机器可读汇总位于 `outputs/medical_project/text_ablation_1k_eval/summary_metrics.{json,csv}`；预测样本与逐模型 PPL/规则评测文件保存在同一目录。
