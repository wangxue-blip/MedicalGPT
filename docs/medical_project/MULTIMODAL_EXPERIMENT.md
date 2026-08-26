# Qwen2.5-VL 中文医疗图文实验

首轮实验选用 SLAKE 的中文 VQA 子集，原因是它提供官方训练、验证、测试划分，并在数据卡中标注为 CC BY 4.0。训练使用 2,048 条中文影像问答和 512 条本项目既有文本 SFT 样本；文本样本仅进入训练集，不进入图像评测。验证和测试分别保留 SLAKE 的 1,046 和 1,033 条中文影像问答，避免图像泄漏。

模型为 `Qwen/Qwen2.5-VL-3B-Instruct`，使用 bf16、梯度检查点、单卡 LoRA（`q_proj,v_proj`，r=8）。每图最多 512 个视觉 token，以便在单张 46 GiB A40 上稳定训练。评测同时运行未微调基础模型与 LoRA 适配器；主指标是规范化答案 Exact Match，并按 SLAKE 内容类型分组输出。

运行顺序：

```bash
bash scripts/medical_project/run_14_vl_slake.sh prepare
# 将 Qwen/Qwen2.5-VL-3B-Instruct 下载到 models/Qwen2.5-VL-3B-Instruct 后：
CUDA_VISIBLE_DEVICES=3 bash scripts/medical_project/run_14_vl_slake.sh all
```

数据来源的二期候选包括 ChiMed-VL（中文、规模更大）和 PubMedVision（含中文版本）。前者的公开页面未给出可核验许可，因此不自动使用；后者为 Apache-2.0 但约 59.5 GB，适合作为完成 SLAKE 首轮验证后的扩展训练语料。

## 已完成：首轮运行（2026-08-26）

本次使用 `models/Qwen2.5-VL-3B-Instruct-ms`（官方 ModelScope 镜像）在 GPU 3 上执行完毕。适配器、训练指标和基座/LoRA 的逐样本预测均保存在 `outputs/medical_project/` 下，未覆盖原有文本实验。

| 项目 | 值 |
| --- | ---: |
| 训练集 | 2,048 条 SLAKE 中文图文问答 + 512 条 MedicalGPT 纯文本 SFT |
| 验证集 | 256 条未参与训练的 SLAKE 中文图文问答 |
| 测试集 | 1,033 条未参与训练的 SLAKE 中文图文问答 |
| 配置 | Q/V LoRA r=8, alpha=16, dropout=0.05；BF16；1 epoch；seed=42 |
| 可训练参数 | 1,843,200（总参数的 0.0491%） |
| 训练耗时 | 927.18 秒 |
| 验证损失 / PPL | 0.5246 / 1.6899 |

### 测试结果

两个模型使用相同图片、中文提示词、图像预处理、贪心解码和 32-token 输出上限。主指标为去空白/标点后字符串精确匹配（strict EM），因此它是本实验的一致性指标，不是 SLAKE 论文的官方分数，也不是临床准确率。

| 测试口径 | 基座 | LoRA | 变化 |
| --- | ---: | ---: | ---: |
| Strict EM（原始规范化字符串） | 378 / 1,033 = 36.59% | 534 / 1,033 = 51.69% | **+15.10pp** |
| 有限同义映射 EM | 535 / 1,033 = 51.79% | 573 / 1,033 = 55.47% | **+3.68pp** |

有限同义映射仅将以下明确等价表达合并：`是/是的/有/包含/存在`、`否/不是/没有/不包含/无`、以及 `MRI/核磁共振/核磁共振成像/磁共振成像`（少量 CT 同义形式也合并）。它不是官方指标，也并不覆盖所有医学同义词；其目的只是避免把答案格式对齐误写成视觉推理增益。

严格 EM 下，两个模型的逐样本交叉结果为：360 条都正确、481 条都错误、174 条仅 LoRA 正确、18 条仅基座正确。174 条“仅 LoRA 正确”中有 98 条在上述有限同义映射下基座已经语义等价（最常见的是标注为“包含”而基座回答“是”，39 条）。因此，这次 LoRA 同时带来了明显的输出格式对齐，以及较小但仍为正向的有限语义匹配增益。

| 题型 | 基座 strict EM | LoRA strict EM | 变化 |
| --- | ---: | ---: | ---: |
| Organ（n=254） | 28.35% | 64.57% | +36.22pp |
| Abnormality（n=161） | 23.60% | 37.89% | +14.29pp |
| Plane（n=58） | 60.34% | 74.14% | +13.79pp |
| Modality（n=120） | 52.50% | 63.33% | +10.83pp |
| Position（n=189） | 18.52% | 23.81% | +5.29pp |
| Size（n=59） | 84.75% | 81.36% | -3.39pp |

其中 Organ 的严格 EM 大幅上升应特别谨慎解读：该题型含大量“包含/不包含”标注，因而最受中文表述规范化影响。Size 的负向变化样本量较小，不能据此单独下结论。

### 可复现产物与下一步

- LoRA：`outputs/medical_project/vl/qwen25_vl_3b_slake_zh_textmix_lora_r8/adapter_model.safetensors`
- 训练指标：同目录的 `train_results.json`、`eval_results.json` 与 `run_config.json`
- 基座预测/指标：`outputs/medical_project/vl_eval/qwen25_vl_3b_base_slake_zh_{predictions,metrics}.json*`
- LoRA 预测/指标：`outputs/medical_project/vl_eval/qwen25_vl_3b_lora_slake_zh_{predictions,metrics}.json*`
- 严格/有限同义映射对照：`outputs/medical_project/vl_eval/qwen25_vl_3b_slake_zh_comparison.json`；可由 `eval/medical_project/summarize_vl_comparison.py` 从两份预测重新生成。

下一轮应固定该评测脚本，增加人工核查的语义评分与不确定性/拒答测试；再在许可清晰的数据上扩展训练并做数据配比消融（仅 SLAKE 图文、图文+文本、以及不同 LoRA target/rank）。任何结果均不应直接用于诊断、分诊或替代临床判断。
