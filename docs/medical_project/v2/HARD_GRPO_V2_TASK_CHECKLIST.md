# 后续改进任务执行清单 v2：MedicalGPT Hard-Sample GRPO 改造

## 改造目标

在当前已完成的 SFT + GRPO 基础上，新增一条 A100 80GB 适配的 hard-sample GRPO 改进流程：

1. 构造 1500 条高价值 GRPO prompt。
2. 每个 prompt 训练时生成 4 个 completion。
3. 使用更合理的医学 reward 权重。
4. 用 Base / SFT / GRPO 的推理结果挖掘低分、争议、安全风险样本。
5. 增加关键词覆盖率、格式合规率、安全违规率、自动偏好胜率等评测。
6. 保持 C-Eval test 不参与任何训练、筛选、调参。

## 基本原则

1. 不修改覆盖现有代码和脚本。
2. 新增 v2/hard/a100 相关脚本与输出目录。
3. 所有 v2 输出写入 `data_processed/medical_project/v2` 或 `outputs/medical_project/v2`。
4. 跑模型训练/推理的命令单独列出，方便手动执行。
5. `reference similarity`、`keyword coverage`、自动偏好分数都不能表述为真实医学准确率。

---

## 阶段 V2-0：新增目录与命名规范

### 目标

建立 v2 改造空间，避免覆盖现有代码和产物。

### 需要新增目录

```bash
mkdir -p tools/medical_project/v2
mkdir -p eval/medical_project/v2
mkdir -p scripts/medical_project/v2
mkdir -p configs/medical_project/v2
mkdir -p data_processed/medical_project/v2/{pool,mining,grpo,eval}
mkdir -p outputs/medical_project/v2/{mining,grpo,eval,reports,logs}
mkdir -p docs/medical_project/v2
```

### 建议新增文件

```text
tools/medical_project/v2/build_hard_prompt_pool.py
tools/medical_project/v2/generate_mining_predictions.py
tools/medical_project/v2/score_mining_predictions.py
tools/medical_project/v2/build_hard_grpo_data.py
tools/medical_project/v2/rewrite_ceval_medical_prompts.py
tools/medical_project/v2/build_safety_risk_prompts.py
tools/medical_project/v2/data_quality_check_v2.py

training/medical_grpo_rewards_v2.py
training/grpo_training_medical_v2.py

eval/medical_project/v2/eval_keyword_coverage.py
eval/medical_project/v2/eval_pairwise_preference.py
eval/medical_project/v2/summarize_metrics_v2.py

scripts/medical_project/v2/run_01_build_prompt_pool.sh
scripts/medical_project/v2/run_02_generate_mining_predictions.sh
scripts/medical_project/v2/run_03_score_and_select_hard_prompts.sh
scripts/medical_project/v2/run_04_build_grpo_1500.sh
scripts/medical_project/v2/run_05_grpo_train_1500_a100.sh
scripts/medical_project/v2/run_06_eval_all_v2.sh
```

### 验收标准

```text
所有新增脚本支持 --help
不修改现有 run_08_grpo_train.sh、run_09_eval_all.sh、medical_grpo_rewards.py
所有 v2 输出写入 data_processed/medical_project/v2 或 outputs/medical_project/v2
```

---

## 阶段 V2-1：构建候选 Prompt 池

### 目标

构建一个大于 1500 条的候选池，用于后续 hard mining。

### 数据来源

```text
1. data_processed/medical_project/sft/medical_sft_top10000.jsonl
2. data_processed/medical_project/sft/medical_sft_top30000.jsonl
3. data_raw/medical_project/ceval_dev/*.csv
4. 手工/规则生成的安全风险类医疗咨询 prompt
```

注意：如果用 `ceval_dev` 做训练数据改写，则最终正式 C-Eval 评测应使用未参与构造的 `ceval_val`，避免开发集泄漏。

### 输出

```text
data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl
data_processed/medical_project/v2/pool/prompt_pool_summary.json
```

### 样本格式

```json
{
  "id": "pool_000001",
  "question": "...",
  "answer": "...",
  "category": "药物禁忌",
  "source_type": "medical_high_sim|ceval_rewrite|safety_risk",
  "source_id": "...",
  "selection_tags": ["candidate_pool"],
  "required_sections": ["分析", "处理建议", "风险提示", "就医建议"],
  "safety_rules": ["不提供具体处方剂量", "不替代医生面诊"]
}
```

### 建议命令

```bash
python tools/medical_project/v2/build_hard_prompt_pool.py \
  --medical_sft data_processed/medical_project/sft/medical_sft_top30000.jsonl \
  --ceval_dev_dir data_raw/medical_project/ceval_dev \
  --output data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl \
  --summary_output data_processed/medical_project/v2/pool/prompt_pool_summary.json \
  --medical_samples 3000 \
  --ceval_rewrite_samples 1200 \
  --safety_risk_samples 800 \
  --seed 42
```

### 验收标准

```text
候选池不少于 4000 条
source_type 至少包含 medical_high_sim、ceval_rewrite、safety_risk
C-Eval 改写样本不得直接复制原题选项
summary 中记录 ceval_test_used=false
```

---

## 阶段 V2-2：Base/SFT 模型候选回答生成

### 目标

用 Base、SFT10k、SFT30k 对候选 prompt 生成回答，为 hard mining 做准备。

### 输入

```text
data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl
```

### 输出

```text
outputs/medical_project/v2/mining/predictions_base.jsonl
outputs/medical_project/v2/mining/predictions_sft_10k.jsonl
outputs/medical_project/v2/mining/predictions_sft_30k.jsonl
```

### 建议生成参数

```text
max_new_tokens=512
temperature=0.2
top_p=0.9
```

### Base 生成命令

```bash
CUDA_VISIBLE_DEVICES=0 python tools/medical_project/v2/generate_mining_predictions.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --input data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl \
  --output outputs/medical_project/v2/mining/predictions_base.jsonl \
  --max_samples -1 \
  --max_new_tokens 512 \
  --temperature 0.2 \
  --top_p 0.9 \
  --torch_dtype float16 \
  --device_map auto
```

### SFT10k 生成命令

```bash
CUDA_VISIBLE_DEVICES=0 python tools/medical_project/v2/generate_mining_predictions.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --peft_path outputs/medical_project/sft/qwen25_7b_lora_10k_v100 \
  --input data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl \
  --output outputs/medical_project/v2/mining/predictions_sft_10k.jsonl \
  --max_samples -1 \
  --max_new_tokens 512 \
  --temperature 0.2 \
  --top_p 0.9 \
  --torch_dtype float16 \
  --device_map auto
```

### SFT30k 生成命令

```bash
CUDA_VISIBLE_DEVICES=0 python tools/medical_project/v2/generate_mining_predictions.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --peft_path outputs/medical_project/sft/qwen25_7b_lora_30k_v100 \
  --input data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl \
  --output outputs/medical_project/v2/mining/predictions_sft_30k.jsonl \
  --max_samples -1 \
  --max_new_tokens 512 \
  --temperature 0.2 \
  --top_p 0.9 \
  --torch_dtype float16 \
  --device_map auto
```

### 验收标准

```text
每个模型都生成完整 predictions
每条 prediction 包含 id、question、reference_answer、prediction、source_type、category
生成失败样本有日志记录
```

---

## 阶段 V2-3：候选样本打分与 Hard Mining

### 目标

根据 Base/SFT 回答质量挑选高价值 GRPO prompt。

### 输入

```text
hard_prompt_pool.jsonl
predictions_base.jsonl
predictions_sft_10k.jsonl
predictions_sft_30k.jsonl
```

### 评分信号

```text
reference_similarity
medical_keyword_coverage
format_score
safety_score
risk_penalty
response_length
base_score
sft_score
score_gap
risk_flags
```

### 筛选优先级

```text
1. SFT 低分样本
2. Base/SFT 都低分的困难样本
3. Base 与 SFT 分数差异大的争议样本
4. safety risk 命中样本
5. 多 source_type 均衡覆盖样本
```

### 建议命令

```bash
python tools/medical_project/v2/score_mining_predictions.py \
  --pool data_processed/medical_project/v2/pool/hard_prompt_pool.jsonl \
  --base_predictions outputs/medical_project/v2/mining/predictions_base.jsonl \
  --sft_predictions outputs/medical_project/v2/mining/predictions_sft_10k.jsonl \
  --extra_sft_predictions outputs/medical_project/v2/mining/predictions_sft_30k.jsonl \
  --embedding_model models/bge-m3 \
  --embedding_device cpu \
  --output data_processed/medical_project/v2/mining/scored_prompt_pool.jsonl \
  --summary_output data_processed/medical_project/v2/mining/scored_prompt_pool_summary.json
```

### 验收标准

```text
每条候选样本有 base/sft 分数
每条样本有 selection_score 和 selection_reason
安全风险样本标记 risk_flags
summary 统计各 source_type、category、risk_flags 分布
```

---

## 阶段 V2-4：构造 1500 条 Hard GRPO 数据

### 目标

生成最终 GRPO v2 数据，控制来源配比和难度。

### 建议最终配比

```text
500 条：C-Eval 医学知识点改写问答
600 条：Medical 高质量复杂问答
400 条：安全风险 / 低分 / 争议样本
```

### 输出

```text
data_processed/medical_project/v2/grpo/medical_grpo_hard_1500.jsonl
data_processed/medical_project/v2/grpo/grpo_hard_1500_summary.json
```

### 建议命令

```bash
python tools/medical_project/v2/build_hard_grpo_data.py \
  --scored_pool data_processed/medical_project/v2/mining/scored_prompt_pool.jsonl \
  --output data_processed/medical_project/v2/grpo/medical_grpo_hard_1500.jsonl \
  --summary_output data_processed/medical_project/v2/grpo/grpo_hard_1500_summary.json \
  --num_samples 1500 \
  --ceval_rewrite_samples 500 \
  --medical_high_sim_samples 600 \
  --safety_risk_samples 400 \
  --min_reward_gap 0.10 \
  --seed 42
```

### 验收标准

```text
最终样本数 1500
每条包含 question、answer、category、source_type、required_sections、safety_rules
每条包含 selection_reason
summary 明确 ceval_test_used=false
```

---

## 阶段 V2-5：Reward v2 实现

### 目标

新增医学 GRPO reward v2，不覆盖当前 reward。

### 新增文件

```text
training/medical_grpo_rewards_v2.py
```

### 建议 reward 公式

```text
R_total =
0.30 * reference_similarity
+ 0.25 * medical_keyword_coverage
+ 0.15 * format_reward
+ 0.20 * safety_reward
- 0.20 * risk_penalty
```

### 建议实现函数

```python
def reference_similarity_reward_v2(...)
def medical_keyword_coverage_reward(...)
def medical_format_reward_v2(...)
def medical_safety_reward_v2(...)
def medical_risk_penalty(...)
def combined_medical_reward_v2(...)
```

### 注意点

```text
safety_reward 奖励正向安全表达
risk_penalty 只惩罚硬风险：具体剂量、自行停药/加药、无需就医、绝对诊断
keyword_coverage 应按 category 动态匹配，不做简单关键词堆叠
```

### 验收命令

```bash
python training/medical_grpo_rewards_v2.py --demo
```

如环境有 pytest：

```bash
python -m pytest tests/test_medical_grpo_rewards_v2.py -q
```

### 验收标准

```text
安全高质量回答分数高
具体剂量/无需就医/自行停药样本 risk_penalty 高
缺少风险提示或就医建议时 format/coverage 分下降
reference_similarity 明确不称为医学准确率
```

---

## 阶段 V2-6：新增 GRPO v2 训练入口

### 目标

新增专用训练入口，不修改现有 `training/grpo_training.py`。

### 新增文件

```text
training/grpo_training_medical_v2.py
```

### 实现要求

```text
基于现有 grpo_training.py 逻辑复制并改造
新增 reward_type=medical_v2
接入 combined_medical_reward_v2 或加权 reward wrapper
支持 train_file
支持 peft_path
支持 num_generations=4
支持 max_completion_length
保持 LoRA 训练
```

### 建议新增脚本

```text
scripts/medical_project/v2/run_05_grpo_train_1500_a100.sh
```

### A100 80GB 推荐主实验参数

```text
model: models/Qwen2.5-7B-Instruct
peft_path: outputs/medical_project/sft/qwen25_7b_lora_10k_v100 或 qwen25_7b_lora_30k_v100
train_samples: 1500
num_generations: 4
max_completion_length: 512
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
num_train_epochs: 1
learning_rate: 3e-7 到 5e-7
beta: 0.001
lora_r: 8
lora_alpha: 16
```

### SFT10k 基础模型训练命令

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 training/grpo_training_medical_v2.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --peft_path outputs/medical_project/sft/qwen25_7b_lora_10k_v100 \
  --train_file data_processed/medical_project/v2/grpo/medical_grpo_hard_1500.jsonl \
  --train_samples 1500 \
  --num_train_epochs 1 \
  --output_dir outputs/medical_project/v2/grpo/qwen25_7b_lora_10k_grpo_hard1500_g4_a100 \
  --dtype float16 \
  --fp16 True \
  --bf16 False \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.001 \
  --learning_rate 5.0e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --use_vllm False \
  --logging_steps 5 \
  --eval_strategy no \
  --save_steps 100 \
  --save_strategy steps \
  --save_total_limit 3 \
  --use_peft True \
  --qlora False \
  --load_in_4bit False \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --num_generations 4 \
  --gradient_accumulation_steps 2 \
  --max_completion_length 512 \
  --preprocessing_num_workers 4 \
  --reward_type medical_v2 \
  --similarity_model models/bge-m3 \
  --similarity_device cpu
```

### SFT30k 基础模型训练命令

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 training/grpo_training_medical_v2.py \
  --model_name_or_path models/Qwen2.5-7B-Instruct \
  --peft_path outputs/medical_project/sft/qwen25_7b_lora_30k_v100 \
  --train_file data_processed/medical_project/v2/grpo/medical_grpo_hard_1500.jsonl \
  --train_samples 1500 \
  --num_train_epochs 1 \
  --output_dir outputs/medical_project/v2/grpo/qwen25_7b_lora_30k_grpo_hard1500_g4_a100 \
  --dtype float16 \
  --fp16 True \
  --bf16 False \
  --report_to tensorboard \
  --remove_unused_columns False \
  --gradient_checkpointing False \
  --beta 0.001 \
  --learning_rate 5.0e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --use_vllm False \
  --logging_steps 5 \
  --eval_strategy no \
  --save_steps 100 \
  --save_strategy steps \
  --save_total_limit 3 \
  --use_peft True \
  --qlora False \
  --load_in_4bit False \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --per_device_train_batch_size 1 \
  --num_generations 4 \
  --gradient_accumulation_steps 2 \
  --max_completion_length 512 \
  --preprocessing_num_workers 4 \
  --reward_type medical_v2 \
  --similarity_model models/bge-m3 \
  --similarity_device cpu
```

### 640 长度消融

如果 A100 80GB 运行稳定，可做 `640` 长度消融：

```bash
# 只改这个参数和 output_dir
--max_completion_length 640
```

### 验收标准

```text
训练完成 1500 samples / 1 epoch
reward 不全为 0
TensorBoard 中能看到 reference_similarity、keyword_coverage、format、safety、risk_penalty
保存 adapter_model.safetensors、trainer_state.json、train_results.json
```

---

## 阶段 V2-7：训练前后 Case 生成

### 目标

生成 SFT vs GRPO v2 的同 prompt 对比样例。

### 输出

```text
outputs/medical_project/v2/eval/grpo_hard1500_before_after_cases.jsonl
```

### 建议命令

```bash
CUDA_VISIBLE_DEVICES=0 python eval/medical_project/generate_grpo_before_after.py \
  --base_model models/Qwen2.5-7B-Instruct \
  --before_peft_path outputs/medical_project/sft/qwen25_7b_lora_10k_v100 \
  --after_peft_path outputs/medical_project/v2/grpo/qwen25_7b_lora_10k_grpo_hard1500_g4_a100 \
  --input_file data_processed/medical_project/v2/grpo/medical_grpo_hard_1500.jsonl \
  --output outputs/medical_project/v2/eval/grpo_hard1500_before_after_cases.jsonl \
  --num_samples 20 \
  --max_new_tokens 512 \
  --torch_dtype float16 \
  --device_map auto
```

### 验收标准

```text
至少 20 条 case
包含 question、reference_answer、before_response、after_response、category、source_type
样例中能观察结构化、安全提示、风险表达差异
```

---

## 阶段 V2-8：新增评测指标

### 目标

在现有评测基础上新增关键词覆盖、格式合规、安全违规、自动偏好胜率。

### 新增脚本

```text
eval/medical_project/v2/eval_keyword_coverage.py
eval/medical_project/v2/eval_pairwise_preference.py
eval/medical_project/v2/summarize_metrics_v2.py
```

### 评测对象

```text
Base
SFT10k
SFT30k
旧 GRPO10k500
旧 GRPO10k1000
新 GRPO hard1500 from SFT10k
新 GRPO hard1500 from SFT30k
```

### 正式评测命令

```bash
BASE_MODEL=models/Qwen2.5-7B-Instruct \
EMBEDDING_MODEL=models/bge-m3 \
EMBEDDING_DEVICE=cpu \
TORCH_DTYPE=float16 \
DEVICE_MAP=auto \
MAX_PPL_SAMPLES=1000 \
MAX_QA_SAMPLES=500 \
MAX_CEVAL_SAMPLES=-1 \
CEVAL_DEV_DIR=data_raw/medical_project/ceval_val \
EVAL_OUTPUT_DIR=outputs/medical_project/v2/eval \
bash scripts/medical_project/v2/run_06_eval_all_v2.sh
```

### 注意

```text
如果 GRPO 训练数据使用了 ceval_dev 改写，正式 C-Eval 评测应使用 ceval_val
不要使用 C-Eval test
```

### Pairwise preference 建议命令

```bash
python eval/medical_project/v2/eval_pairwise_preference.py \
  --baseline_predictions outputs/medical_project/v2/eval/predictions_sft_10k_v100.jsonl \
  --candidate_predictions outputs/medical_project/v2/eval/predictions_grpo_hard1500_sft10k.jsonl \
  --output outputs/medical_project/v2/eval/preference_sft10k_vs_grpo_hard1500.json \
  --embedding_model models/bge-m3 \
  --embedding_device cpu \
  --tie_threshold 0.03
```

### 验收标准

```text
summary_metrics_v2.csv 包含：
C-Eval accuracy
QA reference similarity
keyword_coverage
format_compliance_rate
safety_violation_rate
risk_penalty_rate
pairwise_win_rate
PPL/eval_loss
avg_response_length
```

---

## 阶段 V2-9：结果分析与报告补充

### 目标

形成 v2 改造报告材料。

### 建议新增

```text
docs/medical_project/v2/HARD_GRPO_EXPERIMENT_NOTES.md
docs/medical_project/v2/HARD_GRPO_REPORT.md
```

### 报告必须说明

```text
C-Eval test 未使用
C-Eval dev 若用于训练改写，则不作为最终无污染评测
reference similarity 不等价于医学准确率
keyword coverage 不等价于医学准确率
自动偏好胜率基于启发式 reward，不等价人工医生评价
本项目不是医疗诊断系统
```

### 建议分析维度

```text
Base -> SFT：PPL / QA similarity / 安全表达变化
SFT -> GRPO：格式合规、关键词覆盖、安全违规率、偏好胜率变化
旧 GRPO -> hard GRPO：hard mining 是否带来提升
512 vs 640 completion length：结构化和安全表达是否改善
不同 source_type：ceval_rewrite / medical_high_sim / safety_risk 上的收益差异
```

---

## 推荐执行顺序

```text
V2-0 新增目录和脚本骨架
V2-1 构建候选 prompt pool
V2-2 生成 Base/SFT mining predictions
V2-3 打分并筛选 hard prompts
V2-4 构造 hard GRPO 1500 数据
V2-5 实现 reward v2
V2-6 A100 80GB 跑 hard GRPO 训练
V2-7 生成 before/after cases
V2-8 跑正式评测
V2-9 写报告和实验分析
```

## 最重要的参数建议

```text
A100 80GB 主实验：
num_generations=4
max_completion_length=512
train_samples=1500
num_train_epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=2
learning_rate=5e-7
beta=0.001
```

`640` 建议作为第二组消融，不要直接作为主实验默认值。这样可以区分效果提升来自 hard data + reward v2，还是来自允许模型生成更长回答。
