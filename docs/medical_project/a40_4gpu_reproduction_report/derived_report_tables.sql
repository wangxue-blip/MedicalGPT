-- Reproducible report snapshot derived from the project's checked-in
-- EXPERIMENT_REPORT_ZH.md and stage scripts. The A40 time rows are engineering
-- estimates, not observed benchmark results.

SELECT * FROM (VALUES
  (1, 'Base', 22.9328, 0.9081, 0.930, 0.325, 0.020, '100%（15题）'),
  (2, 'SFT-10k', 17.5387, 0.9111, 0.985, 0.000, 0.010, '100%（15题）'),
  (3, 'SFT-30k', 17.5520, 0.9096, 0.960, 0.000, 0.035, '100%（15题）'),
  (4, 'GRPO-500（SFT-10k）', 17.5459, 0.9127, 0.985, 0.000, 0.015, '100%（15题）'),
  (5, 'GRPO-1000（SFT-10k）', 17.5525, 0.9095, 0.975, 0.005, 0.020, '100%（15题）')
) AS existing_metrics(stage_order, model, ppl, qa_similarity, safety_pass, structure_hit, high_risk, ceval);

SELECT * FROM (VALUES
  (1, 'SFT-10k', 'V100', 10000, 3990.1, 1.11),
  (2, 'SFT-30k', 'V100', 30000, 13077.1, 3.63),
  (3, 'SFT-10k', 'A100单卡', 10000, 3034.1, 0.84),
  (4, 'SFT-30k', 'A100单卡', 30000, 5636.3, 1.57),
  (5, 'GRPO-500', 'V100', 450, 960.2, 0.27),
  (6, 'GRPO-1000', 'V100', 900, 14890.2, 4.14),
  (7, 'GRPO-1000（SFT-30k）', 'A100单卡', 1000, 9549.3, 2.65)
) AS recorded_times(sort_order, task, hardware, samples, seconds, hours);

SELECT * FROM (VALUES
  (1, '环境、下载与校验', 'CPU+单卡', '1～4小时', 'CUDA/NCCL/FlashAttention可用'),
  (2, '126万QA清洗', 'CPU多进程', '0.5～1.5小时', '样本数、UTF-8、无异常字段'),
  (3, 'BGE-M3筛选', '单卡或四卡分片', '0.8～6小时', 'backend必须为sentence_transformers'),
  (4, 'SFT-1k冒烟', '单卡', '10～30分钟', 'loss稳定、Adapter可加载'),
  (5, 'SFT-10k', '4卡DDP BF16', '0.5～1.2小时', '有效batch保持8'),
  (6, 'SFT-30k', '4卡DDP BF16', '1.2～3小时', '独立验证集、无NaN'),
  (7, 'GRPO冒烟', '1～2卡', '20～60分钟', 'reward变化、显存稳定'),
  (8, 'GRPO-500', '4卡DDP', '0.5～2小时', '无reward hacking'),
  (9, 'GRPO-1000', '4卡DDP', '1.5～5小时', '与500 checkpoint比较'),
  (10, '全部模型评测', '4个单卡任务并行', '1.5～4小时', '各模型四类指标完整'),
  (11, '汇总与报告', 'CPU', '5～20分钟', '无旧JSON混入')
) AS a40_times(sort_order, stage, strategy, estimated_time, critical_check);

SELECT * FROM (VALUES
  (0, '环境检查', '驱动、CUDA、Python环境', 'env_check.json', '4张A40和关键依赖可见'),
  (1, '原始数据标准化', '开源医疗QA', 'medical_candidates.jsonl', '字段完整、中文正常'),
  (2, '目标域query', '医学主题+C-Eval开发题', 'medical_topic_queries.jsonl', '主题覆盖合理且无测试集'),
  (3, '筛选', '候选池+query+BGE', 'Top-1k/10k/30k+评分', 'BGE backend、分布和规则统计正确'),
  (4, '格式转换', 'Top-k JSONL', 'ShareGPT数据', '对话角色、系统提示和行数正确'),
  (5, 'SFT', 'ShareGPT+Qwen Base', 'SFT LoRA Adapter', 'loss稳定、Adapter可推理'),
  (6, 'GRPO数据', 'Top-10k高风险样本', 'GRPO-500/1000', '参考答案、规则和类别齐全'),
  (7, 'GRPO训练', 'SFT Adapter+GRPO数据', 'GRPO Adapter', 'reward合理、无NaN/投机'),
  (8, '独立评测', 'Base/SFT/GRPO+held-out', '逐项JSON和预测', '三阶段口径一致'),
  (9, '汇总报告', '指标JSON', 'CSV/JSON/Markdown报告', '无缺模型、无旧结果混入')
) AS artifact_map(sort_order, stage, main_input, main_output, acceptance_gate);
