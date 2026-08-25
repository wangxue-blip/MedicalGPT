# MedicalGPT 三卡数据并行训练

本项目使用单机三卡 DDP，而不是把 Qwen2.5-7B 的层切分到三张卡：
每个 `torchrun` worker 在一张 GPU 上保存一份完整基座模型和 LoRA，
每个 worker 读取不同的数据分片，反向传播后通过 NCCL 同步 LoRA 梯度。

## 环境与 DDP 检查

训练环境固定为：

```bash
export PYTHON_BIN=/home/zzc/miniconda3/envs/lora/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2
bash scripts/medical_project/run_00_env_check.sh
bash scripts/medical_project/run_00_ddp_check_3gpu.sh
```

只有日志出现三个不同的 `local_rank`，且最后出现 `DDP_CHECK_PASSED`，
才进入正式训练。

如果只获得两张卡，只需让可见卡数量与 `NPROC_PER_NODE` 一致。例如：

```bash
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
  bash scripts/medical_project/run_00_ddp_check_3gpu.sh
```

## SFT

```bash
bash scripts/medical_project/run_04_sft_ablation_3gpu.sh 1k
bash scripts/medical_project/run_04_sft_ablation_3gpu.sh 10k
bash scripts/medical_project/run_04_sft_ablation_3gpu.sh 30k
```

双卡示例：

```bash
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
  bash scripts/medical_project/run_04_sft_ablation_3gpu.sh 1k
```

默认参数为 LoRA rank 8、alpha 16、dropout 0.05，目标层为
`q_proj,v_proj`，FP16，最大长度 1024，训练 1 epoch。每卡 batch size
为 1，梯度累积为 8，因此三卡全局有效 batch size 为 24、双卡为 16。

## GRPO

SFT 10k 完成后运行：

```bash
bash scripts/medical_project/run_08_grpo_ablation_3gpu.sh 500
bash scripts/medical_project/run_08_grpo_ablation_3gpu.sh 1000
```

GRPO 默认从三卡 SFT 10k adapter 继续训练。脚本会在启动前检查 adapter
是否存在，避免错误地从 Base 模型开始。

## 日志和产物

- 训练日志：`outputs/medical_project/logs/`
- SFT adapter：`outputs/medical_project/sft/`
- GRPO adapter：`outputs/medical_project/grpo/`
- 每阶段计时和 GPU 峰值：对应的 `*_timing.json`

正式启动前应使用 `nvidia-smi` 确认所选三张卡具有足够空闲显存，且没有
其他用户的高利用率任务。
