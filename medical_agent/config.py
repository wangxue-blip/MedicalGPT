# -*- coding: utf-8 -*-
"""medical_agent 的默认配置。所有项都可用环境变量 MEDAGENT_* 覆盖。

把路径与阈值集中在这里，是为了让"哪个模型、什么阈值"在日志和评测输出里
可以整块打印，而不是散落在各个工具的默认参数中。
"""

import os


def _env(name: str, default):
    value = os.environ.get("MEDAGENT_" + name)
    if value is None:
        return default
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "y")
    if isinstance(default, int):
        try:
            return int(value)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except ValueError:
            return default
    return value


# 运行模式
MOCK = _env("MOCK", False)                 # True 时所有工具走确定性假后端
DEVICE = _env("DEVICE", "cpu")             # cpu / cuda / cuda:0

# 模型路径（相对仓库根目录）
VLM_MODEL = _env("VLM_MODEL", "models/Qwen2.5-VL-7B-Instruct")
TEXT_MODEL = _env("TEXT_MODEL", "models/Qwen2.5-7B-Instruct")
TEXT_PEFT = _env("TEXT_PEFT", "")          # 你的医疗 LoRA/GRPO adapter 目录，空则不挂
XRV_WEIGHTS = _env("XRV_WEIGHTS", "densenet121-res224-all")   # torchxrayvision 权重名

# RAG
RAG_CHUNKS = _env("RAG_CHUNKS", "data_processed/medical_project/rag/chunks.jsonl")
RAG_DENSE_MODEL = _env("RAG_DENSE_MODEL", "")
RAG_TOPK = _env("RAG_TOPK", 5)

# 检测 / 定位
DETECT_TOPK = _env("DETECT_TOPK", 5)
DETECT_MIN_SCORE = _env("DETECT_MIN_SCORE", 0.30)   # 低于此分的病理不作为 finding
CAM_THRESHOLD = _env("CAM_THRESHOLD", 0.50)         # Grad-CAM 归一化后二值化阈值
MIN_BOX_AREA = _env("MIN_BOX_AREA", 0.002)          # 归一化面积下限，过滤噪点框
MAX_MASK_AREA = _env("MAX_MASK_AREA", 0.60)         # 掩码占比上限，超过视为退化（整图高亮）
IMAGE_SIZE = _env("IMAGE_SIZE", 224)

# VQA
VQA_MAX_NEW_TOKENS = _env("VQA_MAX_NEW_TOKENS", 256)
VQA_MAX_PIXELS = _env("VQA_MAX_PIXELS", 640 * 640)

# reporter
REPORT_MAX_NEW_TOKENS = _env("REPORT_MAX_NEW_TOKENS", 512)

# 输出
OUTPUT_DIR = _env("OUTPUT_DIR", "outputs/medical_project/agent")
ROUTER_LOG = _env("ROUTER_LOG", "outputs/medical_project/agent/router_decisions.jsonl")

# 安全
MAX_GRAPH_STEPS = _env("MAX_GRAPH_STEPS", 32)


def snapshot() -> dict:
    """当前生效配置的快照，写进每次运行的输出便于复现。"""
    return {
        key: value for key, value in globals().items()
        if key.isupper() and not key.startswith("_")
    }
