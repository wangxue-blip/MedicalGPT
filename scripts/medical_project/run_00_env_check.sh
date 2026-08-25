#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: bash scripts/medical_project/run_00_env_check.sh

Print Python, torch, CUDA, GPU, and key dependency versions for the
MedicalGPT medical SFT + GRPO project.
EOF
  exit 0
fi

: "${PYTHON_BIN:=python}"

"${PYTHON_BIN}" - <<'PY'
import importlib
import json
import platform
from pathlib import Path


def version_of(package):
    try:
        module = importlib.import_module(package)
    except Exception as exc:
        return {"installed": False, "version": None, "error": str(exc)}
    return {"installed": True, "version": getattr(module, "__version__", "unknown"), "error": None}


stats = {
    "python": platform.python_version(),
    "python_executable": platform.python_implementation(),
}

try:
    import torch

    stats["torch"] = torch.__version__
    stats["cuda_available"] = torch.cuda.is_available()
    stats["cuda_version"] = getattr(torch.version, "cuda", None)
    stats["gpu_count"] = torch.cuda.device_count()
    stats["gpus"] = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        stats["gpus"].append(
            {
                "index": idx,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
            }
        )
except Exception as exc:
    stats["torch"] = None
    stats["cuda_available"] = False
    stats["cuda_error"] = str(exc)
    stats["gpu_count"] = 0
    stats["gpus"] = []

for package in [
    "transformers",
    "peft",
    "trl",
    "datasets",
    "sentence_transformers",
    "faiss",
    "bitsandbytes",
]:
    stats[package] = version_of(package)

print(json.dumps(stats, ensure_ascii=False, indent=2))
Path("outputs/medical_project/logs").mkdir(parents=True, exist_ok=True)
Path("outputs/medical_project/logs/env_check.json").write_text(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
