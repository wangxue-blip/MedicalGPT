#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:=0,1,2}"
: "${NPROC_PER_NODE:=3}"
: "${PYTHON_BIN:=/home/zzc/miniconda3/envs/lora/bin/python}"
: "${TORCHRUN_BIN:=${PYTHON_BIN%/python}/torchrun}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_GPUS[@]}" -ne "${NPROC_PER_NODE}" ]]; then
  echo "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_GPUS[@]} device(s), but NPROC_PER_NODE=${NPROC_PER_NODE}." >&2
  exit 2
fi

RUN_NAME="ddp_check_${NPROC_PER_NODE}gpu"

"${PYTHON_BIN}" tools/medical_project/run_timed_stage.py \
  --stage "${RUN_NAME}" \
  --log "outputs/medical_project/logs/${RUN_NAME}.log" \
  --metadata "outputs/medical_project/logs/${RUN_NAME}_timing.json" \
  -- "${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" \
  tools/medical_project/check_ddp.py --expected-world-size "${NPROC_PER_NODE}"
