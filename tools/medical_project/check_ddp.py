#!/usr/bin/env python
"""Verify one-process-per-GPU DDP/NCCL wiring with a tiny all-reduce."""

import argparse
import json
import os
import socket

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-world-size", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    dist.init_process_group(backend="nccl", init_method="env://")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if world_size != args.expected_world_size:
            raise RuntimeError(
                f"world_size={world_size}, expected {args.expected_world_size}"
            )
        if torch.cuda.device_count() != world_size:
            raise RuntimeError(
                f"visible CUDA devices={torch.cuda.device_count()}, world_size={world_size}"
            )

        torch.cuda.set_device(local_rank)
        value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) / 2
        if value.item() != expected_sum:
            raise RuntimeError(
                f"NCCL all-reduce returned {value.item()}, expected {expected_sum}"
            )

        props = torch.cuda.get_device_properties(local_rank)
        print(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "rank": rank,
                    "local_rank": local_rank,
                    "world_size": world_size,
                    "cuda_device": torch.cuda.current_device(),
                    "gpu_name": props.name,
                    "all_reduce_sum": value.item(),
                    "status": "ok",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        dist.barrier()
        if rank == 0:
            print("DDP_CHECK_PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
