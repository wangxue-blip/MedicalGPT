#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one experiment stage while recording wall time and peak GPU usage."""

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def selected_gpu_indexes():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        return None
    values = []
    for item in visible.split(","):
        item = item.strip()
        if item.isdigit():
            values.append(int(item))
    return set(values) if values else None


def gpu_snapshot(selected=None):
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except Exception as exc:
        return {"error": str(exc), "gpus": []}
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index = int(parts[0])
        if selected is not None and index not in selected:
            continue
        rows.append(
            {
                "index": index,
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_percent": int(parts[5]),
            }
        )
    return {"gpus": rows}


def main():
    args = parse_args()
    log_path = Path(args.log)
    metadata_path = Path(args.metadata)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    selected = selected_gpu_indexes()
    peaks = {}
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            snapshot = gpu_snapshot(selected)
            for gpu in snapshot.get("gpus", []):
                index = str(gpu["index"])
                current = peaks.setdefault(index, {"memory_used_mib": 0, "utilization_percent": 0})
                current["memory_used_mib"] = max(current["memory_used_mib"], gpu["memory_used_mib"])
                current["utilization_percent"] = max(
                    current["utilization_percent"], gpu["utilization_percent"]
                )
            stop_event.wait(2)

    started_at = now_iso()
    started_perf = time.perf_counter()
    before = gpu_snapshot(selected)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    header = [
        f"stage={args.stage}",
        f"started_at={started_at}",
        f"cwd={Path.cwd()}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        f"command={shlex.join(args.command)}",
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in header:
            print(line, flush=True)
            log_file.write(line + "\n")
        log_file.flush()
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        exit_code = process.wait()

    stop_event.set()
    monitor_thread.join(timeout=5)
    ended_at = now_iso()
    duration = time.perf_counter() - started_perf
    after = gpu_snapshot(selected)
    metadata = {
        "stage": args.stage,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(duration, 3),
        "exit_code": exit_code,
        "cwd": str(Path.cwd()),
        "command": args.command,
        "command_shell": shlex.join(args.command),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_before": before,
        "gpu_after": after,
        "gpu_peaks": peaks,
        "log_file": str(log_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"stage={args.stage} ended_at={ended_at} duration_seconds={duration:.3f} exit_code={exit_code}",
        flush=True,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
