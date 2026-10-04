#!/usr/bin/env python3
"""Run a finite queue on authorized GPUs 0 through 4 using the machine env."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


STORAGE = Path(os.environ.get("DOIT_STORAGE", "/home/mlw0719/doit_correction_storage"))
REPO = Path(os.environ.get("DOIT_REPO", "/home/mlw0719/doit_correction"))
PYTHON = REPO / ".venv/bin/python"
STARTUP_GRACE_SECONDS = 45


def now():
    return datetime.now(timezone.utc).isoformat()


def git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def refuse_dirty_code():
    lines = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).splitlines()
    relevant = []
    for line in lines:
        path = line[3:]
        if path.startswith(("scripts/", "configs/")) or path.endswith(
            (".py", ".yaml", ".yml", ".json", ".toml", ".sh")
        ):
            relevant.append(path)
    if relevant:
        raise RuntimeError("Refusing formal runs with uncommitted code/config changes: " + ", ".join(relevant))


def gpu_free_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    free = {}
    for row in output.splitlines():
        index, mib = (part.strip() for part in row.split(",", 1))
        index = int(index)
        if index in range(5):
            free[index] = int(mib)
    if set(free) != set(range(5)):
        raise RuntimeError("nvidia-smi did not report all GPUs 0 through 4")
    return free


def append_event(path, event):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")


def process_status(proc):
    code = proc.poll()
    if code is None:
        return "running"
    return "completed" if code == 0 else "failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--memory-mib", required=True, type=int)
    parser.add_argument("--reserve-mib", type=int, default=2048)
    parser.add_argument("--max-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.memory_mib <= 0 or args.reserve_mib < 0 or args.max_per_gpu < 1 or args.poll_seconds < 1:
        parser.error("memory must be positive; reserve nonnegative; concurrency and polling positive")

    with args.queue.open(encoding="utf-8") as stream:
        items = json.load(stream)
    if not isinstance(items, list):
        parser.error("queue must be a JSON list")

    refuse_dirty_code()
    commit = git_commit()
    jobs_log = STORAGE / "jobs/jobs.jsonl"
    artifacts = STORAGE / "artifacts"
    active = []
    pending = []
    seen_ids = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("config"), str):
            parser.error("each item must contain string id and config fields")
        if item["id"] in seen_ids:
            parser.error(f"duplicate experiment id: {item['id']}")
        seen_ids.add(item["id"])
        output_dir = STORAGE / "runs" / item["id"]
        if output_dir.exists():
            parser.error(f"refusing existing output directory: {output_dir}")
        extra = item.get("args", [])
        if not isinstance(extra, list) or not all(isinstance(arg, str) for arg in extra):
            parser.error(f"args for {item['id']} must be a list of strings")
        pending.append((item, output_dir, extra))

    running_events = set()
    failed_count = 0
    while pending or active:
        for job in active[:]:
            status = process_status(job["proc"])
            if status != job["status"]:
                job["status"] = status
                event = dict(job["event"], status=status, timestamp=now())
                if status in ("completed", "failed"):
                    event["returncode"] = job["proc"].returncode
                    event["finished_at"] = now()
                    job["metadata"].write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
                    job["log"].close()
                    if status == "failed":
                        failed_count += 1
                    active.remove(job)
                append_event(jobs_log, event)

        if pending:
            free = gpu_free_memory()
            own_counts = {gpu: 0 for gpu in range(5)}
            reservations = {gpu: 0 for gpu in range(5)}
            for job in active:
                own_counts[job["gpu"]] += 1
                if time.time() - job["launched"] < STARTUP_GRACE_SECONDS:
                    reservations[job["gpu"]] += args.memory_mib
            next_pending = []
            for item, output_dir, extra in pending:
                available = [
                    gpu for gpu in range(5)
                    if own_counts[gpu] < args.max_per_gpu
                    and free[gpu] - reservations[gpu] - args.reserve_mib >= args.memory_mib
                ]
                if not available:
                    next_pending.append((item, output_dir, extra))
                    continue
                gpu = max(available, key=lambda index: free[index] - reservations[index])
                output_dir.mkdir(parents=True)
                log_path = artifacts / f"{item['id']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                config = Path(item["config"])
                if not config.is_absolute():
                    config = REPO / config
                command = [
                    str(PYTHON), "-m", "doit_correction.run", "--config", str(config),
                    "--output", str(output_dir), *extra,
                ]
                log = log_path.open("w", encoding="utf-8")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME=str(STORAGE / "huggingface"),
                           OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
                event = {
                    "experiment_id": item["id"], "commit": commit, "config": str(config),
                    "machine": os.environ.get("DOIT_MACHINE", "orion"),
                    "gpu": gpu, "pid": proc.pid, "path": str(output_dir),
                    "log_path": str(log_path), "status": "submitted", "timestamp": now(),
                }
                metadata = output_dir / "terminal.json"
                metadata.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
                append_event(jobs_log, event)
                active.append({"proc": proc, "gpu": gpu, "launched": time.time(), "event": event,
                               "status": "submitted", "metadata": metadata, "log": log})
                running = dict(event, status="running", timestamp=now(), started_at=now())
                append_event(jobs_log, running)
                active[-1]["event"] = running
                active[-1]["status"] = "running"
                own_counts[gpu] += 1
                reservations[gpu] += args.memory_mib
            pending = next_pending
        if pending or active:
            time.sleep(args.poll_seconds)

    print(f"Finished {len(items)} queued experiments. Logs: {artifacts}; events: {jobs_log}")
    if failed_count:
        print(f"{failed_count} experiment(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"queue: {exc}", file=sys.stderr)
        sys.exit(1)
