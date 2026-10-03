"""Run one committed image experiment config and save auditable artifacts."""

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import torch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.time()
    metadata = {"commit": commit, "config": args.config, "experiment_id": config["id"],
                "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "started_at": started, "status": "running"}
    write_json(output / "run.json", metadata)
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
        torch.cuda.reset_peak_memory_stats()
        mode = config.get("mode", "image")
        if mode == "checks":
            from .checks import run_checks
            metrics = run_checks(output)
        elif mode == "extension_checks":
            from .extension_checks import run_extension_checks
            metrics = run_extension_checks(output)
        elif mode == "fidelity_checks":
            from .extension_checks import run_fidelity_checks
            metrics = run_fidelity_checks(output)
        elif mode == "image":
            from .image import ImageEngine
            engine = ImageEngine(config.get("image", {}))
            rows = []
            for prompt in config["prompts"]:
                for seed in config["seeds"]:
                    row = engine.generate(prompt, seed, config["estimator"], output / prompt / str(seed))
                    rows.append(row)
                    write_json(output / "images.json", rows)
                    print(json.dumps({k: v for k, v in row.items() if k != "steps"}), flush=True)
            metrics = {"images": len(rows), "mean_reward": sum(r["reward"] for r in rows) / len(rows),
                       "mean_alignment": sum(r["clip_alignment"] for r in rows) / len(rows),
                       "guidance_queries": sum(r["guidance_queries"] for r in rows),
                       "mean_seconds": sum(r["seconds"] for r in rows) / len(rows),
                       "denominator_events": sum(r["denominator_events"] for r in rows)}
        elif mode == "capture":
            from .fixed import capture
            metrics = capture(config, output)
        elif mode == "fixed":
            from .fixed import evaluate
            metrics = evaluate(config, output)
        elif mode == "future_probe":
            from .fixed import probe_future
            metrics = probe_future(config, output)
        else:
            raise ValueError(mode)
        metrics.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        write_json(output / "metrics.json", metrics)
        metadata.update(status="completed", elapsed_seconds=time.time() - started)
        print(json.dumps(metrics), flush=True)
    except Exception as exc:
        metadata.update(status="failed", error=repr(exc), elapsed_seconds=time.time() - started)
        write_json(output / "run.json", metadata)
        traceback.print_exc()
        raise
    write_json(output / "run.json", metadata)


if __name__ == "__main__":
    main()
