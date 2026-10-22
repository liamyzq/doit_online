"""Run one committed image experiment config and save auditable artifacts."""

import argparse
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path
from importlib.metadata import version

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
                "started_at": started, "status": "running",
                "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                                "packages": {name: version(name) for name in
                                             ["numpy", "diffusers", "transformers", "torchvision",
                                              "huggingface_hub", "safetensors", "h5py"]}}}
    write_json(output / "run.json", metadata)
    engine = None
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
        torch.cuda.reset_peak_memory_stats()
        mode = config.get("mode", "image")
        if mode == "uncertainty_image":
            from .uncertainty_image import evaluate
            from .image import ImageEngine
            engine = ImageEngine(config["image"])
            metadata["models"] = engine.provenance
            write_json(output / "run.json", metadata)
            metrics = evaluate(config, output, engine)
        elif mode == "uncertainty_terminal_calibration":
            from .uncertainty_critic import calibrate_terminal
            metrics = calibrate_terminal(config, output)
        elif mode == "uncertainty_critic":
            from .uncertainty_critic import evaluate
            metrics = evaluate(config, output)
        elif mode == "uncertainty_critic_calibration":
            from .uncertainty_critic import calibrate
            metrics = calibrate(config, output)
        elif mode in {"uncertainty_checks", "uncertainty_toy", "uncertainty_local"}:
            from .uncertainty_toy import run_checks, run_toys, run_local
            metrics = {"uncertainty_checks": run_checks, "uncertainty_toy": run_toys,
                       "uncertainty_local": run_local}[mode](config, output)
        elif mode == "checks":
            from .checks import run_checks
            metrics = run_checks(output)
        elif mode == "extension_checks":
            from .extension_checks import run_extension_checks
            metrics = run_extension_checks(output)
        elif mode == "fidelity_checks":
            from .extension_checks import run_fidelity_checks
            metrics = run_fidelity_checks(output)
        elif mode == "paired_fit_checks":
            from .extension_checks import run_paired_fit_checks
            metrics = run_paired_fit_checks(output)
        elif mode == "execution_checks":
            from .extension_checks import run_execution_checks
            metrics = run_execution_checks(output)
        elif mode == "analytic_mechanism":
            from .audit import analytic_mechanism
            metrics = analytic_mechanism(config, output)
        elif mode == "historical_diagnostics":
            from .audit import historical_diagnostics
            metrics = historical_diagnostics(config, output)
        elif mode == "critic_calibration":
            from .critic import calibrate
            metrics = calibrate(config, output)
        elif mode == "critic_fixed":
            from .critic import evaluate_fixed
            metrics = evaluate_fixed(config, output)
        elif mode == "kernel_probe":
            from .kernel_probe import run_probe
            metrics = run_probe(config, output)
        elif mode == "sensitivity_controlled":
            from .reliability import controlled
            metrics = controlled(config, output)
        elif mode == "sensitivity_probe":
            from .reliability import image_probe
            from .image import ImageEngine
            engine = ImageEngine(config["image"])
            metrics = image_probe(config, output, engine)
        elif mode == "image":
            from .image import ImageEngine
            engine = ImageEngine(config.get("image", {}))
            metadata["models"] = engine.provenance
            write_json(output / "run.json", metadata)
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
        if engine is not None:
            metadata.update(reward_queries=engine.reward_queries, score_forwards=engine.score_forwards,
                            vjp_calls=engine.vjp_calls, reward_seconds=engine.reward_seconds,
                            vjp_seconds=engine.vjp_seconds, sensitivity_seconds=engine.sensitivity_seconds)
        write_json(output / "run.json", metadata)
        traceback.print_exc()
        raise
    write_json(output / "run.json", metadata)


if __name__ == "__main__":
    main()
