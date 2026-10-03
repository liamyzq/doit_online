"""Audit-directed mechanism experiments and recovery of existing scalar evidence."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .estimators import MomentEstimator, gaussian_noise


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analytic_mechanism(config, output):
    """Separate exact-surrogate performance from learning with 96 queries.

    F(z)=exp(0.3*z[0]), m=0, rho=1 gives the known ratio 0.3*e_1.
    Every method receives identical fresh paired samples on each online call.
    The oracle surrogate is a mechanism control with externally supplied truth;
    its performance is not a claim about an implementable learned method.
    """
    output = Path(output)
    device = torch.device("cpu")
    repeats = int(config["repeats"])
    rows, summaries = [], []
    slope = float(config.get("slope", 0.3))
    for dimension in config["dimensions"]:
        mean = torch.zeros(dimension, device=device)
        truth = mean.clone()
        truth[0] = slope
        for repeat in range(repeats):
            seed = int(config["seed"]) + repeat * 104729
            generator = torch.Generator(device=device).manual_seed(seed)
            estimators = {}
            for name, settings in config["variants"].items():
                options = dict(samples=8, warmup_samples=32, warmup_steps=1,
                               refresh_every=0, tau=1.0, reward_offset=0.0,
                               antithetic=True, **settings)
                estimator = MomentEstimator(options, seed, device)
                if name == "oracle-exponential":
                    estimator.ensure_surrogate(mean, 1.0)
                    estimator.surrogate.theta[1] = slope
                    estimator.surrogate.fitted = True
                estimators[name] = estimator
            for step in range(9):
                eps = gaussian_noise(32 if step == 0 else 8, dimension, generator, device)
                rewards = slope * eps[:, 0]
                for name, estimator in estimators.items():
                    result = estimator.from_samples(mean, 1.0, eps, rewards)
                    learned = estimator.surrogate
                    direction_error = None
                    if learned is not None:
                        direction_error = float((learned.slope() - truth.double()).square().sum())
                    rows.append({
                        "dimension": dimension, "repeat": repeat, "seed": seed,
                        "online_step": step, "variant": name,
                        "queries": result.diagnostics["queries"],
                        "cumulative_queries": estimator.queries,
                        "ratio_sq_error": float((result.ratio.double() - truth.double()).square().sum()),
                        "zero_ratio_sq_error": slope ** 2,
                        "post_update_slope_sq_error": direction_error,
                        "denominator_floored": result.diagnostics["denominator_floored"],
                    })
        selected = [row for row in rows if row["dimension"] == dimension and row["online_step"] > 0]
        mc_error = np.mean([r["ratio_sq_error"] for r in selected if r["variant"] == "mc"])
        for name in config["variants"]:
            group = [r for r in selected if r["variant"] == name]
            per_repeat = np.array([np.mean([r["ratio_sq_error"] for r in group if r["repeat"] == i])
                                   for i in range(repeats)])
            paired_mc = np.array([np.mean([r["ratio_sq_error"] for r in selected
                                          if r["variant"] == "mc" and r["repeat"] == i])
                                  for i in range(repeats)])
            rng = np.random.default_rng(33107)
            indices = rng.integers(repeats, size=(10000, repeats))
            bootstrap = per_repeat[indices].mean(1) / paired_mc[indices].mean(1)
            summaries.append({"dimension": dimension, "variant": name, "repeats": repeats,
                              "post_warmup_ratio_mse": float(per_repeat.mean()),
                              "mse_over_mc": float(per_repeat.mean() / mc_error),
                              "mse_over_mc_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                              "queries_per_rollout": 96,
                              "final_slope_sq_error": float(np.mean([
                                  r["post_update_slope_sq_error"] for r in rows
                                  if r["dimension"] == dimension and r["variant"] == name
                                  and r["online_step"] == 8])) if name != "mc" else None,
                              "denominator_events": sum(r["denominator_floored"] for r in group)})
        print(json.dumps([r for r in summaries if r["dimension"] == dimension]), flush=True)
    write_rows(output / "mechanism.csv", rows)
    dump(output / "summary.json", summaries)

    # This identity is specific to complete antithetic pairs, an affine CV,
    # and coefficient one. Failure would invalidate the audit interpretation.
    mean = torch.tensor([0.2, -0.3, 0.1, 0.5, -0.1, 0.4])
    physical_slope = torch.tensor([0.1, 0.05, -0.1, 0.2, 0.07, -0.03]).double()
    rho = 0.7
    eps = gaussian_noise(8, 6, torch.Generator().manual_seed(33701), device)
    rewards = 0.3 * (mean[None] + rho * eps)[:, 0]
    options = dict(kind="cv", surrogate="affine", freeze_after_steps=0,
                   tau=1.0, reward_offset=0.0, cv_coefficient=1.0)
    affine = MomentEstimator(options, 33701, device)
    affine.ensure_surrogate(mean, rho)
    affine.surrogate.theta[0] = 1.7
    affine.surrogate.theta[1:] = physical_slope * rho
    affine.surrogate.fitted = True
    cv = affine.from_samples(mean, rho, eps, rewards)
    mc = MomentEstimator(dict(options, kind="mc"), 33701, device).from_samples(mean, rho, eps, rewards)
    covariance = eps[:4].double().T @ eps[:4].double() / 4
    expected_u = mc.u + rho * (physical_slope - covariance @ physical_slope)
    h_error = float(abs(cv.h - mc.h))
    u_error = float((cv.u - expected_u).abs().max())
    assert h_error < 1e-6 and u_error < 1e-6, (h_error, u_error)

    # Expose the in-sample diagnostic's degeneracy; it is not a confidence score.
    single = gaussian_noise(1, 6, torch.Generator().manual_seed(33803), device, False)
    result = MomentEstimator(dict(kind="mc", tau=1.0, reward_offset=0.0,
                                  antithetic=False), 33803, device).from_samples(
                                      torch.zeros(6), 1.0, single, slope * single[:, 0])
    truth = torch.zeros(6)
    truth[0] = slope
    moment = result.diagnostics["in_sample_weighted_residual_moment"]
    true_error = float((result.ratio - truth).square().sum())
    assert moment < 1e-24 and true_error > 0.1, (moment, true_error)
    oracle_max = max(r["ratio_sq_error"] for r in rows if r["variant"] == "oracle-exponential")
    assert oracle_max < 1e-9, oracle_max
    dump(output / "identities.json", {"affine_denominator_abs_error": h_error,
                                     "affine_numerator_max_abs_error": u_error,
                                     "m1_in_sample_weighted_residual_moment": moment,
                                     "m1_true_ratio_sq_error": true_error,
                                     "oracle_max_ratio_sq_error": oracle_max})
    return {"dimensions": len(config["dimensions"]), "repeats": repeats,
            "rows": len(rows), "identity_checks_passed": 3}


def historical_diagnostics(config, output):
    """Recover execution KL from saved applied-guidance norms, without new images."""
    from diffusers import EulerAncestralDiscreteScheduler
    from .image import euler_coefficients

    output = Path(output)
    manifest = json.loads(Path(config["manifest"]).read_text())
    storage = Path(config["run_root"])
    images, steps, groups = [], [], defaultdict(list)
    schedules = {}
    for group, paths in manifest["groups"].items():
        for source in paths:
            directory = storage / Path(source).name
            run = json.loads((directory / "run.json").read_text())
            assert run["status"] == "completed", directory
            run_config = json.loads((directory / "config.json").read_text())
            image_config = run_config["image"]
            key = (image_config.get("model", "sd-legacy/stable-diffusion-v1-5"),
                   int(image_config.get("steps", 20)))
            if key not in schedules:
                scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                    key[0], subfolder="scheduler", revision=config["model_revision"], local_files_only=True)
                scheduler.set_timesteps(key[1])
                schedules[key] = scheduler.sigmas.float().tolist()
            sigmas = schedules[key]
            gamma = float(run_config["estimator"].get("gamma", 4))
            assert image_config.get("proxy", "euler") == "euler", directory
            for original in json.loads((directory / "images.json").read_text()):
                row = dict(original, group=group, source_run=str(directory), source_commit=run["commit"])
                recovered = []
                for original_step in row["steps"]:
                    step = dict(original_step)
                    index = step["index"]
                    sigma, next_sigma = sigmas[index:index + 2]
                    up, _, dt = euler_coefficients(sigma, next_sigma)
                    gain = -dt * sigma
                    displacement = abs(gamma * gain) * step["correction_norm"]
                    step.update(sigma=sigma, sigma_next=next_sigma, noise_std=up,
                                correction_gain=gain, displacement_norm=displacement,
                                displacement_over_noise=displacement / up,
                                execution_kl=0.5 * (displacement / up) ** 2,
                                execution_kl_source="recovered_from_saved_applied_correction_norm")
                    recovered.append(step)
                    steps.append(dict(group=group, prompt=row["prompt"], seed=row["seed"], **step))
                row["steps"] = recovered
                row["max_execution_kl"] = max((s["execution_kl"] for s in recovered), default=0.0)
                row["sum_execution_kl"] = sum(s["execution_kl"] for s in recovered)
                images.append(row)
                groups[group].append(row)
    summaries = []
    for group, rows in groups.items():
        values = np.array([r["max_execution_kl"] for r in rows])
        summaries.append({"group": group, "images": len(rows),
                          "max_step_kl_quantiles": dict(zip(["min", "median", "p90", "p95", "max"],
                                                            np.quantile(values, [0, .5, .9, .95, 1]).tolist())),
                          "per_prompt": {prompt: {
                              "mean_reward": float(np.mean([r["reward"] for r in rows if r["prompt"] == prompt])),
                              "mean_max_step_kl": float(np.mean([r["max_execution_kl"] for r in rows if r["prompt"] == prompt])),
                              "max_step_kl": max(r["max_execution_kl"] for r in rows if r["prompt"] == prompt)}
                              for prompt in sorted({r["prompt"] for r in rows})}})
    dump(output / "images.json", images)
    write_rows(output / "steps.csv", steps)
    dump(output / "summary.json", summaries)
    dump(output / "schedule.json", {"model_revision": config["model_revision"],
                                    "schedules": [{"model": k[0], "steps": k[1], "sigmas": v}
                                                  for k, v in schedules.items()]})
    return {"images": len(images), "guided_steps": len(steps), "new_reward_queries": 0}
