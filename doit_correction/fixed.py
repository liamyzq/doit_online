"""Frozen image-state capture and independent-bank estimator comparisons."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .estimators import MomentEstimator, gaussian_noise
from .image import ImageEngine


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def ratio_from_rewards(eps, rewards, tau, offset=6.0):
    weights = torch.exp((rewards.double() - offset) / tau)
    return (weights[:, None] * eps.double()).sum(0) / weights.sum()


def capture(config, output):
    engine = ImageEngine(config["image"])
    capture_config = config["capture"]
    bank_n = int(capture_config.get("bank_samples", 256))
    reference_n = int(capture_config.get("reference_samples", 512))
    taus = capture_config.get("taus", [0.3, 0.4])
    rows, images = [], []
    previous = None
    output = Path(output)

    def save_state(transition, prompt, seed):
        nonlocal previous
        index = transition.index
        bank_seed = seed * 100003 + index * 997 + 7000001
        ref_seed = bank_seed + 500000003
        gen = torch.Generator(device=engine.device).manual_seed(bank_seed)
        eps = gaussian_noise(bank_n, engine.dimension, gen, engine.device)
        m, rho = transition.endpoint_mean, transition.endpoint_std
        rewards = engine.reward(m[None] + rho * eps)
        previous_rewards = None
        if previous is not None:
            pm, pr = previous
            previous_rewards = engine.reward(pm[None] + pr * eps).cpu()
        ref_gen = torch.Generator(device=engine.device).manual_seed(ref_seed)
        ref_eps = gaussian_noise(reference_n, engine.dimension, ref_gen, engine.device)
        ref_rewards = engine.reward(m[None] + rho * ref_eps)
        refs = {}
        half, quarter = reference_n // 2, reference_n // 4
        first = torch.cat([torch.arange(quarter), torch.arange(half, half + quarter)]).to(engine.device)
        second = torch.cat([torch.arange(quarter, half), torch.arange(half + quarter, reference_n)]).to(engine.device)
        for tau in taus:
            ratio = ratio_from_rewards(ref_eps, ref_rewards, tau).float()
            split_a = ratio_from_rewards(ref_eps[first], ref_rewards[first], tau).float()
            split_b = ratio_from_rewards(ref_eps[second], ref_rewards[second], tau).float()
            guidance = engine.correction(transition, ratio, retain_graph=True).cpu()
            guidance_a = engine.correction(transition, split_a, retain_graph=True).cpu()
            guidance_b = engine.correction(transition, split_b, retain_graph=True).cpu()
            refs[str(tau)] = {
                "ratio": ratio.cpu(), "guidance": guidance,
                "split_ratio_sq_distance": float((split_a - split_b).square().sum()),
                "split_guidance_sq_distance": float((guidance_a - guidance_b).square().sum()),
                "reward_mean": float(ref_rewards.mean()), "reward_std": float(ref_rewards.std()),
            }
        future = None
        if index in capture_config.get("future_indices", []):
            future_n = int(capture_config.get("future_samples", 32))
            future_eps = gaussian_noise(future_n, engine.dimension,
                                       torch.Generator(device=engine.device).manual_seed(ref_seed + 13), engine.device)
            proxy_reward = engine.reward(m[None] + rho * future_eps)
            full_reward = engine.future_reward(transition, future_eps, ref_seed + 29)
            future = {"eps": future_eps.cpu(), "proxy_rewards": proxy_reward.cpu(),
                      "future_rewards": full_reward.cpu()}
        state = {
            "prompt": prompt, "seed": seed, "index": index,
            "latent": transition.x.detach().cpu(), "m": m.cpu(), "rho": rho,
            "sigma_transition": transition.noise_std,
            # Save the actual queried float32 epsilons. CUDA and CPU generators
            # do not produce interchangeable streams from the same integer seed.
            "eps": eps.cpu(), "rewards": rewards.cpu(),
            "previous_rewards": previous_rewards, "references": refs,
            "bank_seed": bank_seed, "reference_seed": ref_seed,
            "reference_samples": reference_n, "future": future,
            "image_config": config["image"], "path_estimator": config["estimator"],
        }
        filename = f"{prompt}-{seed}-step{index:02d}.pt"
        torch.save(state, output / filename)
        row = {"path": str(output / filename), "prompt": prompt, "seed": seed, "index": index,
               "rho": rho, "bank_reward_mean": float(rewards.mean()),
               "bank_samples": bank_n, "reference_samples": reference_n}
        rows.append(row)
        dump(output / "states.json", rows)
        print(json.dumps(row), flush=True)
        previous = (m.detach().clone(), rho)

    for prompt in config["prompts"]:
        for seed in config["seeds"]:
            previous = None
            images.append(engine.generate(prompt, seed, config["estimator"],
                                          output / "path_images" / prompt / str(seed), save_state))
    dump(output / "path_images.json", images)
    return {"states": len(rows), "trajectories": len(images), "capture_queries": engine.reward_queries,
            "score_forwards": engine.score_forwards, "reward_seconds": engine.reward_seconds,
            "vjp_seconds": engine.vjp_seconds}


def sample_indices(bank_n, n, anti, seed):
    generator = torch.Generator().manual_seed(seed)
    half = bank_n // 2
    if anti:
        pairs = torch.randperm(half, generator=generator)[:n // 2]
        pieces = [pairs, pairs + half]
        if n % 2:
            remaining = torch.ones(half, dtype=torch.bool)
            remaining[pairs] = False
            choices = torch.where(remaining)[0]
            pieces.append(choices[torch.randint(len(choices), (1,), generator=generator)])
        return torch.cat(pieces)
    # The first half contains independent draws; the second contains negatives.
    if n > half:
        raise ValueError("The independent part of the stored bank is too small")
    return torch.randperm(half, generator=generator)[:n]


def probe_future(config, output):
    """Couple later noises to resolve conditional-future variation."""
    paths = []
    for directory in config["captures"]:
        paths.extend(row["path"] for row in json.loads((Path(directory) / "states.json").read_text())
                     if row["index"] in config.get("indices", [10, 16]))
    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    engine = ImageEngine(first["image_config"])
    output, rows = Path(output), []
    n, branches = int(config.get("samples", 64)), int(config.get("branches", 2))
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        engine.use_prompt(state["prompt"])
        transition = engine.transition(state["latent"].to(engine.device), state["index"], True)
        seed = state["seed"] * 700001 + state["index"] * 1009 + 11000009
        eps = gaussian_noise(n, engine.dimension,
                             torch.Generator(device=engine.device).manual_seed(seed), engine.device)
        proxy = engine.reward(transition.endpoint_mean[None] + transition.endpoint_std * eps)
        futures = torch.stack([engine.future_reward(transition, eps, seed + 7919 * (branch + 1),
                                                    paired_future=True) for branch in range(branches)])
        ratios = {}
        for tau in config.get("taus", [0.3, 0.4, 0.6]):
            proxy_weights = ((proxy.double() - 6) / tau).exp()
            full_weights = ((futures.double() - 6) / tau).exp().mean(0)
            proxy_ratio = (proxy_weights[:, None] * eps.double()).sum(0) / proxy_weights.sum()
            full_ratio = (full_weights[:, None] * eps.double()).sum(0) / full_weights.sum()
            proxy_guidance = engine.correction(transition, proxy_ratio.float(), retain_graph=True)
            full_guidance = engine.correction(transition, full_ratio.float(), retain_graph=True)
            ratios[str(tau)] = {"proxy_ratio": proxy_ratio.cpu(), "future_ratio": full_ratio.cpu(),
                                "proxy_guidance": proxy_guidance.cpu(), "future_guidance": full_guidance.cpu()}
        filename = f'{state["prompt"]}-{state["seed"]}-step{state["index"]:02d}.pt'
        torch.save({"capture_path": path, "eps": eps.cpu(), "proxy_rewards": proxy.cpu(),
                    "future_rewards": futures.cpu(), "ratios": ratios, "seed": seed,
                    "paired_future": True}, output / filename)
        row = {"path": str(output / filename), "capture_path": path, "samples": n, "branches": branches,
               "proxy_mean_reward": float(proxy.mean()), "future_mean_reward": float(futures.mean()),
               "score_forwards": engine.score_forwards, "reward_queries": engine.reward_queries}
        rows.append(row)
        dump(output / "probes.json", rows)
        print(json.dumps(row), flush=True)
        del transition
    return {"states": len(rows), "reward_queries": engine.reward_queries,
            "score_forwards": engine.score_forwards, "reward_seconds": engine.reward_seconds,
            "vjp_seconds": engine.vjp_seconds}


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config, output):
    variants = config["variants"]
    repeats = int(config.get("repeats", 16))
    paths = []
    for directory in config["captures"]:
        paths.extend(row["path"] for row in json.loads((Path(directory) / "states.json").read_text()))
    output = Path(output)
    use_vjp = bool(config.get("with_vjp", False))
    if not use_vjp and any(v["estimator"]["kind"] == "guidance_ema" for v in variants):
        raise ValueError("Guidance EMA requires an actual VJP comparison")
    needs_oracle = any(v["estimator"]["kind"] in {"is", "cv_is"} for v in variants)
    engine = None
    if use_vjp or needs_oracle:
        engine = ImageEngine(torch.load(paths[0], map_location="cpu", weights_only=False)["image_config"])
    rows, estimators, previous_guidance = [], {}, {}
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)
        trajectory = str(Path(path).parent), state["prompt"], state["seed"]
        m, rho = state["m"], state["rho"]
        transition = None
        if engine is not None:
            engine.use_prompt(state["prompt"])
        if use_vjp:
            transition = engine.transition(state["latent"].to(engine.device), state["index"], True)
        for variant in variants:
            vconfig = variant["estimator"]
            ref = state["references"][str(vconfig.get("tau", 0.3))]
            for repeat in range(repeats):
                key = (*trajectory, variant["name"], repeat)
                if key not in estimators:
                    estimators[key] = MomentEstimator(vconfig, 900001 + repeat * 101, "cpu")
                estimator = estimators[key]
                start = time.perf_counter()
                if vconfig["kind"] in {"is", "cv_is"}:
                    result = estimator.estimate(m, rho, lambda z: engine.reward(z.to(engine.device)).cpu())
                else:
                    n = estimator.budget(estimator.calls)
                    if estimator.kind == "spider" and not estimator._refresh(estimator.calls):
                        n //= 2
                    subset_seed = 2000003 + state["seed"] * 7919 + state["index"] * 97 + repeat * 65537
                    selected = sample_indices(len(state["rewards"]), n,
                                              vconfig.get("antithetic", True), subset_seed)
                    previous_rewards = None
                    if estimator.kind == "spider" and not estimator._refresh(estimator.calls):
                        previous_rewards = state["previous_rewards"][selected]
                    result = estimator.from_samples(m, rho, state["eps"][selected], state["rewards"][selected],
                                                    previous_rewards=previous_rewards)
                ratio_error = float((result.ratio - ref["ratio"]).double().square().sum())
                ref_norm = float(ref["ratio"].double().square().sum())
                guidance_error, guidance_norm = None, None
                if use_vjp:
                    grad = engine.correction(transition, result.ratio.to(engine.device), retain_graph=True).cpu()
                    if estimator.kind == "guidance_ema" and key in previous_guidance:
                        beta = float(vconfig.get("ema_beta", 0.5))
                        grad = beta * grad + (1 - beta) * previous_guidance[key]
                    previous_guidance[key] = grad.clone()
                    guidance_error = float((grad - ref["guidance"]).double().square().sum())
                    guidance_norm = float(ref["guidance"].double().square().sum())
                row = {
                    "variant": variant["name"], "prompt": state["prompt"], "seed": state["seed"],
                    "index": state["index"], "repeat": repeat, "tau": vconfig.get("tau", 0.3),
                    "queries": result.diagnostics["queries"], "cumulative_queries": estimator.queries,
                    "ratio_sq_error": ratio_error, "ratio_reference_sq_norm": ref_norm,
                    "ratio_relative_sq_error": ratio_error / max(ref_norm, 1e-12),
                    "guidance_sq_error": guidance_error, "guidance_reference_sq_norm": guidance_norm,
                    "guidance_relative_sq_error": None if guidance_error is None else guidance_error / max(guidance_norm, 1e-12),
                    "reference_split_ratio_sq_distance": ref["split_ratio_sq_distance"],
                    "reference_split_guidance_sq_distance": ref["split_guidance_sq_distance"],
                    "denominator_floored": result.diagnostics["denominator_floored"],
                    "ess": result.diagnostics["ess"],
                    "heldout_prediction_mse": result.diagnostics["heldout_prediction_mse"],
                    "heldout_residual_mse": result.diagnostics["heldout_residual_mse"],
                    "heldout_pair_difference_mse": result.diagnostics["heldout_pair_difference_mse"],
                    "weighted_residual_moment": result.diagnostics["weighted_residual_moment"],
                    "proposal_norm": result.diagnostics.get("proposal_norm", 0.0),
                    "estimator_seconds": result.diagnostics["estimator_seconds"],
                    "live_oracle_seconds": result.diagnostics.get("oracle_seconds", 0.0),
                    "elapsed_seconds": time.perf_counter() - start,
                }
                rows.append(row)
        del transition
        write_rows(output / "fixed.csv", rows)
        print(json.dumps({"state": path, "rows": len(rows)}), flush=True)
    groups = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    summaries = []
    for name, group in groups.items():
        summary = {"variant": name, "rows": len(group)}
        for field in ("ratio_relative_sq_error", "guidance_relative_sq_error", "heldout_residual_mse",
                      "heldout_pair_difference_mse", "weighted_residual_moment", "proposal_norm",
                      "ess", "queries", "estimator_seconds", "live_oracle_seconds"):
            vals = [row[field] for row in group if row[field] is not None]
            summary[field] = sum(vals) / len(vals) if vals else None
        summary["denominator_events"] = sum(r["denominator_floored"] for r in group)
        summary["total_queries"] = sum(r["queries"] for r in group)
        summaries.append(summary)
    dump(output / "summary.json", summaries)
    return {"variants": len(variants), "states": len(paths), "rows": len(rows),
            "actual_new_oracle_queries": 0 if engine is None else engine.reward_queries}
