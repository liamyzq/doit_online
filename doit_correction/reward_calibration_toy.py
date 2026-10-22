"""Controlled sparse-SMC comparisons for conditional reward calibration."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .reward_calibration import RewardCalibration, check_reward_calibration
from .uncertainty_toy import MixtureChain, dump, normalize, resample_rows, summarize


def features(model, x, k):
    weights, means, _ = model.posterior(x, k)
    clean = np.sum(weights * means, axis=-1)
    proxy = model.log_reward(clean)
    return np.stack([x, clean, proxy], axis=-1), proxy


def continuations(model, x, k, branches, rng):
    y = np.broadcast_to(x[..., None], (*x.shape, branches)).copy()
    for index in range(k, 0, -1):
        y = model.draw(model.kernel(y, index), rng)
    return model.log_reward(y)


def calibrate(model, config):
    rng = np.random.default_rng(config["calibration_seed"])
    x = model.initial((config["calibration_anchors"],), rng)
    banks = {}
    for k in range(model.steps, 0, -1):
        if k in config["checkpoints"]:
            f, proxy = features(model, x, k)
            rewards = continuations(model, x, k, config["calibration_futures"], rng)
            banks[k] = RewardCalibration(f, proxy, rewards)
        x = model.draw(model.kernel(x, k), rng)
    return banks


def potential(model, banks, x, k, method, rng):
    kind = method["value"]
    order = method.get("moment_order", 1.0)
    if kind in {"point", "future", "mixture"}:
        value = model.value(x, k, kind, rng, method.get("branches", 2), power=order) / order
    else:
        f, proxy = features(model, x, k)
        model.cost["value_evaluations"] += x.size
        value, _ = banks[k].predict(f, proxy, 1.0 / order, kind, method.get("neighbours", 8))
        value /= order
    return method.get("strength", 1.0) * value


def sparse_particles(model, banks, config, method, seed):
    initial, path, future, selection = [np.random.default_rng(seed + offset)
                                        for offset in (0, 100003, 200003, 300007)]
    repeats, n = config["repeats"], method["particles"]
    x = model.initial((repeats, n), initial)
    v = np.zeros_like(x)
    logw = np.zeros_like(x)
    checkpoints = config["checkpoints"] if method["value"] != "flat" else []
    ess_history, resamplings = [], 0
    for k in range(model.steps, 0, -1):
        if k in checkpoints:
            new_v = potential(model, banks, x, k, method, future)
            logw += new_v - v
            v = new_v
            x, v, logw, ess, count = resample_rows(x, v, logw, selection, method.get("ess_threshold", 0.5))
            ess_history.append(float(np.mean(ess / n)))
            resamplings += count
        x = model.draw(model.kernel(x, k), path)
    terminal = model.log_reward(x)
    logw += terminal - v
    weights = normalize(logw)
    return x, weights, {"checkpoint_ess_fractions": ess_history,
                        "resamplings_per_repeat": resamplings / repeats,
                        "final_ess_fraction": float(np.mean(1 / np.sum(weights ** 2, axis=-1)) / n)}


def value_diagnostics(model, banks, config):
    rng = np.random.default_rng(config["diagnostic_seed"])
    x = model.initial((config["diagnostic_anchors"],), rng)
    records = []
    for k in range(model.steps, 0, -1):
        if k in config["checkpoints"]:
            truth = model.value(x, k, "mixture")
            truth_weights = normalize(truth)
            for kind in ("point", "mean", "gaussian", "empirical", "direct", "future"):
                predicted = potential(model, banks, x, k, {"value": kind, "neighbours": 8}, rng)
                centered = predicted - predicted.mean() - truth + truth.mean()
                predicted_log_weights = predicted - logsumexp(predicted)
                records.append({"index": k, "value": kind,
                                "centered_log_value_mse": float(np.mean(centered ** 2)),
                                "normalized_value_kl": float(np.sum(truth_weights * (truth - logsumexp(truth) - predicted_log_weights)))})
        x = model.draw(model.kernel(x, k), rng)
    return records


def evaluate(config, output):
    checks = check_reward_calibration()
    dump(output / "checks.json", checks)
    rows, calibration, diagnostics, counts = [], [], [], []
    for problem in config["problems"]:
        model = MixtureChain(problem)
        banks = calibrate(model, config)
        training_cost = dict(model.cost)
        calibration.append({"problem": problem["name"], "costs": training_cost,
                            "anchors_per_checkpoint": config["calibration_anchors"],
                            "futures_per_anchor": config["calibration_futures"]})
        dump(output / f'{problem["name"]}-calibration.json', {str(k): bank.record() for k, bank in banks.items()})
        before = model.cost.copy()
        diagnostics.append({"problem": problem["name"], "values": value_diagnostics(model, banks, config),
                            "costs": dict(model.cost - before)})
        dump(output / "value-diagnostics.json", diagnostics)
        for seed in config["seeds"]:
            for method in config["methods"]:
                before = model.cost.copy()
                started = time.perf_counter()
                x, weights, info = sparse_particles(model, banks, config, method, seed)
                row = {"problem": problem["name"], "seed": seed, "method": method["name"],
                       "particles": method["particles"], **summarize(model, x, weights), **info,
                       "seconds": time.perf_counter() - started,
                       "costs": dict(model.cost - before),
                       "costs_per_repeat": {key: value / config["repeats"] for key, value in (model.cost - before).items()},
                       "shared_calibration_costs": training_cost if method["value"] in {"mean", "gaussian", "empirical", "direct"} else {}}
                rows.append(row)
                dump(output / "comparisons.json", rows)
                print(json.dumps(row), flush=True)
        counts.append(dict(model.cost))
    dump(output / "calibration-costs.json", calibration)
    totals = dict(sum((Counter(row) for row in counts), Counter()))
    return {"conditions": len(rows), "checks": checks, "total_costs": totals,
            "calibration_costs": calibration, "repeats_per_condition": config["repeats"]}


def check_allocation_moments(problems):
    """Check the new reward power and outer root against independent integration."""
    from scipy.integrate import quad

    errors = {}
    for problem in problems:
        model = MixtureChain(problem)
        for k in (1, 5):
            for x in (-1.0, 1.5):
                probabilities, means, variance = model.posterior(x, k)
                integral = 0.0
                for weight, mean in zip(probabilities, means):
                    def integrand(z):
                        normal = math.exp(-0.5 * (z - mean) ** 2 / variance) / math.sqrt(2 * math.pi * variance)
                        return normal * math.exp(-model.precision * (z - model.target) ** 2)
                    integral += weight * quad(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
                got = potential(model, {}, np.array([x]), k,
                                {"value": "mixture", "moment_order": 2}, None)[0]
                errors[f'{problem["name"]}-k{k}-x{x}'] = abs(got - 0.5 * math.log(integral))
    bank = RewardCalibration([[0.0]], [1.0], [[1.0, 3.0]])
    got, _ = bank.predict([[0.0]], [4.0], 0.5, "empirical", 1)
    errors["empirical_second_moment_root"] = abs(got[0] / 2 -
        (logsumexp([8.0, 12.0]) - math.log(2)) / 2)
    if max(errors.values()) > 1e-10:
        raise AssertionError(errors)
    return errors


def evaluate_allocation(config, output):
    checks = check_allocation_moments(config["problems"])
    dump(output / "checks.json", checks)
    source = Path(config["calibration_directory"])
    costs = {row["problem"]: row["costs"] for row in json.loads((source / "calibration-costs.json").read_text())}
    rows, counts = [], []
    for problem in config["problems"]:
        model = MixtureChain(problem)
        saved = json.loads((source / f'{problem["name"]}-calibration.json').read_text())
        banks = {int(index): RewardCalibration.from_record(record) for index, record in saved.items()}
        for seed in config["seeds"]:
            for method in config["methods"]:
                before, started = model.cost.copy(), time.perf_counter()
                x, weights, info = sparse_particles(model, banks, config, method, seed)
                calibrated = method["value"] in {"mean", "gaussian", "empirical", "direct"}
                row = {"problem": problem["name"], "seed": seed, "method": method["name"],
                       "particles": method["particles"], **summarize(model, x, weights), **info,
                       "seconds": time.perf_counter() - started, "costs": dict(model.cost - before),
                       "costs_per_repeat": {key: value / config["repeats"] for key, value in (model.cost - before).items()},
                       "shared_calibration_costs": costs[problem["name"]] if calibrated else {}}
                rows.append(row)
                dump(output / "comparisons.json", rows)
                print(json.dumps(row), flush=True)
        counts.append(dict(model.cost))
    return {"conditions": len(rows), "checks": checks,
            "total_costs": dict(sum((Counter(row) for row in counts), Counter())),
            "calibration_reused_from": str(source), "original_calibration_costs": costs,
            "repeats_per_condition": config["repeats"]}
