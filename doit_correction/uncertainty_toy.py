"""Analytic continuation models for uncertainty and corrected particle sampling.

The mixture model uses its exact reverse mixture density, not a Gaussian
approximation to that density. All arrays and density arithmetic use float64.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, ndtr


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def normal_logpdf(x, mean, variance):
    return -0.5 * ((x - mean) ** 2 / variance + np.log(2 * np.pi * variance))


def normalize(logw):
    return np.exp(logw - logsumexp(logw, axis=-1, keepdims=True))


class MixtureChain:
    """X_t = X_0 + N(0, v_t), with a Gaussian-mixture clean distribution."""

    def __init__(self, config):
        self.means = np.asarray(config["means"], dtype=float)
        self.probabilities = np.asarray(config["probabilities"], dtype=float)
        self.clean_variance = float(config["clean_variance"])
        self.variances = np.asarray(config["noise_variances"], dtype=float)
        self.target = float(config["reward_target"])
        self.precision = float(config["beta"]) / float(config["reward_variance"])
        self.steps = len(self.variances) - 1
        self.cost = Counter()

    def log_reward(self, x):
        self.cost["reward_evaluations"] += np.size(x)
        return -0.5 * self.precision * (x - self.target) ** 2

    def initial(self, shape, rng, tilted=False):
        p, means, variance = self.target_parameters() if tilted else (
            self.probabilities, self.means, self.clean_variance)
        labels = rng.choice(len(p), size=shape, p=p)
        return means[labels] + np.sqrt(variance + self.variances[-1]) * rng.normal(size=shape)

    def posterior(self, x, k):
        x = np.asarray(x)
        v = self.variances[k]
        total = self.clean_variance + v
        weights = normalize(np.log(self.probabilities) + normal_logpdf(x[..., None], self.means, total))
        means = (v * self.means + self.clean_variance * x[..., None]) / total
        variance = self.clean_variance * v / total
        return weights, means, variance

    def kernel(self, x, k):
        weights, _, _ = self.posterior(x, k)
        v, s = self.variances[k], self.variances[k - 1]
        ratio = (self.clean_variance + s) / (self.clean_variance + v)
        means = self.means + ratio * (np.asarray(x)[..., None] - self.means)
        variance = (self.clean_variance + s) * (v - s) / (self.clean_variance + v)
        return weights, means, variance

    def draw(self, parameters, rng):
        weights, means, variance = parameters
        u = rng.random(weights.shape[:-1])
        labels = np.sum(u[..., None] > np.cumsum(weights, axis=-1), axis=-1)
        selected = np.take_along_axis(means, labels[..., None], axis=-1)[..., 0]
        self.cost["transition_samples"] += selected.size
        return selected + math.sqrt(variance) * rng.normal(size=selected.shape)

    def density(self, z, parameters, delta=0.0):
        weights, means, variance = parameters
        return logsumexp(np.log(weights) + normal_logpdf(
            np.asarray(z)[..., None], means + np.asarray(delta)[..., None], variance), axis=-1)

    def value(self, x, k, kind, rng=None, branches=2):
        x = np.asarray(x)
        self.cost["value_evaluations"] += x.size
        if k == 0:
            return self.log_reward(x)
        if kind == "flat":
            return np.zeros_like(x)
        if kind in {"branch", "future"}:
            y = np.broadcast_to(x[..., None], (*x.shape, branches)).copy()
            end = k - 1 if kind == "branch" else 0
            for j in range(k, end, -1):
                y = self.draw(self.kernel(y, j), rng)
            values = self.value(y, end, "gaussian" if end else "point")
            return logsumexp(values, axis=-1) - math.log(branches)
        weights, means, variance = self.posterior(x, k)
        if kind == "mixture":
            self.cost["analytic_component_integrals"] += means.size
            denominator = 1 + self.precision * variance
            logh = -0.5 * np.log(denominator) - 0.5 * self.precision * (means - self.target) ** 2 / denominator
            return logsumexp(np.log(weights) + logh, axis=-1)
        mean = np.sum(weights * means, axis=-1)
        if kind == "point":
            return self.log_reward(mean)
        if kind == "gaussian":
            self.cost["analytic_component_integrals"] += x.size
            variance = variance + np.sum(weights * (means - mean[..., None]) ** 2, axis=-1)
            denominator = 1 + self.precision * variance
            return -0.5 * np.log(denominator) - 0.5 * self.precision * (mean - self.target) ** 2 / denominator
        raise ValueError(kind)

    def target_parameters(self):
        denominator = 1 + self.precision * self.clean_variance
        logh = -0.5 * math.log(denominator) - 0.5 * self.precision * (self.means - self.target) ** 2 / denominator
        weights = normalize(np.log(self.probabilities) + logh)
        means = (self.means + self.precision * self.clean_variance * self.target) / denominator
        return weights, means, self.clean_variance / denominator

    def target_cdf(self, x):
        weights, means, variance = self.target_parameters()
        return np.sum(weights * ndtr((np.asarray(x)[..., None] - means) / math.sqrt(variance)), axis=-1)

    def exact_kernel(self, x, k):
        """Reverse transition after tilting the clean mixture analytically."""
        weights, means, variance = self.target_parameters()
        v, s = self.variances[k], self.variances[k - 1]
        probabilities = normalize(np.log(weights) + normal_logpdf(np.asarray(x)[..., None], means, variance + v))
        ratio = (variance + s) / (variance + v)
        conditional_means = means + ratio * (np.asarray(x)[..., None] - means)
        conditional_variance = (variance + s) * (v - s) / (variance + v)
        return probabilities, conditional_means, conditional_variance


def resample_rows(x, potential, logw, rng, threshold):
    weights = normalize(logw)
    n = x.shape[-1]
    ess = 1 / np.sum(weights ** 2, axis=-1)
    rows = np.flatnonzero(ess < threshold * n)
    for row in rows:
        positions = (rng.random() + np.arange(n)) / n
        indices = np.searchsorted(np.cumsum(weights[row]), positions)
        x[row] = x[row, indices]
        potential[row] = potential[row, indices]
        logw[row] = 0
    return x, potential, logw, ess, len(rows)


def particles(model, method, seed, repeats):
    n = method["particles"]
    initial_rng, pilot_rng, value_rng, path_rng, selection_rng = [
        np.random.default_rng(seed + offset) for offset in (0, 100003, 200003, 300007, 400009)]
    exact = method.get("proposal") == "exact"
    x = model.initial((repeats, n), initial_rng, tilted=exact)
    kind = method.get("value", "flat")
    branches = method.get("branches", 2)
    smc = method.get("smc", False)
    v = model.value(x, model.steps, kind, value_rng, branches) if smc else np.zeros_like(x)
    logw = v.copy()
    ess_history, resamplings = [], 0
    for k in range(model.steps, 0, -1):
        if smc:
            x, v, logw, ess, count = resample_rows(x, v, logw, selection_rng, method.get("ess_threshold", 0.5))
            ess_history.append(ess / n)
            resamplings += count
        parameters = model.kernel(x, k)
        probabilities, means, variance = parameters
        if exact:
            x = model.draw(model.exact_kernel(x, k), path_rng)
            continue
        delta = np.zeros_like(x)
        if method.get("proposal", "base") == "pilot":
            pilots = method["pilots"]
            candidates = model.draw(model.kernel(np.broadcast_to(x[..., None], (*x.shape, pilots)), k), pilot_rng)
            values = model.value(candidates, k - 1, kind, pilot_rng, branches)
            base_mean = np.sum(probabilities * means, axis=-1)
            delta = method.get("scale", 1.0) * (np.sum(normalize(values) * candidates, axis=-1) - base_mean)
        elif method.get("proposal") == "gradient":
            base_mean = np.sum(probabilities * means, axis=-1)
            step = 1e-4
            gradient = (model.value(base_mean + step, k - 1, kind) - model.value(base_mean - step, k - 1, kind)) / (2 * step)
            delta = method.get("scale", 1.0) * variance * gradient
        z = model.draw((probabilities, means + delta[..., None], variance), path_rng)
        if smc:
            new_v = model.value(z, k - 1, kind, value_rng, branches)
            logw += model.density(z, parameters) - model.density(z, parameters, delta) + new_v - v
            v = new_v
        x = z
    weights = normalize(logw) if smc else np.full_like(x, 1 / n)
    return x, weights, {"mean_step_ess_fraction": float(np.mean(ess_history)) if ess_history else 1.0,
                        "final_ess_fraction": float(np.mean(1 / np.sum(weights ** 2, axis=-1)) / n),
                        "resampling_events_per_repeat": resamplings / repeats}


def summarize(model, x, weights):
    grid = np.linspace(-4, 4, 161)
    truth = model.target_cdf(grid)
    empirical = np.sum(weights[..., None] * (x[..., None] <= grid), axis=1)
    per_repeat = np.mean((empirical - truth) ** 2, axis=-1)
    positive = np.sum(weights * (x > 0), axis=-1)
    positive_truth = float(1 - model.target_cdf(0))
    return {"cdf_mse": float(np.mean(per_repeat)),
            "cdf_mse_standard_error": float(np.std(per_repeat, ddof=1) / np.sqrt(len(per_repeat))),
            "pooled_cdf_mse": float(np.mean((empirical.mean(axis=0) - truth) ** 2)),
            "positive_mass_mse": float(np.mean((positive - positive_truth) ** 2)),
            "positive_mass": float(positive.mean()), "target_positive_mass": positive_truth,
            "mean": float(np.mean(np.sum(weights * x, axis=-1))),
            "mean_log_reward": float(np.mean(np.sum(weights * (-0.5 * model.precision * (x - model.target) ** 2), axis=-1)))}


def run_toys(config, output):
    rows = []
    for problem in config["problems"]:
        for seed in config["seeds"]:
            for method in config["methods"]:
                model = MixtureChain(problem)
                started = time.perf_counter()
                x, weights, diagnostics = particles(model, method, seed, config["repeats"])
                row = {"problem": problem["name"], "seed": seed, "method": method["name"],
                       "particles": method["particles"], **summarize(model, x, weights), **diagnostics,
                       "costs": dict(model.cost), "seconds": time.perf_counter() - started,
                       "costs_per_repeat": {key: value / config["repeats"] for key, value in model.cost.items()}}
                rows.append(row)
                dump(output / "comparisons.json", rows)
                print(json.dumps(row), flush=True)
    return {"conditions": len(rows), "repeats_per_condition": config["repeats"],
            "total_costs": dict(sum((Counter(row["costs"]) for row in rows), Counter()))}


def fors_draw(count, rng, mean, variance, target, precision, bound, direct=False):
    anchor = -precision * (mean - target)
    proposal_mean = mean + variance * anchor
    accepted, proposed, queries, clipped, factors = [], 0, 0, 0, 0
    while sum(len(batch) for batch in accepted) < count:
        remaining = count - sum(len(batch) for batch in accepted)
        z = proposal_mean + math.sqrt(variance) * rng.normal(size=max(remaining, 128))
        proposed += len(z)
        if direct:
            probability = np.exp(-0.5 * precision * (z - mean) ** 2)
        else:
            numbers = rng.poisson(2 * bound, size=len(z))
            owners = np.repeat(np.arange(len(z)), numbers)
            u = rng.random(len(owners))
            w = -precision * u * (z[owners] - mean) ** 2
            clipped += int(np.sum(np.abs(w) > bound))
            factors += len(w)
            queries += len(w)
            factor = (bound + np.clip(w, -bound, bound)) / (2 * bound)
            probability = np.ones(len(z))
            np.multiply.at(probability, owners, factor)
        accepted.append(z[rng.random(len(z)) < probability])
        if proposed > count * 20000:
            raise RuntimeError("Local rejection exceeded the declared proposal budget")
    return np.concatenate(accepted)[:count], {"proposals": proposed, "guidance_queries": queries + 1,
                                            "residual_value_evaluations": proposed if direct else 0,
                                            "clipped_factors": clipped, "poisson_factors": factors,
                                            "returned_samples": count}


def run_local(config, output):
    mean, variance, target, precision = [config[key] for key in ("mean", "variance", "target", "precision")]
    exact_variance = 1 / (1 / variance + precision)
    exact_mean = exact_variance * (mean / variance + precision * target)
    proposal_mean = mean - variance * precision * (mean - target)
    grid = np.linspace(exact_mean - 4 * math.sqrt(exact_variance), exact_mean + 4 * math.sqrt(exact_variance), 161)
    truth = ndtr((grid - exact_mean) / math.sqrt(exact_variance))
    rows = []
    methods = ["shift", "exact", "direct"] + [f"fors-{b}" for b in config["bounds"]]
    for seed in config["seeds"]:
        for method in methods:
            rng = np.random.default_rng(seed)
            started = time.perf_counter()
            if method in {"shift", "exact"}:
                mu, var = (proposal_mean, variance) if method == "shift" else (exact_mean, exact_variance)
                x = mu + math.sqrt(var) * rng.normal(size=config["samples"])
                counts = {"proposals": len(x), "guidance_queries": int(method == "shift"),
                          "residual_value_evaluations": 0, "clipped_factors": 0,
                          "poisson_factors": 0, "returned_samples": len(x)}
            else:
                bound = float(method.split("-")[1]) if method.startswith("fors") else 0
                x, counts = fors_draw(config["samples"], rng, mean, variance, target, precision, bound, method == "direct")
            empirical = np.searchsorted(np.sort(x), grid, side="right") / len(x)
            rows.append({"seed": seed, "method": method, "cdf_mse": float(np.mean((empirical - truth) ** 2)),
                         "mean": float(x.mean()), "variance": float(x.var()), "target_mean": exact_mean,
                         "target_variance": exact_variance, **counts, "seconds": time.perf_counter() - started})
            dump(output / "local.json", rows)
    return {"conditions": len(rows), "guidance_queries": sum(row["guidance_queries"] for row in rows),
            "proposals": sum(row["proposals"] for row in rows)}


def run_checks(config, output):
    problem = {"means": [0.0], "probabilities": [1.0], "clean_variance": 0.7,
               "noise_variances": [0.0, 0.2, 1.0, 4.0], "reward_target": 1.3,
               "reward_variance": 0.8, "beta": 2.0}
    model = MixtureChain(problem)
    rng = np.random.default_rng(47001)
    x = np.linspace(-3, 3, 41)
    errors = {}
    errors["gaussian_value_equals_mixture"] = float(np.max(np.abs(model.value(x, 2, "gaussian") - model.value(x, 2, "mixture"))))
    initial_v = model.value(x, model.steps, "gaussian")
    logw, total_ratio, v = initial_v.copy(), np.zeros_like(x), initial_v
    exact_errors = []
    for k in range(model.steps, 0, -1):
        parameters = model.kernel(x, k)
        delta = 0.12 * np.sin(x)
        z = model.draw((parameters[0], parameters[1] + delta[..., None], parameters[2]), rng)
        direct_ratio = model.density(z, parameters) - model.density(z, parameters, delta)
        formula = -delta * (z - parameters[1][..., 0]) / parameters[2] + 0.5 * delta ** 2 / parameters[2]
        errors[f"gaussian_ratio_step{k}"] = float(np.max(np.abs(formula - direct_ratio)))
        new_v = model.value(z, k - 1, "gaussian")
        logw += direct_ratio + new_v - v
        total_ratio += direct_ratio
        exact_errors.append(np.max(np.abs(model.density(z, parameters) - model.density(z, model.exact_kernel(x, k)) + new_v - v)))
        x, v = z, new_v
    errors["path_telescoping"] = float(np.max(np.abs(logw - (model.log_reward(x) + total_ratio))))
    errors["exact_doob_increment"] = float(max(exact_errors))
    # Independent Poisson factors for a bounded W: evaluate the identity's series.
    bound, expected_w = 0.7, -0.2
    terms = [math.exp(-2 * bound) * (bound + expected_w) ** j / math.factorial(j) for j in range(40)]
    errors["poisson_identity"] = abs(sum(terms) - math.exp(expected_w - bound))
    # Independent quadrature of the mixture kernel checks the tower relation.
    mixture = MixtureChain(dict(problem, means=[-2.0, 2.0], probabilities=[0.8, 0.2]))
    nodes, quadrature = np.polynomial.hermite.hermgauss(256)
    state = np.array([-1.0, 0.0, 1.0])
    parameters = mixture.kernel(state, 2)
    candidates = parameters[1][..., None] + math.sqrt(2 * parameters[2]) * nodes
    integrated = np.sum(parameters[0] * np.sum(np.exp(mixture.value(candidates, 1, "mixture")) * quadrature / math.sqrt(math.pi), axis=-1), axis=-1)
    errors["mixture_tower"] = float(np.max(np.abs(np.log(integrated) - mixture.value(state, 2, "mixture"))))
    dump(output / "checks.json", errors)
    if max(errors.values()) > 1e-9:
        raise AssertionError(errors)
    return {"checks_passed": len(errors), "maximum_absolute_error": max(errors.values())}
