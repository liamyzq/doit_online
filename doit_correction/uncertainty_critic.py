"""Uncertainty calibration and particle sampling for frozen real RL critics."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .critic import CriticEngine


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class BatchCritic:
    def __init__(self, task, chunk=2048):
        self.engine = CriticEngine(task)
        self.task = task
        self.chunk = chunk
        self.dimension = self.engine.dimension
        self.steps = len(self.engine.timesteps) - 1
        self.calibration = None

    @torch.no_grad()
    def transition(self, x, index):
        shape = x.shape
        x = x.reshape(-1, self.dimension)
        engine = self.engine
        s, t = engine.timesteps[index:index + 1], engine.timesteps[index + 1:index + 2]
        alpha_s, alpha_t = float(engine.schedule.marginal_alpha(s)), float(engine.schedule.marginal_alpha(t))
        sigma_s, sigma_t = float(engine.schedule.marginal_std(s)), float(engine.schedule.marginal_std(t))
        noise2 = self.task["eta"] ** 2 * sigma_t ** 2 / sigma_s ** 2 * (1 - alpha_s ** 2 / alpha_t ** 2)
        noise = math.sqrt(max(noise2, 0) + 1e-12)
        a = alpha_t / alpha_s
        b = math.sqrt(max(sigma_t ** 2 - noise2, 1e-12)) - a * sigma_s
        epsilon = []
        for batch in x.split(self.chunk):
            epsilon.append(-engine.model(batch, s.expand(len(batch)), engine.state.expand(len(batch), -1)) * sigma_s)
            engine.score_forwards += len(batch)
        epsilon = torch.cat(epsilon)
        mean = a * x + b * epsilon
        clean = (x - sigma_s * epsilon) / alpha_s
        return mean.reshape(shape), noise, clean.reshape(shape)

    def reward(self, x):
        shape = x.shape[:-1]
        return torch.cat([self.engine.reward(batch) for batch in x.reshape(-1, self.dimension).split(self.chunk)]).reshape(shape)

    def generator(self, seed):
        return torch.Generator(device="cuda").manual_seed(seed)

    def noise(self, shape, generator):
        return torch.randn(shape, device="cuda", generator=generator)

    def rollout(self, x, start, generator, collect=False):
        clean_means = []
        for index in range(start, self.steps):
            mean, std, clean = self.transition(x, index)
            if collect:
                clean_means.append(clean)
            x = mean + std * self.noise(mean.shape, generator)
        return (x, clean_means) if collect else x

    def batch_check(self):
        """Compare new batched actor/transition math to the validated scalar adapter."""
        self.engine.use_state(0)
        x = torch.linspace(-0.7, 0.9, 3 * self.dimension, device="cuda").reshape(3, self.dimension)
        batched, std, _ = self.transition(x, 3)
        scalar = [self.engine.transition(row, 3) for row in x]
        error = float((batched - torch.stack([row.mean for row in scalar])).abs().max())
        std_error = max(abs(std - row.noise_std) for row in scalar)
        if error > 2e-5 or std_error > 1e-12:
            raise AssertionError({"batch_mean_error": error, "batch_noise_error": std_error})
        return {"batch_mean_max_error": error, "batch_noise_error": std_error}

    def load_calibration(self, path):
        record = json.loads(Path(path).read_text())
        self.calibration = [{key: torch.tensor(row[key], device="cuda", dtype=torch.float32)
                             for key in ("bias", "root")} for row in record["parameters"]]

    def value(self, x, index, method, generator):
        if index == self.steps:
            return self.reward(x).double() / self.task["tau"]
        kind = method["value"]
        if kind == "flat":
            return torch.zeros(x.shape[:-1], device="cuda", dtype=torch.float64)
        if kind in {"branch", "future"}:
            branches = method.get("branches", 2)
            y = x.unsqueeze(-2).expand(*x.shape[:-1], branches, self.dimension)
            if kind == "future":
                y = self.rollout(y, index, generator)
                values = self.reward(y).double() / self.task["tau"]
            else:
                mean, std, _ = self.transition(y, index)
                y = mean + std * self.noise(mean.shape, generator)
                values = self.value(y, index + 1, dict(method, value="gaussian"), generator)
            return torch.logsumexp(values, dim=-1) - math.log(branches)
        _, _, mean = self.transition(x, index)
        if kind == "point":
            return self.reward(mean).double() / self.task["tau"]
        parameters = self.calibration[index]
        mean = mean + parameters["bias"]
        if kind == "calibrated_point":
            return self.reward(mean).double() / self.task["tau"]
        if kind != "gaussian":
            raise ValueError(kind)
        samples = method.get("uncertainty_samples", 2)
        eps = self.noise((*mean.shape[:-1], samples // 2, self.dimension), generator)
        eps = torch.cat([eps, -eps], dim=-2)
        candidates = mean.unsqueeze(-2) + eps @ parameters["root"].T
        values = self.reward(candidates).double() / self.task["tau"]
        return torch.logsumexp(values, dim=-1) - math.log(samples)


def select_ancestors(x, v, logw, generator, threshold):
    weights = torch.softmax(logw, dim=-1)
    n = x.shape[1]
    ess = 1 / weights.square().sum(dim=-1)
    rows = torch.nonzero(ess < threshold * n).flatten()
    if len(rows):
        positions = (torch.rand((len(rows), 1), device="cuda", generator=generator) + torch.arange(n, device="cuda")) / n
        indices = torch.searchsorted(weights[rows].cumsum(dim=-1).contiguous(), positions.double().contiguous())
        x[rows] = x[rows[:, None], indices]
        v[rows] = v[rows[:, None], indices]
        logw[rows] = 0
    return ess, len(rows)


def run_particles(model, method, seed, repeats):
    n, d = method["particles"], model.dimension
    initial, pilot, value_rng, path, selection = [model.generator(seed + offset) for offset in (0, 100003, 200003, 300007, 400009)]
    x = model.noise((repeats, n, d), initial)
    smc = method.get("smc", True)
    v = model.value(x, 0, method, value_rng) if smc else torch.zeros((repeats, n), device="cuda", dtype=torch.float64)
    logw = v.clone()
    ess_values, resampling_count = [], 0
    max_displacement_kl = 0.0
    for index in range(model.steps):
        if smc:
            ess, resampled = select_ancestors(x, v, logw, selection, method.get("ess_threshold", 0.5))
            ess_values.append(float((ess / n).mean()))
            resampling_count += resampled
        mean, std, _ = model.transition(x, index)
        proposal = method.get("proposal", "base")
        if proposal in {"pilot", "grouped"}:
            m = method["pilots"]
            eps = model.noise((*mean.shape[:-1], m, d), pilot)
            candidates = mean.unsqueeze(-2) + std * eps
            values = model.value(candidates, index + 1, method, value_rng)
            candidate_weights = torch.softmax(values, dim=-1)
            if proposal == "grouped":
                indices = torch.multinomial(candidate_weights.reshape(-1, m), 1, generator=selection).reshape(repeats, n)
                z = torch.gather(candidates, 2, indices[..., None, None].expand(-1, -1, 1, d)).squeeze(2)
                new_v = torch.gather(values, 2, indices[..., None]).squeeze(2)
                if smc:
                    logw += torch.logsumexp(values, dim=-1) - math.log(m) - v
                x, v = z, new_v
                continue
            delta = method.get("scale", 1.0) * std * (candidate_weights[..., None] * eps).sum(dim=-2).float()
        else:
            delta = torch.zeros_like(mean)
        z = mean + delta + std * model.noise(mean.shape, path)
        if smc:
            new_v = model.value(z, index + 1, method, value_rng)
            # The actual fresh Gaussian propagation is conditional on its pilots.
            density_ratio = (-(delta.double() * (z - mean).double()).sum(dim=-1) + 0.5 * delta.double().square().sum(dim=-1)) / std ** 2
            logw += density_ratio + new_v - v
            v = new_v
            max_displacement_kl = max(max_displacement_kl, float((0.5 * delta.double().square().sum(dim=-1) / std ** 2).max()))
        x = z
    weights = torch.softmax(logw, dim=-1) if smc else torch.full_like(logw, 1 / n)
    if smc or method.get("proposal") == "grouped":
        terminal_q = v * model.task["tau"]
    else:
        terminal_q = model.reward(x)
    return x, weights, {"mean_step_ess_fraction": float(np.mean(ess_values)) if ess_values else 1.0,
                        "final_ess_fraction": float((1 / weights.square().sum(dim=-1)).mean() / n),
                        "resamplings_per_repeat": resampling_count / repeats,
                        "max_proposal_kl": max_displacement_kl,
                        "mean_q": float((weights * terminal_q).sum(dim=-1).mean())}


def reference(model, config, row, output):
    generator = model.generator(config["reference_seed"] + row * 1009)
    samples = config["reference_samples"]
    x = model.rollout(model.noise((samples, model.dimension), generator), 0, generator)
    q = model.reward(x).double()
    weights = torch.softmax(q / model.task["tau"], dim=0)
    rng = np.random.default_rng(52001)
    random_directions = rng.normal(size=(4, model.dimension))
    random_directions /= np.linalg.norm(random_directions, axis=-1, keepdims=True)
    directions = torch.tensor(np.concatenate([np.eye(model.dimension), random_directions]), device="cuda", dtype=torch.float32)
    projected = x @ directions.T
    thresholds = torch.quantile(projected, torch.linspace(0.1, 0.9, 9, device="cuda"), dim=0).T
    indicators = projected[..., None] <= thresholds
    cdf = (weights[:, None, None] * indicators).sum(dim=0)
    half = samples // 2
    split_cdf = [(torch.softmax(q[part] / model.task["tau"], dim=0)[:, None, None] * indicators[part]).sum(dim=0)
                 for part in (slice(0, half), slice(half, None))]
    info = {"task": model.task["name"], "state_row": row, "samples": samples,
            "ess": float(1 / weights.square().sum()), "mean_q": float((weights * q).sum()),
            "split_cdf_mse": float((split_cdf[0] - split_cdf[1]).square().mean()),
            "directions": directions.cpu().tolist(), "thresholds": thresholds.cpu().tolist(),
            "cdf": cdf.cpu().tolist()}
    dump(output / f'{model.task["name"]}-state{row}-reference.json', info)
    return directions, thresholds, cdf, info


def evaluate(config, output):
    rows, counts = [], []
    for task in config["tasks"]:
        model = BatchCritic(task)
        model.load_calibration(Path(config["calibration_directory"]) / f'{task["name"]}-calibration.json')
        try:
            for state_row in config["state_rows"]:
                model.engine.use_state(state_row)
                directions, thresholds, target_cdf, ref_info = reference(model, config, state_row, output)
                dump(output / "counts.json", counts + [{"task": task["name"], **model.engine.counts()}])
                for seed in config["seeds"]:
                    for method in config["methods"]:
                        before = model.engine.counts()
                        started = time.perf_counter()
                        x, weights, diagnostics = run_particles(model, method, seed + state_row * 1009, config["repeats"])
                        projected = x @ directions.T
                        indicators = projected[..., None] <= thresholds
                        empirical = (weights[..., None, None] * indicators).sum(dim=1)
                        errors = (empirical - target_cdf).square().mean(dim=(-1, -2))
                        torch.cuda.synchronize()
                        after = model.engine.counts()
                        record = {"task": task["name"], "state_row": state_row, "seed": seed,
                                  "method": method["name"], "repeats": config["repeats"],
                                  "cdf_mse": float(errors.mean()),
                                  "cdf_mse_standard_error": float(errors.std() / math.sqrt(len(errors))),
                                  "pooled_cdf_mse": float((empirical.mean(dim=0) - target_cdf).square().mean()),
                                  "reference_ess": ref_info["ess"], "reference_split_cdf_mse": ref_info["split_cdf_mse"],
                                  **diagnostics, "seconds": time.perf_counter() - started,
                                  "costs": {key: after[key] - before[key] for key in after},
                                  "costs_per_repeat": {key: (after[key] - before[key]) / config["repeats"] for key in after}}
                        rows.append(record)
                        dump(output / "comparisons.json", rows)
                        dump(output / "counts.json", counts + [{"task": task["name"], **model.engine.counts()}])
                        print(json.dumps(record), flush=True)
        finally:
            counts.append({"task": task["name"], **model.engine.counts()})
            dump(output / "counts.json", counts)
        del model
        torch.cuda.empty_cache()
    return {"conditions": len(rows), "score_forwards": sum(c["score_forwards"] for c in counts),
            "reward_queries": sum(c["reward_queries"] for c in counts)}


def calibrate(config, output):
    summaries, counts = [], []
    for task in config["tasks"]:
        model = BatchCritic(task)
        before = time.perf_counter()
        try:
            checks = model.batch_check()
            residuals = [[] for _ in range(model.steps)]
            for row in config["state_rows"]:
                model.engine.use_state(row)
                generator = model.generator(config["seed"] + 1009 * row)
                x = model.noise((config["trajectories_per_state"], model.dimension), generator)
                final, clean_means = model.rollout(x, 0, generator, collect=True)
                for index, mean in enumerate(clean_means):
                    residuals[index].append((final - mean).cpu().double().numpy())
                dump(output / "counts.json", counts + [{"task": task["name"], "last_state_row": row,
                                                        **model.engine.counts()}])
            parameters = []
            for index, batches in enumerate(residuals):
                values = np.concatenate(batches)
                covariance = np.cov(values.T)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
                parameters.append({"index": index, "bias": values.mean(axis=0).tolist(),
                                   "covariance": covariance.tolist(), "root": root.tolist(),
                                   "residual_rms": float(np.sqrt(np.mean(values ** 2))),
                                   "samples": len(values)})
            result = {"task": task["name"], "state_rows": config["state_rows"], "seed": config["seed"],
                      "parameters": parameters, "checks": checks, "models": model.engine.provenance,
                      "counts": model.engine.counts(), "seconds": time.perf_counter() - before,
                      "interpretation": "Pooled calibration-state continuation residuals; a Gaussian proxy, not an exact conditional posterior."}
            dump(output / f'{task["name"]}-calibration.json', result)
            summaries.append(result)
        finally:
            counts.append({"task": task["name"], **model.engine.counts()})
            dump(output / "counts.json", counts)
        del model
        torch.cuda.empty_cache()
    return {"tasks": len(summaries), "score_forwards": sum(c["score_forwards"] for c in counts),
            "reward_queries": sum(c["reward_queries"] for c in counts),
            "batch_checks_passed": len(summaries)}
