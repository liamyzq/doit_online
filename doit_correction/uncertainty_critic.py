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
