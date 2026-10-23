"""Black-box critic evaluation of reward-calibrated probe directions."""

from __future__ import annotations

import time

import numpy as np
import torch

from .complement import stream
from .complement_critic import continue_group
from .geometry_critic import denoise
from .geometry_toy import dump
from .reward_directions import direction_pool, group_ids
from .reward_direction_study import Study
from .uncertainty_critic import BatchCritic


def generator(model, seed_words):
    return model.generator(int(stream(*seed_words).integers(0, 2**63 - 1)))


class CriticContext:
    def __init__(self, model, x, index, pool_seed):
        self.model, self.index, self.dimension = model, index, model.dimension
        self.x = x.detach().cpu().double().numpy()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        transition = model.engine.transition(x, index, True)
        self.jmu = model.engine.jacobian(transition).detach().cpu().double().numpy()
        torch.cuda.synchronize()
        self.jacobian_seconds = time.perf_counter() - tick
        self.mean = transition.mean.detach().cpu().double().numpy()
        self.std, self.gain = transition.noise_std, transition.correction_gain
        tick = time.perf_counter()
        self.pool, self.eigenvalues = direction_pool(self.jmu, pool_seed)
        self.pool_seconds = time.perf_counter() - tick

    def evaluate(self, eps, route, seed_words, groups, capacity):
        shape = eps.shape
        x = torch.as_tensor(self.mean + self.std * eps, dtype=torch.float32, device="cuda")
        if route == "P":
            flat = x.reshape(-1, self.dimension)
            clean = torch.cat([denoise(self.model, batch, self.index + 1) for batch in flat.split(self.model.chunk)])
            x = clean.reshape(shape)
        elif route == "F":
            rng = generator(self.model, seed_words)
            ids = torch.as_tensor(groups, device="cuda", dtype=torch.long)
            rows = torch.arange(len(eps), device="cuda")[:, None]
            for index in range(self.index + 1, self.model.steps):
                mean, std, _ = self.model.transition(x, index)
                noise = self.model.noise((len(eps), capacity, self.dimension), rng)
                x = mean + std * noise[rows, ids]
        else:
            raise ValueError(route)
        return (self.model.reward(x).double() / self.model.task["tau"]).cpu().numpy()

    def analytic_reference(self, route):
        return None

    def counts(self):
        return self.model.engine.counts()

    def describe(self):
        return {"index": self.index, "sigma": self.std, "gain": self.gain,
                "kernel_singular_values2": self.eigenvalues.tolist(),
                "jacobian_seconds": self.jacobian_seconds, "pool_seconds": self.pool_seconds,
                "actual_context_jacobian_vjps": self.dimension,
                "deployed_geometry_vjps_per_estimate": self.dimension,
                "baseline_original_kernel_vjps_per_estimate": 1,
                "active_guidance": True, "reward_gradient_calls": 0}


def smoke(context):
    """New variable group-ID evaluator must match preserved fixed sign groups."""
    model, d = context.model, context.dimension
    repeats, count, capacity = 4, 8, 16
    eps = stream(83101).normal(size=(repeats, count, d))
    ids2, ids4 = group_ids(repeats, count, 2), group_ids(repeats, count, 4)
    mixed = ids2.copy()
    mixed[1::2] = ids4[1::2]
    words = (83103, d)
    before = model.engine.counts()
    actual = context.evaluate(eps, "F", words, mixed, capacity)
    after = model.engine.counts()
    expected = []
    for group in (2, 4):
        x = torch.as_tensor(context.mean + context.std * eps, device="cuda", dtype=torch.float32)
        clean = continue_group(model, x, context.index + 1, generator(model, words), group, capacity)
        expected.append((model.reward(clean).double() / model.task["tau"]).cpu().numpy())
    expected[0][1::2] = expected[1][1::2]
    errors = {"mixed_group_matches_preserved_adapter": float(np.max(np.abs(actual - expected[0]))),
              "full_reward_cost": abs(after["reward_queries"] - before["reward_queries"] - repeats * count),
              "full_actor_cost": abs(after["score_forwards"] - before["score_forwards"] -
                                     repeats * count * (model.steps - context.index - 1))}
    before = model.engine.counts()
    context.evaluate(eps, "P", words, mixed, capacity)
    after = model.engine.counts()
    errors["proxy_reward_cost"] = abs(after["reward_queries"] - before["reward_queries"] - repeats * count)
    errors["proxy_actor_cost"] = abs(after["score_forwards"] - before["score_forwards"] - repeats * count)
    if max(errors.values()) > 1e-6:
        raise AssertionError(errors)
    return errors


def evaluate(config, output):
    study, totals, checks = Study(config, output), {}, {}
    jacobian_vjps, jacobian_seconds = 0, 0.0
    for task in config["tasks"]:
        task_word = 0 if task["name"].startswith("hopper") else 1
        model = BatchCritic(task, config["batch"])
        try:
            dump(output / f"{task['name']}-model.json", model.engine.provenance)
            dump(output / f"{task['name']}-selection.json", model.engine.selection)
            for row in config["state_rows"]:
                model.engine.use_state(row)
                rng = generator(model, (config["path_seed"], task_word, row))
                x = model.noise((model.dimension,), rng)
                for index in range(max(config["indices"]) + 1):
                    if index in config["indices"]:
                        if not 0 < index <= task["guidance_end_index"]:
                            raise ValueError("Configured state is outside the active guidance interval")
                        context = CriticContext(model, x, index, config["pool_seed"])
                        jacobian_vjps += model.dimension
                        jacobian_seconds += context.jacobian_seconds
                        if config.get("smoke"):
                            checks[f"{task['name']}/{row}/{index}"] = smoke(context)
                        # Stable task words survive task-wise queue partitioning.
                        study.context(context, (task_word, row, index),
                                      {"task": task["name"], "state_row": row, "index": index})
                    mean, std, _ = model.transition(x, index)
                    x = mean + std * model.noise(mean.shape, rng)
        finally:
            for key, value in model.engine.counts().items():
                totals[key] = totals.get(key, 0) + value
            dump(output / "costs.json", {"actual_costs": totals, "jacobian_vjps": jacobian_vjps,
                                         "jacobian_seconds": jacobian_seconds})
            del model
            torch.cuda.empty_cache()
    dump(output / "checks.json", checks)
    np.savez_compressed(output / "estimates.npz", **study.estimates)
    metrics = {"conditions": len(study.rows), "contexts": len(study.contexts), "costs": totals,
               "actual_jacobian_vjp_calls": jacobian_vjps, "jacobian_seconds": jacobian_seconds,
               "seconds": time.perf_counter() - study.started,
               "scope": "Fixed authentic actors and black-box critics; no critic differentiation or simulator returns"}
    dump(output / "summary.json", metrics)
    return metrics
