"""Complete-action evaluation of the locked reward-direction constructions."""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .complement import stream
from .geometry_toy import dump
from .reward_direction_critic import CriticContext, generator
from .reward_directions import direction_pool, estimate, select
from .uncertainty_critic import BatchCritic


class ActionContext(CriticContext):
    """Independent action states in one actor batch; free probes pay one VJP."""

    def __init__(self, model, x, index, pool_seed, geometry):
        self.model, self.index, self.dimension = model, index, model.dimension
        self.input = x.detach().requires_grad_(True)
        engine = model.engine
        s, t = engine.timesteps[index:index + 1], engine.timesteps[index + 1:index + 2]
        alpha_s, alpha_t = float(engine.schedule.marginal_alpha(s)), float(engine.schedule.marginal_alpha(t))
        sigma_s, sigma_t = float(engine.schedule.marginal_std(s)), float(engine.schedule.marginal_std(t))
        noise2 = model.task["eta"]**2 * sigma_t**2 / sigma_s**2 * (1 - alpha_s**2 / alpha_t**2)
        self.std = math.sqrt(max(noise2, 0) + 1e-12)
        a = alpha_t / alpha_s
        b = math.sqrt(max(sigma_t**2 - noise2, 1e-12)) - a * sigma_s
        epsilon = -engine.model(self.input, s.expand(len(x)), engine.state.expand(len(x), -1)) * sigma_s
        engine.score_forwards += len(x)
        self.graph_mean = a * self.input + b * epsilon
        self.mean = self.graph_mean.detach().cpu().double().numpy()[:, None, :]
        self.gain = -b * sigma_s
        self.geometry = geometry
        self.jacobian_vjps, self.jacobian_seconds = 0, 0.0
        if geometry:
            torch.cuda.synchronize()
            tick = time.perf_counter()
            rows = []
            for coordinate in range(self.dimension):
                v = torch.zeros_like(x)
                v[:, coordinate] = 1
                rows.append(torch.autograd.grad(self.graph_mean, self.input, v,
                            retain_graph=coordinate + 1 < self.dimension)[0])
            self.jmu = torch.stack(rows, dim=1).detach().cpu().double().numpy()
            torch.cuda.synchronize()
            self.jacobian_seconds = time.perf_counter() - tick
            self.jacobian_vjps = len(x) * self.dimension
            self.pool, self.eigenvalues = direction_pool(self.jmu, pool_seed)
            model.endpoint_jacobian_vjps += self.jacobian_vjps
            model.endpoint_jacobian_seconds += self.jacobian_seconds
        else:
            # The common estimator returns a noise score, mapped below by one VJP.
            self.jmu = np.eye(self.dimension)

    def correction(self, value):
        if self.geometry:
            return torch.as_tensor(value, device="cuda", dtype=torch.float32)
        tick = time.perf_counter()
        result = torch.autograd.grad(self.graph_mean, self.input,
                 torch.as_tensor(value, device="cuda", dtype=torch.float32))[0]
        torch.cuda.synchronize()
        self.model.engine.vjp_seconds += time.perf_counter() - tick
        self.model.endpoint_original_vjps += len(value)
        return result.detach()


def action_statistics(actions, q, candidates):
    actions, q = np.asarray(actions), np.asarray(q)
    groups = q.reshape(-1, candidates)
    chosen = np.arange(len(groups)) * candidates + groups.argmax(axis=1)
    selected, selected_q = actions[chosen], q[chosen]

    def diversity(x):
        if len(x) < 2:
            return 0.0
        distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        return float(distance.sum() / (len(x) * (len(x) - 1)))

    return {"mean_q": float(q.mean()), "q10": float(np.quantile(q, 0.1)),
            "mean_pairwise_action_distance": diversity(actions),
            "selected_mean_q": float(selected_q.mean()), "selected_q10": float(np.quantile(selected_q, 0.1)),
            "selected_pairwise_action_distance": diversity(selected),
            "selected_indices": chosen.tolist()}


def sample_actions(model, config, method, row, output):
    task_word = 0 if model.task["name"].startswith("hopper") else 1
    n = config["actions"]
    path = generator(model, (config["path_seed"], task_word, row))
    x = model.noise((n, model.dimension), path)
    trajectory = [x.cpu().numpy()]
    jacobian_vjps, jacobian_seconds, original_vjps = 0, 0.0, 0
    artifact_seconds = 0.0
    for index in range(model.steps):
        if 0 < index <= model.task["guidance_end_index"]:
            geometry = method["selector"] in {"moments", "first", "no_j", "model"}
            context = ActionContext(model, x, index, config["pool_seed"], geometry)
            value, arrays, info = estimate(context, method, method["route"], method["samples"], n,
                (config["seed"], task_word, row, index, method["samples"]), config)
            guidance = context.correction(value)
            arrays["guidance"] = guidance.cpu().double().numpy()
            if not geometry:
                arrays["main_only_guidance"] = arrays["guidance"]
                original_vjps += n
            jacobian_vjps += context.jacobian_vjps
            jacobian_seconds += context.jacobian_seconds
            mean = context.graph_mean.detach() + model.task["gamma"] * context.gain * guidance
            std = context.std
            arrays.update(state=x.cpu().numpy(), mean=context.mean[:, 0],
                          correction_gain=np.array(context.gain), sigma=np.array(std))
            if geometry:
                arrays.update(jmu=context.jmu, pool=context.pool)
            tick = time.perf_counter()
            np.savez_compressed(output / f"step{index:02d}.npz", **arrays)
            dump(output / f"step{index:02d}.json", info)
            artifact_seconds += time.perf_counter() - tick
            del context
        else:
            mean, std, _ = model.transition(x, index)
        x = mean + std * model.noise(mean.shape, path)
        trajectory.append(x.cpu().numpy())
    q = model.reward(x).cpu().numpy()
    actions = x.cpu().numpy()
    tick = time.perf_counter()
    np.savez_compressed(output / "actions.npz", actions=actions, q=q, trajectory=np.stack(trajectory))
    artifact_seconds += time.perf_counter() - tick
    return {**action_statistics(actions, q, model.task["action_candidates"]),
            "jacobian_vjps": jacobian_vjps, "jacobian_seconds": jacobian_seconds,
            "original_kernel_vjps": original_vjps, "artifact_seconds": artifact_seconds}


def smoke(model, config):
    """Detect batch coupling, J orientation, or per-state pilot-selection errors."""
    model.engine.use_state(config["state_rows"][0])
    d, n, index = model.dimension, 4, 3
    x = torch.linspace(-0.7, 0.9, n * d, device="cuda").reshape(n, d)
    context = ActionContext(model, x, index, config["pool_seed"], True)
    scalar = [CriticContext(model, row, index, config["pool_seed"]) for row in x]
    model.endpoint_jacobian_vjps += n * d
    model.endpoint_jacobian_seconds += sum(c.jacobian_seconds for c in scalar)
    errors = {
        "batched_transition_mean": float(np.max(np.abs(context.mean[:, 0] - np.stack([c.mean for c in scalar])))),
        "batched_transition_jacobian": float(np.max(np.abs(context.jmu - np.stack([c.jmu for c in scalar])))),
        "batched_transition_std": max(abs(context.std - c.std) for c in scalar),
    }
    eps = stream(87001).normal(size=(n, 16, d))
    eps[:, 8:] = -eps[:, :8]
    logs = stream(87003).normal(size=(n, 16))
    actual, info = select(eps, logs, context.pool, context.jmu, context.std,
                          agreement_threshold=config["agreement_threshold"])
    expected = [select(eps[i:i + 1], logs[i:i + 1], context.pool[i], context.jmu[i], context.std,
                agreement_threshold=config["agreement_threshold"]) for i in range(n)]
    errors["batched_direction_selection"] = float(np.max(np.abs(actual - np.concatenate([v[0] for v in expected]))))
    errors["batched_moment_scores"] = float(np.max(np.abs(info["scores"] - np.concatenate([v[1]["scores"] for v in expected]))))
    errors["batched_enabled"] = int(np.count_nonzero(info["enabled"] != np.concatenate([v[1]["enabled"] for v in expected])))
    ratio = stream(87005).normal(size=(n, d))
    free = ActionContext(model, x, index, config["pool_seed"], False)
    mapped = free.correction(ratio / free.std).cpu().double().numpy()
    expected = np.einsum("ni,nij->nj", ratio, context.jmu) / context.std
    errors["batched_single_vjp"] = float(np.max(np.abs(mapped - expected)))
    # Repeated states expose estimator batching without changing its RNG layout.
    repeated = x[:1].expand(n, -1).clone()
    for route in ("P", "F"):
        for method in config["smoke_methods"]:
            batch = ActionContext(model, repeated, index, config["pool_seed"], True)
            reference = CriticContext(model, x[0], index, config["pool_seed"])
            model.endpoint_jacobian_vjps += d
            model.endpoint_jacobian_seconds += reference.jacobian_seconds
            words = (87007, index, method["samples"])
            value, _, _ = estimate(batch, method, route, method["samples"], n, words, config)
            target, _, _ = estimate(reference, method, route, method["samples"], n, words, config)
            errors[f"{route}_{method['name']}_relative_error"] = float(np.max(np.abs(value - target)) / max(1, np.max(np.abs(target))))
    failed = {key: value for key, value in errors.items() if value > 2e-4}
    if failed:
        raise AssertionError(failed)
    return errors


def evaluate(config, output):
    rows, checks, totals = [], {}, {}
    total_jacobian_vjps, total_original_vjps, total_jacobian_seconds = 0, 0, 0.0
    for task in config["tasks"]:
        model = BatchCritic(task, config["batch"])
        model.endpoint_jacobian_vjps = model.endpoint_original_vjps = 0
        model.endpoint_jacobian_seconds = 0.0
        dump(output / f"{task['name']}-model.json", model.engine.provenance)
        dump(output / f"{task['name']}-selection.json", model.engine.selection)
        try:
            if config.get("smoke"):
                checks[task["name"]] = smoke(model, config)
                dump(output / "checks.json", checks)
            for row in config["state_rows"]:
                model.engine.use_state(row)
                for method in config["methods"]:
                    directory = output / "raw" / f"{task['name']}-row{row:03d}-{method['name']}"
                    directory.mkdir(parents=True)
                    before = model.engine.counts()
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    info = sample_actions(model, config, method, row, directory)
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - tick
                    after = model.engine.counts()
                    expected_rewards = config["actions"] * (method["samples"] * task["guidance_end_index"] + 1)
                    actual_rewards = after["reward_queries"] - before["reward_queries"]
                    if actual_rewards != expected_rewards:
                        raise AssertionError({"expected_rewards": expected_rewards, "actual_rewards": actual_rewards})
                    continuation_steps = sum(1 if method["route"] == "P" else model.steps - index - 1
                                             for index in range(1, task["guidance_end_index"] + 1))
                    expected_actor = config["actions"] * (model.steps + method["samples"] * continuation_steps)
                    actual_actor = after["score_forwards"] - before["score_forwards"]
                    if actual_actor != expected_actor:
                        raise AssertionError({"expected_actor": expected_actor, "actual_actor": actual_actor})
                    rows.append({"task": task["name"], "state_row": row, "method": method["name"],
                        "actions": config["actions"], "selected_actions": config["actions"] // task["action_candidates"],
                        "route": method["route"], "samples": method["samples"], **info, "seconds": seconds,
                        "compute_seconds": seconds - info["artifact_seconds"],
                        "costs": {key: after[key] - before[key] for key in before}})
                    dump(output / "results.json", rows)
                    print(str({"task": task["name"], "row": row, "method": method["name"]}), flush=True)
        finally:
            for key, value in model.engine.counts().items():
                totals[key] = totals.get(key, 0) + value
            total_jacobian_vjps += model.endpoint_jacobian_vjps
            total_original_vjps += model.endpoint_original_vjps
            total_jacobian_seconds += model.endpoint_jacobian_seconds
            dump(output / "costs.json", {**totals, "actual_jacobian_vjps": total_jacobian_vjps,
                 "actual_original_kernel_vjps": total_original_vjps, "actual_jacobian_seconds": total_jacobian_seconds})
            del model
            torch.cuda.empty_cache()
    metrics = {"conditions": len(rows), "actions": sum(r["actions"] for r in rows), "costs": totals,
        "jacobian_vjps": sum(r["jacobian_vjps"] for r in rows),
        "original_kernel_vjps": sum(r["original_kernel_vjps"] for r in rows),
        "jacobian_seconds": sum(r["jacobian_seconds"] for r in rows),
        "checks_passed": sum(len(v) for v in checks.values()),
        "actual_jacobian_vjps_including_checks": total_jacobian_vjps,
        "actual_original_kernel_vjps_including_checks": total_original_vjps,
        "actual_jacobian_seconds_including_checks": total_jacobian_seconds,
        "scope": "Complete action generation and frozen-critic scores; not simulator returns"}
    dump(output / "summary.json", metrics)
    return metrics
