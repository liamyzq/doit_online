"""Fixed-state geometry probes using the preserved real RL actors and critics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .geometry import gaussian_probes, response_matrix, ratio_estimate, error_summary
from .uncertainty_critic import BatchCritic


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def denoise(model, y, index, with_grad=False):
    engine = model.engine
    s = engine.timesteps[index:index + 1]
    alpha = float(engine.schedule.marginal_alpha(s))
    sigma = float(engine.schedule.marginal_std(s))
    with torch.set_grad_enabled(with_grad):
        eps = -engine.model(y, s.expand(len(y)), engine.state.expand(len(y), -1)) * sigma
        engine.score_forwards += len(y)
        result = (y - sigma * eps) / alpha
    return result


class Context:
    def __init__(self, model, x, index):
        self.model, self.index, self.dimension = model, index, model.dimension
        engine = model.engine
        transition = engine.transition(x, index, True)
        self.transition = transition
        self.jmu = engine.jacobian(transition).detach().cpu().double().numpy()
        self.vjps = model.dimension
        y = transition.mean.detach().reshape(1, -1).requires_grad_(True)
        clean = denoise(model, y, index + 1, True)
        self.center = clean.detach().flatten().cpu().double().numpy()
        basis = torch.eye(model.dimension, device="cuda")
        rows = [torch.autograd.grad(clean, y, e[None], retain_graph=True)[0].flatten() for e in basis]
        self.response = torch.stack(rows).detach().cpu().double().numpy()
        self.vjps += model.dimension
        self.symmetric = (self.response + self.response.T) / 2
        self.values, self.vectors = np.linalg.eigh(self.symmetric)
        self.values, self.vectors = self.values[::-1], self.vectors[:, ::-1]
        self.mean = transition.mean.detach().cpu().double().numpy()
        self.frozen_center = transition.endpoint_mean.cpu().double().numpy()
        self.std = transition.noise_std
        self.frozen_response = 1 / transition.alpha_t
        # A local finite difference detects a new clean-time/sign error. This
        # is used only in smoke, not repeated for every formal condition.

    def describe(self):
        return {"eigenvalues": self.values.tolist(),
                "jacobian_asymmetry_fraction": float(np.linalg.norm(self.response - self.response.T) / max(np.linalg.norm(self.response), 1e-30)),
                "response_norm": float(np.linalg.norm(self.response)), "sigma": self.std,
                "frozen_response": self.frozen_response, "geometry_vjps": self.vjps}

    def linear(self, proxy):
        d = self.dimension
        if proxy == "frozen":
            return self.frozen_center, self.frozen_response * np.eye(d)
        if proxy == "center_identity":
            return self.center, self.frozen_response * np.eye(d)
        if proxy == "center_isotropic":
            return self.center, np.trace(self.response) / d * np.eye(d)
        if proxy.startswith("geometry_"):
            return self.center, response_matrix(self.response, min(int(proxy.split("_")[-1]), d))
        if proxy == "full_linear":
            return self.center, self.response
        raise ValueError(proxy)

    def clean(self, eps, proxy):
        eps = np.asarray(eps)
        original_shape = eps.shape
        flat = eps.reshape(-1, self.dimension)
        if proxy in {"candidate", "continuation"}:
            y = torch.as_tensor(self.mean + self.std * flat, dtype=torch.float32, device="cuda")
            if proxy == "candidate":
                result = torch.cat([denoise(self.model, batch, self.index + 1) for batch in y.split(self.model.chunk)])
            else:
                raise ValueError("Continuation needs its independent random stream")
            return result.reshape(original_shape)
        center, matrix = self.linear(proxy)
        return torch.as_tensor(center + self.std * flat @ matrix.T, dtype=torch.float32, device="cuda").reshape(original_shape)

    def log_weight(self, eps, proxy, generator):
        shape = eps.shape[:-1]
        if proxy == "continuation":
            y = torch.as_tensor(self.mean + self.std * eps, dtype=torch.float32, device="cuda")
            clean = self.model.rollout(y, self.index + 1, generator)
        else:
            clean = self.clean(eps, proxy)
        return (self.model.reward(clean).double() / self.model.task["tau"]).cpu().numpy().reshape(shape)

    def guidance(self, ratio):
        return ratio @ self.jmu / self.std


def reference(context, config, rng, generator):
    n, d = config["reference_samples"], context.dimension
    # IID references, independent of all estimators and direction-selection pilots.
    eps = gaussian_probes(rng, 1, n, d)[0]
    result = {}
    for proxy in config["proxies"]:
        logs = context.log_weight(eps, proxy, generator)
        ratio = ratio_estimate(eps[None], logs[None])[0]
        halves = np.stack([ratio_estimate(e[None], w[None])[0]
                           for e, w in zip(np.split(eps, 2), np.split(logs, 2))])
        weights = np.exp(logs - logs.max()); weights /= weights.sum()
        guidance, split = context.guidance(ratio), context.guidance(halves)
        result[proxy] = {"guidance": guidance.tolist(), "ratio": ratio.tolist(),
                         "ess": float(1 / np.square(weights).sum()),
                         "split_guidance_difference2": float(np.square(split[0] - split[1]).sum()),
                         "estimated_reference_mse": float(np.square(split[0] - split[1]).sum() / 4),
                         "log_mean_weight": float(logs.max() + np.log(np.exp(logs - logs.max()).mean()))}
    return result


def select_basis(context, method, rng, repeats, proxy, generator):
    name, d = method["name"], context.dimension
    rank = min(method.get("rank", 1), d)
    if name in {"iid", "antithetic"}:
        return None, 0
    if name.startswith("random"):
        return np.linalg.qr(rng.normal(size=(d, rank)))[0], 0
    if name.startswith("top"):
        return context.vectors[:, :rank], 0
    if name.startswith("bottom"):
        return context.vectors[:, -rank:], 0
    if name.startswith("pilot"):
        pilots = method["pilot_samples"]
        eps = gaussian_probes(rng, repeats, pilots, d, antithetic=True)
        signal = ratio_estimate(eps, context.log_weight(eps, proxy, generator)) @ context.vectors
        chosen = np.abs(signal).argmax(axis=1)
        return context.vectors[:, chosen].T[:, :, None], pilots
    raise ValueError(name)


def evaluate(config, output):
    rows, proxies, contexts, cost_rows = [], [], [], []
    checks, total_vjps = {}, 0
    started = time.perf_counter()
    for task in config["tasks"]:
        model = BatchCritic(task, config.get("batch", 2048))
        try:
            dump(output / f'{task["name"]}-model.json', model.engine.provenance)
            for row in config["state_rows"]:
                model.engine.use_state(row)
                path_rng = model.generator(config["path_seed"] + row)
                x = model.noise((model.dimension,), path_rng)
                states = {}
                for index in range(max(config["indices"]) + 1):
                    if index in config["indices"]:
                        states[index] = x.clone()
                    mean, std, _ = model.transition(x, index)
                    x = mean + std * model.noise(mean.shape, path_rng)
                for index, x in states.items():
                    before = model.engine.counts()
                    tick = time.perf_counter()
                    context = Context(model, x, index)
                    geometry_seconds = time.perf_counter() - tick
                    total_vjps += context.vjps
                    info = {"task": task["name"], "state_row": row, "index": index,
                            "geometry_seconds": geometry_seconds, **context.describe()}
                    if config.get("smoke", False):
                        direction = torch.arange(1, model.dimension + 1, device="cuda", dtype=torch.float32)
                        direction /= direction.norm()
                        center = torch.as_tensor(context.mean, dtype=torch.float32, device="cuda")[None]
                        h = 0.002
                        numeric = ((denoise(model, center + h * direction, index + 1) - denoise(model, center - h * direction, index + 1)) / (2 * h)).flatten().cpu().numpy()
                        analytic = context.response @ direction.cpu().numpy()
                        error = float(np.linalg.norm(numeric - analytic) / max(np.linalg.norm(analytic), 1e-8))
                        checks[f'{task["name"]}-row{row}-index{index}'] = error
                        if error > 0.02:
                            raise AssertionError({"clean_response_fd_error": error})
                    refs = reference(context, config, np.random.default_rng(config["reference_seed"] + row * 101 + index),
                                     model.generator(config["reference_seed"] + row * 1009 + index))
                    info["references"] = refs
                    contexts.append(info)
                    for seed in config["seeds"]:
                        for integrand in config["probe_integrands"]:
                            for m in config["samples"]:
                                for mi, method in enumerate(config["methods"]):
                                    rng = np.random.default_rng(seed + row * 1009 + index * 97 + m * 103 + mi)
                                    futures = model.generator(seed + row * 1009 + index * 97 + m * 103 + mi + 900001)
                                    counts_before = model.engine.counts()
                                    tick = time.perf_counter()
                                    basis, pilot_count = select_basis(context, method, rng, config["repeats"], integrand, futures)
                                    eps = gaussian_probes(rng, config["repeats"], m - pilot_count, model.dimension, basis, method.get("antithetic", False))
                                    logs = context.log_weight(eps, integrand, futures)
                                    estimates = context.guidance(ratio_estimate(eps, logs))
                                    counts_after = model.engine.counts()
                                    rows.append({"task": task["name"], "state_row": row, "index": index, "seed": seed,
                                                 "integrand": integrand, "method": method["name"], "samples": m,
                                                 "pilot_samples": pilot_count, "repeats": config["repeats"],
                                                 "seconds": time.perf_counter() - tick,
                                                 "score_forwards": counts_after["score_forwards"] - counts_before["score_forwards"],
                                                 "reward_queries": counts_after["reward_queries"] - counts_before["reward_queries"],
                                                 **error_summary(estimates, np.array(refs[integrand]["guidance"]))})
                        eps = gaussian_probes(np.random.default_rng(seed + 700001 + row + index), config["repeats"], config["proxy_samples"], model.dimension, antithetic=True)
                        actual = context.clean(eps, "candidate").cpu().double().numpy()
                        target_ref = np.array(refs["continuation"]["guidance"])
                        for proxy in config["proxies"]:
                            logs = context.log_weight(eps, proxy, model.generator(seed + 800001 + row + index))
                            estimates = context.guidance(ratio_estimate(eps, logs))
                            reference_value = np.array(refs[proxy]["guidance"])
                            item = {"task": task["name"], "state_row": row, "index": index, "seed": seed, "proxy": proxy,
                                    **error_summary(estimates, reference_value),
                                    "reference_bias2_to_continuation": float(np.square(reference_value - target_ref).sum()),
                                    "total_mse_to_continuation": float(np.square(estimates - target_ref).sum(axis=-1).mean())}
                            if proxy != "continuation":
                                candidate = context.clean(eps, proxy).cpu().double().numpy()
                                item["denoiser_mse"] = float(np.square(candidate - actual).sum(axis=-1).mean())
                            proxies.append(item)
                    info["total_state_costs"] = {key: model.engine.counts()[key] - before[key] for key in before}
                    dump(output / "contexts.json", contexts)
                    dump(output / "probe-results.json", rows)
                    dump(output / "proxy-results.json", proxies)
                    print(json.dumps({"task": task["name"], "state_row": row, "index": index, "conditions": len(rows) + len(proxies)}), flush=True)
        finally:
            cost_rows.append({"task": task["name"], **model.engine.counts()})
            dump(output / "costs.json", {"tasks": cost_rows, "geometry_vjps": total_vjps})
            del model
            torch.cuda.empty_cache()
    metrics = {"probe_conditions": len(rows), "proxy_conditions": len(proxies), "contexts": len(contexts),
               "reward_queries": sum(row["reward_queries"] for row in cost_rows),
               "score_forwards": sum(row["score_forwards"] for row in cost_rows),
               "geometry_vjps": total_vjps, "new_response_checks": checks, "seconds": time.perf_counter() - started,
               "scope": "Frozen real actor/critic mechanism study, not simulator returns"}
    dump(output / "summary.json", metrics)
    return metrics
