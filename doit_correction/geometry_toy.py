"""Analytic VE posteriors with a fixed Gaussian DOIT transition."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm

from .geometry import gaussian_probes, response_matrix, ratio_estimate, error_summary


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Toy:
    def __init__(self, kind, dimension=12):
        self.kind, self.dimension = kind, dimension
        rng = np.random.default_rng(63001)
        self.rotation = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
        variances = np.geomspace(0.03, 3.0, dimension)
        self.cov = (self.rotation * variances) @ self.rotation.T
        self.means = np.zeros((1, dimension))
        if kind == "mixture":
            direction = self.rotation[:, dimension // 2]
            self.means = np.stack([-1.5 * direction, 1.5 * direction])
        self.t, self.delta = 1.0, 0.2
        self.next_t = self.t - self.delta
        self.x = self.rotation @ np.linspace(-0.15, 0.15, dimension)
        denoised, covariance, _, _, _ = self.posterior(self.x, self.t)
        self.score = (denoised - self.x) / self.t
        self.mean = self.x + self.delta * self.score
        self.std = np.sqrt(self.delta)
        self.jmu = np.eye(dimension) + self.delta * (covariance / self.t - np.eye(dimension)) / self.t
        self.center, covariance, _, _, _ = self.posterior(self.mean, self.next_t)
        self.response = covariance / self.next_t
        values, vectors = np.linalg.eigh(self.response)
        self.values, self.vectors = values[::-1], vectors[:, ::-1]
        self.affine = self.cov @ np.linalg.inv(self.cov + self.next_t * np.eye(dimension))

    def posterior(self, x, t):
        inverse = np.linalg.inv(self.cov + t * np.eye(self.dimension))
        a = self.cov @ inverse
        conditional_cov = t * a
        component_means = np.asarray(x)[..., None, :] @ a.T + self.means @ (np.eye(self.dimension) - a).T
        delta = np.asarray(x)[..., None, :] - self.means
        logits = -0.5 * np.einsum("...kd,de,...ke->...k", delta, inverse, delta)
        logp = logits - logsumexp(logits, axis=-1, keepdims=True)
        p = np.exp(logp)
        mean = np.sum(p[..., None] * component_means, axis=-2)
        differences = component_means - mean[..., None, :]
        covariance = conditional_cov + np.einsum("...k,...kd,...ke->...de", p, differences, differences)
        return mean, covariance, component_means, conditional_cov, logp

    def set_reward(self, alignment):
        indices = {"top": 0, "bottom": self.dimension - 1, "middle": self.dimension // 2}
        if alignment == "random":
            self.reward_direction = np.random.default_rng(63007).normal(size=self.dimension)
            self.reward_direction /= np.linalg.norm(self.reward_direction)
        else:
            self.reward_direction = self.vectors[:, indices[alignment]]
        v = self.reward_direction
        scale = np.sqrt(v @ (self.next_t * self.response) @ v)
        self.width = 0.7 * scale + 0.15
        self.target = v @ self.center + 0.75 * scale

    def linear_map(self, proxy):
        if proxy == "frozen":
            return self.mean + self.next_t * self.score, np.eye(self.dimension)
        if proxy == "center_identity":
            return self.center, np.eye(self.dimension)
        if proxy == "center_isotropic":
            return self.center, np.eye(self.dimension) * np.trace(self.response) / self.dimension
        if proxy.startswith("geometry"):
            return self.center, response_matrix(self.response, int(proxy.split("_")[-1]))
        if proxy == "full_linear":
            return self.center, self.response
        raise ValueError(proxy)

    def denoise(self, y, proxy):
        if proxy == "candidate":
            # The reward evaluation needs only the posterior mean, not covariance.
            return self.posterior_mean(y)
        center, matrix = self.linear_map(proxy)
        return center + (y - self.mean) @ matrix.T

    def posterior_mean(self, y):
        inverse = np.linalg.inv(self.cov + self.next_t * np.eye(self.dimension))
        component_means = np.asarray(y)[..., None, :] @ self.affine.T + self.means @ (np.eye(self.dimension) - self.affine).T
        delta = np.asarray(y)[..., None, :] - self.means
        logits = -0.5 * np.einsum("...kd,de,...ke->...k", delta, inverse, delta)
        p = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
        return np.sum(p[..., None] * component_means, axis=-2)

    def log_weight(self, eps, proxy):
        y = self.mean + self.std * eps
        v = self.reward_direction
        if proxy != "continuation":
            return -0.5 * np.square((self.denoise(y, proxy) @ v - self.target) / self.width)
        # Analytic integration of a scalar Gaussian quadratic reward, conditional
        # on each mixture component; posterior component probabilities depend on y.
        inverse = np.linalg.inv(self.cov + self.next_t * np.eye(self.dimension))
        a = self.affine
        component_scalar_means = (y @ (a.T @ v))[..., None] + self.means @ ((np.eye(self.dimension) - a).T @ v)
        delta = y[..., None, :] - self.means
        logits = -0.5 * np.einsum("...kd,de,...ke->...k", delta, inverse, delta)
        logp = logits - logsumexp(logits, axis=-1, keepdims=True)
        variance = v @ (self.next_t * a) @ v
        weights = 0.5 * np.log(self.width ** 2 / (self.width ** 2 + variance))
        weights = weights - 0.5 * np.square(component_scalar_means - self.target) / (self.width ** 2 + variance)
        return logsumexp(logp + weights, axis=-1)

    def reference(self, proxy, nodes=96):
        # The integrand depends on at most two Gaussian projections: the reward
        # projection and the posterior mixture log odds. All orthogonal moments
        # integrate to zero, avoiding a high-dimensional approximate reference.
        if proxy in {"candidate", "continuation"}:
            directions = [self.affine.T @ self.reward_direction]
            if len(self.means) > 1:
                directions.append(np.linalg.solve(self.cov + self.next_t * np.eye(self.dimension), self.means[1] - self.means[0]))
        else:
            _, matrix = self.linear_map(proxy)
            directions = [matrix.T @ self.reward_direction]
        u, s, _ = np.linalg.svd(np.stack(directions, axis=1), full_matrices=False)
        basis = u[:, s > 1e-10]
        nodes1, weights1 = roots_hermitenorm(nodes)
        weights1 /= np.sqrt(2 * np.pi)
        if basis.shape[1] == 1:
            coordinates, weights = nodes1[:, None], weights1
        else:
            x, y = np.meshgrid(nodes1, nodes1, indexing="ij")
            coordinates = np.stack([x.ravel(), y.ravel()], axis=1)
            weights = np.outer(weights1, weights1).ravel()
        eps = coordinates @ basis.T
        w = np.exp(self.log_weight(eps, proxy)) * weights
        ratio = w @ eps / w.sum()
        return {"ratio": ratio, "guidance": self.jmu.T @ ratio / self.std, "mass": float(w.sum()),
                "integrand_evaluations": len(eps), "component_integrals": len(eps) * len(self.means) if proxy == "continuation" else 0}


def run_checks(config, output):
    errors = {}
    for kind in ("gaussian", "mixture"):
        model = Toy(kind, config["dimension"])
        # A changed posterior formula would alter both geometry and references.
        h = 1e-5
        numeric = np.stack([(model.posterior(model.mean + h * e, model.next_t)[0] - model.posterior(model.mean - h * e, model.next_t)[0]) / (2 * h)
                            for e in np.eye(model.dimension)], axis=1)
        errors[kind + "_response_fd"] = float(np.max(np.abs(numeric - model.response)))
        for alignment in ("top", "bottom", "middle", "random"):
            model.set_reward(alignment)
            for proxy in ("frozen", "geometry_4", "candidate", "continuation"):
                first, second = model.reference(proxy, 96), model.reference(proxy, 160)
                errors[f"{kind}_{alignment}_{proxy}_quadrature"] = float(np.max(np.abs(first["guidance"] - second["guidance"])))
        if kind == "gaussian":
            model.set_reward("random")
            # Independent closed form for W(D(Y)) with a linear Gaussian map.
            b = model.std * model.response.T @ model.reward_direction
            offset = model.reward_direction @ model.center - model.target
            exact_ratio = -offset * b / (model.width ** 2 + b @ b)
            errors["gaussian_closed_ratio"] = float(np.max(np.abs(exact_ratio - model.reference("candidate")["ratio"])))
            eps = np.random.default_rng(63101).normal(size=(32, model.dimension))
            errors["gaussian_full_proxy"] = float(np.max(np.abs(model.denoise(model.mean + model.std * eps, "full_linear") - model.denoise(model.mean + model.std * eps, "candidate"))))
    rng = np.random.default_rng(63103)
    basis = np.linalg.qr(rng.normal(size=(12, 4)))[0]
    eps = gaussian_probes(rng, 8192, 16, 12, basis)
    mean = eps.mean(axis=(0, 1))
    covariance = eps.reshape(-1, 12).T @ eps.reshape(-1, 12) / eps.shape[0] / eps.shape[1]
    errors["gaussian_marginal_mean"] = float(np.max(np.abs(mean)))
    errors["gaussian_marginal_covariance"] = float(np.max(np.abs(covariance - np.eye(12))))
    projected = eps @ basis
    from scipy.special import ndtr
    bins = np.floor(ndtr(projected) * 16).astype(int)
    errors["stratum_occupancy"] = float(np.max(np.abs(np.sort(bins, axis=1) - np.arange(16)[None, :, None])))
    paired = gaussian_probes(rng, 32, 16, 12, basis, True)
    errors["antithetic_constant_weight"] = float(np.abs(paired.mean(axis=1)).max())
    tolerances = {"gaussian_marginal_mean": 0.015, "gaussian_marginal_covariance": 0.025}
    failed = {key: value for key, value in errors.items() if value > tolerances.get(key, 1e-7)}
    dump(output / "checks.json", {"errors": errors, "failed": failed})
    if failed:
        raise AssertionError(failed)
    return {"checks_passed": len(errors), "maximum_analytic_error": max(v for k, v in errors.items() if not k.startswith("gaussian_marginal"))}


def design_basis(model, method, rng, repeats, proxy):
    name = method["name"]
    if name in {"iid", "antithetic"}:
        return None, 0
    rank = method.get("rank", 1)
    if name.startswith("random"):
        return np.linalg.qr(rng.normal(size=(model.dimension, rank)))[0], 0
    if name.startswith("top"):
        return model.vectors[:, :rank], 0
    if name.startswith("bottom"):
        return model.vectors[:, -rank:], 0
    if name.startswith("extreme"):
        return model.vectors[:, [0, -1]], 0
    if name.startswith("pilot"):
        pilot_count = method["pilot_samples"]
        pilot_eps = gaussian_probes(rng, repeats, pilot_count, model.dimension, antithetic=True)
        signal = ratio_estimate(pilot_eps, model.log_weight(pilot_eps, proxy)) @ model.vectors
        selected = np.abs(signal).argmax(axis=1)
        return model.vectors[:, selected].T[:, :, None], pilot_count
    raise ValueError(name)


def evaluate(config, output):
    rows, proxy_rows, costs = [], [], {"reward_evaluations": 0, "pilot_reward_evaluations": 0,
                                      "reference_integrand_evaluations": 0, "analytic_component_integrals": 0}
    repeats = config["repeats"]
    started = time.perf_counter()
    for kind in config["models"]:
        model = Toy(kind, config["dimension"])
        for alignment in config["reward_alignments"]:
            model.set_reward(alignment)
            refs = {proxy: model.reference(proxy, config["quadrature_nodes"]) for proxy in config["proxies"]}
            costs["reference_integrand_evaluations"] += sum(ref["integrand_evaluations"] for ref in refs.values())
            costs["analytic_component_integrals"] += sum(ref["component_integrals"] for ref in refs.values())
            for seed in config["seeds"]:
                for samples in config["samples"]:
                    for mi, method in enumerate(config["methods"]):
                        rng = np.random.default_rng(seed + 101 * mi + 10007 * samples)
                        tick = time.perf_counter()
                        basis, pilots = design_basis(model, method, rng, repeats, "candidate")
                        eps = gaussian_probes(rng, repeats, samples - pilots, model.dimension, basis, method.get("antithetic", False))
                        ratio = ratio_estimate(eps, model.log_weight(eps, "candidate"))
                        guidance = ratio @ model.jmu / model.std
                        row = {"model": kind, "alignment": alignment, "seed": seed, "samples": samples,
                               "method": method["name"], "rank": method.get("rank", 0), "repeats": repeats,
                               "pilot_samples": pilots, "reward_evaluations": repeats * samples,
                               "seconds": time.perf_counter() - tick,
                               **error_summary(guidance, refs["candidate"]["guidance"])}
                        rows.append(row)
                        costs["reward_evaluations"] += repeats * samples
                        costs["pilot_reward_evaluations"] += repeats * pilots
                rng = np.random.default_rng(seed + 800009)
                eps = gaussian_probes(rng, repeats, config["proxy_samples"], model.dimension, antithetic=True)
                actual = model.denoise(model.mean + model.std * eps, "candidate")
                true_logs = model.log_weight(eps, "candidate")
                costs["reward_evaluations"] += repeats * config["proxy_samples"]
                for proxy in config["proxies"]:
                    logw = model.log_weight(eps, proxy)
                    guidance = ratio_estimate(eps, logw) @ model.jmu / model.std
                    row = {"model": kind, "alignment": alignment, "seed": seed, "proxy": proxy,
                           "samples": config["proxy_samples"], "repeats": repeats,
                           **error_summary(guidance, refs[proxy]["guidance"]),
                           "reference_bias2_to_continuation": float(np.square(refs[proxy]["guidance"] - refs["continuation"]["guidance"]).sum()),
                           "reference_bias2_to_candidate": float(np.square(refs[proxy]["guidance"] - refs["candidate"]["guidance"]).sum()),
                           "total_mse_to_continuation": float(np.square(guidance - refs["continuation"]["guidance"]).sum(axis=-1).mean()),
                           "log_weight_mse_to_candidate": float(np.square(logw - true_logs).mean())}
                    if proxy != "continuation":
                        row["denoiser_mse"] = float(np.square(model.denoise(model.mean + model.std * eps, proxy) - actual).sum(axis=-1).mean())
                    proxy_rows.append(row)
                    if proxy == "continuation":
                        costs["analytic_component_integrals"] += repeats * config["proxy_samples"] * len(model.means)
                    else:
                        costs["reward_evaluations"] += repeats * config["proxy_samples"]
            dump(output / "probe-results.json", rows)
            dump(output / "proxy-results.json", proxy_rows)
            print(json.dumps({"model": kind, "alignment": alignment, "conditions": len(rows) + len(proxy_rows)}), flush=True)
    metrics = {"probe_conditions": len(rows), "proxy_conditions": len(proxy_rows), "costs": costs,
               "seconds": time.perf_counter() - started,
               "target": "Fixed Gaussian reverse kernel with separate proxy and analytic-continuation integrands"}
    dump(output / "summary.json", metrics)
    return metrics
