"""Bounded complement-cancellation study with exact toy reference moments."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .complement import stream, sign_probes, grouped_draw, loss_summary, paired_difference
from .geometry import gaussian_probes, ratio_estimate
from .geometry_toy import Toy, design_basis, dump


def stochastic_log_weight(model, eps, rng, group, capacity):
    """One exact posterior terminal draw per candidate, with optional coupling."""
    repeats, samples, _ = eps.shape
    y = model.mean + model.std * eps
    inverse = np.linalg.inv(model.cov + model.next_t * np.eye(model.dimension))
    delta = y[..., None, :] - model.means
    logits = -0.5 * np.einsum("...kd,de,...ke->...k", delta, inverse, delta)
    p = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    uniforms = rng.random((repeats, capacity))[:, :samples // group]
    uniforms = np.concatenate([uniforms] * group, axis=1)
    component = (uniforms > p[..., 0]).astype(int) if len(model.means) > 1 else np.zeros_like(uniforms, dtype=int)
    v = model.reward_direction
    means = (y @ (model.affine.T @ v))[..., None] + model.means @ ((np.eye(model.dimension) - model.affine).T @ v)
    scalar = np.take_along_axis(means, component[..., None], axis=-1)[..., 0]
    variance = v @ (model.next_t * model.affine) @ v
    scalar += np.sqrt(variance) * grouped_draw(rng, repeats, samples, group, capacity=capacity)
    return -0.5 * np.square((scalar - model.target) / model.width)


def weights(model, eps, integrand, rng, group, capacity=None):
    capacity = eps.shape[1] if capacity is None else capacity
    if integrand.startswith("stochastic"):
        return stochastic_log_weight(model, eps, rng, group if integrand.endswith("shared") else 1, capacity)
    return model.log_weight(eps, integrand)


def choose_basis(model, method, rng, repeats):
    name, d = method["basis"], model.dimension
    rank = method.get("rank", 1)
    if name == "none":
        return None, 0
    if name == "spectral":
        return model.vectors[:, :rank], 0
    if name == "random":
        return np.linalg.qr(rng.normal(size=(repeats, d, rank)))[0], 0
    if name == "oracle":
        angle = np.deg2rad(method.get("angle_degrees", 0))
        return (np.cos(angle) * model.reward_direction + np.sin(angle) * model.vectors[:, -1])[:, None], 0
    if name == "pilot":
        # Fit a continuous direction from independent central log-weight
        # differences, rather than choosing a precomputed spectral axis.
        pilots = method["pilot_samples"]
        directions = np.linalg.qr(rng.normal(size=(repeats, d, pilots // 2)))[0]
        offsets = method["pilot_radius"] * directions.transpose(0, 2, 1)
        logs = model.log_weight(np.concatenate([offsets, -offsets], axis=1), "candidate")
        contrast = (logs[:, :pilots // 2] - logs[:, pilots // 2:]) / (2 * method["pilot_radius"])
        direction = np.einsum("ndr,nr->nd", directions, contrast)
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.where(norm > 1e-14, direction / np.maximum(norm, 1e-14), directions[:, :, 0])
        return direction[..., None], pilots
    raise ValueError(name)


def replay(config, output):
    """Reproduce only the old top-aligned M32 rows needed for new decomposition."""
    old = json.loads(Path(config["replay_config"]).read_text())
    recorded = json.loads(Path(config["replay_results"]).read_text())
    rows, arrays, largest, reference_evaluations = [], {}, 0.0, 0
    for kind in old["models"]:
        model = Toy(kind, old["dimension"])
        model.set_reward("top")
        ref = model.reference("candidate", old["quadrature_nodes"])
        reference = ref["guidance"]
        reference_evaluations += ref["integrand_evaluations"]
        for seed in old["seeds"]:
            for mi, method in enumerate(old["methods"]):
                if method["name"] not in config["replay_methods"]:
                    continue
                rng = np.random.default_rng(seed + 101 * mi + 10007 * 32)
                basis, pilots = design_basis(model, method, rng, old["repeats"], "candidate")
                eps = gaussian_probes(rng, old["repeats"], 32 - pilots, model.dimension, basis, method.get("antithetic", False))
                estimates = ratio_estimate(eps, model.log_weight(eps, "candidate")) @ model.jmu / model.std
                item = {"model": kind, "seed": seed, "method": method["name"],
                        **loss_summary(estimates, reference, model.reward_direction[:, None])}
                prior = next(x for x in recorded if x["model"] == kind and x["alignment"] == "top" and x["samples"] == 32 and x["seed"] == seed and x["method"] == method["name"])
                item["original_mse_discrepancy"] = abs(item["mse"] - prior["mse"])
                largest = max(largest, item["original_mse_discrepancy"])
                arrays[f"{kind}_{seed}_{method['name']}"] = estimates
                rows.append(item)
    dump(output / "replay.json", rows)
    np.savez_compressed(output / "replay-estimates.npz", **arrays)
    if largest > 1e-9:
        raise AssertionError({"original_mse_discrepancy": largest})
    return {"replay_conditions": len(rows), "replay_reward_evaluations": len(rows) * 32 * old["repeats"],
            "replay_reference_integrand_evaluations": reference_evaluations,
            "maximum_replay_discrepancy": largest}


def checks(config, output):
    errors = {}
    d, repeats = 12, 16384
    basis = np.linalg.qr(stream(70001, 0).normal(size=(d, 2)))[0]
    for lhs in (False, True):
        eps = sign_probes(stream(70001, int(lhs) + 1), repeats, 16, d, basis, True, lhs)
        # Fixed-index marginals, not the automatically zero sign-group mean.
        first = eps[:, 0]
        errors[f"marginal_mean_{lhs}"] = float(abs(first.mean(axis=0)).max())
        errors[f"marginal_covariance_{lhs}"] = float(abs(first.T @ first / repeats - np.eye(d)).max())
        errors[f"constant_cancellation_{lhs}"] = float(abs(eps.mean(axis=1)).max())
        logs = 0.7 * (eps @ basis[:, 0])
        ratio = ratio_estimate(eps, logs)
        complement = ratio - (ratio @ basis) @ basis.T
        errors[f"active_weight_complement_{lhs}"] = float(abs(complement).max())
        # A nonzero orthogonal reward gradient must remain in the numerator.
        direction = np.eye(d)[0] - basis @ basis[0]
        direction /= np.linalg.norm(direction)
        t = 0.15 * basis[:, 0] + 0.2 * direction
        unnormalized = np.einsum("nm,nmd->nd", np.exp(eps @ t - np.dot(t, t) / 2), eps) / 16
        errors[f"full_weight_moment_{lhs}"] = float(abs(unnormalized.mean(axis=0) - t).max())
    oracle_reward_calls, oracle_component_integrals = 0, 0
    for ki, kind in enumerate(("gaussian", "mixture")):
        model = Toy(kind, d)
        model.set_reward("top")
        eps = sign_probes(stream(70003, ki), 8, 16, d, model.reward_direction[:, None], True)
        for ii, integrand in enumerate(("candidate", "continuation", "stochastic_shared")):
            logs = weights(model, eps, integrand, stream(70007, ki, ii), 4)
            errors[f"oracle_complement_invariance_{kind}_{integrand}"] = float(max(abs(logs[:, :4] - logs[:, 4:8]).max(), abs(logs[:, 8:12] - logs[:, 12:]).max()))
            if integrand == "continuation":
                oracle_component_integrals += 128 * len(model.means)
            else:
                oracle_reward_calls += 128
    tolerances = {key: 0.04 if key.startswith("marginal") else 0.015 if key.startswith("full_weight") else 1e-10 for key in errors}
    failed = {key: value for key, value in errors.items() if value > tolerances[key]}
    dump(output / "checks.json", {"errors": errors, "tolerances": tolerances, "failed": failed})
    if failed:
        raise AssertionError(failed)
    return {"checks_passed": len(errors), "synthetic_moment_weight_evaluations": 2 * 2 * repeats * 16,
            "oracle_check_reward_evaluations": oracle_reward_calls,
            "oracle_check_analytic_component_integrals": oracle_component_integrals,
            **replay(config, output)}


def evaluate(config, output):
    started = time.perf_counter()
    rows, differences, arrays, references = [], [], {}, []
    costs = {"reward_evaluations": 0, "pilot_reward_evaluations": 0,
             "analytic_component_integrals": 0, "reference_integrand_evaluations": 0}
    repeats = config["repeats"]
    for ki, kind in enumerate(config["models"]):
        model = Toy(kind, config["dimension"])
        model.set_reward("top")
        refs = {name: model.reference(name, config["quadrature_nodes"]) for name in ("candidate", "continuation")}
        costs["reference_integrand_evaluations"] += sum(ref["integrand_evaluations"] for ref in refs.values())
        costs["analytic_component_integrals"] += sum(ref["component_integrals"] for ref in refs.values())
        references.append({"model": kind, **{name: ref["guidance"].tolist() for name, ref in refs.items()}})
        for ii, integrand in enumerate(config["integrands"]):
            reference = refs["candidate" if integrand == "candidate" else "continuation"]["guidance"]
            for seed in config["seeds"]:
                for m in config["samples"]:
                    values = {}
                    for mi, method in enumerate(config["methods"]):
                        basis, pilots = choose_basis(model, method, stream(seed, ki, ii, m, 1, mi), repeats)
                        eps = sign_probes(stream(seed, ki, ii, m, 2), repeats, m - pilots, model.dimension,
                                          basis, method.get("four", False), method.get("lhs", False), capacity=m)
                        group = 4 if method.get("four", False) else 2
                        logs = weights(model, eps, integrand, stream(seed, ki, ii, m, 3), group, capacity=m)
                        ratio = ratio_estimate(eps, logs)
                        if method.get("rao_blackwell", False):
                            # Oracle only: these top-aligned integrands depend in
                            # expectation on the known reward direction alone.
                            v = model.reward_direction
                            ratio = (ratio @ v)[:, None] * v
                        estimates = ratio @ model.jmu / model.std
                        key = f"c{len(rows):04d}"
                        arrays[key] = estimates
                        values[method["name"]] = estimates
                        row = {"array_key": key, "model": kind, "integrand": integrand, "seed": seed,
                               "samples": m, "main_samples": m - pilots, "method": method["name"],
                               "repeats": repeats, "pilot_queries_per_estimate": pilots,
                               "reward_queries_per_estimate": m if integrand != "continuation" else pilots,
                               "conditional_oracle_evaluations_per_estimate": m - pilots if integrand == "continuation" else 0,
                               **loss_summary(estimates, reference, model.reward_direction[:, None])}
                        if basis is not None:
                            row["mean_direction_capture"] = float(np.square(np.einsum("d,...dr->...r", model.reward_direction, basis)).sum(axis=-1).mean())
                        rows.append(row)
                        costs["pilot_reward_evaluations"] += repeats * pilots
                        costs["reward_evaluations"] += repeats * (pilots if integrand == "continuation" else m)
                        if integrand == "continuation":
                            costs["analytic_component_integrals"] += repeats * (m - pilots) * len(model.means)
                    for name, estimate in values.items():
                        for control in config["controls"]:
                            if name != control:
                                differences.append({"model": kind, "integrand": integrand, "seed": seed,
                                                    "samples": m, "method": name, "control": control,
                                                    **paired_difference(estimate, values[control], reference)})
        dump(output / "results.json", rows)
        dump(output / "paired-differences.json", differences)
        dump(output / "references.json", references)
        np.savez_compressed(output / "estimates.npz", **arrays)
        print(json.dumps({"model": kind, "conditions": len(rows)}), flush=True)
    metrics = {"conditions": len(rows), "costs": costs, "seconds": time.perf_counter() - started,
               "scope": "Exact fixed-kernel toy; oracle projections are mechanism bounds only"}
    dump(output / "summary.json", metrics)
    return metrics
