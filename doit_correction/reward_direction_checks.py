"""Focused checks for the new moments, adaptive design, and discrete references."""

import numpy as np
from scipy.special import logsumexp

from .complement import stream, sign_probes
from .geometry import ratio_estimate
from .geometry_toy import dump
from .reward_directions import direction_pool, moments, select, estimate, group_ids
from .reward_direction_toy import ToyProcess, ToyContext


def checks(config, output):
    errors, tolerances = {}, {}

    def record(name, error, tolerance):
        errors[name], tolerances[name] = float(error), tolerance

    d, n = 6, config["repeats"]
    jacobian = stream(81101).normal(size=(d, d))
    pool, eigenvalues = direction_pool(jacobian, 81103)
    record("output_space_spectral_equation", np.max(np.abs(
        jacobian @ jacobian.T @ pool[:, :d] - pool[:, :d] * eigenvalues)), 1e-10)
    eps = sign_probes(stream(81107), n, 32, d, capacity=32)
    first, second, _ = moments(eps, np.zeros(eps.shape[:2]), pool)
    record("constant_empirical_moments", max(np.abs(first).max(), np.abs(second).max()), 1e-13)
    _, selected = select(eps, np.zeros(eps.shape[:2]), pool, jacobian, 0.4)
    record("constant_reward_fallback", selected["enabled"].sum(), 0)
    axis = pool[:, d + 1]
    z = eps @ axis
    logs = -0.5 * 0.35 * (z - 0.8)**2
    pooled = eps.reshape(1, -1, d)
    first, second, _ = moments(pooled, logs.reshape(1, -1), axis[:, None])
    target_mean = 0.35 * 0.8 / 1.35
    record("gaussian_tilt_first", abs(first.item() - target_mean), 0.004)
    record("gaussian_tilt_second", abs(second.item() - (1 / 1.35 - 1 + target_mean**2)), 0.008)

    class Context:
        dimension, jmu, std = d, jacobian, 0.4

        def evaluate(self, e, route, seed_words, groups, capacity):
            t = 0.25 if route == "P" else -0.2
            return t * (e @ axis) - t**2 / 2

    context = Context()
    context.pool = pool
    for route in ("P", "PF"):
        for layout in ("pair_lhs", "four_lhs"):
            _, arrays, _ = estimate(context, {"selector": "moments", "layout": layout},
                                   route, 32, n, (81109,), config)
            if route == "P":
                all_eps = np.concatenate([arrays["pilot_eps"], arrays["main_eps"]], axis=1)
                all_logs = np.concatenate([arrays["pilot_logs"], arrays["main_logs"]], axis=1)
                t = 0.25
            else:
                all_eps, all_logs, t = arrays["main_eps"], arrays["main_logs"], -0.2
            mass = np.exp(all_logs)
            record(f"adaptive_mass_{route}_{layout}", abs(mass.mean() - 1), 0.004)
            numerator = np.einsum("nm,nmd->nd", mass, all_eps) / all_eps.shape[1]
            record(f"adaptive_numerator_{route}_{layout}", np.max(np.abs(numerator.mean(axis=0) - t * axis)), 0.007)
            manual = ratio_estimate(all_eps, all_logs) @ jacobian / context.std
            record(f"ratio_merge_{route}_{layout}", np.max(np.abs(manual - arrays["guidance"])), 1e-12)
            # The first main position is Gaussian even after adaptive selection.
            sample = arrays["main_eps"][:, 0]
            record(f"adaptive_main_mean_{route}_{layout}", abs(sample.mean(axis=0)).max(), 0.025)
            record(f"adaptive_main_covariance_{route}_{layout}", abs(sample.T @ sample / n - np.eye(d)).max(), 0.04)
            record(f"separate_pilot_main_{route}_{layout}", float(np.array_equal(
                arrays["pilot_eps"][:, 0], arrays["main_eps"][:, 0])), 0)
    model = ToyProcess("gaussian")
    for route in ("P", "F"):
        context = ToyContext(model, 7, 1, "off_pool", "shift", 81111)
        draws = sign_probes(stream(81113), n, 8, d)
        logs = context.evaluate(draws, route, (81117,), group_ids(n, 8, 2), 8)
        weights = np.exp(logs - logsumexp(logs))
        ratio = np.einsum("nm,nmd->d", weights, draws)
        ref = context.analytic_reference(route)
        record(f"actual_discrete_reference_{route}", np.max(np.abs(ratio - ref["ratio"])), 0.015)
    dump(output / "checks.json", {"errors": errors, "tolerances": tolerances,
                                   "failed": {k: v for k, v in errors.items() if v > tolerances[k]}})
    failed = {k: v for k, v in errors.items() if v > tolerances[k]}
    if failed:
        raise AssertionError(failed)
    return {"checks_passed": len(errors), "synthetic_reward_evaluations": n * (4 * 32 + 32),
            "toy_costs": model.counts()}
