"""Reward-calibrated Gaussian probe directions, with original DOIT ratios."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .complement import stream, sign_probes
from .geometry import ratio_estimate


def direction_pool(jacobian, seed, random_count=16):
    """Columns live in transition-output/noise space, including nonsymmetric J."""
    values, vectors = np.linalg.eigh(jacobian @ jacobian.T)
    random = stream(seed).normal(size=(jacobian.shape[0], random_count))
    random /= np.linalg.norm(random, axis=0)
    return np.column_stack([vectors[:, ::-1], random]), values[::-1]


def moments(eps, logs, pool):
    weights = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
    # Subtract the empirical Gaussian baseline, not an identity covariance.
    centered = weights - 1 / eps.shape[1]
    projection = eps @ pool
    first = np.einsum("nm,nmk->nk", centered, projection)
    second = np.einsum("nm,nmk->nk", centered, projection * projection - 1)
    return first, second, 1 / np.square(weights).sum(axis=1)


def select(eps, logs, pool, jacobian, sigma, selector="moments",
           agreement_threshold=0.8, ess_fraction=0.25):
    scale = np.square(pool.T @ jacobian).sum(axis=1) / sigma**2
    if selector == "no_j":
        scale = np.ones_like(scale)

    def choose(e, l):
        first, second, ess = moments(e, l, pool)
        score = scale * (first**2 + (0 if selector == "first" else 0.5 * second**2))
        return score.argmax(axis=1), first, second, ess, score

    chosen, first, second, ess, score = choose(eps, logs)
    count = eps.shape[1] // 2
    half = count // 2
    parts = [np.r_[0:half, count:count + half], np.r_[half:count, count + half:2 * count]]
    split = [choose(eps[:, part], logs[:, part])[0] for part in parts]
    agreement = np.abs(np.einsum("dn,dn->n", pool[:, split[0]], pool[:, split[1]]))
    enabled = ((agreement >= agreement_threshold) & (ess >= eps.shape[1] * ess_fraction)
               & (score.max(axis=1) > 1e-24))
    return pool[:, chosen].T[..., None], {
        "chosen": chosen, "first_moment": first, "second_moment": second,
        "pilot_ess": ess, "scores": score, "split_chosen": np.stack(split, axis=1),
        "agreement": agreement, "enabled": enabled,
    }


def group_ids(repeats, samples, group):
    return np.broadcast_to(np.tile(np.arange(samples // group), group), (repeats, samples)).copy()


def estimate(context, method, route, budget, repeats, seed_words, config):
    """Fresh main randomness; retain every same-evaluator pilot unchanged.

    evaluator(eps, route, future_seed_words, group_ids, capacity) returns detached
    log rewards. Integer group IDs share futures within each repeat only.
    """
    selector, layout = method["selector"], method["layout"]
    pilot_count = budget // 4 if selector in {"moments", "first", "no_j"} else 0
    main_count = budget - pilot_count
    d = context.dimension
    arrays = {}
    pilot_route, main_route = ("P", "F") if route == "PF" else (route, route)
    enabled = np.ones(repeats, dtype=bool)
    if pilot_count:
        pilot = sign_probes(stream(*seed_words, 11), repeats, pilot_count, d, capacity=budget)
        pilot_groups = group_ids(repeats, pilot_count, 2)
        pilot_logs = context.evaluate(pilot, pilot_route, (*seed_words, 12), pilot_groups, budget)
        basis, diagnostics = select(pilot, pilot_logs, context.pool, context.jmu, context.std,
                                    selector, config["agreement_threshold"], config["ess_fraction"])
        enabled = diagnostics["enabled"]
        arrays.update(pilot_eps=pilot, pilot_logs=pilot_logs, pilot_group_ids=pilot_groups, **diagnostics)
    elif selector == "random":
        basis = stream(*seed_words, 13).normal(size=(repeats, d, 1))
        basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    elif selector == "model":
        basis = context.pool[:, :1]
    elif selector == "oracle":
        basis = context.oracle[:, None]
    elif selector in {"none", "zero"}:
        basis = None
    else:
        raise ValueError(selector)
    four = layout == "four_lhs"
    main = sign_probes(stream(*seed_words, 21), repeats, main_count, d, basis,
                       four=four, lhs=layout != "pair", capacity=budget)
    groups = group_ids(repeats, main_count, 4 if four else 2)
    if pilot_count and not enabled.all():
        fallback = sign_probes(stream(*seed_words, 21), repeats, main_count, d, capacity=budget)
        main[~enabled] = fallback[~enabled]
        groups[~enabled] = group_ids(repeats, main_count, 2)[~enabled]
    logs = context.evaluate(main, main_route, (*seed_words, 22), groups, budget)
    if pilot_count and pilot_route == main_route:
        all_eps = np.concatenate([pilot, main], axis=1)
        all_logs = np.concatenate([pilot_logs, logs], axis=1)
    else:
        all_eps, all_logs = main, logs
    final_weights = np.exp(all_logs - logsumexp(all_logs, axis=1, keepdims=True))
    ratio = ratio_estimate(all_eps, all_logs)
    guidance = ratio @ context.jmu / context.std
    arrays.update(main_eps=main, main_logs=logs, main_group_ids=groups, final_weights=final_weights,
                  guidance=guidance, main_only_guidance=ratio_estimate(main, logs) @ context.jmu / context.std,
                  selected_direction=np.zeros((repeats, d)) if basis is None else
                  np.broadcast_to(basis, (repeats, d, 1))[..., 0], enabled=enabled)
    independent_groups = np.max(groups, axis=1) + 1
    future_groups = independent_groups if main_route == "F" else np.zeros(repeats)
    if pilot_count and pilot_route == "F":
        future_groups = future_groups + pilot_count // 2
    info = {"pilot_queries": pilot_count, "main_queries": main_count,
            "queries_per_estimate": budget, "pilot_retained": bool(pilot_count and pilot_route == main_route),
            "fallback_rate": float(1 - enabled.mean()) if pilot_count else 0.0,
            "mean_independent_future_groups": float(np.mean(future_groups)),
            "mean_final_ess": float(np.mean(1 / np.square(final_weights).sum(axis=1)))}
    if pilot_count:
        info.update(mean_pilot_ess=float(arrays["pilot_ess"].mean()),
                    mean_split_agreement=float(arrays["agreement"].mean()))
    return guidance, arrays, info
