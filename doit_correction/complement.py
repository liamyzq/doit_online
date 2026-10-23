"""Gaussian sign-group probes with independently chosen active subspaces."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def stream(seed, *coordinates):
    """Explicit streams: context coordinates are separate SeedSequence words."""
    return np.random.default_rng(np.random.SeedSequence([seed, *coordinates]))


def sign_probes(rng, repeats, samples, dimension, basis=None, four=False, lhs=False):
    """Return block-ordered pairs or {Ua+b, Ua-b, -Ua+b, -Ua-b}.

    A basis can be fixed or selected per repeat from an independent pilot.
    Every indexed probe has N(0,I) marginal conditional on that basis. All four
    weights must be evaluated; the method never discards complement scores.
    """
    group = 4 if four else 2
    if samples % group:
        raise ValueError("Sample count must contain complete sign groups")
    count = samples // group
    eps = rng.normal(size=(repeats, count, dimension))
    if basis is None:
        if four or lhs:
            raise ValueError("A subspace is required for four-point or LHS probes")
        return np.concatenate([eps, -eps], axis=1)
    basis = np.asarray(basis)
    if basis.ndim == 2:
        basis = np.broadcast_to(basis, (repeats, *basis.shape))
    a = np.einsum("nmd,ndr->nmr", eps, basis)
    b = eps - np.einsum("nmr,ndr->nmd", a, basis)
    if lhs:
        strata = np.argsort(rng.random(a.shape), axis=1)
        uniforms = (strata + rng.random(a.shape)) / count
        a = ndtri(np.maximum(uniforms, np.finfo(float).tiny))
    active = np.einsum("nmr,ndr->nmd", a, basis)
    if four:
        return np.concatenate([active + b, active - b, -active + b, -active - b], axis=1)
    return np.concatenate([active + b, -active - b], axis=1)


def grouped_draw(rng, repeats, samples, group, dimension=None):
    """Share later randomness within each sign group, retaining its marginal."""
    shape = (repeats, samples // group) + (() if dimension is None else (dimension,))
    values = rng.normal(size=shape)
    return np.concatenate([values] * group, axis=1)


def loss_summary(estimates, reference, active=None):
    delta = estimates - reference
    loss = np.square(delta).sum(axis=-1)
    result = {"mse": float(loss.mean()),
              "estimated_squared_bias": float(np.square(delta.mean(axis=0)).sum()),
              "variance": float(np.square(estimates - estimates.mean(axis=0)).sum(axis=-1).mean())}
    if active is not None:
        active_loss = np.square(delta @ active).sum(axis=-1)
        result.update(active_mse=float(active_loss.mean()),
                      complement_mse=float((loss - active_loss).mean()))
    return result


def paired_difference(estimates, control, reference):
    """Fixed-context MC interval, conditional on the supplied reference.

    The common reference's squared norm cancels from the paired loss difference.
    This interval does not quantify reference bias or observation generalization.
    """
    difference = np.sum((estimates - control) * (estimates + control - 2 * reference), axis=-1)
    mean = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
    return {"mse_difference": mean, "mc_se": se,
            "mc_interval95": [mean - 1.96 * se, mean + 1.96 * se]}
