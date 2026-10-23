"""Gaussian probe designs and local response matrices for the geometry study."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


def gaussian_probes(rng, repeats, samples, dimension, basis=None, antithetic=False):
    """Randomized LHS in a subspace; each indexed probe is standard Gaussian.

    Each coordinate gets an independent random stratum permutation. The
    orthogonal complement is independent Gaussian. Pairing retains the same
    marginals while cancelling the constant-weight numerator exactly.
    """
    if antithetic and samples % 2:
        raise ValueError("Antithetic sample counts must be even")
    count = samples // 2 if antithetic else samples
    eps = rng.normal(size=(repeats, count, dimension))
    if basis is not None:
        basis = np.asarray(basis)
        # A per-repeat direction selected with an independent pilot is allowed.
        if basis.ndim == 2:
            basis = np.broadcast_to(basis, (repeats, *basis.shape))
        rank = basis.shape[-1]
        strata = np.argsort(rng.random((repeats, count, rank)), axis=1)
        uniforms = (strata + rng.random((repeats, count, rank))) / count
        coordinates = ndtri(np.maximum(uniforms, np.finfo(float).tiny))
        projections = np.einsum("nmd,ndr->nmr", eps, basis)
        eps += np.einsum("nmr,ndr->nmd", coordinates - projections, basis)
    return np.concatenate([eps, -eps], axis=1) if antithetic else eps


def response_matrix(matrix, rank):
    """Top algebraic directions plus the unresolved subspace's mean response."""
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if rank == len(values):
        return matrix.copy()
    background = values[rank:].mean()
    u = vectors[:, :rank]
    return background * np.eye(len(values)) + (u * (values[:rank] - background)) @ u.T


def ratio_estimate(epsilon, log_weights):
    values = np.exp(log_weights - log_weights.max(axis=-1, keepdims=True))
    weights = values / values.sum(axis=-1, keepdims=True)
    return np.einsum("nm,nmd->nd", weights, epsilon)


def error_summary(estimates, reference):
    mean = estimates.mean(axis=0)
    squared_bias = float(np.square(mean - reference).sum())
    variance = float(np.square(estimates - mean).sum(axis=-1).mean())
    return {"mse": squared_bias + variance, "squared_bias": squared_bias,
            "variance": variance, "reference_norm2": float(np.square(reference).sum())}
