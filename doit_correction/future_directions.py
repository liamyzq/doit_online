"""Gaussian sign blocks whose basis may depend on independent future noise."""

import numpy as np
from scipy.special import ndtri


def layout(method, budget, dimension):
    kind = method['direction']
    setup = 2 * dimension if kind == 'candidate_fd' else 0
    per_block = {'path_gradient': 1, 'path_fd': 2 * dimension,
                 'path_pilot4': 4}.get(kind, 0)
    size = method['group']
    blocks = (budget - setup) // (size + per_block)
    return {'blocks': blocks, 'group': size, 'main_queries': blocks * size,
            'direction_queries': setup + blocks * per_block,
            'total_queries': setup + blocks * (size + per_block),
            'unused_queries': budget - setup - blocks * (size + per_block)}


def unit(vector):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    fallback = np.zeros_like(vector)
    fallback[..., 0] = 1
    return np.where(norm > 1e-12, vector / np.maximum(norm, 1e-12), fallback)


def probe_blocks(rng, repeats, capacity, dimension, blocks, group, basis=None, lhs=False):
    """Return [repeat, future block, sign point, dimension].

    The basis is [repeat, future block, dimension], determined without using
    this generator. Conditional on future noise/pilots, each probe is Gaussian.
    Fixed-capacity draws keep common randomness within matching repeat indices.
    """
    raw = rng.normal(size=(repeats, capacity, dimension))[:, :blocks]
    if basis is None:
        if group != 2 or lhs:
            raise ValueError('A direction is needed for four points or LHS')
        return np.stack([raw, -raw], axis=2)
    active_scalar = np.sum(raw * basis, axis=-1)
    complement = raw - active_scalar[..., None] * basis
    if lhs:
        permutation = np.argsort(rng.random((repeats, capacity))[:, :blocks], axis=1)
        jitter = rng.random((repeats, capacity))[:, :blocks]
        active_scalar = ndtri(np.maximum((permutation + jitter) / blocks, np.finfo(float).tiny))
    active = active_scalar[..., None] * basis
    if group == 4:
        return np.stack([active + complement, active - complement,
                         -active + complement, -active - complement], axis=2)
    return np.stack([active + complement, -active - complement], axis=2)
