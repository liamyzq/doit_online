"""Exact joint Gaussian/future toy for continuation-conditioned directions."""

import time

import numpy as np
from scipy.special import roots_hermitenorm

from .complement import stream, loss_summary, paired_difference
from .future_directions import layout, unit, probe_blocks
from .geometry import ratio_estimate
from .geometry_toy import dump


class FutureToy:
    def __init__(self, dimension, degrees):
        self.dimension, self.degrees = dimension, degrees
        self.angle = np.deg2rad(degrees)
        self.target, self.scale, self.width = 1., .7, .7

    def directions(self, future):
        direction = np.zeros((*future.shape[:-1], self.dimension))
        direction[..., 0] = np.cos(self.angle)
        direction[..., 1] = np.where(future[..., 0] >= 0, 1., -1.) * np.sin(self.angle)
        return direction

    def log_weight(self, eps, future):
        scalar = np.sum(eps * self.directions(future)[..., None, :], axis=-1)
        target = self.target + self.scale * future[..., 1]
        return -.5 * ((scalar - target[..., None]) / self.width) ** 2

    def reference(self):
        variance = self.width ** 2 + 1 + self.scale ** 2
        mean = np.zeros(self.dimension)
        mean[0] = self.target * np.cos(self.angle) / variance
        mass = self.width / np.sqrt(variance) * np.exp(-.5 * self.target ** 2 / variance)
        return mean, mass

    def basis(self, method, future, rng, capacity):
        repeats, blocks = future.shape[:2]
        kind = method['direction']
        if kind == 'none':
            return None
        if kind == 'random':
            return unit(rng.normal(size=(repeats, capacity, self.dimension, 2))[:, :blocks, :, 0])
        if kind == 'candidate_fd':
            offsets = .25 * np.concatenate([np.eye(self.dimension), -np.eye(self.dimension)])
            # Actually evaluate all charged candidate contrasts per repeat.
            logs = -.5 * ((np.broadcast_to(offsets[:, 0], (repeats, 2 * self.dimension)) - self.target) / self.width) ** 2
            gradient = (logs[:, :self.dimension] - logs[:, self.dimension:]) / .5
            return np.broadcast_to(unit(gradient)[:, None], (repeats, blocks, self.dimension))
        if kind == 'path_gradient':
            self.log_weight(np.zeros((repeats, blocks, 1, self.dimension)), future)
            return unit((self.target + self.scale * future[..., 1, None]) * self.directions(future) / self.width ** 2)
        if kind == 'path_fd':
            directions = np.broadcast_to(np.eye(self.dimension), (repeats, blocks, self.dimension, self.dimension))
        elif kind == 'path_pilot4':
            directions = np.linalg.qr(rng.normal(size=(repeats, capacity, self.dimension, 2)))[0][:, :blocks].swapaxes(-1, -2)
        else:
            raise ValueError(kind)
        offsets = .25 * np.concatenate([directions, -directions], axis=2)
        logs = self.log_weight(offsets, future)
        rank = directions.shape[2]
        gradient = np.sum(directions * ((logs[..., :rank] - logs[..., rank:]) / .5)[..., None], axis=2)
        return unit(gradient)


def evaluate(config, output):
    started = time.perf_counter()
    rows, comparisons, references, vectors = [], [], [], {}
    counts = {'main_reward_evaluations': 0, 'direction_reward_evaluations': 0,
              'oracle_derivative_evaluations': 0, 'exact_reference_formulas': 0}
    for di, dimension in enumerate(config['dimensions']):
        for ai, degrees in enumerate(config['angles']):
            toy = FutureToy(dimension, degrees)
            target, mass = toy.reference()
            references.append({'dimension': dimension, 'angle': degrees, 'guidance': target.tolist(), 'mean_weight': mass})
            counts['exact_reference_formulas'] += 1
            for seed in config['seeds']:
                for budget in config['budgets']:
                    methods = {}
                    for method in config['methods']:
                        cost = layout(method, budget, dimension)
                        repeats, groups = config['repeats'], cost['blocks']
                        future = stream(seed, di, ai, budget, 1).normal(size=(repeats, budget, 2))[:, :groups]
                        basis = toy.basis(method, future, stream(seed, di, ai, budget, 2), budget)
                        eps = probe_blocks(stream(seed, di, ai, budget, 3), repeats, budget,
                                           dimension, groups, cost['group'], basis, method.get('lhs', False))
                        logs = toy.log_weight(eps, future)
                        estimates = ratio_estimate(eps.reshape(repeats, -1, dimension), logs.reshape(repeats, -1))
                        methods[method['name']] = estimates
                        key = f'c{len(rows):04d}'
                        vectors[key] = estimates
                        record = {'dimension': dimension, 'angle': degrees, 'seed': seed, 'budget': budget,
                                  'method': method['name'], 'array_key': key, 'repeats': repeats,
                                  'queries_per_call': cost, **loss_summary(estimates, target)}
                        if basis is not None:
                            record['mean_path_direction_capture'] = float(np.mean(np.sum(basis * toy.directions(future), axis=-1) ** 2))
                        rows.append(record)
                        counts['main_reward_evaluations'] += repeats * cost['main_queries']
                        counts['direction_reward_evaluations'] += repeats * cost['direction_queries']
                        if method['direction'] == 'path_gradient':
                            counts['oracle_derivative_evaluations'] += repeats * groups
                    for method, estimates in methods.items():
                        for control in config['controls']:
                            if method != control:
                                comparisons.append({'dimension': dimension, 'angle': degrees, 'seed': seed,
                                                    'budget': budget, 'method': method, 'control': control,
                                                    **paired_difference(estimates, methods[control], target)})
    dump(output / 'results.json', rows)
    dump(output / 'paired-differences.json', comparisons)
    dump(output / 'references.json', references)
    np.savez_compressed(output / 'estimates.npz', **vectors)
    counts['reward_evaluations'] = counts['main_reward_evaluations'] + counts['direction_reward_evaluations']
    metrics = {'conditions': len(rows), **counts, 'seconds': time.perf_counter() - started}
    dump(output / 'summary.json', metrics)
    return metrics


def checks(config, output):
    repeats, dimension = 65536, 6
    toy = FutureToy(dimension, 60)
    future = stream(75001, 0).normal(size=(repeats, 1, 2))
    basis = toy.directions(future)
    eps = probe_blocks(stream(75001, 1), repeats, 4, dimension, 1, 4, basis, True)
    errors, failed = {}, {}
    for point in range(4):
        values = eps[:, 0, point]
        errors[f'point{point}_mean'] = float(abs(values.mean(axis=0)).max())
        errors[f'point{point}_covariance'] = float(abs(values.T @ values / repeats - np.eye(dimension)).max())
        errors[f'point{point}_future_cross_moment'] = float(abs(values.T @ future[:, 0] / repeats).max())
    for key, value in errors.items():
        if value > .025:
            failed[key] = value
    logs = toy.log_weight(eps, future)
    weights = np.exp(logs)
    numerator = (weights[..., None] * eps).mean(axis=2)[:, 0]
    denominator = weights.mean(axis=2)[:, 0]
    projection = numerator - np.sum(numerator * basis[:, 0], axis=-1, keepdims=True) * basis[:, 0]
    errors['correct_direction_complement_max'] = float(abs(projection).max())
    if errors['correct_direction_complement_max'] > 1e-10:
        failed['correct_direction_complement_max'] = errors['correct_direction_complement_max']
    target, mass = toy.reference()
    for key, samples, exact in [('numerator', numerator, mass * target), ('denominator', denominator, mass)]:
        zscore = float(np.max(abs(samples.mean(axis=0) - exact) / (samples.std(axis=0, ddof=1) / np.sqrt(repeats) + 1e-12)))
        errors[f'joint_{key}_max_zscore'] = zscore
        if zscore > 6:
            failed[f'joint_{key}_max_zscore'] = zscore
    # Hold the future sign positive: a wrong e1 basis must retain the nonzero
    # e2 target rather than silently project the full score onto its basis.
    positive = future.copy(); positive[..., 0] = abs(positive[..., 0])
    wrong = np.zeros_like(basis); wrong[..., 0] = 1
    wrong_eps = probe_blocks(stream(75001, 2), repeats, 4, dimension, 1, 4, wrong, True)
    wrong_weight = np.exp(toy.log_weight(wrong_eps, positive))
    component = (wrong_weight * wrong_eps[..., 1]).mean(axis=(1, 2))
    expected = mass * toy.target * np.sin(toy.angle) / (toy.width ** 2 + 1 + toy.scale ** 2)
    errors['wrong_direction_retained_component_zscore'] = float(abs(component.mean() - expected) / (component.std(ddof=1) / np.sqrt(repeats)))
    if errors['wrong_direction_retained_component_zscore'] > 6:
        failed['wrong_direction_retained_component_zscore'] = errors['wrong_direction_retained_component_zscore']
    nodes, quadrature = roots_hermitenorm(64)
    quadrature = quadrature / np.sqrt(2 * np.pi)
    grid = np.exp(-.5 * ((nodes[:, None] - toy.target - toy.scale * nodes[None]) / toy.width) ** 2)
    measured_mass = quadrature @ grid @ quadrature
    measured_active = (quadrature * nodes) @ grid @ quadrature
    errors['exact_mass_quadrature'] = float(abs(measured_mass - mass))
    errors['exact_moment_quadrature'] = float(abs(measured_active / measured_mass * np.cos(toy.angle) - target[0]))
    for key in ['exact_mass_quadrature', 'exact_moment_quadrature']:
        if errors[key] > 1e-10:
            failed[key] = errors[key]
    future_small = stream(75001, 4).normal(size=(32, 64, 2))
    basis_small = toy.directions(future_small)
    pair = probe_blocks(stream(75001, 5), 32, 64, dimension, 16, 2, basis_small[:, :16], True)
    four = probe_blocks(stream(75001, 5), 32, 64, dimension, 7, 4, basis_small[:, :7], True)
    def residual(v):
        return v - np.sum(v * basis_small[:, :7], axis=-1, keepdims=True) * basis_small[:, :7]
    errors['repeat_aligned_complement_prefix'] = float(abs(residual(pair[:, :7, 0]) - residual(four[:, :, 0])).max())
    if errors['repeat_aligned_complement_prefix'] > 1e-10:
        failed['repeat_aligned_complement_prefix'] = errors['repeat_aligned_complement_prefix']
    dump(output / 'checks.json', {'errors': errors, 'failed': failed})
    if failed:
        raise AssertionError(failed)
    return {'checks_passed': len(errors), 'toy_reward_evaluations': repeats * 8,
            'reference_quadrature_integrand_evaluations': len(nodes) ** 2,
            'purpose': 'Validate future-conditioned Gaussian marginals, full moments and repeat alignment'}
