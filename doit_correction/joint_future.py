"""Cost-free sign couplings of the current probe and all future Gaussian noise."""

import time

import numpy as np

from .complement import stream, loss_summary, paired_difference
from .future_directions import probe_blocks, unit
from .future_toy import FutureToy
from .geometry import ratio_estimate
from .geometry_toy import dump


def sign_map(method):
    if method == 'cartesian_four':
        return [1, -1, 1, -1], [1, 1, -1, -1]
    if method == 'joint_antithetic':
        return [1, -1], [1, -1]
    if method == 'independent_future':
        return [1, -1], None
    if method in ('shared_antithetic', 'random_paired_lhs'):
        return [1, -1], [1, 1]
    raise ValueError(method)


def main_blocks(rng, direction_rng, repeats, budget, dimension, method):
    main_signs, _ = sign_map(method)
    group = len(main_signs)
    blocks = budget // group
    if method == 'random_paired_lhs':
        basis = unit(direction_rng.normal(size=(repeats, budget, dimension))[:, :blocks])
        return probe_blocks(rng, repeats, budget, dimension, blocks, 2, basis, True)
    raw = rng.normal(size=(repeats, budget, dimension))[:, :blocks]
    return raw[:, :, None] * np.array(main_signs)[None, None, :, None]


def future_blocks(raw, method, blocks):
    main_signs, future_signs = sign_map(method)
    group = len(main_signs)
    if future_signs is None:
        return raw[:, :blocks * group].reshape(raw.shape[0], blocks, group, *raw.shape[2:])
    signs = np.array(future_signs).reshape(1, 1, group, *([1] * (raw.ndim - 2)))
    return raw[:, :blocks, None] * signs


def toy_logs(toy, eps, future):
    scalar = np.sum(eps * toy.directions(future), axis=-1)
    return -.5 * ((scalar - toy.target - toy.scale * future[..., 1]) / toy.width) ** 2


def evaluate(config, output):
    started = time.perf_counter()
    rows, differences, references, vectors = [], [], [], {}
    for di, dimension in enumerate(config['dimensions']):
        for ai, angle in enumerate(config['angles']):
            toy = FutureToy(dimension, angle)
            target, mass = toy.reference()
            references.append({'dimension': dimension, 'angle': angle, 'guidance': target.tolist(), 'mean_weight': mass})
            for seed in config['seeds']:
                for budget in config['budgets']:
                    estimates_by_method = {}
                    for method in config['methods']:
                        eps = main_blocks(stream(seed, di, ai, budget, 1), stream(seed, di, ai, budget, 2),
                                          config['repeats'], budget, dimension, method)
                        raw_future = stream(seed, di, ai, budget, 3).normal(size=(config['repeats'], budget, 2))
                        future = future_blocks(raw_future, method, eps.shape[1])
                        logs = toy_logs(toy, eps, future)
                        estimates = ratio_estimate(eps.reshape(config['repeats'], budget, dimension), logs.reshape(config['repeats'], budget))
                        key = f'c{len(rows):04d}'
                        vectors[key] = estimates
                        estimates_by_method[method] = estimates
                        weights = np.exp(logs)
                        rows.append({'dimension': dimension, 'angle': angle, 'seed': seed, 'budget': budget,
                                     'method': method, 'array_key': key, 'repeats': config['repeats'],
                                     'queries_per_call': budget, 'direction_queries': 0,
                                     'mean_numerator': np.mean(weights[..., None] * eps, axis=(0, 1, 2)).tolist(),
                                     'mean_denominator': float(weights.mean()), **loss_summary(estimates, target)})
                    for method, estimates in estimates_by_method.items():
                        for control in config['controls']:
                            if method != control:
                                differences.append({'dimension': dimension, 'angle': angle, 'seed': seed, 'budget': budget,
                                                    'method': method, 'control': control,
                                                    **paired_difference(estimates, estimates_by_method[control], target)})
    dump(output / 'results.json', rows)
    dump(output / 'paired-differences.json', differences)
    dump(output / 'references.json', references)
    np.savez_compressed(output / 'estimates.npz', **vectors)
    metrics = {'conditions': len(rows), 'reward_evaluations': sum(x['repeats'] * x['budget'] for x in rows),
               'direction_reward_evaluations': 0, 'exact_reference_formulas': len(references),
               'seconds': time.perf_counter() - started}
    dump(output / 'summary.json', metrics)
    return metrics


def checks(config, output):
    errors, failed = {}, {}
    repeats, budget, dimension = 32768, 4, 6
    toy = FutureToy(dimension, 60)
    target, mass = toy.reference()
    for mi, method in enumerate(config['methods']):
        eps = main_blocks(stream(77001, mi, 1), stream(77001, mi, 2), repeats, budget, dimension, method)
        future = future_blocks(stream(77001, mi, 3).normal(size=(repeats, budget, 2)), method, eps.shape[1])
        weights = np.exp(toy_logs(toy, eps, future))
        numerator = (weights[..., None] * eps).mean(axis=(1, 2))
        denominator = weights.mean(axis=(1, 2))
        errors[f'{method}_constant_weight_cancellation'] = float(abs(eps.sum(axis=(1, 2))).max())
        if errors[f'{method}_constant_weight_cancellation'] > 1e-10:
            failed[f'{method}_constant_weight_cancellation'] = errors[f'{method}_constant_weight_cancellation']
        for label, values, exact in [('numerator', numerator, mass * target), ('denominator', denominator, mass)]:
            statistic = float(np.max(abs(values.mean(axis=0) - exact) / (values.std(axis=0, ddof=1) / np.sqrt(repeats) + 1e-12)))
            errors[f'{method}_{label}_max_zscore'] = statistic
            if statistic > 6:
                failed[f'{method}_{label}_max_zscore'] = statistic
    # Distinct values encode repeats/slots, making sign or repeat-index mixing
    # observable without another stochastic distribution check.
    raw = np.arange(3 * 8 * 2, dtype=float).reshape(3, 8, 2)
    paired = future_blocks(raw, 'joint_antithetic', 4)
    four = future_blocks(raw, 'cartesian_four', 2)
    shared = future_blocks(raw, 'shared_antithetic', 4)
    independent = future_blocks(raw, 'independent_future', 4)
    relations = {'joint_opposite': paired[:, :, 0] + paired[:, :, 1],
                 'cartesian_sign': four[:, :, :2] + four[:, :, 2:],
                 'cartesian_first_two_shared': four[:, :, 0] - four[:, :, 1],
                 'repeat_prefix': paired[:, :2, 0] - four[:, :, 0],
                 'shared_equal': shared[:, :, 0] - shared[:, :, 1],
                 'independent_slots': independent.reshape(raw.shape) - raw}
    for label, values in relations.items():
        errors[label] = float(abs(values).max())
        if errors[label] > 0:
            failed[label] = errors[label]
    dump(output / 'checks.json', {'errors': errors, 'failed': failed})
    if failed:
        raise AssertionError(failed)
    return {'checks_passed': len(errors), 'toy_reward_evaluations': repeats * budget * len(config['methods']),
            'scope': 'New current/future sign maps and exact target moments; no repeated reference quadrature'}
