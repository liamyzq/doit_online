"""Evaluate the predeclared M32 future-coupling toy gate by task dimension."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run / 'results.json').read_text())
    paired = json.loads((args.run / 'paired-differences.json').read_text())
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('dimension', 'angle', 'budget', 'method'))
        groups.setdefault(key, []).append(row)
    summaries, gates = [], []
    for key, values in groups.items():
        dimension, angle, budget, method = key
        row = dict(zip(('dimension', 'angle', 'budget', 'method'), key))
        row.update({k: mean(x[k] for x in values) for k in ('mse', 'estimated_squared_bias', 'variance')})
        row['queries_per_call'] = budget
        row['comparisons'] = {}
        for control in ('shared_antithetic', 'random_paired_lhs'):
            if method == control:
                continue
            selected = [x for x in paired if tuple(x[k] for k in ('dimension', 'angle', 'budget', 'method', 'control')) == (*key, control)]
            delta = mean(x['mse_difference'] for x in selected)
            se = math.sqrt(sum(x['mc_se'] ** 2 for x in selected)) / len(selected)
            baseline = mean(x['mse'] for x in groups[(dimension, angle, budget, control)])
            row['comparisons'][control] = {'relative_mse_gain': 1 - row['mse'] / baseline,
                                            'mse_difference': delta, 'mc_interval95': [delta - 1.96 * se, delta + 1.96 * se]}
        summaries.append(row)
        if angle == 60 and budget == 32 and method in ('joint_antithetic', 'cartesian_four'):
            gates.append({'dimension': dimension, 'method': method,
                          'passes': all(x['relative_mse_gain'] >= .2 and x['mc_interval95'][1] < 0 for x in row['comparisons'].values()),
                          'comparisons': row['comparisons']})
    eligible = [x for x in gates if x['passes']]
    selected_method = next((m for m in ('joint_antithetic', 'cartesian_four') if any(x['method'] == m for x in eligible)), None)
    selected_dimensions = [x['dimension'] for x in eligible if x['method'] == selected_method]
    result = {'comparisons': summaries, 'mechanism_gate': gates,
              'selected_method': selected_method, 'selected_dimensions': selected_dimensions,
              'advance_to_bounded_critic': bool(selected_dimensions),
              'scope': 'Exact joint toy; per-dimension M32 rotating-target gate; aligned/M64 controls remain reported; no generalization claim'}
    (args.run / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('mechanism_gate', 'selected_method', 'selected_dimensions', 'advance_to_bounded_critic')}, indent=2))


if __name__ == '__main__':
    main()
