"""Apply the predeclared conditional-direction toy gate."""

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
        row['queries_per_call'] = values[0]['queries_per_call']
        if 'mean_path_direction_capture' in values[0]:
            row['mean_path_direction_capture'] = mean(x['mean_path_direction_capture'] for x in values)
        row['comparisons'] = {}
        for control in ('antithetic', 'random_paired_lhs'):
            if method == control:
                continue
            selected = [x for x in paired if tuple(x[k] for k in ('dimension', 'angle', 'budget', 'method', 'control')) == (*key, control)]
            delta = mean(x['mse_difference'] for x in selected)
            se = math.sqrt(sum(x['mc_se'] ** 2 for x in selected)) / len(selected)
            baseline = mean(x['mse'] for x in groups[(dimension, angle, budget, control)])
            row['comparisons'][control] = {'relative_mse_gain': 1 - row['mse'] / baseline,
                                            'mse_difference': delta, 'mc_interval95': [delta - 1.96 * se, delta + 1.96 * se]}
        summaries.append(row)
        if angle == 60 and budget == 64 and method == 'path_gradient_four':
            gates.append({'dimension': dimension, 'passes': all(x['relative_mse_gain'] >= .2 and x['mc_interval95'][1] < 0 for x in row['comparisons'].values()),
                          'comparisons': row['comparisons']})
    result = {'comparisons': summaries, 'mechanism_gate': gates,
              'advance_to_bounded_critic': len(gates) == 2 and all(x['passes'] for x in gates),
              'scope': 'Fixed exact joint toy; oracle derivatives are counted but not a black-box implementation; independent-repeat MC intervals, no generalization claim'}
    (args.run / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'advance_to_bounded_critic': result['advance_to_bounded_critic'], 'mechanism_gate': gates}, indent=2))


if __name__ == '__main__':
    main()
