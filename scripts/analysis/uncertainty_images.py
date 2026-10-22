"""Summarize complete uncertainty-image waves and render review contact sheets."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path('results/uncertainty_smc')
    data = {}
    for folder in sorted(root.glob(args.prefix + '*')):
        if not folder.is_dir():
            continue
        if json.loads((folder / 'run.json').read_text())['status'] != 'completed':
            raise ValueError(f'Incomplete experiment: {folder}')
        for row in json.loads((folder / 'images.json').read_text()):
            method = row['method']
            key = row['prompt'], row['seed']
            row['directory'] = str(folder / method / key[0] / str(key[1]))
            data.setdefault(method, {})[key] = row
    methods = list(data)
    keys = sorted(data['mc8'])
    if any(set(rows) != set(keys) for rows in data.values()):
        raise ValueError('Methods have different prompt/seed coverage')
    seeds = sorted({seed for _, seed in keys})
    prompts = list(dict.fromkeys(prompt for prompt, _ in keys))
    summaries = {}
    for method, records in data.items():
        rows = [records[key] for key in keys]
        summary = {'images': len(rows)}
        for field in ['reward', 'clip_alignment', 'seconds', 'all_queries', 'score_forwards', 'vjp_calls']:
            summary['mean_' + field] = float(np.mean([row[field] for row in rows]))
        summary['reward_10th_percentile'] = float(np.quantile([r['reward'] for r in rows], .1))
        summary['by_seed'] = {str(seed): {field: float(np.mean([r[field] for r in rows if r['seed'] == seed]))
                                        for field in ['reward', 'clip_alignment', 'seconds']} for seed in seeds}
        distances = []
        for prompt in prompts:
            embeddings = [np.load(Path(records[prompt, seed]['directory']) / 'embedding.npy') for seed in seeds]
            distances += [float(1 - a @ b) for a, b in itertools.combinations(embeddings, 2)]
        summary['mean_within_prompt_embedding_distance'] = float(np.mean(distances))
        if all('weighted_reward' in row for row in rows):
            summary['mean_weighted_reward_diagnostic'] = float(np.mean([r['weighted_reward'] for r in rows]))
            summary['mean_final_ess'] = float(np.mean([r['final_ess'] for r in rows]))
            summary['mean_initial_ancestors'] = float(np.mean([r['initial_ancestors'] for r in rows]))
        summary['comparisons'] = {}
        for comparator in [name for name in ['mc8', 'mc32', 'terminal-n8'] if name in data and name != method]:
            comparisons = {}
            for field in ['reward', 'clip_alignment', 'seconds']:
                by_seed = np.array([np.mean([records[prompt, seed][field] - data[comparator][prompt, seed][field]
                                            for prompt in prompts]) for seed in seeds])
                result = {'mean_delta': float(by_seed.mean()), 'seed_deltas': by_seed.tolist()}
                if args.confirmation:
                    rng = np.random.default_rng(59099)
                    samples = rng.choice(by_seed, (20000, len(by_seed)), replace=True).mean(axis=1)
                    result['seed_block_95_percent_interval'] = np.quantile(samples, [.025, .975]).tolist()
                comparisons[field] = result
            summary['comparisons'][comparator] = comparisons
        summaries[method] = summary
    result = {'prefix': args.prefix, 'seeds': seeds, 'prompts': prompts, 'confirmation': args.confirmation,
              'interval_scope': 'Seed-block bootstrap conditional on these fixed prompts; no discovery interval from two seeds.',
              'groups': summaries,
              'total_queries': sum(row['all_queries'] for records in data.values() for row in records.values()),
              'total_score_forwards': sum(row['score_forwards'] for records in data.values() for row in records.values()),
              'total_vjp_calls': sum(row['vjp_calls'] for records in data.values() for row in records.values())}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    cell_w, cell_h = 256, 290
    for seed in seeds:
        for start in range(0, len(prompts), 4):
            selected_prompts = prompts[start:start + 4]
            sheet = Image.new('RGB', (cell_w * len(methods), cell_h * len(selected_prompts)), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, method in enumerate(methods):
                for i, prompt in enumerate(selected_prompts):
                    row = data[method][prompt, seed]
                    picture = Image.open(Path(row['directory']) / 'image.png').convert('RGB')
                    sheet.paste(picture.resize((256, 256)), (j * cell_w, i * cell_h + 34))
                    label = f'{method} | {prompt} | s{seed}\nR {row["reward"]:.3f} A {row["clip_alignment"]:.3f} t {row["seconds"]:.1f}s'
                    draw.text((j * cell_w + 3, i * cell_h + 1), label, fill='black')
            sheet.save(output / f'review-s{seed}-{start // 4 + 1}.png')
    print(json.dumps({name: {k: v for k, v in row.items() if k not in {'comparisons', 'by_seed'}}
                      for name, row in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
