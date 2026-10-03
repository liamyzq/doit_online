"""Summarize matched image runs and seed-block uncertainty from saved artifacts.

Bootstrap intervals condition on the fixed prompt set and resample entire seed
blocks, preserving the shared seeds across prompts and paired methods. They are
not prompt-population confidence intervals. Discovery runs with fewer than four
seeds receive no interval.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def read_run(path):
    path = Path(path)
    metadata = json.loads((path / "run.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError(f"Only completed image runs can be summarized: {path}")
    config = json.loads((path / "config.json").read_text())
    if config.get("mode", "image") != "image":
        raise ValueError(f"Capture timings are not image-generation timings: {path}")
    rows = json.loads((path / "images.json").read_text())
    return {(r["prompt"], r["seed"]): r for r in rows}, config


def diversity(path, keys):
    prompt_values = []
    for prompt in sorted({p for p, _ in keys}):
        features = np.stack([np.load(Path(path) / p / str(s) / "embedding.npy")
                             for p, s in sorted(keys) if p == prompt]).astype(np.float64)
        if len(features) < 2:
            continue
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        distance = 1 - np.clip(features @ features.T, -1, 1)
        prompt_values.append(float(distance[np.triu_indices(len(features), 1)].mean()))
    return float(np.mean(prompt_values)) if prompt_values else None


def paired_interval(differences, keys):
    seeds = sorted({s for _, s in keys})
    if len(seeds) < 4:
        return None
    blocks = np.array([np.mean([d for d, (_, seed) in zip(differences, keys) if seed == s])
                       for s in seeds])
    generator = np.random.default_rng(17011)
    bootstrap = generator.choice(blocks, size=(10000, len(blocks)), replace=True).mean(1)
    return np.quantile(bootstrap, [0.025, 0.975]).tolist()


def summarize(paths, baseline):
    reference, _ = read_run(baseline)
    results = []
    for path in paths:
        rows, config = read_run(path)
        if rows.keys() != reference.keys():
            raise ValueError(f"Paired prompt/seed sets differ: {path}")
        keys = sorted(rows)
        result = {"id": config["id"], "images": len(keys), "seeds": len({s for _, s in keys}),
                  "baseline": Path(baseline).name}
        for metric in ["reward", "clip_alignment"]:
            values = np.array([rows[k][metric] for k in keys])
            differences = values - np.array([reference[k][metric] for k in keys])
            result[f"mean_{metric}"] = float(values.mean())
            result[f"paired_{metric}_delta"] = float(differences.mean())
            result[f"paired_{metric}_seed_ci95"] = paired_interval(differences, keys)
            result[f"{metric}_delta_by_prompt"] = {
                p: float(np.mean([d for d, (prompt, _) in zip(differences, keys) if prompt == p]))
                for p in sorted({p for p, _ in keys})}
        for metric in ["guidance_queries", "score_forwards", "seconds"]:
            result[f"mean_{metric}"] = float(np.mean([rows[k][metric] for k in keys]))
        result["denominator_events"] = sum(rows[k]["denominator_events"] for k in keys)
        result["within_prompt_embedding_distance"] = diversity(path, keys)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    output = json.dumps(summarize(args.runs, args.baseline), indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
