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
    return {(r["prompt"], r["seed"]): dict(r, artifact_dir=str(path)) for r in rows}, config


def read_group(paths):
    combined = {}
    first_config = None
    for path in paths:
        rows, config = read_run(path)
        if first_config is None:
            first_config = config
        if (config["estimator"] != first_config["estimator"]
                or config["image"] != first_config["image"]):
            raise ValueError(f"A comparison group mixes estimator or image settings: {path}")
        if combined.keys() & rows.keys():
            raise ValueError(f"Repeated prompt/seed within a comparison group: {path}")
        combined.update(rows)
    return combined


def diversity(rows, keys):
    prompt_values = []
    for prompt in sorted({p for p, _ in keys}):
        features = np.stack([np.load(Path(rows[(p, s)]["artifact_dir"]) / p / str(s) / "embedding.npy")
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


def summarize(groups, baseline):
    reference = read_group(groups[baseline])
    results = []
    for name, paths in groups.items():
        rows = read_group(paths)
        if rows.keys() != reference.keys():
            raise ValueError(f"Paired prompt/seed sets differ: {name}")
        keys = sorted(rows)
        result = {"id": name, "images": len(keys), "seeds": len({s for _, s in keys}),
                  "baseline": baseline, "source_runs": [str(p) for p in paths]}
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
        result["within_prompt_embedding_distance"] = diversity(rows, keys)
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="JSON with baseline group name and groups of run directories")
    parser.add_argument("--output")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    output = json.dumps(summarize(manifest["groups"], manifest["baseline"]), indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
