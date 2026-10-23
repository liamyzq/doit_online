"""Observation-level paired summaries of complete-action critic endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons, means = [], []
    metrics = ["mean_q", "q10", "mean_pairwise_action_distance", "selected_mean_q",
               "selected_q10", "selected_pairwise_action_distance", "compute_seconds", "seconds"]
    for run in args.runs:
        rows = json.loads((run / "results.json").read_text())
        for task in sorted({r["task"] for r in rows}):
            methods = sorted({r["method"] for r in rows if r["task"] == task})
            groups = {name: {r["state_row"]: r for r in rows if r["task"] == task and r["method"] == name}
                      for name in methods}
            for name, group in groups.items():
                summary = {metric: float(np.mean([r[metric] for r in group.values()])) for metric in metrics}
                summary.update({"reward_queries": sum(r["costs"]["reward_queries"] for r in group.values()),
                    "actor_forwards": sum(r["costs"]["score_forwards"] for r in group.values()),
                    "jacobian_vjps": sum(r["jacobian_vjps"] for r in group.values()),
                    "original_kernel_vjps": sum(r["original_kernel_vjps"] for r in group.values())})
                means.append({"run": run.name, "task": task, "method": name, "observations": len(group), **summary})
                for control in ("antithetic", "random_pair"):
                    if control == name or control not in groups:
                        continue
                    observation_ids = sorted(group)
                    if observation_ids != sorted(groups[control]):
                        raise ValueError("Endpoint methods must cover the same observations")
                    for metric in metrics:
                        a = np.array([group[i][metric] for i in observation_ids])
                        b = np.array([groups[control][i][metric] for i in observation_ids])
                        difference = a - b
                        rng = np.random.default_rng(89001)
                        boot = difference[rng.integers(0, len(a), size=(20000, len(a)))].mean(axis=1)
                        comparisons.append({"run": run.name, "task": task, "method": name, "control": control,
                            "metric": metric, "observations": len(a), "mean": float(a.mean()),
                            "control_mean": float(b.mean()), "difference": float(difference.mean()),
                            "observation_bootstrap_interval95": np.quantile(boot, [0.025, 0.975]).tolist(),
                            "observations_positive": int((difference > 0).sum()),
                            "observation_differences": dict(zip(map(str, observation_ids), difference.tolist()))})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("means", means), ("comparisons", comparisons)):
        (args.output / f"{name}.json").write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"means": len(means), "comparisons": len(comparisons)}))


if __name__ == "__main__":
    main()
