"""Summarize locked reward-direction comparisons without MC pseudoreplication."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries, contexts_out = [], []
    for run in args.runs:
        rows, contexts = load(run / "results.json"), load(run / "contexts.json")
        lookup = {(r["context_key"], r["route"], r["samples"], r["method"]): r for r in rows}
        contexts = {c["context_key"]: c for c in contexts}
        arrays = np.load(run / "estimates.npz")
        real = "task" in rows[0]
        conditions = {}
        for row in rows:
            name = row["method"]
            if name in {"antithetic", "random_pair", "random_four"}:
                continue
            matched = "random_four" if (name.endswith("four") or
                       (row["route"] == "P" and name in {"model", "first", "no_j"})) else "random_pair"
            for control in ("antithetic", matched):
                route_b = "F" if row["route"] == "PF" else row["route"]
                baseline = lookup[(row["context_key"], route_b, row["samples"], control)]
                context = contexts[row["context_key"]]
                gain2 = context["gain"]**2
                estimate = arrays[row["array_key"]]
                other = arrays[baseline["array_key"]]
                metrics = ["same_target"] + (["proxy_to_full"] if row["route"] == "P" else [])
                for metric in metrics:
                    ref_route = "P" if row["route"] == "P" and metric == "same_target" else "F"
                    ref = context["references"][ref_route]
                    target = np.array(ref["guidance"])
                    a = gain2 * np.square(estimate - target).sum(axis=1).mean()
                    b = gain2 * np.square(other - target).sum(axis=1).mean()
                    block_difference = [gain2 * np.sum((estimate - other) *
                        (estimate + other - 2 * np.array(block["guidance"])), axis=1).mean()
                        for block in ref.get("blocks", [])]
                    labels = (row["task"],) if real else (row["model"], row["alignment"], row["reward"])
                    key = (*labels, row["route"], row["samples"], name, control, metric)
                    conditions.setdefault(key, []).append({"observation": row.get("state_row", row["context_key"]),
                        "mse": float(a), "control_mse": float(b), "difference": float(a - b),
                        "zero_mse": float(gain2 * (target @ target)),
                        "fallback": row["fallback_rate"], "reference_block_differences": block_difference,
                        "context_key": row["context_key"]})
        for key, entries in conditions.items():
            labels = dict(zip(["task"] if real else ["model", "alignment", "reward"], key[:-5]))
            route, budget, method, control, metric = key[-5:]
            observations = sorted(set(e["observation"] for e in entries))
            grouped = [[e for e in entries if e["observation"] == obs] for obs in observations]
            diff = np.array([np.mean([e["difference"] for e in block]) for block in grouped])
            mse = float(np.mean([e["mse"] for e in entries]))
            baseline = float(np.mean([e["control_mse"] for e in entries]))
            blocks = np.array([e["reference_block_differences"] for e in entries])
            summary = {"run": run.name, **labels, "route": route, "samples": budget,
                "method": method, "control": control, "metric": metric, "contexts": len(entries),
                "mse": mse, "control_mse": baseline, "difference": float(diff.mean()),
                "relative_improvement": 1 - mse / baseline if baseline > 1e-20 else None,
                "zero_mse": float(np.mean([e["zero_mse"] for e in entries])),
                "fallback_rate": float(np.mean([e["fallback"] for e in entries])),
                "units_improved": int((diff < 0).sum()), "units": len(diff),
                "reference_block_differences": blocks.mean(axis=0).tolist() if blocks.size else []}
            if real:
                rng = np.random.default_rng(85001)
                boot = diff[rng.integers(0, len(diff), size=(20000, len(diff)))].mean(axis=1)
                summary["observation_bootstrap_interval95"] = np.quantile(boot, [0.025, 0.975]).tolist()
                summary["observation_differences"] = dict(zip(map(str, observations), diff.tolist()))
            summaries.append(summary)
            contexts_out.append({"comparison": {**labels, "route": route, "samples": budget,
                                 "method": method, "control": control, "metric": metric}, "entries": entries})
        arrays.close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparisons.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    (args.output / "context-comparisons.json").write_text(json.dumps(contexts_out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"comparisons": len(summaries), "output": str(args.output)}))


if __name__ == "__main__":
    main()
