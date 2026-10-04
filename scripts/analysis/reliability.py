#!/usr/bin/env python3
"""Summarize the locked SD sensitivity probe without inventing a true Jacobian."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run / "sensitivity.json").read_text())
    raw = {(r["prompt"], r["seed"], r["index"], r["group"]): r
           for r in rows if r["method"] == "raw"}
    summaries = []
    for index in sorted({r["index"] for r in rows}):
        for method in dict.fromkeys(r["method"] for r in rows):
            group = [r for r in rows if r["method"] == method and r["index"] == index]
            def values(field):
                return np.asarray([r[field] for r in group])
            keys = [(r["prompt"], r["seed"], r["index"], r["group"]) for r in group]
            item = {"index": index, "method": method, "vectors": len(group),
                    "mean_relative_difference": float(values("relative_difference").mean()),
                    "max_relative_difference": float(values("relative_difference").max()),
                    "mean_raw_cosine": float(values("raw_cosine").mean()),
                    "min_raw_cosine": float(values("raw_cosine").min()),
                    "mean_independent_group_cosine": float(values("independent_group_cosine").mean()),
                    "mean_kl_ratio_to_raw": float(np.mean([
                        r["uncapped_execution_kl"] / raw[k]["uncapped_execution_kl"]
                        for k, r in zip(keys, group)])),
                    "mean_sensitivity_seconds": float(values("sensitivity_seconds").mean()),
                    "mean_backward_seconds": float(values("vjp_seconds").mean()),
                    "score_forwards": int(values("score_forwards").sum()),
                    "vjps": int(values("vjp_calls").sum())}
            summaries.append(item)
    finite = []
    for path in sorted(args.run.glob("fd-*.json")):
        entries = json.loads(path.read_text())
        finite.append({"file": path.name, **{
            f"{direction}_minimum_relative_error": min(r["relative_error"] for r in entries
                                                       if r["direction"] == direction)
            for direction in ["raw", "random"]}})
    ratio_groups = json.loads((args.run / "ratio-groups.json").read_text())
    report = {"scope": "Eight discovery states on four unguided paths; no ideal SD Jacobian reference or endpoint claim.",
              "methods_by_index": summaries, "finite_difference": finite,
              "ratio_groups": ratio_groups,
              "identity_learned_cosine_range": [min(r["identity_learned_cosine"] for r in raw.values()),
                                                 max(r["identity_learned_cosine"] for r in raw.values())],
              "metrics": json.loads((args.run / "metrics.json").read_text())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
