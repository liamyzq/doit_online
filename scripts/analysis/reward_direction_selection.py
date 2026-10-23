"""Read saved selection artifacts to separate pilot failure from pool coverage."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        contexts = {c["context_key"]: c for c in json.loads((run / "contexts.json").read_text())}
        conditions = json.loads((run / "results.json").read_text())
        for condition in conditions:
            if condition["method"] in {"antithetic", "random_pair", "random_four"}:
                continue
            context = contexts[condition["context_key"]]
            ref = context["references"]["P" if condition["route"] == "P" else "F"]
            axis = np.array(ref.get("oracle", context.get("reward_axis", [])))
            with np.load(run / "raw" / (condition["array_key"] + ".npz")) as raw:
                enabled = raw["enabled"]
                result = {"run": run.name, **{k: condition[k] for k in
                    ("context_key", "array_key", "route", "samples", "method")},
                    "enabled_fraction": float(enabled.mean())}
                for key in ("task", "state_row", "model", "alignment", "reward", "state", "index"):
                    if key in condition:
                        result[key] = condition[key]
                if "agreement" in raw:
                    result.update(mean_agreement=float(raw["agreement"].mean()),
                                  mean_pilot_ess=float(raw["pilot_ess"].mean()),
                                  pilot_ess_quantiles=np.quantile(raw["pilot_ess"], [0,.1,.5,.9,1]).tolist())
                if len(axis):
                    capture = np.square(raw["selected_direction"] @ axis)
                    result.update(mean_direction_capture=float(capture.mean()),
                                  exact_axis_fraction=float((capture > 1 - 1e-10).mean()),
                                  enabled_direction_capture=float(capture[enabled].mean()) if enabled.any() else None)
            if len(axis):
                with np.load(run / "raw" / (condition["context_key"] + "-matrices.npz")) as matrix:
                    result["best_pool_capture"] = float(np.square(axis @ matrix["pool"]).max())
                result["direction_reference"] = "exact_gaussian_pullback" if "oracle" in ref else "structural_reward_axis"
            rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
