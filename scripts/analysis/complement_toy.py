"""Summarize locked toy comparisons without selecting favorable conditions."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run / "results.json").read_text())
    differences = json.loads((args.run / "paired-differences.json").read_text())
    groups, comparisons = {}, []
    for row in rows:
        key = (row["model"], row["integrand"], row["samples"], row["method"])
        groups.setdefault(key, []).append(row)
    for key, values in groups.items():
        kind, integrand, samples, method = key
        entry = {"model": kind, "integrand": integrand, "samples": samples, "method": method,
                 **{field: mean(x[field] for x in values) for field in
                    ("mse", "active_mse", "complement_mse", "estimated_squared_bias", "variance")}}
        if "mean_direction_capture" in values[0]:
            entry["mean_direction_capture"] = mean(x["mean_direction_capture"] for x in values)
        entry["comparisons"] = {}
        for control in ("antithetic", "oracle_paired_r1"):
            if method == control:
                continue
            baseline = mean(x["mse"] for x in groups[(kind, integrand, samples, control)])
            paired = [x for x in differences if (x["model"], x["integrand"], x["samples"], x["method"], x["control"]) == (*key, control)]
            delta = mean(x["mse_difference"] for x in paired)
            se = math.sqrt(sum(x["mc_se"] ** 2 for x in paired)) / len(paired)
            entry["comparisons"][control] = {"relative_mse_gain": 1 - entry["mse"] / baseline,
                                              "mse_difference": delta,
                                              "mc_interval95": [delta - 1.96 * se, delta + 1.96 * se]}
        comparisons.append(entry)
    gate = []
    for row in comparisons:
        if row["samples"] == 32 and row["method"] == "oracle_four_lhs_r1" and row["integrand"] in ("candidate", "continuation"):
            gate.append({"model": row["model"], "integrand": row["integrand"],
                         "passes": all(x["relative_mse_gain"] >= 0.2 and x["mc_interval95"][1] < 0 for x in row["comparisons"].values()),
                         "comparisons": row["comparisons"]})
    result = {"comparisons": comparisons, "mechanism_gate": gate,
              "advance_to_bounded_critic": len(gate) == 4 and all(x["passes"] for x in gate),
              "interval_scope": "Fixed analytic model Monte Carlo uncertainty; independent seeds pooled; no generalization claim"}
    if args.replay:
        replay = json.loads((args.replay / "replay.json").read_text())
        result["old_top_m32_decomposition"] = []
        for kind in ("gaussian", "mixture"):
            for method in ("antithetic", "random_paired_lhs_r2", "top_paired_lhs_r2"):
                values = [x for x in replay if x["model"] == kind and x["method"] == method]
                item = {"model": kind, "method": method,
                        **{k: mean(x[k] for x in values) for k in ("mse", "active_mse", "complement_mse")}}
                item["complement_fraction"] = item["complement_mse"] / item["mse"]
                result["old_top_m32_decomposition"].append(item)
    (args.run / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"advance_to_bounded_critic": result["advance_to_bounded_critic"], "mechanism_gate": gate}, indent=2))


if __name__ == "__main__":
    main()
