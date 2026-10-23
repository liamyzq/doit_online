"""Apply the locked active-interval, observation and paired-comparison gates."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run / "results.json").read_text())
    differences = json.loads((args.run / "paired-differences.json").read_text())
    contexts = json.loads((args.run / "contexts.json").read_text())
    by_context = {row["context_key"]: row for row in contexts}
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["integrand"], row["samples"], row["method"]), []).append(row)
    comparisons, gates = [], []
    feasible = {"random_four_lhs_r1", "fd_full_four_lhs_r1", "pilot4_four_lhs_r1"}
    for key, values in groups.items():
        task, integrand, budget, method = key
        item = {"task": task, "integrand": integrand, "samples": budget, "method": method,
                "observations": len({x["state_row"] for x in values}),
                "fixed_contexts": len({x["context_key"] for x in values}),
                "mean_guidance_mse": mean(x["mse"] for x in values),
                "mean_unit_gamma_displacement_mse": mean(x["unit_gamma_displacement_mse"] for x in values),
                "mean_queries_per_call": mean(x["total_queries_per_call"] for x in values),
                "mean_score_forwards_per_call": mean(x["total_score_forwards_per_call"] for x in values),
                "mean_additional_direction_vjps": mean(x["additional_direction_vjps_per_call"] for x in values),
                "comparisons": {}}
        if "direction_capture_at_center" in values[0]:
            item["mean_center_gradient_capture"] = mean(x["direction_capture_at_center"] for x in values)
        for control in ("antithetic", "random_paired_r1", "oracle_paired_r1", "fd_full_paired_r1"):
            if method == control:
                continue
            selected = [row for row in differences if (row["task"], row["integrand"], row["samples"], row["method"], row["control"]) == (*key, control)]
            control_mse = mean(x["unit_gamma_displacement_mse"] for x in groups[(task, integrand, budget, control)])
            gains2 = [by_context[x["context_key"]]["correction_gain"] ** 2 for x in selected]
            delta = mean(x["mse_difference"] * gain for x, gain in zip(selected, gains2))
            se = math.sqrt(sum((x["mc_se"] * gain) ** 2 for x, gain in zip(selected, gains2))) / len(selected)
            blocks = [mean(x["reference_block_differences"][block] * gain for x, gain in zip(selected, gains2))
                      for block in range(len(selected[0]["reference_block_differences"]))]
            observation_differences = {str(row): mean(x["unit_gamma_displacement_difference"] for x in selected if x["state_row"] == row)
                                       for row in sorted({x["state_row"] for x in selected})}
            item["comparisons"][control] = {"relative_displacement_mse_gain": 1 - item["mean_unit_gamma_displacement_mse"] / control_mse,
                                             "displacement_mse_difference": delta,
                                             "conditional_mc_interval95": [delta - 1.96 * se, delta + 1.96 * se],
                                             "reference_block_replacement_differences": blocks,
                                             "reference_block_range": [min(blocks), max(blocks)],
                                             "observation_displacement_differences": observation_differences,
                                             "observations_improved": sum(x < 0 for x in observation_differences.values())}
        comparisons.append(item)
        if method in feasible and budget == 32 and integrand == "continuation_shared":
            comparison_checks = {}
            for control in ("antithetic", "random_paired_r1"):
                c = item["comparisons"][control]
                comparison_checks[control] = {"at_least20percent": c["relative_displacement_mse_gain"] >= .2,
                                              "at_least3of4_observations": c["observations_improved"] >= 3,
                                              "conditional_mc_difference_resolved": c["conditional_mc_interval95"][1] < 0,
                                              "all_reference_block_replacements_negative": c["reference_block_range"][1] < 0}
            improving_observations = set.intersection(*[
                {row for row, delta in item["comparisons"][control]["observation_displacement_differences"].items() if delta < 0}
                for control in ("antithetic", "random_paired_r1")])
            gates.append({"task": task, "method": method, "checks": comparison_checks,
                          "same_observations_improved_against_both": sorted(improving_observations),
                          "passes": len(improving_observations) >= 3 and all(all(x.values()) for x in comparison_checks.values())})
    result = {"comparisons": comparisons, "discovery_gates": gates,
              "eligible_for_independent_confirmation": [x for x in gates if x["passes"]],
              "reference_diagnostics": [{"task": x["task"], "state_row": x["state_row"], "index": x["index"],
                                         "candidate_ess": x["references"]["combined"]["candidate"]["ess"],
                                         "continuation_ess": x["references"]["combined"]["continuation_shared"]["ess"],
                                         "continuation_block_ess": x["references"]["combined"]["continuation_shared"]["block_ess"]} for x in contexts],
              "uncertainty_scope": "MC intervals condition on fixed contexts and pooled reference; reference-block replacements are a sensitivity diagnostic, not calibrated ratio-bias intervals; four observations per task"}
    (args.run / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"eligible_for_independent_confirmation": result["eligible_for_independent_confirmation"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
