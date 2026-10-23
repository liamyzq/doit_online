"""Shared fixed-state evaluation and detailed artifacts for reward directions."""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.special import logsumexp

from .complement import stream, sign_probes, loss_summary, paired_difference
from .geometry import ratio_estimate
from .geometry_toy import dump
from .reward_directions import estimate, group_ids
from .reward_direction_toy import ToyProcess, ToyContext


def references(context, config, coordinates, output):
    result, raw = {}, {}
    for route in ("P", "F"):
        exact = context.analytic_reference(route)
        if exact is not None:
            result[route] = exact
            continue
        entries = []
        for block in range(config["reference_blocks"]):
            n = config["reference_block_samples"]
            eps = sign_probes(stream(config["reference_seed"], *coordinates, block, 1), 1, n, context.dimension)
            groups = group_ids(1, n, 2)
            logs = context.evaluate(eps, route, (config["reference_seed"], *coordinates, block, 2), groups, n)
            weights = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
            ratio = ratio_estimate(eps, logs)[0]
            entries.append({"ratio": ratio.tolist(), "guidance": (ratio @ context.jmu / context.std).tolist(),
                            "log_mass": float(logsumexp(logs) - np.log(n)),
                            "ess": float(1 / np.square(weights).sum()),
                            "independent_future_groups": n // 2 if route == "F" else 0})
            raw.update({f"{route}{block}_{key}": value for key, value in
                        {"eps": eps, "logs": logs, "group_ids": groups}.items()})
        logmass = np.array([x["log_mass"] for x in entries])
        weights = np.exp(logmass - logsumexp(logmass))
        ratio = weights @ np.array([x["ratio"] for x in entries])
        result[route] = {"kind": "independent_simulation", "blocks": entries,
                         "guidance": (ratio @ context.jmu / context.std).tolist(),
                         "ratio": ratio.tolist(), "log_mass": float(logsumexp(logmass) - np.log(len(entries))),
                         "ess": float(1 / np.sum(weights**2 / np.array([x["ess"] for x in entries]))),
                         "independent_future_groups": sum(x["independent_future_groups"] for x in entries)}
    if raw:
        np.savez_compressed(output, **raw)
    return result


class Study:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.rows, self.contexts, self.differences = [], [], []
        self.estimates = {}
        self.started = time.perf_counter()
        (output / "raw").mkdir()

    def context(self, context, coordinates, metadata):
        config, output = self.config, self.output
        key = f"context{len(self.contexts):04d}"
        refs = references(context, config, coordinates, output / "raw" / f"{key}-references.npz")
        np.savez_compressed(output / "raw" / f"{key}-matrices.npz", jmu=context.jmu,
                            pool=context.pool, mean=context.mean, x=context.x)
        self.contexts.append({"context_key": key, **metadata, **context.describe(), "references": refs})
        for route, methods in config["routes"].items():
            target_route = "P" if route == "P" else "F"
            reference = np.array(refs[target_route]["guidance"])
            if "oracle" in refs[target_route]:
                context.oracle = np.array(refs[target_route]["oracle"])
            for budget in config["samples"]:
                values = {}
                for method in methods:
                    words = (config["seed"], *coordinates, budget)
                    # Paired methods share stream coordinates; different budgets,
                    # contexts, pilot/main and reference streams are separate.
                    before = context.model.counts()
                    tick = time.perf_counter()
                    guidance, arrays, info = estimate(context, method, route, budget,
                                                     config["repeats"], words, config)
                    seconds = time.perf_counter() - tick
                    after = context.model.counts()
                    array_key = f"condition{len(self.rows):05d}"
                    np.savez_compressed(output / "raw" / f"{array_key}.npz", **arrays)
                    self.estimates[array_key] = guidance
                    values[method["name"]] = guidance
                    row = {"context_key": key, "array_key": array_key, **metadata,
                           "route": route, "samples": budget, "method": method["name"],
                           "repeats": config["repeats"], **info,
                           "actual_batch_costs": {k: after[k] - before[k] for k in before},
                           "evaluation_seconds": seconds, **loss_summary(guidance, reference)}
                    row["displacement_mse"] = context.gain**2 * row["mse"]
                    row["zero_displacement_mse"] = context.gain**2 * float(reference @ reference)
                    if route == "P":
                        full = np.array(refs["F"]["guidance"])
                        row["proxy_to_full_displacement_mse"] = context.gain**2 * loss_summary(guidance, full)["mse"]
                    self.rows.append(row)
                for name, guidance in values.items():
                    for control in config["controls"]:
                        if control not in values or name == control:
                            continue
                        self.differences.append({"context_key": key, **metadata, "route": route,
                            "samples": budget, "method": name, "control": control,
                            "gain2": context.gain**2, **paired_difference(guidance, values[control], reference),
                            "reference_block_mse_differences": [paired_difference(guidance, values[control],
                                np.array(block["guidance"]))["mse_difference"] for block in refs[target_route].get("blocks", [])]})
        self.save()
        print(json.dumps({"context": key, **metadata, "conditions": len(self.rows)}), flush=True)

    def save(self):
        dump(self.output / "results.json", self.rows)
        dump(self.output / "contexts.json", self.contexts)
        dump(self.output / "paired-differences.json", self.differences)
        np.savez_compressed(self.output / "estimates.npz", **self.estimates)


def evaluate_toy(config, output):
    study, totals = Study(config, output), {"reward_queries": 0, "score_forwards": 0}
    for ki, kind in enumerate(config["models"]):
        model = ToyProcess(kind, config["dimension"])
        for index in config["indices"]:
            for state in config["states"]:
                for ai, alignment in enumerate(config["alignments"]):
                    for ri, reward in enumerate(config["rewards"]):
                        context = ToyContext(model, index, state, alignment, reward, config["pool_seed"])
                        study.context(context, (ki, index, state, ai, ri),
                                      {"model": kind, "index": index, "state": state,
                                       "alignment": alignment, "reward": reward})
        for key, value in model.counts().items():
            totals[key] += value
    metrics = {"conditions": len(study.rows), "contexts": len(study.contexts), "costs": totals,
               "seconds": time.perf_counter() - study.started,
               "scope": "Actual sixteen-step Gaussian transition process; analytic references only for the Gaussian model"}
    dump(output / "summary.json", metrics)
    return metrics
