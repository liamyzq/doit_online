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
        self.reference_contexts = None
        if "reference_source" in config:
            from pathlib import Path
            source = Path(config["reference_source"])
            original = json.loads((source / "config.json").read_text())
            # This bounded gate ablation changes no target, state, seed or budget.
            # Reject reuse if anything beyond the declared method knobs differs.
            omitted = {"id", "purpose", "routes", "agreement_threshold", "reference_source"}
            if ({k: v for k, v in config.items() if k not in omitted} !=
                    {k: v for k, v in original.items() if k not in omitted}):
                raise ValueError("Reference reuse requires identical target and fixed-state settings")
            self.reference_contexts = {c["context_key"]: c for c in json.loads((source / "contexts.json").read_text())}

    def context(self, context, coordinates, metadata):
        config, output = self.config, self.output
        key = f"context{len(self.contexts):04d}"
        if self.reference_contexts is None:
            refs = references(context, config, coordinates, output / "raw" / f"{key}-references.npz")
        else:
            source = self.reference_contexts[key]
            if any(source[k] != v for k, v in metadata.items()):
                raise ValueError("Reference context order differs from the source run")
            refs = source["references"]
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
                    before = context.counts()
                    tick = time.perf_counter()
                    guidance, arrays, info = estimate(context, method, route, budget,
                                                     config["repeats"], words, config)
                    seconds = time.perf_counter() - tick
                    after = context.counts()
                    array_key = f"condition{len(self.rows):05d}"
                    np.savez_compressed(output / "raw" / f"{array_key}.npz", **arrays)
                    self.estimates[array_key] = guidance
                    values[method["name"]] = guidance
                    row = {"context_key": key, "array_key": array_key, **metadata,
                           "route": route, "samples": budget, "method": method["name"],
                           "repeats": config["repeats"], **info,
                           "actual_batch_costs": {k: after[k] - before[k] for k in before},
                           "evaluation_seconds": seconds, **loss_summary(guidance, reference)}
                    if hasattr(context, "jacobian_seconds"):
                        uses_pool = method["selector"] in {"model", "moments", "first", "no_j"}
                        row.update(deployed_jacobian_vjps=context.dimension if uses_pool else 1,
                                   measured_context_jacobian_seconds=context.jacobian_seconds,
                                   extra_geometry_required=uses_pool)
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


def evaluate_toy(config, output):
    study, totals = Study(config, output), {"reward_queries": 0, "score_forwards": 0}
    for kind in config["models"]:
        ki = 0 if kind == "gaussian" else 1
        model = ToyProcess(kind, config["dimension"])
        if kind == "gaussian" and "reference_source" not in config:
            # A new polynomial reference must integrate the same terminal reward
            # for P and F. Sixteen Gaussian quadrature nodes integrate this
            # degree-five numerator exactly, independently of its closed form.
            check_context = ToyContext(model, 7, 1, "off_pool", "hermite4", config["pool_seed"])
            nodes, weights = np.polynomial.hermite.hermgauss(16)
            nodes, weights = nodes * np.sqrt(2), weights / np.sqrt(np.pi)
            errors = {}
            for route in ("P", "F"):
                center, response, covariance = check_context.gaussian_endpoint(route)
                axis = check_context.axis
                mean = center @ axis
                variance = axis @ (response @ response.T + covariance) @ axis
                z = mean + np.sqrt(variance) * nodes
                terminal = z[:, None] * axis
                w = np.exp(check_context.logs(terminal)) * weights
                numeric = (response.T @ axis) / variance * np.sum(w * (z - mean)) / w.sum()
                exact = check_context.analytic_reference(route)
                errors[route] = max(float(abs(w.sum() - exact["mass"])),
                                    float(np.max(np.abs(numeric - exact["ratio"]))))
            dump(output / "hermite-reference-check.json", errors)
            if max(errors.values()) > 1e-10:
                raise AssertionError(errors)
        for index in config["indices"]:
            for state in config["states"]:
                for alignment in config["alignments"]:
                    ai = 0 if alignment == "in_pool" else 1
                    for ri, reward in enumerate(config["rewards"]):
                        context = ToyContext(model, index, state, alignment, reward, config["pool_seed"])
                        study.context(context, (ki, index, state, ai, ri),
                                      {"model": kind, "index": index, "state": state,
                                       "alignment": alignment, "reward": reward})
        for key, value in model.counts().items():
            totals[key] += value
    np.savez_compressed(output / "estimates.npz", **study.estimates)
    metrics = {"conditions": len(study.rows), "contexts": len(study.contexts), "costs": totals,
               "seconds": time.perf_counter() - study.started,
               "scope": "Actual sixteen-step Gaussian transition process; analytic references only for the Gaussian model"}
    dump(output / "summary.json", metrics)
    return metrics
