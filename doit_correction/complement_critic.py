"""Direction acquisition and full-score sign groups on authentic RL critics."""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.special import logsumexp

from .complement import stream, sign_probes, loss_summary, paired_difference
from .geometry import ratio_estimate
from .geometry_critic import Context, denoise, dump
from .uncertainty_critic import BatchCritic


def generator(model, seed, *coordinates):
    return model.generator(int(stream(seed, *coordinates).integers(0, 2**63 - 1)))


def normalized(direction):
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    fallback = np.zeros_like(direction)
    fallback[..., 0] = 1
    return np.where(norm > 1e-12, direction / np.maximum(norm, 1e-12), fallback)


def continue_group(model, x, start, rng, group, capacity=None):
    """Block-ordered groups share future noises, never their reward evaluations."""
    capacity = x.shape[1] if capacity is None else capacity
    for index in range(start, model.steps):
        mean, std, _ = model.transition(x, index)
        noise = model.noise((x.shape[0], capacity, x.shape[2]), rng)[:, :x.shape[1] // group]
        x = mean + std * torch.cat([noise] * group, dim=1)
    return x


def log_weights(context, eps, integrand, rng, group, capacity=None):
    if integrand == "candidate":
        clean = context.clean(eps, "candidate")
    else:
        x = torch.as_tensor(context.mean + context.std * eps, dtype=torch.float32, device="cuda")
        clean = continue_group(context.model, x, context.index + 1, rng,
                               group if integrand == "continuation_shared" else 1, capacity)
    return (context.model.reward(clean).double() / context.model.task["tau"]).cpu().numpy()


def acquisition(context, radius):
    """Measure fixed directions once; record what each algorithm must pay anew."""
    model, d = context.model, context.dimension
    torch.cuda.synchronize()
    tick = time.perf_counter()
    center = torch.as_tensor(context.mean[None], dtype=torch.float32, device="cuda").requires_grad_(True)
    clean = denoise(model, center, context.index + 1, True)
    value = model.engine.reward(clean, with_grad=True).sum() / model.task["tau"]
    gradient = torch.autograd.grad(value, center)[0].detach().flatten().cpu().double().numpy() * context.std
    torch.cuda.synchronize()
    oracle_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    eps = radius * np.concatenate([np.eye(d), -np.eye(d)])[None]
    logs = log_weights(context, eps, "candidate", None, 1)[0]
    finite_difference = (logs[:d] - logs[d:]) / (2 * radius)
    torch.cuda.synchronize()
    fd_seconds = time.perf_counter() - tick
    _, singular_values, vh = np.linalg.svd(context.response, full_matrices=False)
    bases = {"symmetric": context.vectors[:, :1], "svd": vh[:1].T,
             "oracle": normalized(gradient)[:, None], "fd_full": normalized(finite_difference)[:, None]}
    costs = {
        "symmetric": {"reward_queries": 0, "score_forwards": 1, "additional_vjp_calls": d, "seconds": context.response_seconds},
        "svd": {"reward_queries": 0, "score_forwards": 1, "additional_vjp_calls": d, "seconds": context.response_seconds},
        "oracle": {"reward_queries": 1, "score_forwards": 1, "additional_vjp_calls": 1, "seconds": oracle_seconds},
        "fd_full": {"reward_queries": 2 * d, "score_forwards": 2 * d, "additional_vjp_calls": 0, "seconds": fd_seconds},
    }
    return bases, costs, {"candidate_log_weight_gradient": gradient,
                          "finite_difference_gradient": finite_difference,
                          "singular_values": singular_values, "right_singular_vectors": vh.T}


def references(context, config, seed_words):
    blocks = []
    for block in range(config["reference_blocks"]):
        eps = sign_probes(stream(config["reference_seed"], *seed_words, block, 1),
                          1, config["reference_block_samples"], context.dimension)
        entry = {}
        for ii, integrand in enumerate(("candidate", "continuation_shared")):
            logs = log_weights(context, eps, integrand,
                               generator(context.model, config["reference_seed"], *seed_words, block, ii, 2), 2)[0]
            ratio = ratio_estimate(eps, logs[None])[0]
            logmass = float(logsumexp(logs) - np.log(len(logs)))
            weights = np.exp(logs - logsumexp(logs))
            entry[integrand] = {"ratio": ratio.tolist(), "guidance": context.guidance(ratio).tolist(),
                                "log_mean_weight": logmass, "ess": float(1 / np.square(weights).sum())}
        blocks.append(entry)
    combined = {}
    for integrand in ("candidate", "continuation_shared"):
        entries = [block[integrand] for block in blocks]
        masses = np.array([x["log_mean_weight"] for x in entries])
        relative_mass = np.exp(masses - logsumexp(masses))
        ratios = np.array([x["ratio"] for x in entries])
        combined[integrand] = {
            "guidance": context.guidance(relative_mass @ ratios).tolist(),
            "ess": float(1 / np.sum(np.square(relative_mass) / np.array([x["ess"] for x in entries]))),
            "log_mean_weight": float(logsumexp(masses) - np.log(len(entries))),
            "block_guidance": [x["guidance"] for x in entries],
            "block_ess": [x["ess"] for x in entries],
            "block_log_mean_weights": masses.tolist(),
        }
    return {"blocks": blocks, "combined": combined}


def smoke_checks(context, bases, info, config):
    """Only new reward-gradient orientation and future-coupling paths."""
    model, d = context.model, context.dimension
    # These FP32 critics have large constant Q offsets. At h=1e-3 the Q
    # difference is near quantization resolution; exp-072's diagnostic found
    # h=0.1 resolves the derivative on both tasks without changing the oracle.
    h = 0.1
    direction = normalized(np.arange(1, d + 1, dtype=float))
    eps = np.stack([h * direction, -h * direction])[None]
    logs = log_weights(context, eps, "candidate", None, 1)[0]
    numeric = (logs[0] - logs[1]) / (2 * h)
    analytic = np.dot(info["candidate_log_weight_gradient"], direction)
    derivative_error = abs(numeric - analytic) / max(abs(analytic), 1e-4)
    x = torch.as_tensor(context.mean + context.std * stream(72001, d).normal(size=(2, 8, d)), dtype=torch.float32, device="cuda")
    actual = continue_group(model, x.clone(), context.index + 1, model.generator(72003), 1)
    expected = model.rollout(x.clone(), context.index + 1, model.generator(72003))
    duplicate = x[:, :2].repeat(1, 4, 1)
    shared = continue_group(model, duplicate, context.index + 1, model.generator(72007), 4)
    sharing_error = max(float((shared[:, :2] - shared[:, k:k + 2]).abs().max()) for k in (2, 4, 6))
    errors = {"candidate_gradient_fd_relative": float(derivative_error),
              "candidate_gradient_fd_step": h,
              "candidate_gradient_numeric": float(numeric),
              "candidate_gradient_autograd": float(analytic),
              "independent_continuation_adapter_max": float((actual - expected).abs().max()),
              "shared_duplicate_continuation_max": sharing_error}
    if derivative_error > 0.04 or errors["independent_continuation_adapter_max"] > 2e-6 or sharing_error > 2e-6:
        raise AssertionError(errors)
    return errors


def evaluate(config, output):
    started = time.perf_counter()
    rows, differences, contexts, cost_rows, vectors, matrices = [], [], [], [], {}, {}
    checks, additional_vjps = {}, 0
    for ti, task in enumerate(config["tasks"]):
        model = BatchCritic(task, config["batch"])
        try:
            dump(output / f"{task['name']}-model.json", model.engine.provenance)
            for state_row in config["state_rows"]:
                model.engine.use_state(state_row)
                path_rng = generator(model, config["path_seed"], ti, state_row)
                x = model.noise((model.dimension,), path_rng)
                for index in range(max(config["indices"]) + 1):
                    if index in config["indices"]:
                        if not 0 < index <= task["guidance_end_index"]:
                            raise ValueError("The configured comparison must be in the active guidance interval")
                        context = Context(model, x, index)
                        bases, setup, acquisition_info = acquisition(context, config["pilot_radius"])
                        additional_vjps += context.vjps + 1
                        ci = len(contexts)
                        context_key = f"context{ci:03d}"
                        matrices.update({f"{context_key}_{key}": value for key, value in
                                         {"response": context.response, "kernel_jacobian": context.jmu,
                                          "mean": context.mean, "symmetric_eigenvectors": context.vectors,
                                          **acquisition_info}.items()})
                        for name, basis in bases.items():
                            matrices[f"{context_key}_basis_{name}"] = basis
                        if config.get("smoke"):
                            checks[context_key] = smoke_checks(context, bases, acquisition_info, config)
                        refs = references(context, config, (ti, state_row, index))
                        gain = context.transition.correction_gain
                        info = {"context_key": context_key, "task": task["name"], "state_row": state_row,
                                "index": index, "correction_gain": gain, "active_guidance": True,
                                **context.describe(), "direction_setup_per_guidance_call": setup,
                                "direction_alignment_to_candidate_gradient": {
                                    name: float(np.square(basis.T @ bases["oracle"]).sum()) for name, basis in bases.items()},
                                "references": refs}
                        contexts.append(info)
                        for seed in config["seeds"]:
                            for ii, integrand in enumerate(config["integrands"]):
                                ref = refs["combined"]["candidate" if integrand == "candidate" else "continuation_shared"]
                                target = np.array(ref["guidance"])
                                for budget in config["samples"]:
                                    estimates_by_method = {}
                                    for method in config["methods"]:
                                        name, direction_kind = method["name"], method["basis"]
                                        before = model.engine.counts()
                                        torch.cuda.synchronize()
                                        tick = time.perf_counter()
                                        if direction_kind in bases:
                                            basis = bases[direction_kind]
                                            direction_cost = setup[direction_kind]
                                        elif direction_kind == "random":
                                            basis = np.linalg.qr(stream(seed, ti, state_row, index, budget, 4).normal(size=(config["repeats"], model.dimension, 1)))[0]
                                            direction_cost = {"reward_queries": 0, "score_forwards": 0, "additional_vjp_calls": 0, "seconds": 0}
                                        elif direction_kind == "pilot4":
                                            rng = stream(seed, ti, state_row, index, budget, 5)
                                            directions = np.linalg.qr(rng.normal(size=(config["repeats"], model.dimension, 2)))[0]
                                            offsets = config["pilot_radius"] * directions.transpose(0, 2, 1)
                                            pilot_logs = log_weights(context, np.concatenate([offsets, -offsets], axis=1), "candidate", None, 1)
                                            gradient = np.einsum("ndr,nr->nd", directions, (pilot_logs[:, :2] - pilot_logs[:, 2:]) / (2 * config["pilot_radius"]))
                                            basis = normalized(gradient)[..., None]
                                            direction_cost = {"reward_queries": 4, "score_forwards": 4, "additional_vjp_calls": 0, "seconds": 0}
                                        else:
                                            basis = None
                                            direction_cost = {"reward_queries": 0, "score_forwards": 0, "additional_vjp_calls": 0, "seconds": 0}
                                        group = 4 if method.get("four", False) else 2
                                        main_samples = group * ((budget - direction_cost["reward_queries"]) // group)
                                        eps = sign_probes(stream(seed, ti, state_row, index, budget, 6), config["repeats"], main_samples,
                                                          model.dimension, basis, method.get("four", False), method.get("lhs", False), capacity=budget)
                                        logs = log_weights(context, eps, integrand, generator(model, seed, ti, state_row, index, budget, 7), group, capacity=budget)
                                        estimates = context.guidance(ratio_estimate(eps, logs))
                                        torch.cuda.synchronize()
                                        seconds = time.perf_counter() - tick
                                        after = model.engine.counts()
                                        array_key = f"c{len(rows):04d}"
                                        vectors[array_key] = estimates
                                        estimates_by_method[name] = estimates
                                        counts = {key: after[key] - before[key] for key in before}
                                        record = {"context_key": context_key, "array_key": array_key,
                                                  "task": task["name"], "state_row": state_row, "index": index,
                                                  "seed": seed, "integrand": integrand, "samples": budget,
                                                  "main_samples": main_samples, "method": name, "repeats": config["repeats"],
                                                  "direction_queries_per_call": direction_cost["reward_queries"],
                                                  "total_queries_per_call": main_samples + direction_cost["reward_queries"],
                                                  "total_score_forwards_per_call": 1 + direction_cost["score_forwards"] + main_samples * (1 if integrand == "candidate" else model.steps - index - 1),
                                                  "additional_direction_vjps_per_call": direction_cost["additional_vjp_calls"],
                                                  "common_kernel_vjp_calls_per_deployed_estimate": 1,
                                                  "actual_batch_costs": counts, "actual_batch_seconds": seconds,
                                                  "fixed_direction_setup_seconds": direction_cost["seconds"],
                                                  **loss_summary(estimates, target)}
                                        record["unit_gamma_displacement_mse"] = gain ** 2 * record["mse"]
                                        if basis is not None:
                                            record["direction_capture_at_center"] = float(np.square(np.einsum("d,...dr->...r", bases["oracle"][:, 0], basis)).sum(axis=-1).mean())
                                        rows.append(record)
                                    for name, estimates in estimates_by_method.items():
                                        for control in config["controls"]:
                                            if name == control:
                                                continue
                                            baseline = estimates_by_method[control]
                                            uncertainty = paired_difference(estimates, baseline, target)
                                            block_differences = [paired_difference(estimates, baseline, np.array(reference))["mse_difference"]
                                                                 for reference in ref["block_guidance"]]
                                            differences.append({"context_key": context_key, "task": task["name"], "state_row": state_row,
                                                                "index": index, "seed": seed, "integrand": integrand,
                                                                "samples": budget, "method": name, "control": control,
                                                                **uncertainty,
                                                                "reference_block_differences": block_differences,
                                                                "reference_block_range": [min(block_differences), max(block_differences)],
                                                                "unit_gamma_displacement_difference": gain ** 2 * uncertainty["mse_difference"]})
                        dump(output / "contexts.json", contexts)
                        dump(output / "results.json", rows)
                        dump(output / "paired-differences.json", differences)
                        np.savez_compressed(output / "estimates.npz", **vectors)
                        np.savez_compressed(output / "matrices.npz", **matrices)
                        print(f"Completed {context_key}: {task['name']} row{state_row} index{index}; {len(rows)} conditions", flush=True)
                    mean, std, _ = model.transition(x, index)
                    x = mean + std * model.noise(mean.shape, path_rng)
        finally:
            cost_rows.append({"task": task["name"], **model.engine.counts()})
            dump(output / "costs.json", {"tasks": cost_rows, "actual_jacobian_and_oracle_vjp_calls": additional_vjps})
            del model
            torch.cuda.empty_cache()
    metrics = {"conditions": len(rows), "contexts": len(contexts), "checks": checks,
               "reward_queries": sum(x["reward_queries"] for x in cost_rows),
               "score_forwards": sum(x["score_forwards"] for x in cost_rows),
               "actual_jacobian_and_oracle_vjp_calls": additional_vjps,
               "seconds": time.perf_counter() - started,
               "scope": "Fixed actor/critic contexts in the active interval; no simulator return or standalone runtime claim"}
    dump(output / "summary.json", metrics)
    return metrics
