"""Current/future noise sign coupling under the unchanged real-critic kernel."""

import time

import numpy as np
import torch

from .complement import stream, loss_summary, paired_difference
from .complement_critic import generator, continue_group, references
from .geometry import ratio_estimate
from .geometry_critic import dump
from .joint_future import sign_map, main_blocks, future_blocks
from .uncertainty_critic import BatchCritic


class KernelContext:
    def __init__(self, model, x, index):
        self.model, self.index, self.dimension = model, index, model.dimension
        self.transition = model.engine.transition(x, index, True)
        self.jmu = model.engine.jacobian(self.transition).detach().cpu().double().numpy()
        self.mean = self.transition.mean.detach().cpu().double().numpy()
        self.std = self.transition.noise_std

    def guidance(self, ratio):
        return ratio @ self.jmu / self.std


def torch_future_blocks(raw, method, blocks):
    main_signs, future_signs = sign_map(method)
    group = len(main_signs)
    if future_signs is None:
        return raw[:, :blocks * group].reshape(raw.shape[0], blocks, group, raw.shape[-1])
    signs = raw.new_tensor(future_signs).reshape(1, 1, group, 1)
    return raw[:, :blocks, None] * signs


def continue_coupled(model, x, start, rng, method, capacity):
    """Keep [repeat, pair, sign, coordinate] order throughout the rollout."""
    for index in range(start, model.steps):
        mean, std, _ = model.transition(x, index)
        raw = model.noise((x.shape[0], capacity, x.shape[-1]), rng)
        x = mean + std * torch_future_blocks(raw, method, x.shape[1])
    return x


def log_weights(context, eps, rng, method):
    x = torch.as_tensor(context.mean + context.std * eps, dtype=torch.float32, device="cuda")
    clean = continue_coupled(context.model, x, context.index + 1, rng, method, eps.shape[1] * eps.shape[2])
    return (context.model.reward(clean).double() / context.model.task["tau"]).cpu().numpy()


def smoke_checks(context, config, output):
    model, d = context.model, context.dimension
    errors = {}
    raw = np.arange(3 * 8 * d, dtype=float).reshape(3, 8, d)
    for method in config["methods"]:
        actual = torch_future_blocks(torch.as_tensor(raw, device="cuda"), method, 4).cpu().numpy()
        errors[f"{method}_future_map_max"] = float(abs(actual - future_blocks(raw, method, 4)).max())
        eps = main_blocks(stream(79019, 1), stream(79019, 2), 3, 8, d, method)
        constant = context.guidance(ratio_estimate(eps.reshape(3, 8, d), np.zeros((3, 8))))
        errors[f"{method}_constant_guidance_max"] = float(abs(constant).max())
    eps = main_blocks(stream(79023, 1), stream(79023, 2), 2, 8, d, "shared_antithetic")
    x = torch.as_tensor(context.mean + context.std * eps, dtype=torch.float32, device="cuda")
    actual = continue_coupled(model, x, context.index + 1, model.generator(79029), "independent_future", 8)
    expected = model.rollout(x.reshape(2, 8, d), context.index + 1, model.generator(79029))
    errors["independent_rollout_max"] = float((actual.reshape(2, 8, d) - expected).abs().max())
    actual = continue_coupled(model, x, context.index + 1, model.generator(79031), "shared_antithetic", 8)
    block_order = x.transpose(1, 2).reshape(2, 8, d)
    expected = continue_group(model, block_order, context.index + 1, model.generator(79031), 2, capacity=8)
    expected = expected.reshape(2, 2, 4, d).transpose(1, 2)
    errors["shared_rollout_max"] = float((actual - expected).abs().max())
    dump(output / "checks.json", errors)
    if any(value > 2e-6 for value in errors.values()):
        raise AssertionError(errors)
    return errors


def evaluate(config, output):
    started = time.perf_counter()
    rows, differences, contexts, cost_rows, vectors, matrices = [], [], [], [], {}, {}
    checks, jacobian_vjps = {}, 0
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
                            raise ValueError("Comparison must be inside the active guidance interval")
                        context = KernelContext(model, x, index)
                        jacobian_vjps += model.dimension
                        context_key = f"context{len(contexts):03d}"
                        matrices[f"{context_key}_kernel_jacobian"] = context.jmu
                        matrices[f"{context_key}_mean"] = context.mean
                        if config["smoke"]:
                            checks[context_key] = smoke_checks(context, config, output)
                        refs = references(context, config, (ti, state_row, index), ("continuation_shared",))
                        ref = refs["combined"]["continuation_shared"]
                        target = np.array(ref["guidance"])
                        gain = context.transition.correction_gain
                        contexts.append({"context_key": context_key, "task": task["name"], "state_row": state_row,
                                         "index": index, "correction_gain": gain, "active_guidance": True,
                                         "sigma": context.std, "actual_kernel_jacobian_vjp_calls": model.dimension,
                                         "references": refs})
                        for seed in config["seeds"]:
                            for budget in config["samples"]:
                                estimates_by_method = {}
                                for method in config["methods"]:
                                    before = model.engine.counts()
                                    torch.cuda.synchronize()
                                    tick = time.perf_counter()
                                    eps = main_blocks(stream(seed, ti, state_row, index, budget, 1),
                                                      stream(seed, ti, state_row, index, budget, 2),
                                                      config["repeats"], budget, model.dimension, method)
                                    logs = log_weights(context, eps, generator(model, seed, ti, state_row, index, budget, 3), method)
                                    estimates = context.guidance(ratio_estimate(eps.reshape(config["repeats"], budget, model.dimension),
                                                                              logs.reshape(config["repeats"], budget)))
                                    torch.cuda.synchronize()
                                    seconds = time.perf_counter() - tick
                                    after = model.engine.counts()
                                    counts = {key: after[key] - before[key] for key in before}
                                    if config["smoke"]:
                                        errors = {"reward_queries": counts["reward_queries"] - config["repeats"] * budget,
                                                  "score_forwards": counts["score_forwards"] - config["repeats"] * budget * (model.steps - index - 1)}
                                        checks[f"{context_key}_{seed}_{budget}_{method}_counts"] = errors
                                        dump(output / "checks.json", checks)
                                        if any(errors.values()):
                                            raise AssertionError(errors)
                                    array_key = f"c{len(rows):04d}"
                                    vectors[array_key] = estimates
                                    estimates_by_method[method] = estimates
                                    record = {"context_key": context_key, "array_key": array_key,
                                              "task": task["name"], "state_row": state_row, "index": index,
                                              "seed": seed, "samples": budget, "method": method, "repeats": config["repeats"],
                                              "direction_queries_per_call": 0, "total_queries_per_call": budget,
                                              "total_score_forwards_per_call": 1 + budget * (model.steps - index - 1),
                                              "common_kernel_vjp_calls_per_deployed_estimate": 1,
                                              "actual_batch_costs": counts, "actual_batch_seconds": seconds,
                                              **loss_summary(estimates, target)}
                                    record["unit_gamma_displacement_mse"] = gain ** 2 * record["mse"]
                                    rows.append(record)
                                for method, estimates in estimates_by_method.items():
                                    for control in config["controls"]:
                                        if method == control:
                                            continue
                                        baseline = estimates_by_method[control]
                                        uncertainty = paired_difference(estimates, baseline, target)
                                        block_differences = [paired_difference(estimates, baseline, np.array(reference))["mse_difference"]
                                                             for reference in ref["block_guidance"]]
                                        differences.append({"context_key": context_key, "task": task["name"], "state_row": state_row,
                                                            "index": index, "seed": seed, "samples": budget, "method": method,
                                                            "control": control, **uncertainty,
                                                            "reference_block_differences": block_differences,
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
            dump(output / "costs.json", {"tasks": cost_rows, "actual_kernel_jacobian_vjp_calls": jacobian_vjps})
            del model
            torch.cuda.empty_cache()
    metrics = {"conditions": len(rows), "contexts": len(contexts), "checks": checks,
               "reward_queries": sum(row["reward_queries"] for row in cost_rows),
               "score_forwards": sum(row["score_forwards"] for row in cost_rows),
               "actual_kernel_jacobian_vjp_calls": jacobian_vjps, "seconds": time.perf_counter() - started,
               "scope": "Fixed active-interval contexts; timing batches is not individual-call deployment timing; no simulator return claim"}
    dump(output / "summary.json", metrics)
    return metrics
