"""Conditional reward calibration on held-out frozen-critic observations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from .reward_calibration import RewardCalibration
from .uncertainty_critic import BatchCritic, dump


def difference(after, before):
    return {key: after[key] - before[key] for key in after}


def features(model, x, index):
    _, _, clean = model.transition(x, index)
    proxy = model.reward(clean).cpu().double().numpy()
    flat = x.reshape(-1, model.dimension)
    state = model.engine.state.expand(len(flat), -1)
    data = torch.cat([flat, clean.reshape_as(flat), state], dim=-1).cpu().double().numpy()
    return np.concatenate([data, proxy.reshape(-1, 1)], axis=-1), proxy


def future_rewards(model, x, index, branches, generator):
    y = x.unsqueeze(-2).expand(*x.shape[:-1], branches, model.dimension)
    return model.reward(model.rollout(y, index, generator)).cpu().double().numpy()


def anchors(model, row, count, checkpoints, seed):
    model.engine.use_state(row)
    path = model.generator(seed + row * 1009)
    x = model.noise((count, model.dimension), path)
    for index in range(max(checkpoints) + 1):
        if index in checkpoints:
            yield index, x
        if index < max(checkpoints):
            mean, std, _ = model.transition(x, index)
            x = mean + std * model.noise(mean.shape, path)


def calibrate(model, config, output):
    collected = {index: [] for index in config["checkpoints"]}
    for row in config["calibration_state_rows"]:
        future = model.generator(config["calibration_seed"] + 1000003 + row * 1009)
        for index, x in anchors(model, row, config["calibration_anchors_per_state"],
                                config["checkpoints"], config["calibration_seed"]):
            f, proxy = features(model, x, index)
            rewards = future_rewards(model, x, index, config["calibration_futures"], future)
            collected[index].append((f, proxy, rewards))
    banks = {index: RewardCalibration(*(np.concatenate([row[k] for row in records])
                                       for k in range(3)))
             for index, records in collected.items()}
    dump(output / f'{model.task["name"]}-reward-calibration.json',
         {str(index): bank.record() for index, bank in banks.items()})
    return banks


def value_error(predicted, truth):
    centered = predicted - predicted.mean() - truth + truth.mean()
    truth_log_weights = truth - logsumexp(truth)
    return {"centered_log_value_mse": float(np.mean(centered ** 2)),
            "normalized_value_kl": float(np.sum(np.exp(truth_log_weights) *
                (truth_log_weights - predicted + logsumexp(predicted))))}


def diagnose(model, banks, config, output):
    records = []
    tau = model.task["tau"]
    for row in config["diagnostic_state_rows"]:
        reference_rng = model.generator(config["diagnostic_seed"] + 1000003 + row * 1009)
        value_rng = model.generator(config["diagnostic_seed"] + 2000003 + row * 1009)
        for index, x in anchors(model, row, config["diagnostic_anchors_per_state"],
                                config["checkpoints"], config["diagnostic_seed"]):
            f, proxy = features(model, x, index)
            rewards = future_rewards(model, x, index, config["reference_futures"], reference_rng)
            scaled = rewards / tau
            truth = logsumexp(scaled, axis=-1) - math.log(scaled.shape[-1])
            half = scaled.shape[-1] // 2
            split = [logsumexp(part, axis=-1) - math.log(part.shape[-1])
                     for part in (scaled[:, :half], scaled[:, half:])]
            reference_weights = np.exp(scaled - logsumexp(scaled, axis=-1, keepdims=True))
            ess = 1 / np.sum(reference_weights ** 2, axis=-1)
            estimates = {"point": proxy / tau}
            moments = {}
            for kind in ("mean", "gaussian", "empirical", "direct"):
                estimates[kind], moments = banks[index].predict(f, proxy, tau, kind, config["neighbours"])
            estimates["pooled_action_gaussian"] = model.value(
                x, index, {"value": "gaussian", "uncertainty_samples": 2}, value_rng).cpu().numpy()
            fresh = future_rewards(model, x, index, 2, value_rng)
            estimates["future2"] = logsumexp(fresh / tau, axis=-1) - math.log(2)
            record = {"task": model.task["name"], "state_row": row, "index": index,
                      "anchors": len(x), "reference_futures": scaled.shape[-1],
                      "reference_ess_min": float(ess.min()), "reference_ess_median": float(np.median(ess)),
                      "reference_split_error": value_error(split[0], split[1]),
                      "values": {name: value_error(pred, truth) for name, pred in estimates.items()},
                      "proxy_rewards": proxy.tolist(), "reference_log_values": truth.tolist(),
                      "reference_split_log_values": [part.tolist() for part in split],
                      "reference_ess": ess.tolist(),
                      "predicted_log_values": {name: pred.tolist() for name, pred in estimates.items()},
                      "predicted_reward_mean": moments["mean"].tolist(),
                      "predicted_reward_variance": moments["variance"].tolist(),
                      "reference_reward_mean": rewards.mean(axis=-1).tolist(),
                      "reference_reward_variance": rewards.var(axis=-1).tolist()}
            records.append(record)
            dump(output / f'{model.task["name"]}-value-diagnostics.json', records)
    return records


def evaluate(config, output):
    counts, stages, summaries = [], [], []
    for task in config["tasks"]:
        model = BatchCritic(task)
        dump(output / f'{task["name"]}-models.json', model.engine.provenance)
        model.load_calibration(Path(config["pooled_calibration_directory"]) / f'{task["name"]}-calibration.json')
        try:
            before = model.engine.counts()
            banks = calibrate(model, config, output)
            stages.append({"task": task["name"], "stage": "calibration",
                           "costs": difference(model.engine.counts(), before)})
            dump(output / "stage-costs.json", stages)
            before = model.engine.counts()
            records = diagnose(model, banks, config, output)
            stages.append({"task": task["name"], "stage": "diagnostics",
                           "costs": difference(model.engine.counts(), before)})
            summaries.extend({key: record[key] for key in
                ("task", "state_row", "index", "anchors", "reference_futures", "reference_ess_min",
                 "reference_ess_median", "reference_split_error", "values")} for record in records)
            dump(output / "summary.json", summaries)
            dump(output / "stage-costs.json", stages)
        finally:
            counts.append({"task": task["name"], **model.engine.counts()})
            dump(output / "counts.json", counts)
        del model
        torch.cuda.empty_cache()
    return {"conditions": len(summaries), "score_forwards": sum(c["score_forwards"] for c in counts),
            "reward_queries": sum(c["reward_queries"] for c in counts), "stage_costs": stages,
            "interpretation": "Independent continuation-value diagnostics; no simulator or endpoint success claim."}
