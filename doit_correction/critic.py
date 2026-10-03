"""Low-dimensional action diffusion with authentic, fixed observation states."""

from __future__ import annotations

import functools
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .estimators import IntegrableSurrogate, MomentEstimator, gaussian_noise


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass
class ActionTransition:
    x: torch.Tensor
    mean: torch.Tensor
    endpoint_mean: torch.Tensor
    endpoint_std: float
    noise_std: float
    correction_gain: float
    epsilon: torch.Tensor
    alpha_s: float
    alpha_t: float
    sigma_s: float
    sigma_t: float
    epsilon_coefficient: float


class CriticEngine:
    def __init__(self, config):
        self.config = config
        self.device = "cuda"
        source = Path(config["source"])
        sys.path.insert(0, str(source))
        from diffusion_SDE.model import ScoreNet
        from diffusion_SDE.schedule import marginal_prob_std

        self.states = np.load(Path(config["states"]) / "observations.npy")
        self.actions = np.load(Path(config["states"]) / "actions.npy")
        self.selection = json.loads((Path(config["states"]) / "selection.json").read_text())
        self.dimension = self.actions.shape[1]
        args = SimpleNamespace(device=self.device, alpha=1.0)
        marginal = functools.partial(marginal_prob_std, device=self.device)
        self.model = ScoreNet(input_dim=self.states.shape[1] + self.dimension,
                              output_dim=self.dimension, marginal_prob_std=marginal,
                              args=args).to(self.device)
        actor = source / config["actor"]
        critic = source / config["critic"]
        self.model.load_state_dict(torch.load(actor, map_location=self.device, weights_only=True))
        incompatible = self.model.q[0].load_state_dict(
            torch.load(critic, map_location=self.device, weights_only=True), strict=False)
        # The source stores q in a Python list, outside model.parameters().
        self.model.eval().requires_grad_(False)
        self.model.q[0].eval().requires_grad_(False)
        self.model.q[0].guidance_scale = 0.0
        self.schedule = self.model.noise_schedule
        self.timesteps = self.model.dpm_solver.get_time_steps(
            skip_type="time_uniform", t_T=1.0, t_0=0.001,
            N=int(config.get("steps", 15)), device=self.device)
        self.reward_queries = 0
        self.reward_seconds = 0.0
        self.score_forwards = 0
        self.vjp_seconds = 0.0
        self.provenance = {
            "source": str(source),
            "source_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
            "actor": str(actor), "critic": str(critic), "states": config["states"],
            "state_dimension": self.states.shape[1], "action_dimension": self.dimension,
            "critic_missing_keys": incompatible.missing_keys,
            "critic_unexpected_keys": incompatible.unexpected_keys,
            "timesteps": self.timesteps.cpu().tolist(),
            "observation_normalization": "none", "reward": "minimum of the two source q0 heads",
            "eta": float(config["eta"]), "guidance_end_index": int(config["guidance_end_index"]),
        }

    def use_state(self, row):
        self.state = torch.as_tensor(self.states[row:row + 1], device=self.device)
        self.model.condition = self.state

    def reward(self, actions, *, with_grad=False):
        started = time.perf_counter()
        with torch.set_grad_enabled(with_grad):
            values = self.model.q[0].q0(actions.float(), self.state.expand(len(actions), -1)).flatten()
        torch.cuda.synchronize()
        self.reward_seconds += time.perf_counter() - started
        self.reward_queries += len(actions)
        return values

    def transition(self, x, index, with_grad=False):
        x = x.detach().reshape(-1).requires_grad_(with_grad)
        s, t = self.timesteps[index:index + 1], self.timesteps[index + 1:index + 2]
        alpha_s, alpha_t = float(self.schedule.marginal_alpha(s)), float(self.schedule.marginal_alpha(t))
        sigma_s, sigma_t = float(self.schedule.marginal_std(s)), float(self.schedule.marginal_std(t))
        noise2 = float(self.config["eta"]) ** 2 * sigma_t ** 2 / sigma_s ** 2 * (1 - alpha_s ** 2 / alpha_t ** 2)
        noise = (max(noise2, 0.0) + 1e-12) ** 0.5
        a = alpha_t / alpha_s
        b = max(sigma_t ** 2 - noise2, 1e-12) ** 0.5 - a * sigma_s
        with torch.set_grad_enabled(with_grad):
            epsilon = (-self.model(x[None], s, self.state) * sigma_s).flatten()
            mean = a * x + b * epsilon
        self.score_forwards += 1
        endpoint = (mean.detach() - sigma_t * epsilon.detach()) / alpha_t
        return ActionTransition(x, mean, endpoint, noise / alpha_t, noise,
                                -b * sigma_s, epsilon, alpha_s, alpha_t,
                                sigma_s, sigma_t, b)

    def correction(self, transition, ratio, retain_graph=False):
        started = time.perf_counter()
        result = torch.autograd.grad(transition.mean, transition.x,
                                     grad_outputs=ratio / transition.noise_std,
                                     retain_graph=retain_graph)[0]
        torch.cuda.synchronize()
        self.vjp_seconds += time.perf_counter() - started
        return result

    def jacobian(self, transition):
        basis = torch.eye(self.dimension, device=self.device)
        return torch.stack([torch.autograd.grad(transition.mean, transition.x,
                                                grad_outputs=e, retain_graph=True)[0]
                            for e in basis])

    def counts(self):
        return {"reward_queries": self.reward_queries, "reward_seconds": self.reward_seconds,
                "score_forwards": self.score_forwards, "vjp_seconds": self.vjp_seconds}


def adapter_checks(engine):
    """Detect a source-adapter sign, endpoint, or VJP mismatch before study use."""
    engine.use_state(0)
    x = torch.linspace(-0.7, 0.9, engine.dimension, device=engine.device)
    transition = engine.transition(x, 3, True)
    s, t = engine.timesteps[3:4], engine.timesteps[4:5]
    _, source = engine.model.dpm_solver.singlestep_dpm_solver_update(
        x[None], s, t, solver_type="dpm_solver", order=1, return_intermediate=True)
    x0 = source["model_s"].flatten().detach()
    source_epsilon = (x - transition.alpha_s * x0) / transition.sigma_s
    eps = torch.linspace(-0.5, 0.5, engine.dimension, device=engine.device)
    candidate = transition.mean.detach() + transition.noise_std * eps
    source_endpoint = (candidate - transition.sigma_t * source_epsilon) / transition.alpha_t
    adapter_endpoint = transition.endpoint_mean + transition.endpoint_std * eps
    ratio = torch.linspace(-0.15, 0.25, engine.dimension, device=engine.device)
    jacobian = engine.jacobian(transition)
    guidance = engine.correction(transition, ratio, retain_graph=True)
    direct = jacobian.T @ ratio / transition.noise_std
    gamma = 0.4
    corrected_eps = transition.epsilon.detach() - gamma * transition.sigma_s * guidance
    corrected_x0 = (x - transition.sigma_s * corrected_eps) / transition.alpha_s
    source_mean = transition.alpha_t * corrected_x0 + (
        transition.epsilon_coefficient + transition.alpha_t / transition.alpha_s * transition.sigma_s
    ) * corrected_eps
    adapter_mean = transition.mean.detach() + gamma * transition.correction_gain * guidance
    errors = {"epsilon_max_error": float((source_epsilon - transition.epsilon).abs().max()),
              "endpoint_max_error": float((source_endpoint - adapter_endpoint).abs().max()),
              "vjp_max_error": float((direct - guidance).abs().max()),
              "execution_max_error": float((source_mean - adapter_mean).abs().max())}
    for name, error in errors.items():
        if error > 2e-4:
            raise AssertionError(f"Source adapter {name}: {error}")
    errors["passed"] = True
    return errors


def calibrate(config, output):
    rows, summaries = [], []
    for task in config["tasks"]:
        engine = CriticEngine(task)
        checks = adapter_checks(engine)
        engine.score_forwards = 0
        engine.vjp_seconds = 0.0
        dump(Path(output) / f'{task["name"]}-models.json', engine.provenance)
        for row in config["state_rows"]:
            engine.use_state(row)
            seed = int(config["seed"]) + row * 1009
            generator = torch.Generator(device=engine.device).manual_seed(seed)
            x = torch.randn(engine.dimension, generator=generator, device=engine.device)
            recorded_q = float(engine.reward(torch.as_tensor(engine.actions[row:row + 1], device=engine.device))[0])
            for index in range(len(engine.timesteps) - 1):
                transition = engine.transition(x, index)
                if 0 < index <= task["guidance_end_index"]:
                    noise_generator = torch.Generator(device=engine.device).manual_seed(seed + 7000001 + index * 97)
                    eps = gaussian_noise(config["samples"], engine.dimension, noise_generator, engine.device)
                    q = engine.reward(transition.endpoint_mean[None] + transition.endpoint_std * eps)
                    rows.append({"task": task["name"], "state_row": row, "index": index,
                                 "seed": seed, "rho": transition.endpoint_std,
                                 "q_mean": float(q.mean()), "q_std": float(q.std()),
                                 "q_min": float(q.min()), "q_max": float(q.max()),
                                 "recorded_action_q": recorded_q})
                x = transition.mean.detach() + transition.noise_std * torch.randn(
                    engine.dimension, generator=generator, device=engine.device)
        group = [r for r in rows if r["task"] == task["name"]]
        summary = {"task": task["name"], "states": len(config["state_rows"]),
                   "guided_states": len(group), "adapter_checks": checks,
                   "median_local_q_std": float(np.median([r["q_std"] for r in group])),
                   "q_std_quantiles": np.quantile([r["q_std"] for r in group], [0, .1, .5, .9, 1]).tolist(),
                   "q_mean_range": [min(r["q_mean"] for r in group), max(r["q_mean"] for r in group)],
                   **engine.counts()}
        summaries.append(summary)
        dump(Path(output) / "calibration.json", {"rows": rows, "summary": summaries})
        print(json.dumps(summary), flush=True)
        del engine
        torch.cuda.empty_cache()
    return {"tasks": len(summaries), "guided_states": len(rows),
            "adapter_checks_passed": sum(r["adapter_checks"]["passed"] for r in summaries),
            "reward_queries": sum(r["reward_queries"] for r in summaries)}


def ratio_from_samples(eps, q, tau):
    weights = torch.softmax(q.double() / tau, dim=0)
    return (weights[:, None] * eps.double()).sum(0).float()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def error_record(ratio, reference, jacobian, transition):
    ratio, reference = ratio.double(), reference.double()
    guidance = jacobian.double().T @ ratio / transition.noise_std
    reference_guidance = jacobian.double().T @ reference / transition.noise_std
    result = {}
    for label, estimate, truth in [("ratio", ratio, reference),
                                    ("guidance", guidance, reference_guidance)]:
        error = float((estimate - truth).square().sum())
        estimate_norm = float(estimate.square().sum())
        reference_norm = float(truth.square().sum())
        result.update({f"{label}_sq_error": error,
                       f"{label}_reference_sq_norm": reference_norm,
                       f"{label}_relative_sq_error": error / max(reference_norm, 1e-20),
                       f"{label}_zero_sq_error": reference_norm,
                       f"{label}_quarter_shrink_sq_error": float((.25 * estimate - truth).square().sum()),
                       f"{label}_direction_cosine": float(estimate @ truth) / (estimate_norm * reference_norm) ** .5
                       if estimate_norm > 0 and reference_norm > 0 else None})
    result["unit_gamma_displacement_sq_error"] = transition.correction_gain ** 2 * result["guidance_sq_error"]
    return result


def summarize_fixed(rows, config):
    result = []
    for task in config["tasks"]:
        task_rows = [r for r in rows if r["task"] == task["name"] and r["online_step"] > 0]
        baseline = [r for r in task_rows if r["variant"] == "mc8"]
        for variant in config["variants"]:
            group = [r for r in task_rows if r["variant"] == variant["name"]]
            summary = {"task": task["name"], "variant": variant["name"], "post_warmup_rows": len(group),
                       "denominator_events": sum(r["denominator_floored"] for r in group)}
            for field in ["ratio_sq_error", "guidance_sq_error", "ratio_zero_sq_error", "guidance_zero_sq_error",
                          "ratio_quarter_shrink_sq_error", "guidance_quarter_shrink_sq_error",
                          "ratio_direction_cosine", "guidance_direction_cosine", "reference_split_ratio_sq_distance",
                          "reference_split_guidance_sq_distance", "reference_ess", "queries",
                          "pilot_queries", "critic_gradients", "estimator_seconds", "heldout_prediction_mse",
                          "heldout_pair_difference_mse", "unit_gamma_displacement_sq_error"]:
                values = [r[field] for r in group if r[field] is not None]
                summary[field] = float(np.mean(values)) if values else None
            for field in ["ratio_sq_error", "guidance_sq_error", "unit_gamma_displacement_sq_error"]:
                numerator = np.array([np.mean([r[field] for r in group if r["state_row"] == index])
                                      for index in config["state_rows"]])
                denominator = np.array([np.mean([r[field] for r in baseline if r["state_row"] == index])
                                        for index in config["state_rows"]])
                generator = np.random.default_rng(41047)
                indices = generator.integers(0, len(numerator), size=(4000, len(numerator)))
                values = numerator[indices].mean(1) / denominator[indices].mean(1)
                summary[field + "_over_mc8"] = float(numerator.mean() / denominator.mean())
                summary[field + "_over_mc8_ci95"] = np.quantile(values, [.025, .975]).tolist()
            result.append(summary)
    return result


def evaluate_fixed(config, output):
    """Shared unguided paths, independent reference/pilot/current query streams."""
    rows, task_counts = [], []
    for task in config["tasks"]:
        engine = CriticEngine(task)
        dump(Path(output) / f'{task["name"]}-models.json', engine.provenance)
        tau = float(task["tau"])
        for state_row in config["state_rows"]:
            engine.use_state(state_row)
            for path_seed in config["path_seeds"]:
                seed = int(path_seed) + state_row * 100003
                path_generator = torch.Generator(device=engine.device).manual_seed(seed)
                x = torch.randn(engine.dimension, generator=path_generator, device=engine.device)
                estimators = {}
                online_step = 0
                for index in range(len(engine.timesteps) - 1):
                    guided = 0 < index <= task["guidance_end_index"]
                    transition = engine.transition(x, index, with_grad=guided)
                    if guided:
                        m, rho = transition.endpoint_mean, transition.endpoint_std
                        if online_step == 0:
                            # Fix the offset at the first guided mean. It scales
                            # weights without fitting a direction or using future data.
                            offset = float(engine.reward(m[None])[0])
                            for variant in config["variants"]:
                                settings = dict(variant["estimator"], tau=tau, reward_offset=offset)
                                for repeat in range(config["repeats"]):
                                    estimators[variant["name"], repeat] = MomentEstimator(
                                        settings, seed + repeat * 7919, engine.device)
                        jacobian = engine.jacobian(transition).detach()
                        reference_seed = seed + index * 65537 + 11000009
                        reference_generator = torch.Generator(device=engine.device).manual_seed(reference_seed)
                        ref_eps = gaussian_noise(config["reference_samples"], engine.dimension,
                                                 reference_generator, engine.device)
                        ref_q = engine.reward(m[None] + rho * ref_eps)
                        reference = ratio_from_samples(ref_eps, ref_q, tau)
                        half, quarter = len(ref_eps) // 2, len(ref_eps) // 4
                        part_a = torch.cat([torch.arange(quarter), torch.arange(half, half + quarter)]).to(engine.device)
                        part_b = torch.cat([torch.arange(quarter, half), torch.arange(half + quarter, len(ref_eps))]).to(engine.device)
                        split_a = ratio_from_samples(ref_eps[part_a], ref_q[part_a], tau)
                        split_b = ratio_from_samples(ref_eps[part_b], ref_q[part_b], tau)
                        split_delta = split_a - split_b
                        reference_ess = float(1 / torch.softmax(ref_q.double() / tau, 0).square().sum())
                        pilot_generator = torch.Generator(device=engine.device).manual_seed(reference_seed + 500000003)
                        pilot_eps = gaussian_noise(config["pilot_samples"], engine.dimension, pilot_generator, engine.device)
                        pilot_z = m[None] + rho * pilot_eps
                        pilot_q = engine.reward(pilot_z)
                        tangent_m = m.detach().clone().requires_grad_(True)
                        tangent_q = engine.reward(tangent_m[None], with_grad=True)[0]
                        tangent_slope = torch.autograd.grad(tangent_q, tangent_m)[0].detach()
                        for repeat in range(config["repeats"]):
                            current_generator = torch.Generator(device=engine.device).manual_seed(
                                seed + index * 1009 + repeat * 15485863 + 17000023)
                            bank_eps = gaussian_noise(32, engine.dimension, current_generator, engine.device)
                            bank_q = engine.reward(m[None] + rho * bank_eps)
                            for variant in config["variants"]:
                                estimator = estimators[variant["name"], repeat]
                                n = estimator.budget(online_step)
                                selection = torch.cat([torch.arange(n // 2), torch.arange(16, 16 + n // 2)]).to(engine.device)
                                eps, q = bank_eps[selection], bank_q[selection]
                                assistance = variant.get("assistance")
                                if assistance:
                                    surrogate = IntegrableSurrogate(m, rho, estimator.config, estimator.feature_generator)
                                    if assistance == "pilot":
                                        features = surrogate.features(pilot_z)
                                        target = (pilot_q.double() - offset) / tau
                                        surrogate.theta = torch.linalg.lstsq(features, target).solution
                                    elif assistance == "tangent":
                                        surrogate.theta[0] = (tangent_q.detach().double() - offset) / tau
                                        surrogate.theta[1:] = tangent_slope.double() * surrogate.scale / tau
                                    else:
                                        raise ValueError(assistance)
                                    surrogate.fitted = True
                                    estimator.surrogate = surrogate
                                estimate = estimator.from_samples(m, rho, eps, q)
                                row = {"task": task["name"], "variant": variant["name"],
                                       "state_row": state_row, "path_seed": path_seed, "index": index,
                                       "online_step": online_step, "repeat": repeat, "tau": tau,
                                       "reward_offset": offset, "rho": rho,
                                       "queries": n, "cumulative_queries": estimator.queries,
                                       "offset_queries_per_path": 1,
                                       "pilot_queries": config["pilot_samples"] if assistance == "pilot" else (1 if assistance == "tangent" else 0),
                                       "critic_gradients": int(assistance == "tangent"),
                                       "reference_samples": len(ref_eps), "reference_ess": reference_ess,
                                       "reference_split_ratio_sq_distance": float(split_delta.double().square().sum()),
                                       "reference_split_guidance_sq_distance": float((jacobian.T @ split_delta / transition.noise_std).double().square().sum()),
                                       "denominator_floored": estimate.diagnostics["denominator_floored"],
                                       "heldout_prediction_mse": estimate.diagnostics["heldout_prediction_mse"],
                                       "heldout_pair_difference_mse": estimate.diagnostics["heldout_pair_difference_mse"],
                                       "estimator_seconds": estimate.diagnostics["estimator_seconds"]}
                                row.update(error_record(estimate.ratio, reference, jacobian, transition))
                                rows.append(row)
                        online_step += 1
                    x = transition.mean.detach() + transition.noise_std * torch.randn(
                        engine.dimension, generator=path_generator, device=engine.device)
                write_csv(Path(output) / "fixed.csv", rows)
                print(json.dumps({"task": task["name"], "state_row": state_row,
                                  "path_seed": path_seed, "rows": len(rows)}), flush=True)
        task_counts.append({"task": task["name"], **engine.counts()})
        dump(Path(output) / "counts.json", task_counts)
        del engine
        torch.cuda.empty_cache()
    dump(Path(output) / "summary.json", summarize_fixed(rows, config))
    return {"tasks": len(task_counts), "rows": len(rows), "variants": len(config["variants"]),
            "actual_reward_queries": sum(r["reward_queries"] for r in task_counts),
            "reference_samples_per_state": config["reference_samples"],
            "pilot_samples_per_state": config["pilot_samples"],
            "denominator_events": sum(r["denominator_floored"] for r in rows)}
