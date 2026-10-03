"""Low-dimensional action diffusion with authentic, fixed observation states."""

from __future__ import annotations

import functools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .estimators import gaussian_noise


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
