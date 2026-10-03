"""Compare information use at common image states with identical candidates."""

import json
from pathlib import Path

import torch

from .estimators import gaussian_noise
from .execution import categorical_transition, gaussian_displacement
from .fixed import dump
from .image import ImageEngine, euler_coefficients


@torch.no_grad()
def continue_with_common_noise(engine, latents, start_index, seed):
    """Reevaluate next states with the same independent later noise per branch."""
    rewards = []
    for batch in latents.split(int(engine.config.get("future_batch", 2))):
        generator = torch.Generator(device=engine.device).manual_seed(seed)
        x = batch.clone()
        for index in range(start_index, engine.steps):
            prediction = engine.predict_epsilon(x, index)
            up, _, dt = euler_coefficients(*engine.sigmas[index:index + 2])
            noise = torch.randn((1, *x.shape[1:]), device=engine.device, generator=generator)
            x = x + dt * prediction + up * noise
        rewards.append(engine.reward(x.flatten(1)))
    return torch.cat(rewards)


def run_probe(config, output):
    first = torch.load(config["states"][0], map_location="cpu", weights_only=False)
    engine = ImageEngine(first["image_config"])
    dump(Path(output) / "models.json", engine.provenance)
    rows = []
    for path in config["states"]:
        state = torch.load(path, map_location="cpu", weights_only=False)
        engine.use_prompt(state["prompt"])
        transition = engine.transition(state["latent"].to(engine.device), state["index"], True)
        for repeat in range(config["repeats"]):
            seed = state["seed"] * 100003 + state["index"] * 997 + repeat * 15485863 + 23000009
            candidate_generator = torch.Generator(device=engine.device).manual_seed(seed)
            eps = gaussian_noise(config["samples"], engine.dimension, candidate_generator, engine.device)
            candidates = transition.mean.detach() + transition.noise_std * eps.reshape(-1, *transition.mean.shape[1:])
            execution_generator = torch.Generator(device=engine.device).manual_seed(seed + 2000003)
            execution_noise = torch.randn(transition.mean.shape, generator=execution_generator, device=engine.device)
            base = transition.mean.detach() + transition.noise_std * execution_noise
            for paired in [True, False]:
                counters_before = engine.reward_queries, engine.score_forwards, engine.vjp_seconds
                rewards = engine.future_reward(transition, eps, seed + 11000009, paired_future=paired)
                log_weights = rewards.double() / config["tau"]
                probabilities = torch.softmax(log_weights, dim=0)
                ratio = (probabilities[:, None] * eps.double()).sum(0).float()
                correction = engine.correction(transition, ratio, retain_graph=True)
                free, free_diagnostics = gaussian_displacement(
                    correction, transition.correction_gain, config["gamma"], transition.noise_std)
                capped, capped_diagnostics = gaussian_displacement(
                    correction, transition.correction_gain, config["gamma"], transition.noise_std,
                    config["execution_kl_limit"])
                selection_generator = torch.Generator(device=engine.device).manual_seed(seed + 3000017)
                _, selected = categorical_transition(transition.mean.detach(), transition.noise_std,
                                                      eps, log_weights, selection_generator)
                # Reevaluate every candidate, allowing both the actual stochastic
                # selection and its conditional categorical mean to be reported.
                next_states = torch.cat([candidates, base, base + free, base + capped])
                evaluation = torch.stack([
                    continue_with_common_noise(engine, next_states, state["index"] + 1,
                                               seed + 500000003 + branch * 7919)
                    for branch in range(config["evaluation_branches"])])
                n = config["samples"]
                candidate_values = evaluation[:, :n].double()
                weighted = (candidate_values * probabilities[None]).sum(1)
                index = selected["selected_candidate"]
                row = {"capture_path": path, "prompt": state["prompt"], "seed": state["seed"],
                       "index": state["index"], "repeat": repeat, "paired_future": paired,
                       "samples": n, "tau": config["tau"], "gamma": config["gamma"],
                       "selection_rewards": rewards.cpu().tolist(), "probabilities": probabilities.cpu().tolist(),
                       "evaluation_rewards": evaluation.cpu().tolist(),
                       "candidate_uniform_reward": float(candidate_values.mean()),
                       "categorical_conditional_reward": float(weighted.mean()),
                       "categorical_selected_reward": float(candidate_values[:, index].mean()),
                       "base_reward": float(evaluation[:, n].mean()),
                       "gradient_free_reward": float(evaluation[:, n + 1].mean()),
                       "gradient_capped_reward": float(evaluation[:, n + 2].mean()),
                       "categorical_minus_uniform_reward": float(weighted.mean() - candidate_values.mean()),
                       "selected": selected, "gradient_free": free_diagnostics, "gradient_capped": capped_diagnostics,
                       "selection_queries": n, "evaluation_queries": len(next_states) * config["evaluation_branches"],
                       "reward_queries": engine.reward_queries - counters_before[0],
                       "score_forwards": engine.score_forwards - counters_before[1],
                       "vjp_seconds": engine.vjp_seconds - counters_before[2]}
                rows.append(row)
                dump(Path(output) / "probes.json", rows)
                print(json.dumps({k: row[k] for k in ["prompt", "seed", "index", "repeat", "paired_future",
                                                     "categorical_minus_uniform_reward", "reward_queries"]}), flush=True)
        del transition
    return {"states": len(config["states"]), "rows": len(rows),
            "reward_queries": engine.reward_queries, "score_forwards": engine.score_forwards,
            "reward_seconds": engine.reward_seconds, "vjp_seconds": engine.vjp_seconds}
