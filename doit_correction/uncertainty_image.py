"""Sparse image SMC with cached point or full-continuation potentials."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .image import euler_coefficients


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generator(engine, seed):
    return torch.Generator(device=engine.device).manual_seed(seed)


def noise(x, rng):
    return torch.randn(x.shape, device=x.device, generator=rng)


def predict(engine, x, index):
    return torch.cat([engine.predict_epsilon(batch, index)
                      for batch in x.split(engine.config.get("particle_batch", 1))])


def rollout(engine, x, start, rng):
    for index in range(start, engine.steps):
        epsilon = predict(engine, x, index)
        up, _, dt = euler_coefficients(*engine.sigmas[index:index + 2])
        x = x + dt * epsilon + up * noise(x, rng)
    return x


def potential(engine, x, index, method, rng):
    if method["value"] == "point":
        clean = x - engine.sigmas[index] * predict(engine, x, index)
        return engine.reward(clean.flatten(1)).double() / method["tau"]
    if method["value"] == "future":
        branches = method["branches"]
        y = x.repeat_interleave(branches, dim=0)
        y = rollout(engine, y, index, rng)
        values = engine.reward(y.flatten(1)).double().reshape(len(x), branches) / method["tau"]
        return torch.logsumexp(values, dim=-1) - math.log(branches)
    raise ValueError(method["value"])


@torch.no_grad()
def generate(engine, prompt, seed, method, output):
    engine.use_prompt(prompt)
    n = method["particles"]
    path, value_rng, selection = [generator(engine, seed + offset) for offset in (0, 1000003, 2000003)]
    shape = (n, 4, engine.size // 8, engine.size // 8)
    x = torch.randn(shape, device=engine.device, generator=path) * engine.sigmas[0]
    v = torch.zeros(n, device=engine.device, dtype=torch.float64)
    logw = v.clone()
    lineage = torch.arange(n, device=engine.device)
    steps = []
    before_q, before_score = engine.reward_queries, engine.score_forwards
    before_reward = engine.reward_seconds
    torch.cuda.synchronize(engine.device)
    started = time.perf_counter()
    for index in range(engine.steps):
        if index in method.get("checkpoints", []):
            new_v = potential(engine, x, index, method, value_rng)
            logw += new_v - v
            v = new_v
            weights = torch.softmax(logw, dim=0)
            ess = float(1 / weights.square().sum())
            resampled = ess < method.get("ess_threshold", 0.5) * n
            if resampled:
                positions = (torch.rand((), device=engine.device, generator=selection) + torch.arange(n, device=engine.device)) / n
                indices = torch.searchsorted(weights.cumsum(0), positions.double())
                x, v, lineage = x[indices], v[indices], lineage[indices]
                logw.zero_()
            steps.append({"index": index, "ess": ess, "resampled": resampled,
                          "initial_ancestors": len(lineage.unique())})
        epsilon = predict(engine, x, index)
        up, _, dt = euler_coefficients(*engine.sigmas[index:index + 2])
        x = x + dt * epsilon + up * noise(x, path)
    # Evaluate every terminal candidate once and reuse its reward for selection
    # and reporting. This includes decoder/CLIP work in the full query budget.
    terminal_started = time.perf_counter()
    images, scores, embeddings, alignments = [], [], [], []
    for batch in x.split(engine.chunk):
        decoded = engine.decode(batch.flatten(1))
        reward, embedding, alignment = engine.score_images(decoded, return_embeddings=True)
        images.extend(decoded)
        scores.append(reward)
        embeddings.append(embedding)
        alignments.append(alignment)
        engine.reward_queries += len(batch)
    scores, embeddings, alignments = map(torch.cat, (scores, embeddings, alignments))
    torch.cuda.synchronize(engine.device)
    terminal_seconds = time.perf_counter() - terminal_started
    weights = torch.softmax(logw + scores.double() / method["tau"] - v, dim=0)
    chosen = int(torch.multinomial(weights, 1, generator=selection))
    torch.cuda.synchronize(engine.device)
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    images[chosen].save(output / "image.png")
    np.save(output / "embedding.npy", embeddings[chosen].cpu().numpy())
    return {"prompt": prompt, "seed": seed, "method": method["name"],
            "reward": float(scores[chosen]), "clip_alignment": float(alignments[chosen]),
            "seconds": elapsed, "guidance_queries": engine.reward_queries - before_q - n,
            "evaluation_queries": n, "all_queries": engine.reward_queries - before_q,
            "score_forwards": engine.score_forwards - before_score, "vjp_calls": 0,
            "reward_seconds": engine.reward_seconds - before_reward,
            "terminal_evaluation_seconds": terminal_seconds,
            "weighted_reward": float((weights * scores).sum()),
            "weighted_alignment": float((weights * alignments).sum()),
            "final_ess": float(1 / weights.square().sum()),
            "initial_ancestors": len(lineage.unique()), "chosen": chosen,
            "candidate_rewards": scores.cpu().tolist(), "weights": weights.cpu().tolist(),
            "steps": steps}


def evaluate(config, output, engine):
    rows = []
    for prompt in config["prompts"]:
        for seed in config["seeds"]:
            for method in config["methods"]:
                destination = output / method["name"] / prompt / str(seed)
                if "estimator" in method:
                    row = engine.generate(prompt, seed, method["estimator"], destination)
                    row["method"] = method["name"]
                else:
                    row = generate(engine, prompt, seed, method, destination)
                rows.append(row)
                dump(output / "images.json", rows)
                print(json.dumps({key: value for key, value in row.items() if key != "steps"}), flush=True)
    checks = {}
    if config.get("execution_smoke"):
        # N=1 must execute exactly the existing unguided sampler. A mismatch
        # invalidates the image adapter and blocks endpoint comparisons.
        by_name = {row["method"]: row for row in rows}
        base, one = by_name["base"], by_name["terminal-n1"]
        checks["n1_reward_error"] = abs(base["reward"] - one["reward"])
        checks["n1_alignment_error"] = abs(base["clip_alignment"] - one["clip_alignment"])
        if max(checks.values()) > 1e-7:
            raise AssertionError(checks)
        for method, row in zip(config["methods"], rows):
            if "estimator" in method:
                continue
            n = method["particles"]
            checkpoints = method.get("checkpoints", [])
            branches = method.get("branches", 1)
            expected_q = n * (1 + len(checkpoints) * branches)
            extra_scores = len(checkpoints) if method["value"] == "point" else branches * sum(engine.steps - k for k in checkpoints)
            expected_score = n * (engine.steps + extra_scores)
            if row["all_queries"] != expected_q or row["score_forwards"] != expected_score:
                raise AssertionError({"method": method["name"], "actual": row, "expected": [expected_q, expected_score]})
        checks["cost_conditions_passed"] = len(rows) - 1
        dump(output / "checks.json", checks)
    return {"images": len(rows), "reward_queries": engine.reward_queries,
            "score_forwards": engine.score_forwards, "vjp_calls": engine.vjp_calls,
            "checks": checks}
