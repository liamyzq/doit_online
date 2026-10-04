"""Fixed-vector sensitivity diagnostics and explicit biased VJP controls."""

import json
import math
import time
from pathlib import Path

import torch


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cosine(a, b):
    return float(torch.dot(a.flatten().double(), b.flatten().double()) /
                 (a.double().norm() * b.double().norm()).clamp_min(1e-30))


def sensitivity_correction(engine, transition, ratio, specification, seed, retain_graph=False):
    torch.cuda.synchronize(engine.device)
    started = time.perf_counter()
    result = _sensitivity_correction(engine, transition, ratio, specification, seed, retain_graph)
    torch.cuda.synchronize(engine.device)
    engine.sensitivity_seconds += time.perf_counter() - started
    return result


def _sensitivity_correction(engine, transition, ratio, specification, seed, retain_graph):
    """Keep the reward vector fixed while changing the image mean sensitivity.

    Euler's explicit linear coefficient is one. Shrinkage scales only the
    learned component. Smoothing averages K VJPs at Gaussian perturbations,
    with complete opposite pairs, without requerying the reward or changing
    the current transition mean. Both modifications are biased controls.
    """
    kind = specification["kind"]
    ratio = ratio.detach()
    if kind == "raw":
        return engine.correction(transition, ratio, retain_graph)
    if kind in {"shrink", "shrink_norm_matched"}:
        if kind == "shrink" and float(specification["coefficient"]) == 0:
            return ratio.reshape_as(transition.mean) / transition.noise_std
        raw = engine.correction(transition, ratio, retain_graph)
        identity = ratio.reshape_as(raw) / transition.noise_std
        result = identity + float(specification["coefficient"]) * (raw - identity)
        if kind == "shrink_norm_matched":
            result = result * (raw.norm() / result.norm().clamp_min(1e-30))
        return result
    if kind == "smooth":
        count = int(specification["samples"])
        if count < 2 or count % 2:
            raise ValueError("Smoothing uses complete opposite perturbation pairs")
        generator = torch.Generator(device=engine.device).manual_seed(seed)
        delta = float(specification["noise_fraction"]) * transition.noise_std
        half = torch.randn((count // 2, *transition.x.shape), device=engine.device,
                           generator=generator)
        result = torch.zeros_like(transition.x)
        for perturbation in torch.cat([half, -half]):
            shifted = engine.transition(transition.x.detach() + delta * perturbation,
                                        transition.index, True)
            result += engine.correction(shifted, ratio) / count
        return result
    raise ValueError(kind)


def controlled(config, output):
    """Compare known ideal derivatives with a small high-frequency value error."""
    device, dtype = "cuda", torch.float64
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    rows = []
    for dimension in config["dimensions"]:
        x = torch.randn((config["repeats"], dimension), generator=generator, device=device, dtype=dtype)
        v = torch.randn(x.shape, generator=generator, device=device, dtype=dtype)
        v = v / v.norm(dim=-1, keepdim=True)
        ideal = .8 * v + .15 * v.roll(-1, dims=-1) + .05 * (1 - x.tanh().square()) * v
        for amplitude in config["amplitudes"]:
            frequency = config["frequency"]
            def derivative(at):
                return (.8 * v + .15 * v.roll(-1, dims=-1) +
                        (.05 * (1 - at.tanh().square()) + amplitude * frequency *
                         (frequency * at).cos()) * v)
            raw = derivative(x)
            outputs = {"raw": raw, "shrink-half": v + .5 * (raw - v)}
            count = config["smoothing_samples"]
            noise = torch.randn((count // 2, *x.shape), generator=generator, device=device, dtype=dtype)
            delta = config["noise_fraction"] * config["transition_noise_std"]
            outputs["smooth-four"] = torch.stack([
                derivative(x + delta * u) for u in torch.cat([noise, -noise])]).mean(0)
            for name, estimate in outputs.items():
                errors = (estimate - ideal).square().sum(-1)
                rows.append({"dimension": dimension, "amplitude": amplitude, "method": name,
                             "mean_squared_error": float(errors.mean()),
                             "squared_error_standard_error": float(errors.std() / math.sqrt(len(errors))),
                             "relative_mse": float(errors.mean() / ideal.square().sum(-1).mean()),
                             "value_error_rms": float((amplitude * (frequency * x).sin()).square().mean().sqrt()),
                             "perturbation_norm_typical": delta * math.sqrt(dimension)})
    # This nonsymmetric map detects a swapped Jacobian transpose or an
    # accidentally differentiated v. The central-difference sweep is fixed-v.
    x = torch.linspace(-.7, .8, 6, device=device, dtype=dtype).requires_grad_()
    v = torch.linspace(.2, -.4, 6, device=device, dtype=dtype)
    u = torch.arange(1, 7, device=device, dtype=dtype); u = u / u.norm()
    def mean(at):
        return .8 * at + .15 * at.roll(1) + .05 * at.tanh() + .02 * (40 * at).sin()
    raw = torch.autograd.grad(mean(x), x, v)[0]
    exact = .8 * v + .15 * v.roll(-1) + (.05 * (1 - x.tanh().square()) + .8 * (40 * x).cos()) * v
    differences = []
    for h in [1e-2, 1e-3, 1e-4, 1e-5]:
        fd = torch.dot(v, mean(x + h * u) - mean(x - h * u)) / (2 * h)
        differences.append({"h": h, "relative_error": float((fd - torch.dot(u, raw)).abs() /
                                                               torch.dot(u, raw).abs().clamp_min(1e-30))})
    if float((raw - exact).abs().max()) > 1e-12 or min(r["relative_error"] for r in differences) > 1e-6:
        raise AssertionError("Controlled fixed-vector adjoint check failed")
    dump(Path(output) / "controlled.json", {"rows": rows, "directional_differences": differences})
    return {"rows": len(rows), "adjoint_max_error": float((raw - exact).abs().max()),
            "minimum_fd_relative_error": min(r["relative_error"] for r in differences),
            "reward_queries": 0, "model_score_forwards": 0}


def image_probe(config, output, engine):
    """Audit actual SD states, fixing each vector for every sensitivity variant."""
    from .estimators import gaussian_noise
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dump(Path(output) / "models.json", engine.provenance)
    states = []
    # Capture base trajectories independently of which sensitivity looks best.
    for prompt in config["prompts"]:
        for seed in config["seeds"]:
            def capture(transition, name, path_seed):
                if transition.index in config["indices"]:
                    states.append((name, path_seed, transition.index, transition.x.detach().cpu()))
            engine.generate(prompt, seed, {"kind": "mc", "gamma": 0},
                            Path(output) / "base" / prompt / str(seed), capture)
    rows = []
    for prompt, seed, index, latent in states:
        engine.use_prompt(prompt)
        transition = engine.transition(latent.to(engine.device), index, True)
        ratios = []
        for group in range(2):
            generator = torch.Generator(device=engine.device).manual_seed(seed * 100003 + index * 997 + group * 7919)
            eps = gaussian_noise(config["samples"], engine.dimension, generator, engine.device)
            rewards = engine.reward(transition.endpoint_mean + transition.endpoint_std * eps)
            weights = torch.softmax(rewards.double() / config["tau"], dim=0)
            ratios.append((weights[:, None] * eps.double()).sum(0).float())
        outputs = []
        for group, ratio in enumerate(ratios):
            estimates = {}
            for name, specification in config["variants"].items():
                before = engine.score_forwards, engine.vjp_calls, engine.vjp_seconds, engine.sensitivity_seconds
                correction = sensitivity_correction(engine, transition, ratio, specification,
                                                      seed * 997 + index * 97, retain_graph=True)
                estimates[name] = correction.detach()
                raw = estimates["raw"]
                identity = ratio.reshape_as(raw) / transition.noise_std
                rows.append({"prompt": prompt, "seed": seed, "index": index, "group": group,
                             "method": name, "guidance_norm": float(correction.norm()),
                             "raw_cosine": cosine(correction, raw),
                             "relative_difference": float((correction - raw).norm() / raw.norm()),
                             "identity_norm": float(identity.norm()),
                             "learned_norm": float((raw - identity).norm()),
                             "identity_learned_cosine": cosine(identity, raw - identity),
                             "uncapped_execution_kl": float((config["gamma"] * transition.correction_gain * correction).square().sum()
                                                            / (2 * transition.noise_std ** 2)),
                             "noise_std": transition.noise_std,
                             "smooth_delta": specification.get("noise_fraction", 0) * transition.noise_std,
                             "smooth_perturbation_norm_typical": specification.get("noise_fraction", 0) *
                                                                  transition.noise_std * math.sqrt(engine.dimension),
                             "score_forwards": engine.score_forwards - before[0],
                             "vjp_calls": engine.vjp_calls - before[1],
                             "sensitivity_seconds": engine.sensitivity_seconds - before[3],
                             "vjp_seconds": engine.vjp_seconds - before[2]})
            outputs.append(estimates)
        for row in rows[-2 * len(config["variants"]):]:
            row["independent_group_cosine"] = cosine(outputs[0][row["method"]], outputs[1][row["method"]])
        # One random and one raw-guidance direction. The latter avoids a tiny
        # projection obscuring a valid FP32 finite-difference plateau.
        raw = outputs[0]["raw"]
        vector = ratios[0].reshape_as(transition.mean) / transition.noise_std
        random = torch.randn(raw.shape, generator=generator, device=engine.device)
        directional = []
        for direction_name, direction in [("raw", raw), ("random", random)]:
            direction = direction / direction.norm()
            adjoint = (raw.double() * direction.double()).sum()
            for fraction in config["fd_rms_fractions"]:
                h = fraction * transition.noise_std * math.sqrt(engine.dimension)
                plus = engine.transition(transition.x.detach() + h * direction, index, False)
                minus = engine.transition(transition.x.detach() - h * direction, index, False)
                finite = (vector.double() * (plus.mean.double() - minus.mean.double())).sum() / (2 * h)
                directional.append({"direction": direction_name, "rms_fraction": fraction, "h": h,
                                    "adjoint": float(adjoint), "finite_difference": float(finite),
                                    "relative_error": float((finite - adjoint).abs() / adjoint.abs().clamp_min(1e-30))})
        dump(Path(output) / f"fd-{prompt}-{seed}-{index}.json", directional)
        dump(Path(output) / "sensitivity.json", rows)
        torch.save({"latent": latent, "ratios": torch.stack(ratios).cpu(), "index": index,
                    "prompt": prompt, "seed": seed}, Path(output) / f"state-{prompt}-{seed}-{index}.pt")
        print(json.dumps({"prompt": prompt, "seed": seed, "index": index,
                          "raw_independent_cosine": cosine(outputs[0]["raw"], outputs[1]["raw"]),
                          "best_raw_fd_error": min(r["relative_error"] for r in directional if r["direction"] == "raw")}), flush=True)
        del transition
    return {"states": len(states), "rows": len(rows), "reward_queries": engine.reward_queries,
            "score_forwards": engine.score_forwards, "vjp_calls": engine.vjp_calls,
            "sensitivity_seconds": engine.sensitivity_seconds,
            "reward_seconds": engine.reward_seconds, "vjp_seconds": engine.vjp_seconds}
