"""Bounded SD1.5 probe and denoiser-response experiments at fixed base states."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .geometry import gaussian_probes, ratio_estimate, error_summary


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def counts(engine):
    return {"score_forwards": engine.score_forwards, "reward_queries": engine.reward_queries,
            "vjp_calls": engine.vjp_calls, "reward_seconds": engine.reward_seconds,
            "vjp_seconds": engine.vjp_seconds}


class ImageContext:
    def __init__(self, engine, latent, index, config, seed):
        self.engine, self.index, self.config = engine, index, config
        self.dimension = engine.dimension
        self.visual_decodes = 0
        self.visual_artifact_seconds = 0.0
        preliminary = engine.transition(latent, index, False)
        self.mean = preliminary.mean.detach().flatten()
        self.frozen = preliminary.endpoint_mean.detach()
        self.std = preliminary.noise_std
        self.shape = preliminary.mean.shape
        self.h = config["fd_rms_fraction"] * math.sqrt(self.dimension) * max(engine.sigmas[index + 1], 0.1)
        before = counts(engine)
        torch.cuda.synchronize(engine.device)
        tick = time.perf_counter()
        self._sketch(seed)
        torch.cuda.synchronize(engine.device)
        self.geometry_seconds = time.perf_counter() - tick
        self.geometry_costs = {key: counts(engine)[key] - before[key] for key in before}
        # Release the denoiser graph before retaining the parent kernel graph.
        self.transition = engine.transition(latent, index, True)

    def denoise(self, y, with_grad=False):
        with torch.set_grad_enabled(with_grad):
            x = y.reshape(-1, *self.shape[1:])
            epsilon = self.engine.predict_epsilon(x, self.index + 1)
            return (x - self.engine.sigmas[self.index + 1] * epsilon).flatten(1)

    def _sketch(self, seed):
        engine, d, rank = self.engine, self.dimension, self.config["sketch_rank"]
        rng = np.random.default_rng(seed)
        initial = np.linalg.qr(rng.normal(size=(d, rank)))[0]
        initial = torch.as_tensor(initial, device=engine.device, dtype=torch.float32)
        center = self.mean[None].detach().requires_grad_(True)
        clean = self.denoise(center, True)
        self.center = clean.detach().flatten()
        asymmetries = []

        def operator(vector):
            # Central directional differences avoid unsupported double-backward
            # attention paths. The transpose action is an ordinary exact VJP.
            positive = self.denoise(center.detach() + self.h * vector)
            negative = self.denoise(center.detach() - self.h * vector)
            forward = (positive - negative).flatten() / (2 * self.h)
            torch.cuda.synchronize(engine.device)
            started = time.perf_counter()
            transpose = torch.autograd.grad(clean, center, vector[None], retain_graph=True)[0].flatten()
            torch.cuda.synchronize(engine.device)
            engine.vjp_seconds += time.perf_counter() - started
            engine.vjp_calls += 1
            asymmetries.append(float((forward - transpose).norm() / forward.norm().clamp_min(1e-12)))
            return (forward + transpose) / 2

        actions = torch.stack([operator(vector) for vector in initial.T], dim=1)
        self.trace_estimate = float(d * (initial * actions).sum() / rank)
        random_projected = (initial.T @ actions)
        self.random_projected = (random_projected + random_projected.T).detach() / 2
        self.random_basis = initial.detach()
        refined = torch.linalg.qr(actions).Q
        refined_actions = torch.stack([operator(vector) for vector in refined.T], dim=1)
        projected = refined.T @ refined_actions
        projected = (projected + projected.T) / 2
        values, vectors = torch.linalg.eigh(projected)
        self.values = values.flip(0).detach()
        self.basis = (refined @ vectors.flip(1)).detach()
        self.asymmetry = float(np.mean(asymmetries))
        self.ritz_residual_fraction = float((refined_actions @ vectors.flip(1) - self.basis * self.values).norm() / refined_actions.norm().clamp_min(1e-12))
        self.fd_discrepancy = None
        if self.config.get("smoke", False):
            vector = initial[:, 0]
            full = (self.denoise(center.detach() + self.h * vector) - self.denoise(center.detach() - self.h * vector)).flatten() / (2 * self.h)
            half = (self.denoise(center.detach() + self.h / 2 * vector) - self.denoise(center.detach() - self.h / 2 * vector)).flatten() / self.h
            self.fd_discrepancy = float((full - half).norm() / half.norm().clamp_min(1e-12))
            if self.fd_discrepancy > 0.05:
                raise AssertionError({"denoiser_directional_difference_instability": self.fd_discrepancy})

    def tensor(self, eps):
        return torch.as_tensor(eps.reshape(-1, self.dimension), device=self.engine.device, dtype=torch.float32)

    def proxy_clean(self, eps, proxy):
        perturbation = self.std * self.tensor(eps)
        if proxy == "frozen":
            return self.frozen + perturbation
        if proxy == "center_identity":
            return self.center + perturbation
        if proxy == "center_isotropic":
            return self.center + perturbation * self.trace_estimate / self.dimension
        if proxy == "random_geometry":
            background = (self.trace_estimate - float(self.random_projected.trace())) / (self.dimension - self.random_basis.shape[1])
            correction = self.random_projected - background * torch.eye(len(self.random_projected), device=self.engine.device)
            return self.center + background * perturbation + ((perturbation @ self.random_basis) @ correction.T) @ self.random_basis.T
        if proxy.startswith("geometry_"):
            rank = int(proxy.split("_")[-1])
            u, values = self.basis[:, :rank], self.values[:rank]
            background = (self.trace_estimate - float(values.sum())) / (self.dimension - rank)
            return self.center + background * perturbation + ((perturbation @ u) * (values - background)) @ u.T
        if proxy == "candidate":
            return torch.cat([self.denoise(self.mean + batch) for batch in perturbation.split(self.config["denoiser_batch"])])
        raise ValueError(proxy)

    def log_weight(self, eps, proxy, seed):
        if proxy == "continuation":
            reward = self.engine.future_reward(self.transition, self.tensor(eps), seed, paired_future=True)
        else:
            reward = self.engine.reward(self.proxy_clean(eps, proxy))
        return reward.cpu().double().numpy().reshape(eps.shape[:-1]) / self.config["tau"]

    def guidance(self, ratios):
        original = ratios.shape
        vectors = np.asarray(ratios).reshape(-1, self.dimension)
        results = [self.engine.correction(self.transition, torch.as_tensor(vector, dtype=torch.float32, device=self.engine.device), retain_graph=True).flatten().cpu().double().numpy()
                   for vector in vectors]
        return np.stack(results).reshape(original)

    def describe(self):
        return {"sigma": self.std, "noise_index": self.index + 1,
                "eigenvalues_in_refined_subspace": self.values.cpu().tolist(),
                "trace_estimate": self.trace_estimate, "jacobian_asymmetry_fraction": self.asymmetry,
                "ritz_residual_fraction": self.ritz_residual_fraction,
                "fd_half_step_discrepancy": self.fd_discrepancy,
                "geometry_seconds": self.geometry_seconds, "geometry_costs": self.geometry_costs,
                "visual_decodes": self.visual_decodes, "visual_artifact_seconds": self.visual_artifact_seconds,
                "interpretation": "Symmetric denoiser-response Ritz sketch; learned CFG Jacobian is not assumed to be a posterior covariance"}


def paired_splits(n):
    half, quarter = n // 2, n // 4
    return [np.r_[0:quarter, half:half + quarter], np.r_[quarter:half, half + quarter:n]]


def references(context, config, seed, directory):
    n, d = config["reference_samples"], context.dimension
    eps = gaussian_probes(np.random.default_rng(seed), 1, n, d, antithetic=True)[0]
    refs, vectors, logs_by_proxy, proxy_errors = {}, {}, {}, []
    clean = context.proxy_clean(eps, "candidate").cpu().double().numpy()
    for proxy in config["proxies"]:
        logs = context.log_weight(eps, proxy, seed + 100003)
        ratios = [ratio_estimate(eps[None], logs[None])[0]]
        ratios += [ratio_estimate(eps[part][None], logs[part][None])[0] for part in paired_splits(n)]
        guidance = context.guidance(np.stack(ratios))
        weights = np.exp(logs - logs.max()); weights /= weights.sum()
        refs[proxy] = {"samples": n, "ess": float(1 / np.square(weights).sum()),
                       "estimated_reference_mse": float(np.square(guidance[1] - guidance[2]).sum() / 4),
                       "guidance_norm2": float(np.square(guidance[0]).sum())}
        vectors[proxy] = guidance[0]
        logs_by_proxy[proxy] = logs
        proxy_clean = context.proxy_clean(eps, proxy).cpu().double().numpy()
        proxy_errors.append({"proxy": proxy, "denoiser_mse": float(np.square(proxy_clean - clean).sum(axis=-1).mean()),
                             "denoiser_relative_mse": float(np.square(proxy_clean - clean).sum() / max(np.square(clean).sum(), 1e-30))})
        # Retain a small, identical candidate subset for visual inspection.
        visual_started = time.perf_counter()
        images = context.engine.decode(torch.as_tensor(proxy_clean[:4], device=context.engine.device, dtype=torch.float32))
        for i, image in enumerate(images):
            image.save(directory / f"{proxy}-candidate{i}.png")
        context.visual_decodes += len(images)
        context.visual_artifact_seconds += time.perf_counter() - visual_started
    future_n = config["continuation_samples"]
    future_eps = gaussian_probes(np.random.default_rng(seed + 200003), 1, future_n, d, antithetic=True)[0]
    logs = context.log_weight(future_eps, "continuation", seed + 300007)
    ratios = [ratio_estimate(future_eps[None], logs[None])[0]]
    ratios += [ratio_estimate(future_eps[part][None], logs[part][None])[0] for part in paired_splits(future_n)]
    guidance = context.guidance(np.stack(ratios))
    weights = np.exp(logs - logs.max()); weights /= weights.sum()
    refs["continuation"] = {"samples": future_n, "ess": float(1 / np.square(weights).sum()),
                            "estimated_reference_mse": float(np.square(guidance[1] - guidance[2]).sum() / 4),
                            "guidance_norm2": float(np.square(guidance[0]).sum())}
    vectors["continuation"] = guidance[0]
    for row in proxy_errors:
        proxy = row["proxy"]
        row["reward_mse_to_candidate"] = float(np.square(config["tau"] * (logs_by_proxy[proxy] - logs_by_proxy["candidate"])).mean())
        row["reference_difference2_to_candidate"] = float(np.square(vectors[proxy] - vectors["candidate"]).sum())
        row["reference_difference2_to_continuation"] = float(np.square(vectors[proxy] - vectors["continuation"]).sum())
    np.savez(directory / "reference-vectors.npz", **vectors)
    return refs, vectors, proxy_errors


def select_basis(context, method, rng, repeats, seed):
    name, d = method["name"], context.dimension
    if name in {"iid", "antithetic"}:
        return None, 0
    rank = method.get("rank", 1)
    if name.startswith("random"):
        return np.linalg.qr(rng.normal(size=(d, rank)))[0], 0
    basis = context.basis.cpu().double().numpy()
    if name.startswith("top"):
        return basis[:, :rank], 0
    if name.startswith("pilot"):
        m = method["pilot_samples"]
        eps = gaussian_probes(rng, repeats, m, d, antithetic=True)
        signal = ratio_estimate(eps, context.log_weight(eps, "candidate", seed)) @ basis
        selected = np.abs(signal).argmax(axis=1)
        return basis[:, selected].T[:, :, None], m
    raise ValueError(name)


def evaluate(config, output, engine):
    rows, proxy_rows, contexts = [], [], []
    started = time.perf_counter()
    for prompt in config["prompts"]:
        engine.use_prompt(prompt)
        for path_seed in config["path_seeds"]:
            rng = torch.Generator(device=engine.device).manual_seed(path_seed)
            latent = torch.randn((1, 4, engine.size // 8, engine.size // 8), device=engine.device, generator=rng) * engine.sigmas[0]
            states = {}
            for index in range(max(config["indices"]) + 1):
                if index in config["indices"]:
                    states[index] = latent.clone()
                transition = engine.transition(latent, index, False)
                latent = transition.mean + transition.noise_std * torch.randn(latent.shape, device=engine.device, generator=rng)
            for index, latent in states.items():
                directory = output / f"{prompt}-s{path_seed}-step{index}"
                directory.mkdir(parents=True, exist_ok=True)
                before = counts(engine)
                context = ImageContext(engine, latent, index, config, config["geometry_seed"] + index)
                refs, ref_vectors, proxy_errors = references(context, config, config["reference_seed"] + index, directory)
                info = {"prompt": prompt, "path_seed": path_seed, "index": index, **context.describe(), "references": refs}
                for row in proxy_errors:
                    proxy_rows.append({"prompt": prompt, "path_seed": path_seed, "index": index, **row})
                for seed in config["seeds"]:
                    for m in config["samples"]:
                        for mi, method in enumerate(config["methods"]):
                            counts_before = counts(engine)
                            tick = time.perf_counter()
                            sample_rng = np.random.default_rng(seed + index * 97 + mi)
                            basis, pilots = select_basis(context, method, sample_rng, config["repeats"], seed + mi)
                            eps = gaussian_probes(sample_rng, config["repeats"], m - pilots, engine.dimension, basis, method.get("antithetic", False))
                            logs = context.log_weight(eps, "candidate", seed + mi)
                            estimates = context.guidance(ratio_estimate(eps, logs))
                            actual_costs = {key: counts(engine)[key] - counts_before[key] for key in counts_before}
                            item = {"prompt": prompt, "path_seed": path_seed, "index": index,
                                    "seed": seed, "samples": m, "method": method["name"], "repeats": config["repeats"],
                                    "pilot_samples": pilots, "seconds": time.perf_counter() - tick, "costs": actual_costs,
                                    **error_summary(estimates, ref_vectors["candidate"])}
                            rows.append(item)
                            dump(output / "probe-results.json", rows)
                info["state_costs"] = {key: counts(engine)[key] - before[key] for key in before}
                contexts.append(info)
                dump(output / "contexts.json", contexts)
                dump(output / "proxy-results.json", proxy_rows)
                dump(output / "costs.json", counts(engine))
                print(json.dumps({"prompt": prompt, "path_seed": path_seed, "index": index, "conditions": len(rows), "costs": counts(engine)}), flush=True)
                del context
                torch.cuda.empty_cache()
    metrics = {"contexts": len(contexts), "probe_conditions": len(rows), "proxy_conditions": len(proxy_rows),
               "seconds": time.perf_counter() - started, **counts(engine),
               "visual_decodes": sum(row["visual_decodes"] for row in contexts),
               "visual_artifact_seconds": sum(row["visual_artifact_seconds"] for row in contexts),
               "scope": "Fixed-state image mechanism evidence; references carry finite-sample uncertainty"}
    dump(output / "summary.json", metrics)
    return metrics
