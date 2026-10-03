"""Full-future and two-level moment estimates with explicit oracle costs."""

import time

import torch

from .estimators import Estimate, gaussian_noise


def two_level_moments(cheap_eps, cheap_weights, fine_eps, coupled_cheap, fine_weights, coefficient):
    """E[F_fine] = a E[F_cheap] + E[F_fine - a F_cheap]."""
    residual = fine_weights.double() - coefficient * coupled_cheap.double()
    h = coefficient * cheap_weights.double().mean() + residual.mean()
    u = (coefficient * (cheap_weights.double()[:, None] * cheap_eps.double()).mean(0)
         + (residual[:, None] * fine_eps.double()).mean(0))
    return h, u, residual


def estimate_fidelity(estimator, m, rho, proxy_reward, future_reward):
    """Use independent cheap/correction batches; any learned scale is historical.

    A full future costs one terminal reward query plus every remaining score
    forward. A two-level correction also evaluates the cheap proxy at that same
    first-transition noise, and counts that extra reward query.
    """
    config = estimator.config
    n = estimator.budget(estimator.calls)
    anti = config.get("antithetic", True)
    if config["oracle"] == "future":
        eps = gaussian_noise(n, m.numel(), estimator.generator, m.device, anti)
        started = time.perf_counter()
        rewards = future_reward(eps)
        elapsed = time.perf_counter() - started
        result = estimator.from_samples(m, rho, eps, rewards)
        result.eps = eps
        result.log_weights = ((rewards.double() - float(config.get("reward_offset", 6.0))) /
                              float(config.get("tau", 0.3)))
        result.diagnostics.update(oracle_seconds=elapsed, proposal_norm=0.0, future_queries=n,
                                  proxy_queries=0, oracle="future")
        return result

    warm = estimator.calls < int(config.get("warmup_steps", 1))
    nf = int(config.get("warmup_future_samples", 8) if warm else config.get("future_samples", 4))
    eps = gaussian_noise(n, m.numel(), estimator.generator, m.device, anti)
    fine_eps = gaussian_noise(nf, m.numel(), estimator.generator, m.device, anti)
    coefficient = float(config.get("proxy_coefficient", 1.0))
    if config.get("coefficient_mode", "fixed") == "previous_mass":
        coefficient *= getattr(estimator, "previous_mass_ratio", 1.0)
    oracle_started = time.perf_counter()
    cheap_rewards = proxy_reward(m[None] + float(rho) * eps)
    coupled_rewards = proxy_reward(m[None] + float(rho) * fine_eps)
    full_rewards = future_reward(fine_eps)
    oracle_seconds = time.perf_counter() - oracle_started
    started = time.perf_counter()
    cheap, coupled, full = (estimator.weights(r) for r in (cheap_rewards, coupled_rewards, full_rewards))
    h, u, residual = two_level_moments(eps, cheap, fine_eps, coupled, full, coefficient)
    floor = float(config.get("denominator_floor", 1e-8))
    ratio = u / h.clamp_min(floor)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Nonfinite two-level moment ratio")
    raw_ratio = (full[:, None] * fine_eps.double()).mean(0) / full.mean()
    # This mass estimate can only affect the NEXT step's coefficient.
    estimator.previous_mass_ratio = float(full.mean() / coupled.mean())
    estimator.calls += 1
    estimator.queries += n + 2 * nf
    diagnostics = {
        "queries": n + 2 * nf, "future_queries": nf, "proxy_queries": n + nf,
        "oracle": "multifidelity", "proxy_coefficient": coefficient,
        "h": float(h), "raw_h": float(full.mean()), "denominator_floored": int(h < floor),
        "ratio_norm": float(ratio.norm()), "raw_ratio_norm": float(raw_ratio.norm()),
        "ess": float(full.sum().square() / full.square().sum()),
        "heldout_residual_mse": float(residual.square().mean()),
        "oracle_seconds": oracle_seconds, "estimator_seconds": time.perf_counter() - started,
        "proposal_norm": 0.0,
    }
    return Estimate(ratio.float(), h, u, diagnostics)
