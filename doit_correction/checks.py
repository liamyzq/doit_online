"""Bounded analytic and scheduler checks for the correction estimators."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from diffusers import EulerAncestralDiscreteScheduler

from .estimators import IntegrableSurrogate, MomentEstimator
from .image import euler_coefficients


def _assert_close(actual, expected, *, atol, label):
    actual = torch.as_tensor(actual).detach().double()
    expected = torch.as_tensor(expected, device=actual.device).detach().double()
    error = float((actual - expected).abs().max())
    scale = float(expected.abs().max())
    assert torch.allclose(actual, expected, atol=atol, rtol=atol), (
        f"{label}: max_abs_error={error:.6g}, expected_scale={scale:.6g}, atol={atol}"
    )
    return {"max_abs_error": error, "expected_scale": scale, "atol": atol}


def _check(name, description, fn):
    try:
        stats = fn()
        return {"name": name, "failure_detected": description, "passed": True, "stats": stats}
    except Exception as exc:
        return {
            "name": name,
            "failure_detected": description,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _surrogate(config, m, rho, *, theta, projection=None, scale=None):
    gen = torch.Generator(device=m.device).manual_seed(701)
    surrogate = IntegrableSurrogate(m, rho, config, gen)
    if scale is not None:
        surrogate.scale = scale
    if projection is not None:
        surrogate.projection = projection.to(device=m.device, dtype=m.dtype)
        surrogate.k = projection.shape[1]
    surrogate.theta = torch.as_tensor(theta, device=m.device, dtype=torch.float64).clone()
    surrogate.fitted = True
    return surrogate


def _analytic_surrogate_case(device, exponential, projected):
    d, rho, scale = 7, 0.65, 1.7
    m = torch.linspace(-0.3, 0.4, d, device=device)
    k = 3 if projected else d
    if projected:
        raw = torch.arange(1, d * k + 1, device=device, dtype=torch.float32).reshape(d, k)
        projection = torch.linalg.qr(raw, mode="reduced").Q
    else:
        projection = None
    theta = torch.linspace(-0.2, 0.25, k + 1, device=device, dtype=torch.float64)
    config = {"surrogate": "exponential" if exponential else "affine", "projection_dim": k if projected else 0}
    surrogate = _surrogate(config, m, rho, theta=theta, projection=projection, scale=scale)
    got_h, got_u = surrogate.moments(m, rho)
    feature_slope = theta[1:]
    slope = feature_slope if projection is None else projection.double() @ feature_slope
    slope = slope / scale
    if exponential:
        expected_h = torch.exp(theta[0] + 0.5 * rho**2 * slope.square().sum())
        expected_u = expected_h * rho * slope
    else:
        expected_h = theta[0]
        expected_u = rho * slope
    return {
        "h": _assert_close(got_h, expected_h, atol=1e-10, label="analytic H"),
        "u": _assert_close(got_u, expected_u, atol=1e-10, label="analytic U"),
        "dimension": d,
        "projection_dim": k if projected else d,
        "exponential": exponential,
    }


def _cv_exact_case(device, exponential, n):
    d, rho, scale = 5, 0.8, 1.3
    m = torch.linspace(-0.1, 0.2, d, device=device)
    tau, offset = 0.4, 6.0
    slope = torch.linspace(-0.12, 0.16, d, device=device, dtype=torch.float64)
    if exponential:
        theta = torch.cat([torch.tensor([0.15], device=device, dtype=torch.float64), slope * scale])
        surrogate_cfg = {"kind": "cv", "surrogate": "exponential", "tau": tau, "reward_offset": offset}
        feature_eps = torch.tensor([[-1.5, 0.5, 1.0, -0.5, 1.5]], device=device, dtype=torch.float64)
        surrogate = _surrogate(surrogate_cfg, m, rho, theta=theta, scale=scale)
        rewards_fn = lambda eps: offset + tau * (theta[0] + (eps * (rho * slope)).sum(-1))
        expected_h = torch.exp(theta[0] + 0.5 * rho**2 * slope.square().sum())
        expected_u = expected_h * rho * slope
    else:
        intercept = 3.0
        theta = torch.cat([torch.tensor([intercept], device=device, dtype=torch.float64), slope * scale])
        surrogate_cfg = {"kind": "cv", "surrogate": "affine", "tau": tau, "reward_offset": offset}
        feature_eps = torch.tensor([[-1.5, 0.5, 1.0, -0.5, 1.5]], device=device, dtype=torch.float64)
        surrogate = _surrogate(surrogate_cfg, m, rho, theta=theta, scale=scale)
        rewards_fn = lambda eps: offset + tau * torch.log(intercept + (eps * (rho * slope)).sum(-1))
        expected_h = torch.tensor(intercept, device=device, dtype=torch.float64)
        expected_u = rho * slope
    del feature_eps
    estimator = MomentEstimator(surrogate_cfg, 19, device)
    estimator.surrogate = surrogate
    eps = torch.randn(n, d, generator=torch.Generator(device=device).manual_seed(119 + n), device=device)
    if n % 2 == 0:
        eps[n // 2:] = -eps[: n // 2]
    rewards = rewards_fn(eps).float()
    result = estimator.from_samples(m, rho, eps.float(), rewards)
    return {
        "h": _assert_close(result.h, expected_h, atol=2e-6, label="exact CV H"),
        "u": _assert_close(result.u, expected_u, atol=2e-6, label="exact CV U"),
        "samples": n,
        "exponential": exponential,
    }


def _importance_case(device):
    d, n, rho, tau, offset = 4, 20000, 0.7, 0.5, 6.0
    m = torch.zeros(d, device=device)
    c = torch.tensor([0.12, -0.09, 0.08, 0.1], device=device, dtype=torch.float64)
    config = {"kind": "is", "antithetic": False, "surrogate": "exponential", "tau": tau,
              "reward_offset": offset, "proposal_scale": 1.0, "proposal_kl": 0.3, "base_mass": 0.25}
    theta = torch.cat([torch.tensor([0.0], device=device, dtype=torch.float64), c])
    est = MomentEstimator(config, 20261002, device)
    est.surrogate = _surrogate(config, m, rho, theta=theta)
    eps, importance, shift_norm = est.proposal(m, rho, n)
    rewards = offset + tau * (eps.double() @ c)
    result = est.from_samples(m, rho, eps, rewards, importance=importance)
    h_true = torch.exp(0.5 * c.square().sum())
    u_true = h_true * c
    h_se = (importance * torch.exp(eps.double() @ c)).std(unbiased=True) / math.sqrt(n)
    weighted = importance[:, None] * torch.exp(eps.double() @ c)[:, None] * eps.double()
    u_se = weighted.std(dim=0, unbiased=True) / math.sqrt(n)
    h_error = float((result.h - h_true).abs())
    u_error = (result.u - u_true).abs()
    assert h_error <= 6 * float(h_se), f"IS E[1] moment error {h_error:.6g} > 6 SE {6 * float(h_se):.6g}"
    assert torch.all(u_error <= 6 * u_se), f"IS E[eps] moment errors={u_error.tolist()}, 6SE={(6 * u_se).tolist()}"
    return {"samples": n, "h_error": h_error, "h_se": float(h_se), "u_error": u_error.tolist(),
            "u_se": u_se.tolist(), "shift_norm": shift_norm, "base_mass": config["base_mass"]}


def _constant_antithetic_cases(device):
    m = torch.zeros(4, device=device)
    rho = 0.5
    eps = torch.tensor([[0.2, -0.4, 0.1, 0.7], [-0.2, 0.4, -0.1, -0.7]], device=device)
    reward = torch.full((2,), 6.0 + 0.3 * math.log(2.0), device=device, dtype=torch.float64)
    mc = MomentEstimator({"kind": "mc", "antithetic": True}, 1, device)
    result_mc = mc.from_samples(m, rho, eps, reward)
    _assert_close(result_mc.u, torch.zeros(4, device=device), atol=1e-7, label="constant anti MC U")

    cfg = {"kind": "cv", "surrogate": "affine", "tau": 0.3, "reward_offset": 6.0}
    cv = MomentEstimator(cfg, 2, device)
    cv.surrogate = _surrogate(cfg, m, rho, theta=[2.0, 0, 0, 0, 0])
    result_cv = cv.from_samples(m, rho, eps, reward)
    _assert_close(result_cv.ratio, result_mc.ratio, atol=1e-7, label="constant CV vs raw ratio")

    one_eps = torch.tensor([[0.3, -0.2, 0.5, 0.7]], device=device)
    one_raw = MomentEstimator({"kind": "mc", "antithetic": False}, 3, device)
    raw = one_raw.from_samples(m, rho, one_eps, reward[:1])
    _assert_close(raw.ratio, one_eps[0], atol=1e-7, label="iid M=1 raw ratio")
    one_cv = MomentEstimator(cfg, 4, device)
    one_cv.surrogate = _surrogate(cfg, m, rho, theta=[2.0, 0, 0, 0, 0])
    exact = one_cv.from_samples(m, rho, one_eps, reward[:1])
    _assert_close(exact.ratio, torch.zeros(4, device=device), atol=1e-7, label="M=1 exact CV informative ratio")
    return {"anti_mc_ratio_norm": float(result_mc.ratio.norm()), "iid_m1_raw_ratio": raw.ratio.tolist(),
            "exact_cv_m1_ratio_norm": float(exact.ratio.norm())}


def _update_order_case(device):
    d, m, rho, tau, offset = 3, torch.zeros(3, device=device), 0.6, 0.4, 6.0
    eps = torch.tensor([[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5], [0.5, 0.5, -1.0], [-0.5, -0.5, 1.0]], device=device)
    rewards = offset + tau * torch.tensor([0.1, -0.2, 0.3, -0.1], device=device)
    cold_cfg = {"kind": "cv", "surrogate": "affine", "tau": tau, "reward_offset": offset}
    cold = MomentEstimator(cold_cfg, 7, device)
    cold_result = cold.from_samples(m, rho, eps, rewards)
    f = cold.weights(rewards)
    _assert_close(cold_result.h, f.mean(), atol=1e-7, label="cold CV H equals raw")
    _assert_close(cold_result.u, (f[:, None] * eps.double()).mean(0), atol=1e-7, label="cold CV U equals raw")

    theta = torch.tensor([0.2, 0.1, -0.15, 0.08], device=device, dtype=torch.float64)
    est = MomentEstimator(cold_cfg, 8, device)
    est.surrogate = _surrogate(cold_cfg, m, rho, theta=theta)
    old_theta = est.surrogate.theta.clone()
    old_f = est.surrogate.predict(m[None] + rho * eps)
    old_h, old_u = est.surrogate.moments(m, rho)
    actual_f = est.weights(rewards)
    expected_h = old_h + (actual_f - old_f).mean()
    expected_u = old_u + ((actual_f - old_f)[:, None] * eps.double()).mean(0)
    result = est.from_samples(m, rho, eps, rewards)
    _assert_close(result.h, expected_h, atol=1e-7, label="update-order H uses old surrogate")
    _assert_close(result.u, expected_u, atol=1e-7, label="update-order U uses old surrogate")
    change = float((est.surrogate.theta - old_theta).norm())
    assert change > 0, "surrogate did not update for the subsequent estimate"
    return {"cold_raw_h": float(cold_result.h), "cold_raw_ratio_norm": float(cold_result.ratio.norm()),
            "post_update_theta_change": change}


def _spider_case(device):
    d, m, rho, tau, offset = 4, torch.zeros(4, device=device), 0.5, 0.3, 6.0
    cfg = {"kind": "spider", "antithetic": True, "samples": 8, "warmup_samples": 8,
           "warmup_steps": 1, "refresh_every": 10, "tau": tau, "reward_offset": offset}
    est = MomentEstimator(cfg, 28, device)
    eps0 = torch.cat([torch.eye(d, device=device), -torch.eye(d, device=device)], dim=0)
    previous_reward = torch.full((len(eps0),), offset, device=device)
    initial = est.from_samples(m, rho, eps0, previous_reward, step=0)
    delta0 = 0.7
    slope = torch.tensor([0.08, -0.06, 0.04, 0.1], device=device)
    eps = math.sqrt(d) * torch.cat([torch.eye(d, device=device), -torch.eye(d, device=device)], dim=0)
    delta = delta0 + eps @ slope
    assert torch.all(delta > 0), "SPIDER affine increment must keep synthetic F positive"
    current_rewards = offset + tau * torch.log1p(delta)
    old_rewards = torch.full_like(current_rewards, offset)
    result = est.from_samples(m, rho, eps, current_rewards, previous_rewards=old_rewards, step=1)
    _assert_close(result.h, initial.h + delta0, atol=2e-6, label="SPIDER exact affine H")
    _assert_close(result.u, initial.u + slope, atol=2e-6, label="SPIDER exact affine U")
    assert result.diagnostics["queries"] == 2 * len(eps), (
        f"SPIDER query count omitted one reward call: got {result.diagnostics['queries']}, expected {2 * len(eps)}"
    )
    return {"queries_refresh": initial.diagnostics["queries"], "queries_delta": result.diagnostics["queries"],
            "exact_increment_h": delta0, "exact_increment_u": slope.tolist()}


def _euler_case(device):
    base = EulerAncestralDiscreteScheduler.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", subfolder="scheduler", local_files_only=True
    )
    base.set_timesteps(4, device=device)
    count = len(base.timesteps)
    indices = sorted(set([0, max(0, count // 2), count - 1]))
    comparisons = []
    for idx in indices:
        scheduler = EulerAncestralDiscreteScheduler.from_config(base.config)
        scheduler.set_timesteps(4, device=device)
        scheduler.set_begin_index(idx)
        sigma, sigma_next = (float(v) for v in scheduler.sigmas[idx : idx + 2])
        up, down, dt = euler_coefficients(sigma, sigma_next)
        sample = torch.linspace(-1, 1, 4 * 8 * 8, device=device).reshape(1, 4, 8, 8)
        eps = torch.full_like(sample, 0.125)
        gen_explicit = torch.Generator(device=device).manual_seed(400 + idx)
        noise = torch.randn(sample.shape, generator=gen_explicit, device=device, dtype=sample.dtype)
        explicit = sample + dt * eps + up * noise
        gen_scheduler = torch.Generator(device=device).manual_seed(400 + idx)
        actual = scheduler.step(eps, scheduler.timesteps[idx], sample, generator=gen_scheduler).prev_sample
        stats = _assert_close(actual, explicit, atol=2e-6, label=f"Euler step idx={idx}")
        comparisons.append({"index": idx, "sigma": sigma, "sigma_next": sigma_next, **stats})
        if idx == count - 1:
            assert sigma_next == 0.0 and up == 0.0 and down == 0.0, (
                f"Euler final transition should be deterministic with sigma_next=0; got {sigma_next}, {up}, {down}"
            )
    return {"comparisons": comparisons, "final_sigma_next": comparisons[-1]["sigma_next"]}


def _vjp_case(device):
    matrix = torch.tensor([[1.0, -0.5, 0.25], [0.2, 0.7, -1.1]], device=device)
    x = torch.tensor([0.4, -0.2, 0.8], device=device, requires_grad=True)
    ratio = torch.tensor([0.3, -0.6], device=device)
    sigma = 0.45
    mean = matrix @ x
    actual = torch.autograd.grad(mean, x, ratio / sigma)[0]
    expected = matrix.T @ ratio / sigma
    return {"vjp": _assert_close(actual, expected, atol=1e-7, label="linear mean VJP"),
            "correction": actual.tolist()}


def run_checks(output):
    """Run small algebraic checks on CUDA and save individual results."""
    if not torch.cuda.is_available():
        raise RuntimeError("Checks require CUDA; this run would detect estimator/scheduler mismatches before experiments.")
    device = torch.device("cuda")
    cases = [
        _check("gaussian_moments_affine_full", "incorrect full-coordinate affine Gaussian moments", lambda: _analytic_surrogate_case(device, False, False)),
        _check("gaussian_moments_exponential_full", "incorrect full-coordinate exponential Gaussian moments", lambda: _analytic_surrogate_case(device, True, False)),
        _check("gaussian_moments_affine_projected", "incorrect projected affine Gaussian moments", lambda: _analytic_surrogate_case(device, False, True)),
        _check("gaussian_moments_exponential_projected", "incorrect projected exponential Gaussian moments", lambda: _analytic_surrogate_case(device, True, True)),
        _check("cv_exact_fit", "CV H/U fail exact affine or exponential fits at one and eight samples", lambda: {
            "affine_n1": _cv_exact_case(device, False, 1), "affine_n8": _cv_exact_case(device, False, 8),
            "exponential_n1": _cv_exact_case(device, True, 1), "exponential_n8": _cv_exact_case(device, True, 8)}),
        _check("importance_mixture", "mixture proposal importance weights fail to recover base-Gaussian moments", lambda: _importance_case(device)),
        _check("constant_and_single_sample", "constant rewards or one-sample CV do not have their exact moment ratios", lambda: _constant_antithetic_cases(device)),
        _check("cv_update_order", "cold CV differs from raw MC or the current output uses its newly updated fit", lambda: _update_order_case(device)),
        _check("spider_coupled_delta", "coupled SPIDER affine difference is inexact or query cost omits a reward call", lambda: _spider_case(device)),
        _check("euler_scheduler", "explicit Euler ancestral transition disagrees with diffusers at supported steps", lambda: _euler_case(device)),
        _check("linear_vjp", "linear transition correction differs from explicit Jacobian-transpose product", lambda: _vjp_case(device)),
    ]
    summary = {"device": str(device), "passed": sum(c["passed"] for c in cases), "failed": sum(not c["passed"] for c in cases),
               "checks": len(cases)}
    payload = {"summary": summary, "results": cases}
    Path(output, "checks.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if summary["failed"]:
        failures = "; ".join(f"{case['name']}: {case['error']}" for case in cases if not case["passed"])
        raise AssertionError(f"{summary['failed']} of {summary['checks']} checks failed; details are in checks.json. {failures}")
    return summary
