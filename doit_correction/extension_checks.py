"""Focused checks for the structured surrogate and coupled future extension."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .checks import _assert_close, _check
from .estimators import IntegrableSurrogate, spatial_dct_projection
from .image import ImageEngine


def check_dct(device):
    dimension, k, rho = 256, 16, 0.7
    projection = spatial_dct_projection(dimension, k, device)
    orthogonality = _assert_close(projection.double().T @ projection.double(),
                                  torch.eye(k, device=device), atol=2e-7, label="DCT orthogonality")
    center = torch.zeros(dimension, device=device)
    m = torch.linspace(-0.2, 0.3, dimension, device=device)
    results = {}
    for kind in ["affine", "exponential"]:
        config = {"surrogate": kind, "projection_kind": "spatial_dct", "projection_dim": k,
                  "update": "rls", "ridge": 1.0}
        surrogate = IntegrableSurrogate(center, rho, config,
                                         torch.Generator(device=device).manual_seed(105))
        surrogate.theta = torch.linspace(-0.1, 0.2, k + 1, device=device, dtype=torch.float64)
        surrogate.fitted = True
        b = projection.double() @ surrogate.theta[1:] / rho
        value = surrogate.theta[0] + m.double() @ b
        expected_h = value.exp() * torch.exp(0.5 * rho**2 * b.square().sum()) if kind == "exponential" else value
        expected_u = rho * b * (expected_h if kind == "exponential" else 1)
        h, u = surrogate.moments(m, rho)
        results[kind] = {"h": _assert_close(h, expected_h, atol=2e-7, label="DCT H"),
                         "u": _assert_close(u, expected_u, atol=2e-7, label="DCT U")}
    return {"orthogonality": orthogonality, "moments": results}


def check_future_pairing(device):
    # The toy reward exposes the first final coordinate. With zero score, its
    # paired difference must cancel every shared future noise exactly.
    engine = object.__new__(ImageEngine)
    engine.device, engine.steps = device, 3
    engine.config, engine.sigmas = {"future_batch": 4}, [2.0, 1.5, 0.7, 0.0]
    engine.predict_epsilon = lambda x, index: torch.zeros_like(x)
    engine.reward = lambda x: x[:, 0]
    first = torch.tensor([[0.3, 0.4, 0.1, -0.2], [0.7, -0.3, 0.2, 0.1],
                          [-0.2, 0.9, -0.4, 0.2]], device=device)
    eps = torch.cat([first, -first])
    transition = SimpleNamespace(mean=torch.zeros(1, 4, device=device), noise_std=0.8, index=0)
    values = engine.future_reward(transition, eps, 9107, paired_future=True)
    difference = values[:3] - values[3:]
    stats = _assert_close(difference, 1.6 * first[:, 0], atol=2e-6,
                           label="shared future noise cancels in antithetic differences")
    # Independent pairs must not all receive the same continuation noise.
    noises = values[:3] - 0.8 * first[:, 0]
    assert float(noises.std()) > 0.01, "future noise was shared across unrelated pairs"
    return {"pair_difference": stats, "distinct_pair_noise_std": float(noises.std())}


def run_extension_checks(output):
    device = torch.device("cuda")
    cases = [
        _check("spatial_dct_moments", "incorrect spatial projection or Gaussian moments after moving the mean",
               lambda: check_dct(device)),
        _check("paired_future_noise", "wrong pairing, output order, or shared noise across unrelated candidates",
               lambda: check_future_pairing(device)),
    ]
    Path(output, "checks.json").write_text(json.dumps(cases, indent=2) + "\n")
    failed = sum(not case["passed"] for case in cases)
    if failed:
        raise AssertionError(f"{failed} extension checks failed; repair before using the extensions")
    return {"checks_passed": len(cases), "checks_failed": 0}
