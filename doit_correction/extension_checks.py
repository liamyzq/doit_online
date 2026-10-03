"""Focused checks for the structured surrogate and coupled future extension."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .checks import _assert_close, _check
from .estimators import IntegrableSurrogate, spatial_dct_projection
from .image import ImageEngine
from .fidelity import estimate_fidelity, two_level_moments
from .estimators import MomentEstimator


def run_execution_checks(output):
    from .execution import categorical_transition, gaussian_displacement
    from .image import euler_coefficients

    device = torch.device("cuda")

    def gaussian_cap():
        correction = torch.tensor([[3.0, -4.0]], device=device)
        delta, free = gaussian_displacement(correction, 2.0, 3.0, 0.5)
        assert torch.equal(delta, 6 * correction)
        capped, stats = gaussian_displacement(correction, 2.0, 3.0, 0.5, max_kl=8.0)
        _assert_close(capped, torch.tensor([[1.2, -1.6]], device=device), atol=2e-6, label="bounded displacement")
        _assert_close(stats["execution_kl"], 8.0, atol=2e-6, label="applied KL cap")
        assert stats["execution_clipped"] == 1 and free["execution_clipped"] == 0
        for sigma, next_sigma in [(2.0, 1.5), (0.7, 0.3), (0.34, 0.03)]:
            up, _, dt = euler_coefficients(sigma, next_sigma)
            _assert_close((-dt * sigma) / up ** 2, (sigma / next_sigma) ** 2,
                          atol=1e-10, label="Euler gain/noise relation")
        return {"raw_kl": free["execution_kl"], "applied_kl": stats["execution_kl"]}

    def resample_realized_candidate():
        mean = torch.tensor([[0.2, -0.1]], device=device)
        eps = torch.tensor([[1., 2.], [-1., -2.], [0.3, -0.4]], device=device)
        log_weights = torch.tensor([0.0, 0.2, -0.1], device=device)
        choices = []
        for seed in range(16):
            generator = torch.Generator(device=device).manual_seed(36101 + seed)
            value, stats = categorical_transition(mean, 0.7, eps, log_weights, generator)
            expected = mean + 0.7 * eps[stats["selected_candidate"]].reshape_as(mean)
            assert torch.equal(value, expected), "selected candidate was shifted or given extra noise"
            shifted, shifted_stats = categorical_transition(
                mean, 0.7, eps, log_weights.double() + 20,
                torch.Generator(device=device).manual_seed(36101 + seed))
            assert torch.equal(value, shifted), "categorical selection depends on reward offset"
            assert stats["execution_kl"] is None
            choices.append(stats["selected_candidate"])
        assert len(set(choices)) > 1, "selection appears to use argmax instead of a categorical draw"
        return {"selected_indices": choices}

    def future_candidate_reuse():
        config = {"kind": "mc", "oracle": "future", "warmup_steps": 0,
                  "samples": 8, "tau": 0.6, "reward_offset": 6.0}
        estimator = MomentEstimator(config, 36203, device)
        observed = []

        def future(eps):
            observed.append(eps.detach().clone())
            return 6 + 0.1 * eps[:, 0]

        result = estimate_fidelity(estimator, torch.zeros(4, device=device), 0.7,
                                   lambda z: (_ for _ in ()).throw(AssertionError("unexpected proxy queries")), future)
        assert len(observed) == 1 and estimator.queries == 8
        assert torch.equal(result.eps, observed[0])
        expected = (0.1 * observed[0][:, 0]).double() / 0.6
        _assert_close(result.log_weights, expected, atol=1e-6, label="future candidate log weights")
        _assert_close(torch.softmax(result.log_weights, 0) @ result.eps.double(),
                      result.ratio.double(), atol=1e-6, label="same weights for ratio and categorical execution")
        return {"future_calls": len(observed), "reward_queries": estimator.queries}

    cases = [
        _check("gaussian_execution_cap", "wrong gain/noise KL, changed uncapped displacement, or a violated cap", gaussian_cap),
        _check("categorical_execution", "extra transition noise, wrong candidate, offset dependence, or argmax selection", resample_realized_candidate),
        _check("same_future_information", "different candidates/weights or additional oracle calls between execution rules", future_candidate_reuse),
    ]
    Path(output, "checks.json").write_text(json.dumps(cases, indent=2) + "\n")
    if any(not case["passed"] for case in cases):
        raise AssertionError("Execution checks failed; repair before the image kernel experiments")
    return {"checks_passed": len(cases), "checks_failed": 0}


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


def check_two_level_moments(device):
    d = 4
    eps = d**0.5 * torch.cat([torch.eye(d, device=device), -torch.eye(d, device=device)])
    cheap_slope = torch.tensor([0.1, -0.05, 0.03, 0.07], device=device)
    full_slope = torch.tensor([-0.08, 0.12, 0.06, 0.03], device=device)
    cheap = 2 + eps @ cheap_slope
    full = 3 + eps @ full_slope
    h, u, _ = two_level_moments(eps, cheap, eps, cheap, full, 1.3)
    return {"h": _assert_close(h, 3.0, atol=2e-7, label="two-level H"),
            "u": _assert_close(u, full_slope, atol=2e-7, label="two-level U")}


def check_fidelity_cost_and_history(device):
    m, rho = torch.zeros(4, device=device), 0.7
    counts = {"proxy": 0, "future": 0}

    def proxy(z):
        counts["proxy"] += len(z)
        return 6 + 0.3 * (torch.log(torch.tensor(2.0, device=device)) + 0.05 * z[:, 0] / rho)

    def future(eps):
        counts["future"] += len(eps)
        return 6 + 0.3 * (torch.log(torch.tensor(3.0, device=device)) + 0.05 * eps[:, 0])

    config = {"kind": "mc", "oracle": "multifidelity", "samples": 4, "warmup_steps": 0,
              "future_samples": 2, "tau": 0.3, "reward_offset": 6.0,
              "coefficient_mode": "previous_mass", "proxy_coefficient": 0.5}
    estimator = MomentEstimator(config, 7051, device)
    first = estimate_fidelity(estimator, m, rho, proxy, future)
    second = estimate_fidelity(estimator, m, rho, proxy, future)
    _assert_close(first.diagnostics["proxy_coefficient"], 0.5, atol=1e-7, label="first coefficient is fixed")
    _assert_close(second.diagnostics["proxy_coefficient"], 0.75, atol=2e-6, label="next coefficient uses previous mass")
    assert counts == {"proxy": 12, "future": 4} and estimator.queries == 16, (counts, estimator.queries)
    counts["proxy"] = counts["future"] = 0
    full = MomentEstimator(dict(config, oracle="future"), 7052, device)
    estimate_fidelity(full, m, rho, proxy, future)
    assert counts == {"proxy": 0, "future": 4} and full.queries == 4, (counts, full.queries)
    return {"two_level_queries_per_step": first.diagnostics["queries"],
            "full_future_queries": full.queries, "next_coefficient": second.diagnostics["proxy_coefficient"]}


def run_fidelity_checks(output):
    device = torch.device("cuda")
    cases = [
        _check("two_level_moments", "wrong sign, scaling, or Gaussian score in two-level moments",
               lambda: check_two_level_moments(device)),
        _check("fidelity_cost_and_history", "omitted proxy/full queries or fitting the current coefficient on current samples",
               lambda: check_fidelity_cost_and_history(device)),
    ]
    Path(output, "checks.json").write_text(json.dumps(cases, indent=2) + "\n")
    failed = sum(not case["passed"] for case in cases)
    if failed:
        raise AssertionError(f"{failed} fidelity checks failed; repair before fidelity experiments")
    return {"checks_passed": len(cases), "checks_failed": 0}


def run_paired_fit_checks(output):
    """Expose pair ordering, difference scaling, and slope/intercept separation."""
    device = torch.device("cuda")

    def fit_known_function():
        center = torch.tensor([0.2, -0.1, 0.05, 0.1], device=device)
        mean = torch.tensor([-0.1, 0.2, 0.4, 0.7], device=device)
        slope = torch.tensor([0.1, -0.15, 0.2, 0.08], device=device)
        eps = 2 * torch.cat([torch.eye(4, device=device), -torch.eye(4, device=device)])
        z = mean + 0.4 * eps
        results = {}
        for kind in ["affine", "exponential"]:
            config = {"surrogate": kind, "update": "paired_lms", "learning_rate": 1.0,
                      "ridge": 0.0, "tau": 0.3, "reward_offset": 6.0}
            model = IntegrableSurrogate(center, 0.7, config,
                                        torch.Generator(device=device).manual_seed(607))
            values = 2 + z.double() @ slope.double()
            weights = values.exp() if kind == "exponential" else values
            rewards = 6 + 0.3 * weights.log()
            model.update(z, rewards, weights)
            results[kind] = _assert_close(model.slope(), slope, atol=2e-7, label="paired slope")
            _assert_close(model.predict(z), weights, atol=2e-6, label="paired mean and slope fit")
        return results

    cases = [_check("paired_surrogate_fit", "wrong plus/minus pairing, slope scaling, or intercept fit",
                    fit_known_function)]
    Path(output, "checks.json").write_text(json.dumps(cases, indent=2) + "\n")
    if not cases[0]["passed"]:
        raise AssertionError("Paired-fit check failed; repair before the contrast regression experiment")
    return {"checks_passed": 1, "checks_failed": 0}
