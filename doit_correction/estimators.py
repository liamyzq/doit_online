"""Gaussian moment estimators with memory updated after each fresh estimate.

The target is E[F] and E[F epsilon], not the ratio's expectation. All proposals
use the original base-Gaussian epsilon in the numerator. A surrogate fitted
after an estimate is available only to subsequent calls.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class Estimate:
    ratio: torch.Tensor
    h: torch.Tensor
    u: torch.Tensor
    diagnostics: dict
    eps: torch.Tensor | None = None
    log_weights: torch.Tensor | None = None


def gaussian_noise(n, dimension, generator, device, antithetic=True):
    if not antithetic:
        return torch.randn(n, dimension, generator=generator, device=device)
    pairs = n // 2
    base = torch.randn(pairs, dimension, generator=generator, device=device)
    pieces = [base, -base]
    if n % 2:
        pieces.append(torch.randn(1, dimension, generator=generator, device=device))
    return torch.cat(pieces)


def spatial_dct_projection(dimension, k, device):
    """Low spatial frequencies in each of the four image latent channels."""
    side, band = math.isqrt(dimension // 4), math.isqrt(k // 4)
    if 4 * side**2 != dimension or 4 * band**2 != k or band > side:
        raise ValueError("Image DCT dimensions must be 4 * side**2 and 4 * band**2")
    positions = torch.arange(side, device=device, dtype=torch.float64) + 0.5
    frequencies = torch.arange(band, device=device, dtype=torch.float64)
    basis = torch.cos(math.pi * positions[:, None] * frequencies[None] / side)
    basis *= math.sqrt(2 / side)
    basis[:, 0] /= math.sqrt(2)
    spatial = torch.einsum("hi,wj->hwij", basis, basis).reshape(side**2, band**2)
    projection = torch.zeros(dimension, k, device=device, dtype=torch.float64)
    for channel in range(4):
        projection[channel * side**2:(channel + 1) * side**2,
                   channel * band**2:(channel + 1) * band**2] = spatial
    return projection.float()


class IntegrableSurrogate:
    """Affine or exponential-affine model with exact Gaussian moments.

    Coordinates are anchored once per rollout. A fixed linear projection keeps
    RLS small while preserving analytic integration in the original latent space.
    """

    def __init__(self, m, rho, config, generator):
        self.center = m.detach().clone()
        self.scale = max(float(rho), 1e-4)
        self.config = config
        self.exponential = config.get("surrogate", "affine") == "exponential"
        self.update_rule = config.get("update", "lms")
        k = int(config.get("projection_dim", 0))
        self.projection = None
        if k and k < m.numel():
            if config.get("projection_kind", "random") == "spatial_dct":
                self.projection = spatial_dct_projection(m.numel(), k, m.device)
            else:
                matrix = torch.randn(m.numel(), k, device=m.device, generator=generator)
                self.projection = torch.linalg.qr(matrix, mode="reduced").Q
        self.k = k if self.projection is not None else m.numel()
        if self.update_rule == "rls" and self.k > 512:
            raise ValueError("Use a projection of at most 512 dimensions for RLS")
        self.theta = torch.zeros(self.k + 1, device=m.device, dtype=torch.float64)
        self.fitted = False
        if self.update_rule == "rls":
            self.gram = torch.eye(self.k + 1, device=m.device, dtype=torch.float64)
            self.gram *= float(config.get("ridge", 1.0))
            self.rhs = torch.zeros_like(self.theta)

    def features(self, z):
        x = (z - self.center) / self.scale
        if self.projection is not None:
            x = x @ self.projection
        x = x.double()
        return torch.cat([torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype), x], -1)

    def slope(self):
        slope = self.theta[1:]
        if self.projection is not None:
            slope = self.projection.double() @ slope
        return slope / self.scale

    def predict(self, z):
        if not self.fitted:
            return torch.zeros(z.shape[0], device=z.device, dtype=torch.float64)
        value = self.features(z) @ self.theta
        return value.exp() if self.exponential else value

    def moments(self, m, rho):
        if not self.fitted:
            return torch.zeros((), device=m.device, dtype=torch.float64), torch.zeros_like(m, dtype=torch.float64)
        value = (self.features(m[None]) @ self.theta)[0]
        b = self.slope()
        if self.exponential:
            h = torch.exp(value + 0.5 * float(rho) ** 2 * b.square().sum())
            return h, h * float(rho) * b
        return value, float(rho) * b

    def update(self, z, rewards, weights):
        x = self.features(z)
        y = ((rewards.double() - float(self.config.get("reward_offset", 6.0))) /
             float(self.config.get("tau", 0.3))) if self.exponential else weights.double()
        if self.update_rule == "lms":
            if not self.fitted:
                self.theta[0] = y.mean()
            rate = float(self.config.get("learning_rate", 0.5))
            ridge = float(self.config.get("ridge", 1.0))
            for xi, yi in zip(x, y):
                self.theta += rate * (yi - xi @ self.theta) * xi / (ridge + xi.square().sum())
        elif self.update_rule == "paired_lms":
            if len(x) % 2:
                raise ValueError("Paired LMS requires complete antithetic pairs")
            half = len(x) // 2
            contrasts = (x[:half, 1:] - x[half:, 1:]) / 2
            targets = (y[:half] - y[half:]) / 2
            rate = float(self.config.get("learning_rate", 0.5))
            ridge = float(self.config.get("ridge", 1.0))
            for xi, yi in zip(contrasts, targets):
                self.theta[1:] += rate * (yi - xi @ self.theta[1:]) * xi / (ridge + xi.square().sum())
            # The mean is fitted separately so moving rollout centers cannot
            # dominate the slope update's normalization or regression loss.
            self.theta[0] = (y - x[:, 1:] @ self.theta[1:]).mean()
        elif self.update_rule == "rls":
            forgetting = float(self.config.get("forgetting", 1.0))
            self.gram = forgetting * self.gram + x.T @ x
            self.rhs = forgetting * self.rhs + x.T @ y
            self.theta = torch.linalg.solve(self.gram, self.rhs)
        else:
            raise ValueError(self.update_rule)
        self.fitted = True


class MomentEstimator:
    def __init__(self, config, seed, device):
        self.config = dict(config)
        self.kind = self.config.get("kind", "mc")
        self.generator = torch.Generator(device=device).manual_seed(seed)
        # Projection generation never changes the candidate-noise stream.
        self.feature_generator = torch.Generator(device=device).manual_seed(seed + 104729)
        self.surrogate = None
        self.previous = None
        self.previous_ratio = None
        self.constant = None
        self.calls = 0
        self.queries = 0

    def budget(self, step):
        warmup_steps = int(self.config.get("warmup_steps", 1))
        refresh = int(self.config.get("refresh_every", 0))
        is_refresh = step < warmup_steps or (refresh > 0 and step % refresh == 0)
        return int(self.config.get("warmup_samples", 32) if is_refresh else self.config.get("samples", 8))

    def weights(self, rewards):
        return torch.exp((rewards.double() - float(self.config.get("reward_offset", 6.0))) /
                         float(self.config.get("tau", 0.3)))

    def ensure_surrogate(self, m, rho):
        if self.surrogate is None:
            self.surrogate = IntegrableSurrogate(m, rho, self.config, self.feature_generator)

    def proposal(self, m, rho, n):
        self.ensure_surrogate(m, rho)
        eps = gaussian_noise(n, m.numel(), self.generator, m.device,
                             self.config.get("antithetic", True))
        shift = torch.zeros_like(m)
        if self.surrogate.fitted:
            shift = (float(rho) * self.surrogate.slope()).float()
            shift *= float(self.config.get("proposal_scale", 1.0))
            radius = math.sqrt(2 * float(self.config.get("proposal_kl", 0.5)))
            norm = float(shift.norm())
            if norm > radius:
                shift *= radius / norm
        base_mass = float(self.config.get("base_mass", 0.2))
        if base_mass < 0 or base_mass > 1:
            raise ValueError("base_mass must be in [0,1]")
        # Independent mixture indicators preserve the mixture marginal for
        # every antithetic Gaussian draw. They are not importance weights.
        guided = torch.rand(n, generator=self.generator, device=m.device) >= base_mass
        eps = eps + guided[:, None] * shift
        log_qtilt_over_p = eps.double() @ shift.double() - 0.5 * shift.double().square().sum()
        if base_mass == 0:
            log_q_over_p = log_qtilt_over_p
        elif base_mass == 1:
            log_q_over_p = torch.zeros_like(log_qtilt_over_p)
        else:
            log_q_over_p = torch.logaddexp(
                torch.full_like(log_qtilt_over_p, math.log(base_mass)),
                math.log1p(-base_mass) + log_qtilt_over_p)
        return eps, (-log_q_over_p).exp(), float(shift.norm())

    def estimate(self, m, rho, reward: Callable, *, step=None):
        """Evaluate an estimate, then fit memory with its queries.

        reward receives endpoint latents [N,D]. For SPIDER the previous and
        current endpoint are evaluated using the same *fresh* epsilons.
        """
        step = self.calls if step is None else step
        n = self.budget(step)
        anti = self.config.get("antithetic", True)
        proposal_norm = 0.0
        if self.kind in {"is", "cv_is"}:
            eps, importance, proposal_norm = self.proposal(m, rho, n)
        else:
            if self.kind == "spider" and not self._refresh(step):
                if n % 2:
                    raise ValueError("SPIDER difference budgets must be even")
                n //= 2
            eps = gaussian_noise(n, m.numel(), self.generator, m.device, anti)
            importance = torch.ones(n, device=m.device, dtype=torch.float64)
        z = m[None] + float(rho) * eps
        started = time.perf_counter()
        rewards = reward(z).detach().double()
        previous_rewards = None
        if self.kind == "spider" and not self._refresh(step):
            pm, pr, _, _ = self.previous
            previous_rewards = reward(pm[None] + pr * eps).detach().double()
        oracle_seconds = time.perf_counter() - started
        result = self.from_samples(m, rho, eps, rewards, previous_rewards=previous_rewards,
                                   importance=importance, step=step)
        result.diagnostics.update(oracle_seconds=oracle_seconds, proposal_norm=proposal_norm)
        return result

    def _refresh(self, step):
        interval = int(self.config.get("refresh_every", 3))
        return self.previous is None or step < int(self.config.get("warmup_steps", 1)) or (interval > 0 and step % interval == 0)

    def from_samples(self, m, rho, eps, rewards, *, previous_rewards=None, importance=None, step=None):
        """Same estimator for live queries or stored, independent reward banks."""
        started = time.perf_counter()
        step = self.calls if step is None else step
        if importance is None:
            importance = torch.ones(len(rewards), device=m.device, dtype=torch.float64)
        f = self.weights(rewards)
        e = eps.double()
        h = (importance * f).mean()
        u = ((importance * f)[:, None] * e).mean(0)
        raw_h, raw_u = h, u
        predictions = torch.zeros_like(f)
        coefficient = float(self.config.get("cv_coefficient", 1.0))
        use_surrogate = self.kind in {"cv", "cv_reset", "is", "cv_is"}
        if use_surrogate:
            if self.kind == "cv_reset":
                self.surrogate = None
            self.ensure_surrogate(m, rho)
            predictions = self.surrogate.predict(m[None] + float(rho) * eps)
        if self.kind in {"cv", "cv_reset", "cv_is"}:
            hu, uu = self.surrogate.moments(m, rho)
            residual = f - coefficient * predictions
            h = coefficient * hu + (importance * residual).mean()
            u = coefficient * uu + ((importance * residual)[:, None] * e).mean(0)
        elif self.kind == "constant" and self.constant is not None:
            predictions = torch.full_like(f, self.constant)
            h = self.constant + (f - predictions).mean()
            u = ((f - predictions)[:, None] * e).mean(0)
        elif self.kind == "spider" and not self._refresh(step):
            if previous_rewards is None:
                raise ValueError("SPIDER requires fresh coupled previous-state rewards")
            difference = f - self.weights(previous_rewards)
            _, _, ph, pu = self.previous
            h = ph + difference.mean()
            u = pu + (difference[:, None] * e).mean(0)
        elif self.kind not in {"mc", "ema", "guidance_ema", "spider", "is", "constant"}:
            raise ValueError(self.kind)
        floor = float(self.config.get("denominator_floor", 1e-8))
        ratio = u / h.clamp_min(floor)
        if self.kind == "ema" and self.previous_ratio is not None:
            beta = float(self.config.get("ema_beta", 0.5))
            ratio = beta * ratio + (1 - beta) * self.previous_ratio
        if not torch.isfinite(ratio).all():
            raise FloatingPointError("Nonfinite Gaussian moment ratio")
        nqueries = len(rewards) + (0 if previous_rewards is None else len(previous_rewards))
        normalized = importance * f
        ess = normalized.sum().square() / normalized.square().sum().clamp_min(1e-30)
        model_residual_mse = (f - predictions).square().mean()
        applied_residual = f
        if self.kind in {"cv", "cv_reset", "cv_is"}:
            applied_residual = f - coefficient * predictions
        elif self.kind == "constant":
            applied_residual = f - predictions
        elif self.kind == "spider" and previous_rewards is not None:
            applied_residual = f - self.weights(previous_rewards)
        pair_difference_mse = None
        if (self.config.get("antithetic", True) and len(f) % 2 == 0
                and self.kind not in {"is", "cv_is"}):
            half = len(f) // 2
            pair_difference_mse = float((applied_residual[:half] - applied_residual[half:]).square().mean())
        # Prediction residuals are held out from the current fit. The weighted
        # moment below centers on this SAME batch's estimated ratio; it is not
        # a variance estimate or confidence score (raw M1 makes it zero).
        # None of these diagnostics chooses the current coefficient.
        diagnostics = {
            "queries": nqueries, "h": float(h), "raw_h": float(raw_h),
            "denominator_floored": int(h < floor), "ess": float(ess),
            "ratio_norm": float(ratio.norm()), "raw_ratio_norm": float((raw_u / raw_h).norm()),
            "heldout_prediction_mse": float(model_residual_mse),
            "heldout_residual_mse": float(applied_residual.square().mean()),
            "heldout_pair_difference_mse": pair_difference_mse,
            "in_sample_weighted_residual_moment": float(((importance * applied_residual).square() * (e - ratio).square().sum(-1)).mean()),
        }
        # All current outputs above use only previously fitted memory.
        update_started = time.perf_counter()
        freeze_after = int(self.config.get("freeze_after_steps", -1))
        if use_surrogate and (freeze_after < 0 or step < freeze_after):
            self.surrogate.update(m[None] + float(rho) * eps, rewards, f)
        self.constant = float(f.mean()) if self.constant is None else 0.5 * self.constant + 0.5 * float(f.mean())
        self.previous = (m.detach().clone(), float(rho), h.detach().clone(), u.detach().clone())
        self.previous_ratio = ratio.detach().clone()
        self.calls += 1
        self.queries += nqueries
        diagnostics["update_seconds"] = time.perf_counter() - update_started
        diagnostics["estimator_seconds"] = time.perf_counter() - started
        return Estimate(ratio.float(), h, u, diagnostics)
