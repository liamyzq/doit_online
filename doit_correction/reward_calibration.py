"""Conditional terminal-reward residuals for inexpensive positive potentials.

Each calibration anchor has one proxy reward and independent continuation
rewards. Neighbouring anchors predict the mean correction and within-anchor
variance separately, so disagreement between anchor means is not mistaken for
future randomness. Empirical UC transports standardized residuals to the
predicted mean and variance. No diffusion or reward-model calls occur here.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp


class RewardCalibration:
    def __init__(self, features, proxy_rewards, future_rewards):
        self.features = np.asarray(features, dtype=np.float64)
        self.proxy = np.asarray(proxy_rewards, dtype=np.float64)
        self.futures = np.asarray(future_rewards, dtype=np.float64)
        self.center = self.features.mean(axis=0)
        self.scale = self.features.std(axis=0)
        # A constant feature carries no neighbourhood information.
        self.scale[self.scale == 0] = 1
        self.standard_features = (self.features - self.center) / self.scale
        self.means = self.futures.mean(axis=1)
        self.variances = self.futures.var(axis=1)
        self.residuals = np.divide(
            self.futures - self.means[:, None], np.sqrt(self.variances[:, None]),
            out=np.zeros_like(self.futures), where=self.variances[:, None] > 0)

    def record(self):
        return {"features": self.features.tolist(), "proxy_rewards": self.proxy.tolist(),
                "future_rewards": self.futures.tolist()}

    @classmethod
    def from_record(cls, record):
        return cls(record["features"], record["proxy_rewards"], record["future_rewards"])

    def predict(self, features, proxy_rewards, tau, kind, neighbours):
        proxy = np.asarray(proxy_rewards, dtype=np.float64)
        query = (np.asarray(features).reshape(-1, self.features.shape[1]) - self.center) / self.scale
        distance = ((query[:, None] - self.standard_features[None]) ** 2).mean(axis=-1)
        indices = np.argsort(distance, axis=1)[:, :neighbours]
        selected_distance = np.take_along_axis(distance, indices, axis=1)
        log_weights = -0.5 * selected_distance
        log_weights -= logsumexp(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        mean = proxy.reshape(-1) + np.sum(weights * (self.means - self.proxy)[indices], axis=1)
        variance = np.sum(weights * self.variances[indices], axis=1)
        if kind == "mean":
            value = mean / tau
        elif kind == "gaussian":
            value = mean / tau + variance / (2 * tau ** 2)
        elif kind == "empirical":
            residual = np.sqrt(variance[:, None, None]) * self.residuals[indices]
            terms = log_weights[:, :, None] - math.log(self.futures.shape[1]) + residual / tau
            value = mean / tau + logsumexp(terms, axis=(1, 2))
        elif kind == "direct":
            # Existing learned-value control: local regression of exp(reward/tau).
            terms = log_weights[:, :, None] - math.log(self.futures.shape[1]) + self.futures[indices] / tau
            value = logsumexp(terms, axis=(1, 2))
        else:
            raise ValueError(kind)
        return value.reshape(proxy.shape), {"mean": mean.reshape(proxy.shape),
                                            "variance": variance.reshape(proxy.shape)}


def check_reward_calibration():
    """New moment algebra, deterministic futures, and reward-offset invariance."""
    features = np.array([[0.0, 1.0]])
    bank = RewardCalibration(features, [1.0], [[1.0, 3.0]])
    query, tau = np.array([4.0]), 0.7
    expected = {"mean": 5 / tau, "gaussian": 5 / tau + 1 / (2 * tau ** 2),
                "empirical": logsumexp(np.array([4.0, 6.0]) / tau) - math.log(2),
                "direct": logsumexp(np.array([1.0, 3.0]) / tau) - math.log(2)}
    errors = {}
    for kind, target in expected.items():
        got, moments = bank.predict(features, query, tau, kind, 1)
        errors[kind] = abs(float(got[0]) - target)
        errors[kind + "_variance"] = abs(float(moments["variance"][0]) - 1)
    shifted = RewardCalibration(features, [11.0], [[11.0, 13.0]])
    got, _ = shifted.predict(features, query + 10, tau, "empirical", 1)
    errors["reward_translation"] = abs(float(got[0]) - expected["empirical"] - 10 / tau)
    deterministic = RewardCalibration(features, [1.0], [[2.0, 2.0]])
    got, _ = deterministic.predict(features, query, tau, "empirical", 1)
    errors["deterministic_future"] = abs(float(got[0]) - 5 / tau)
    if max(errors.values()) > 1e-12:
        raise AssertionError(errors)
    return errors
