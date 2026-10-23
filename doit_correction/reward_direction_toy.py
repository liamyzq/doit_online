"""Six-dimensional fixed states of an actual sixteen-step Gaussian sampler."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr

from .complement import stream
from .reward_directions import direction_pool


class ToyProcess:
    def __init__(self, kind, dimension=6):
        self.kind, self.dimension = kind, dimension
        self.rotation = np.linalg.qr(stream(81001).normal(size=(dimension, dimension)))[0]
        self.cov = (self.rotation * np.geomspace(0.08, 2.0, dimension)) @ self.rotation.T
        self.means = np.zeros((1, dimension)) if kind == "gaussian" else np.stack(
            [-1.25 * self.rotation[:, 2], 1.25 * self.rotation[:, 2]])
        self.times = np.linspace(2.0, 0.0, 17)
        self.steps = 16
        self.inverse = [np.linalg.inv(self.cov + t * np.eye(dimension)) for t in self.times]
        self.actor_forwards = 0
        self.reward_queries = 0

    def posterior(self, x, index, covariance=False):
        t, inverse = self.times[index], self.inverse[index]
        a = self.cov @ inverse
        delta = x[..., None, :] - self.means
        logits = -0.5 * np.einsum("...kd,de,...ke->...k", delta, inverse, delta)
        p = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
        centers = (x @ a.T)[..., None, :] + self.means @ (np.eye(self.dimension) - a).T
        mean = np.sum(p[..., None] * centers, axis=-2)
        if not covariance:
            return mean
        diff = centers - mean[..., None, :]
        cov = t * a + np.einsum("...k,...kd,...ke->...de", p, diff, diff)
        return mean, cov

    def transition(self, x, index):
        dt = self.times[index] - self.times[index + 1]
        clean = self.posterior(x, index)
        self.actor_forwards += int(np.prod(x.shape[:-1]))
        return x + dt * (clean - x) / self.times[index], np.sqrt(dt)

    def counts(self):
        return {"reward_queries": self.reward_queries, "score_forwards": self.actor_forwards}


class ToyContext:
    def __init__(self, model, index, state_number, alignment, reward, pool_seed):
        self.model, self.index, self.dimension = model, index, model.dimension
        self.reward = reward
        # Four deterministic states per noise band, including the symmetric center.
        x = model.rotation @ (0.45 * stream(81007, index, state_number).normal(size=self.dimension))
        if state_number == 0:
            x = np.zeros(self.dimension)
        self.x = x
        self.mean, self.std = model.transition(x, index)
        _, cov = model.posterior(x, index, True)
        t = model.times[index]
        dt = model.times[index] - model.times[index + 1]
        self.jmu = np.eye(self.dimension) + dt * (cov / t - np.eye(self.dimension)) / t
        self.gain = dt
        self.pool, self.eigenvalues = direction_pool(self.jmu, pool_seed)
        self.axis = model.rotation[:, 2] if alignment == "in_pool" else (
            model.rotation[:, 2] + model.rotation[:, -1]) / np.sqrt(2)
        # In-pool cases use a shared eigenaxis of every transition. Off-pool cases
        # deliberately mix separated eigenvalues; the terminal direction then
        # pulls back differently for P and F, recorded by exact references below.
        self.oracle = self.axis
        self.scale = np.sqrt(self.axis @ model.cov @ self.axis)

    def logs(self, terminal):
        self.model.reward_queries += int(np.prod(terminal.shape[:-1]))
        z = terminal @ self.axis / self.scale
        if self.reward == "shift":
            return -0.5 * ((z - 1.0) / 0.6)**2
        if self.reward == "narrow":
            return -0.5 * (z / 0.35)**2
        if self.reward == "intervals":
            return np.where(((z > -1.6) & (z < -0.7)) | ((z > 0.9) & (z < 1.5)), 0.0, -3.0)
        if self.reward == "constant":
            return np.zeros(z.shape)
        if self.reward == "hermite4":
            # Center/scale under this context's Gaussian endpoint law. This gives
            # exactly unchanged first two Gaussian moments for the Gaussian toy.
            # The route-specific standardization is set in evaluate/reference.
            return np.log1p(0.05 * (z**4 - 6 * z**2 + 3))
        raise ValueError(self.reward)

    def gaussian_endpoint(self, route):
        d = self.dimension
        if route == "P":
            a = self.model.cov @ self.model.inverse[self.index + 1]
            return a @ self.mean, self.std * a, np.zeros((d, d))
        mean, response, covariance = self.mean.copy(), self.std * np.eye(d), np.zeros((d, d))
        for index in range(self.index + 1, self.model.steps):
            dt = self.model.times[index] - self.model.times[index + 1]
            a = np.eye(d) - dt * self.model.inverse[index]
            mean, response = a @ mean, a @ response
            covariance = a @ covariance @ a.T + dt * np.eye(d)
        return mean, response, covariance

    def evaluate(self, eps, route, seed_words, groups, capacity):
        x = self.mean + self.std * eps
        if route == "P":
            x = self.model.posterior(x, self.index + 1)
            self.model.actor_forwards += int(np.prod(eps.shape[:-1]))
        else:
            rng = stream(*seed_words)
            for index in range(self.index + 1, self.model.steps):
                mean, std = self.model.transition(x, index)
                noise = rng.normal(size=(len(eps), capacity, self.dimension))
                x = mean + std * noise[np.arange(len(eps))[:, None], groups]
        if self.reward == "hermite4":
            center, response, covariance = self.gaussian_endpoint(route)
            sd = np.sqrt(self.axis @ (response @ response.T + covariance) @ self.axis)
            z = ((x - center) @ self.axis) / sd
            self.model.reward_queries += int(np.prod(z.shape))
            return np.log1p(0.05 * (z**4 - 6 * z**2 + 3))
        return self.logs(x)

    def analytic_reference(self, route):
        if self.model.kind != "gaussian":
            return None
        center, response, covariance = self.gaussian_endpoint(route)
        b = response.T @ self.axis / self.scale
        mean = self.axis @ center / self.scale
        variance = b @ b + self.axis @ covariance @ self.axis / self.scale**2
        if self.reward in {"constant", "hermite4"}:
            ratio, mass = np.zeros(self.dimension), 1.0
        elif self.reward in {"shift", "narrow"}:
            target, width = (1.0, 0.6) if self.reward == "shift" else (0.0, 0.35)
            ratio = b * (target - mean) / (width**2 + variance)
            mass = width / np.sqrt(width**2 + variance) * np.exp(-0.5 * (mean - target)**2 / (width**2 + variance))
        else:
            sd, probability, first = np.sqrt(variance), 0.0, 0.0
            for low, high in ((-1.6, -0.7), (0.9, 1.5)):
                lo, hi = (low - mean) / sd, (high - mean) / sd
                probability += ndtr(hi) - ndtr(lo)
                first += sd * (np.exp(-0.5 * lo**2) - np.exp(-0.5 * hi**2)) / np.sqrt(2 * np.pi)
            floor = np.exp(-3)
            mass = floor + (1 - floor) * probability
            ratio = b / variance * (1 - floor) * first / mass
        oracle = b / np.linalg.norm(b)
        return {"guidance": (ratio @ self.jmu / self.std).tolist(), "ratio": ratio.tolist(),
                "mass": float(mass), "kind": "analytic_actual_discrete_sampler", "oracle": oracle.tolist()}

    def describe(self):
        return {"model": self.model.kind, "index": self.index, "reward": self.reward,
                "sigma": self.std, "gain": self.gain, "x": self.x.tolist(),
                "reward_axis": self.axis.tolist(), "kernel_singular_values2": self.eigenvalues.tolist()}
