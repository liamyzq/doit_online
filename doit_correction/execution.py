"""Explicit Gaussian displacement control and finite-candidate resampling."""

import math

import torch


def gaussian_displacement(correction, gain, gamma, noise_std, max_kl=None):
    displacement = gamma * gain * correction
    raw_norm = float(displacement.double().norm())
    scale = 1.0
    if max_kl is not None:
        if max_kl < 0:
            raise ValueError("Execution KL limit must be nonnegative")
        radius = math.sqrt(2 * max_kl) * noise_std
        if raw_norm > radius:
            scale = radius / raw_norm
            displacement = displacement * scale
    applied_norm = float(displacement.double().norm())
    return displacement, {"raw_displacement_norm": raw_norm,
                          "raw_execution_kl": 0.5 * (raw_norm / noise_std) ** 2,
                          "execution_scale": scale,
                          "execution_clipped": int(scale < 1),
                          "displacement_norm": applied_norm,
                          "displacement_over_noise": applied_norm / noise_std,
                          "execution_kl": 0.5 * (applied_norm / noise_std) ** 2}


def categorical_transition(mean, noise_std, eps, log_weights, generator):
    """Select one realized base-transition candidate, without an extra noise draw.

    This is finite-candidate categorical resampling, not an exact tilted kernel.
    Conditional on the candidate set it is discrete, so the Gaussian mean-shift
    KL formula does not apply to this execution rule.
    """
    probabilities = torch.softmax(log_weights.double(), dim=0)
    index = int(torch.multinomial(probabilities, 1, generator=generator))
    selected_noise = eps[index].reshape_as(mean)
    return mean + noise_std * selected_noise, {
        "selected_candidate": index, "candidate_count": len(eps),
        "selected_probability": float(probabilities[index]),
        "selected_noise_norm": float(selected_noise.double().norm()),
        "categorical_ess": float(1 / probabilities.square().sum()),
        "execution_kl": None,
    }
