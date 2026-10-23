"""Focused regressions for independent repeat blocks in paired comparisons."""

import numpy as np
import torch

from .complement import sign_probes, stream
from .complement_critic import continue_group
from .complement_toy import stochastic_log_weight
from .geometry_toy import Toy, dump


def run(config, output):
    repeats, dimension, budget = 32, 12, 32
    basis = np.linalg.qr(stream(74001, 0).normal(size=(dimension, 2)))[0]
    ordinary = sign_probes(stream(74003, 0), repeats, 32, dimension, capacity=budget)
    four = sign_probes(stream(74003, 0), repeats, 28, dimension, basis[:, :1], True, True, capacity=budget)
    # Varying main counts must still reuse randomness only within the same
    # repeat. LHS changes active coordinates, so compare the common complement.
    def complement(eps):
        return eps - (eps @ basis) @ basis.T
    errors = {"repeat_aligned_complement_prefix": float(abs(complement(ordinary[:, :7]) - complement(four[:, :7])).max())}
    one = sign_probes(stream(74003, 1), repeats, 32, dimension, basis[:, :1], False, True, capacity=budget)
    two = sign_probes(stream(74003, 1), repeats, 32, dimension, basis, False, True, capacity=budget)
    errors["repeat_aligned_lhs_coordinate_across_ranks"] = float(abs(one @ basis[:, 0] - two @ basis[:, 0]).max())

    model = Toy("mixture", dimension)
    model.set_reward("top")
    independent = stochastic_log_weight(model, np.zeros((repeats, 32, dimension)), stream(74007, 0), 1, budget)
    shared = stochastic_log_weight(model, np.zeros((repeats, 28, dimension)), stream(74007, 0), 4, budget)
    errors["repeat_aligned_posterior_randomness"] = float(abs(independent[:, :7] - shared[:, :7]).max())

    class IdentityTransition:
        steps = 3

        def transition(self, x, index):
            return x, 0.3, x

        def noise(self, shape, rng):
            return torch.randn(shape, device="cuda", generator=rng)

    identity = IdentityTransition()
    def future(count, group):
        return continue_group(identity, torch.zeros((repeats, count, 6), device="cuda"), 0,
                              torch.Generator(device="cuda").manual_seed(74011), group, capacity=budget)
    full, reduced, grouped = future(32, 1), future(28, 1), future(28, 4)
    errors["repeat_aligned_torch_independent_future"] = float((full[:, :28] - reduced).abs().max())
    errors["repeat_aligned_torch_grouped_future"] = float((full[:, :7] - grouped[:, :7]).abs().max())
    errors["torch_shared_group_equality"] = float(max((grouped[:, :7] - grouped[:, k:k + 7]).abs().max() for k in (7, 14, 21)))
    failed = {key: value for key, value in errors.items() if value > 1e-10}
    dump(output / "checks.json", {"errors": errors, "failed": failed})
    if failed:
        raise AssertionError(failed)
    return {"checks_passed": len(errors), "toy_reward_evaluations": repeats * (32 + 28),
            "synthetic_future_normal_draws": 3 * repeats * budget * 6 * identity.steps,
            "purpose": "Detect cross-repeat random reuse when main count, sign-group size or LHS rank differs"}
