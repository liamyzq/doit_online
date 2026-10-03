# Estimators and interpretation

The implemented target and experiment decisions are specified in
[the study protocol](study.md) and [PROGRESS.md](../PROGRESS.md). This note
explains the identities needed to interpret the saved experiments.

At a fixed guided step, write the Gaussian endpoint proxy as
`Z = m + rho * epsilon`, with standard Gaussian epsilon, and use the fixed
weight function `F(z) = exp((reward(z) - reward_offset) / tau)`. The estimator
tracks `H = E[F(Z)]` and `U = E[F(Z) * epsilon]`. The image engine applies
`J_mu.T @ (U / H) / sigma_transition` through a VJP. The endpoint noise scale
rho and transition noise scale sigma_transition have different roles.

## Online control variates

For an affine surrogate `u(z) = a + b.T @ z`, the analytic moments are
`H_u = a + b.T @ m` and `U_u = rho * b`. For an exponential-affine surrogate
`u(z) = exp(a + b.T @ z)`, they are
`H_u = exp(a + b.T @ m + rho**2 * ||b||**2 / 2)` and
`U_u = H_u * rho * b`.

The estimate adds these analytic moments to fresh residual averages. A fixed
coefficient multiplies both the analytic surrogate and its sampled subtraction.
Coordinates are anchored once per rollout; full coordinates and fixed linear
projections preserve analytic integration. Normalized LMS, projected recursive
least squares, spatial DCT projections, and direct antithetic contrast fitting
are implemented in `doit_correction/estimators.py`. Memory is updated only after
the current estimate is formed. Reset and frozen-memory controls use the same
ordering.

The conditional moment identity survives a poor surrogate fit. The ratio and
the denominator floor still introduce finite-sample bias. Moreover, pointwise
prediction error is not the relevant loss by itself: the numerator multiplies
the residual by a high-dimensional noise vector. With antithetic sampling, the
plus/minus residual contrast is especially relevant. Exp-026 fits this contrast
directly; its negative result is recorded rather than inferred from regression
error alone.

A constant control variate is exactly redundant with complete antithetic pairs.
For any historical constant c, `c + mean(F - c) = mean(F)`, and
`mean((F - c) * epsilon) = mean(F * epsilon)` because `mean(epsilon) = 0`.
The agreement observed in the experiments is correct algebra.

## Recursive moments and adaptive proposals

SPIDER estimates the difference between current and previous Gaussian moments
using the same fresh epsilon at both states. Each difference costs two reward
queries. Historical error persists until refresh, and a small coupled difference
variance must repay this doubled query cost. The comparison includes matched
MC schedules and records negative denominators.

An exponential surrogate gives the exact tilted proposal
`q = N(m + rho**2 * b, rho**2 * I)`. The numerator always uses the base coordinate
`epsilon = (z - m) / rho`, including the shift. Gaussian and mixture proposals
use the exact `p / q` likelihood ratio. A KL limit can reduce the proposal shift,
and a base-Gaussian mixture can limit weight growth; neither proves an accurate
guidance direction.

Pure IS and CV-IS are exactly equal for the full exponential tilt with complete
antithetic pairs and a unit CV coefficient, provided the KL limit does not alter
the tilt. Let `delta = rho * b`. Then `p(z) * u(z) / q(z) = H_u`, while the paired
sample mean of base epsilon equals delta. Hence the sampled surrogate H term
equals `H_u`, and its U term equals `H_u * delta = U_u`. Both analytic additions
cancel their sampled subtractions. Mixture sampling or a modified tilt removes
this exact identity. Exp-018's equality is therefore expected.

## Biased controls and future fidelity

`ema` smooths the moment ratio before the VJP. `guidance_ema` instead smooths the
actual VJP output. They are distinct biased controls. A low normalized guidance
MSE alone is insufficient evidence of effective steering: a zero estimate has
normalized error one, and the measured noisy-reference errors can exceed one.
Only independent endpoint improvements can establish practical value here.

Direct future sampling starts at each one-step candidate and executes every
remaining base denoising step before evaluating its terminal reward. Paired
futures share later noises within each plus/minus pair; independent futures do
not. Both preserve each candidate's marginal conditional rollout. The two-level
extension estimates `a * E[F_proxy] + E[F_future - a * F_proxy]`, with independent
cheap and correction batches. Any historical mass coefficient is frozen before
these batches and updated afterward.

The cost record includes each future denoiser evaluation, not just its terminal
reward query. Image score counts are CFG calls counted by candidate, so each
count includes the unconditional and conditional prediction branches. Total
image times include final evaluation and exclude model loading. Reported
component timers overlap as described in the study protocol; they are not a
partition to be summed into total time.
