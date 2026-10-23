# Joint current/future noise coupling

At `ea193ab`, exp-077 passes all21 new sign-map, target-moment and repeat-
alignment checks. It uses655,360 toy reward evaluations with zero GPU tensor
allocation. Exp-078 completes80 conditions with15,728,640 reward evaluations,
zero direction queries and four exact reference formulas in5.40seconds. Raw
vectors, paired differences and all secondary conditions are retained.

## Locked M32 rotating-target decision

| Dimension | Shared antithetic MSE | Random paired LHS | Joint antithetic | Cartesian four-point |
|---|---:|---:|---:|---:|
| 3 | 0.105783 | 0.104017 | 0.078615 | 0.075075 |
| 6 | 0.202663 | 0.204605 | 0.166906 | 0.143912 |

In dimension3, joint antithetic improves25.68%/24.42% against the two simple
controls, and Cartesian four-point improves29.03%/27.82%. Both pass. Joint
antithetic's paired difference against shared antithetic is-0.027169, with
95% Monte Carlo interval[-0.029578,-0.024759]; the interval against random
paired LHS is also strictly negative.

In dimension6, joint antithetic improves17.64%/18.43%, below the20% threshold.
Cartesian four-point improves28.99%/29.66% and passes. The predeclared rule
prefers the simpler joint pair if both types have eligible dimensions and
advances at most one coupling. Therefore select joint antithetic for the
three-dimensional Hopper task only. Do not switch to Cartesian four-point
or add HalfCheetah after seeing this selection.

The aligned-direction control is unfavorable and limits the mechanism claim.
At M32, joint antithetic MSE is0.120185/0.264880 in dimensions3/6, versus
shared antithetic0.096226/0.192923: increases of24.90%/37.30%. Cartesian
four-point is also worse. Joint sign coupling can improve the rotated-future
example at fixed cost; it is not a uniform variance reduction. M64 results
remain secondary. The [full analysis](../../results/joint_future/exp-078-joint-future-toy/analysis.json)
contains all comparisons and the unchanged gate decision.

This selection fixes the real-critic comparison below. It does not establish
an image, endpoint or simulator result.

## Real-critic adapter validation

Exp-079 passes at `6e9db3c` on Orion. All ten future-sign, constant-weight
guidance and preserved-rollout comparisons have zero discrepancy. All eight
condition-level reward and actor count comparisons also match exactly. The
independent and shared controls therefore retain their established execution
laws, and the Torch future-sign map matches the checked NumPy construction.

The smoke uses 3,328 reward queries, 37,317 actor forwards and three exact
kernel-Jacobian VJPs, including references and validation rollouts. Peak allocated
and reserved memory is 33.16/40 MiB and total evaluation time is 9.12 seconds.
The inherited `vjp_seconds` counter measures deployed correction calls; exact
Jacobian construction is not separately timed here. Its actual three VJPs are
counted, and total elapsed time includes them. No runtime advantage is inferred.

Exp-080 uses four fresh Hopper observations, three active indices, two seeds,
256 repeats and M32/M64, giving 12 contexts and 192 conditions. Each comparison
uses all budgeted points with no direction-acquisition queries. The stored full
kernel Jacobian serves the fixed-state comparison; a deployed estimate would
use one kernel VJP. Paired Monte Carlo intervals condition on the pooled reference.
The eight independent reference-block replacements describe sensitivity, without
providing a calibrated bound on ratio-reference bias.

## Fresh Hopper result and closure

Exp-080 completes all 192 conditions at `6e9db3c`. Joint antithetic fails every
part of the locked M32 gate against both simple controls. The mean unit-gamma
displacement MSE, averaged equally over four observations, three active indices
and two seeds, is:

| Reward budget | Independent future | Shared antithetic | Random paired LHS | Joint antithetic |
|---|---:|---:|---:|---:|
| M32, primary | 3.098965 | 2.393425 | 2.436307 | 3.126653 |
| M64, secondary | 2.872112 | 2.373460 | 2.371673 | 2.959018 |

At M32, joint antithetic increases error by 30.64% relative to shared antithetic
and 28.34% relative to random paired LHS. Its paired displacement-error
differences are +0.733228, with conditional Monte Carlo 95% interval
[0.626090, 0.840367], and +0.690346, with interval [0.576937, 0.803756].
All four observations worsen against both controls. All eight reference-block
replacement differences remain positive: ranges [0.688574, 1.135439] and
[0.660493, 1.109051]. The secondary M64 comparison also worsens all observations,
with aggregate error increases of 24.67% and 24.77%.

References remain uneven. Pooled ESS is 8.79--24.82 on row41 and 1.02--2.96
on row53, whereas rows45/49 have ESS 9,775--41,040. All contexts are retained.
The adverse result is also present on both observations with high ESS:
their M32 differences against shared antithetic are +0.452590 and +0.138586.
The reported intervals and block sensitivity do not resolve reference bias
on the concentrated contexts.

Shared future noise performs better than independent future noise on all four
observations. The failure of joint sign flipping is consistent with losing
that cancellation: when a reward component depends only on future noise, a
shared-future pair cancels its full-score numerator exactly, while a joint
pair generally does not. This is an interpretation of the comparison, not a
measured decomposition or a universal preference for shared noise. The toy
gain remains correct under its stated rotating-target conditions.

The discovery consumes 3,145,728 actual reward queries, 28,311,596 actor forwards
and 36 kernel-Jacobian VJPs, including all reference work and context paths.
Its evaluation takes 158.19 seconds under shared-GPU contention, with peak
allocated/reserved memory of 57.50/78 MiB. An M32 deployment estimate would
use 32 reward queries, an average of 289 actor forwards over these indices,
and one kernel VJP. Corresponding M64 counts are 64, 577 and one. These equal
costs across methods do not compensate for the measured error increase.

Decision: close joint antithetic on these fresh Hopper observations at M32/M64.
No setting advances to reserved confirmation, individual-call timing, images
or simulator returns. Do not switch to Cartesian coupling or HalfCheetah after
this failure. The [complete analysis](../../results/joint_future/exp-080-joint-future-critic-discovery/analysis.json),
raw vectors, matrices, reference blocks and costs are preserved for reproduction.
Together with the separate paid-direction closure in exp-076, this completes
the bounded continuation-aware follow-up. Terminal N4 remains the practical
image baseline; broader coupling families remain an open research question.
