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

Next: validate the new explicit future-sign path on Orion, then compare the
selected joint pair on Hopper rows41/45/49/53 at active indices3/5/7. Use the
predeclared M32 primary gate with all contexts retained. No image, endpoint or
simulator result follows from this toy selection.
