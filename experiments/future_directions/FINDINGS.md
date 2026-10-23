# Continuation-conditioned directions: evidence

Exp-075 passes all19 checks of the new conditional Gaussian construction and
exact joint moments. Exp-076 completes128 conditions at `8ac4337`, with
24,510,464 actual toy reward evaluations:16,711,680 main and7,798,784 direction
calls. Analytic oracle derivatives are evaluated1,605,632 times. These counts
are toy evaluations, not real neural-network queries. Runtime is17.03seconds
and GPU allocation is zero. Full per-repeat vectors and all costs are saved.

## Locked mechanism gate

At M64 and60-degree future rotation, the paid path-gradient four-point oracle
has MSE0.045402 in dimension3, versus antithetic0.051816 and random paired
LHS0.051768. Its gains are12.38%/12.30%, below the predeclared20% threshold.
In dimension6, oracle MSE0.045772 versus0.101670/0.101720 gives54.98%/55.00%
gains. All four conditional paired intervals favor the oracle, but only one
of the two required dimension gates passes. Do not relax this gate afterward.

The direction mechanism is useful in the six-dimensional example, but paid
black-box acquisition is much worse under these budgets. Full path finite
differences at M64 give MSE0.099384/0.161690 in dimensions3/6, compared with
antithetic0.051816/0.101670. They spend36/48 of64 available queries acquiring
directions, leaving24/16 main points and only6/4 independent future blocks.
The four-query pilot quadruple gives0.124443/0.288127. It spends half the
budget on acquisition and still captures only a random two-dimensional part
of the direction. Paid path-gradient pairs also lose,0.075795/0.149728.
All M32 conditions and aligned-direction controls remain in the complete
[analysis](../../results/future_directions/exp-076-future-direction-toy/analysis.json).

Decision: close this per-future paid-direction design at the tested budgets.
No new real-critic path gradient, image or simulator run follows it. Preserve
the favorable six-dimensional oracle result without treating it as a usable
black-box method. The next separately stated hypothesis concerns free coupling
of future noise, rather than another direction-acquisition parameter setting.
