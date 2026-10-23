# Posterior geometry: current evidence

The initial implementation is correct on its analytic checks. Spectral probe
stratification has not yet established a benefit beyond antithetic pairing and
matched random directions. The local response proxy substantially reduces
candidate reconstruction error, but that does not automatically improve the
finite-query continuation estimator. Real-model verification remains active.

## Exp-063 and exp-064

At commit `1c413e1`, all 40 new checks pass, with maximum analytic discrepancy
2.47e-11. These cover the posterior response identity, independent Gaussian
closed forms, converged low-dimensional quadrature, Gaussian probe marginals,
stratum occupancy and exact antithetic constant-weight cancellation.

Exp-064 completes 960 probe conditions and 144 proxy conditions, including both
seeds and every prescribed reward alignment. Costs are 8,552,448 reward
evaluations, including 65,536 direction-pilot queries, plus 61,632 reference
integrand evaluations and 252,480 analytic component integrals. Component
integrals are computation inside the analytic continuation oracle, not extra
simulated trajectories. Both CPU runs allocate zero GPU tensor memory.

Top-two paired spectral strata do not meet the 20% improvement criterion over
both ordinary antithetic probes and matched random paired strata in any of the
32 model/alignment/budget averages. Against matched random strata, the gain
ranges from -9.91% to +14.71%. At M32 with a top-direction reward, Gaussian MSE
is 0.35263 for antithetic, 0.32531 for random paired strata and 0.30132 for
spectral paired strata. The mixture values are 0.61981/0.58660/0.54432. These
are modest orientation-specific gains; they do not meet the advance rule.
The four-query independent reward pilot also fails to repay its budget here.

The response proxy behaves as its local approximation predicts. Average
Gaussian candidate reconstruction squared error falls from 1.21012 (frozen)
to 0.15759 (center isotropic), 0.02520 (rank four) and essentially zero for the
full response. For the mixture the corresponding errors are
1.14714/0.35237/0.06293/0.02010; nonlinearity leaves error even with the exact
local Jacobian.

However, the Gaussian full-response proxy still has continuation-reference
guidance bias squared 0.05445. Accurate conditional-mean denoising does not
restore the distribution of future terminal rewards. On the mixture, rank
four lowers continuation-reference bias from 0.11122 (isotropic) to 0.07144,
but its finite-M32 continuation MSE is worse, 0.34464 versus 0.28625. On the
Gaussian those MSEs are 0.22506 versus 0.17331. The candidate-denoiser bias and
Monte Carlo variance must remain separate conclusions.

Decision: retain the implementation for the predeclared bounded real-critic
and image mechanism verification. No endpoint improvement or novel algorithm
claim follows from these toy results. Full per-condition comparisons are in
`results/posterior_geometry/exp-064-geometry-toy/analysis.json`.
