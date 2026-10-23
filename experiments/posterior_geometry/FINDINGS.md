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

## Exp-065 and exp-066: real RL actors and critics

The new next-state clean-response smoke passes on both tasks (relative
directional errors 0.00331/0.00528), with 46 MiB reserved memory. Exp-066 at
`b57396f` completes 2304 probe and 216 proxy conditions at 12 fixed contexts.
It costs 10,616,832 Q evaluations and 20,119,628 actor forwards, including
references, pilots and base-state construction. The 108 recorded geometry
VJPs materialize both the kernel-mean and clean-response Jacobians for these
low-dimensional diagnostics. Peak reserved memory is 88 MiB.

The learned response is strongly contractive and often nonsymmetric; measured
asymmetry fractions range from 0.1767 to 1.2602. Several symmetric-part
eigenvalues are negative. The implementation evaluates the actual local
network response correctly, but these matrices are not literal posterior
covariances. This limits the statistical interpretation of spectral directions.

On Hopper, average M32 candidate-integrand MSE is 0.53487 for antithetic,
0.46155 for random paired strata and 0.41866 for spectral paired strata.
For the full-continuation integrand those values are
2.03524/2.00455/1.98434. HalfCheetah candidate values are
0.35486/0.35030/0.34175 and continuation values
1.79530/1.76972/1.80108. Thus no task-wide 20% improvement over both controls
is established. Only Hopper row28/index9, with candidate denoising and M8 or
M32, passes that local threshold with sufficiently small reference error.

Changing the frozen-score proxy is a stronger signal. Mean M32 correction
error against the independent continuation reference is 30.46581 for frozen,
1.95839 for isotropic and 0.81975 for rank-one response on Hopper. The
zero-correction error is 2.58702. The rank-one advantage is driven mainly by
row28/index12 (2.341 versus isotropic 9.243); it does not consistently improve
all discovery contexts. HalfCheetah favors isotropic: 1.18377 versus rank-one
1.36216, with frozen 204.787 and zero correction 1.91067.

These finite-M errors are measured against the continuation reference. They
are not inferred from the frozen proxy's own integration reference, whose ESS
is near one at several early states and cannot support fine bias estimates.
Some low-variance candidate comparisons are also reference-limited despite
8192 iid reference samples. Retain all conditions and their reference errors.

Decision: lock exp-069 on four new observations per task with larger paired
references, selecting rank one on Hopper and treating HalfCheetah as transfer.
The apparent local proxy improvement is not a simulator-return result or a
practical algorithm claim. Image verification remains independent and active.
