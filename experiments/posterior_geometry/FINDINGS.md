# Posterior geometry: current evidence

The completed exp-063--069 pass tests spectral probe arrangements and local
response proxies under a fixed Gaussian kernel on toy, SD1.5 image states, and
real Hopper/HalfCheetah actors and critics. The
analytic implementation is correct. Spectral probe stratification does not
establish a robust gain over antithetic and matched random controls at these
budgets. Response proxies improve reconstruction, but most image improvement
comes from isotropic contraction, and the selected Hopper rank-one advantage
fails fresh-observation confirmation. No endpoint or simulator expansion is
justified by the locked criteria. All finite queues have exited.

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
practical algorithm claim. The independent image verification follows below.

## Exp-067 and exp-068: image mechanism verification

The FP32 smoke passes with directional half-step discrepancy 0.000596 and
16288 MiB peak reserved memory. Discovery at `a48c54f` uses an 18336 MiB job
allowance plus 2048 MiB reserve and peaks at 16284 MiB. Four fixed contexts
(cafe/lighthouse, indices 10/18) complete all 28 probe and 32 proxy conditions:
11520 reward queries, 10162 score forwards, 620 VJPs and 128 additional visual
decodes. Runtime is 8241.3 seconds on a shared GPU, including 65.3 seconds for
saved visual artifacts. This is not a dedicated-device speed benchmark.

Spectral paired strata improve candidate-guidance MSE relative to matched
random paired strata by +11.13%, -10.36%, -8.45% and +3.44% respectively.
No context meets the locked 20% gain over both random and ordinary antithetic
controls. The reward pilot does not establish an advantage. These are small,
fixed-state comparisons; they do not support an endpoint claim.

The local response shows substantial early contraction. At index 10,
isotropic response reduces denoiser reconstruction error from 6491.54 to
605.82 for cafe and from 6529.57 to 512.89 for lighthouse. Adding rank-eight
spectral information gives 605.76 and 517.24, respectively. Late reconstruction
errors likewise barely change with spectral rank. The saved images show
strong texture corruption in early frozen proxies and similar clean-looking,
soft outputs from the isotropic and low-rank controls. All 128 candidates were
reviewed; see the [visual record](../../results/posterior_geometry/exp-068-geometry-image-discovery/visual-review.md).

The sketch costs 33 score forwards and 16 VJPs per state, taking 12.95--22.80
seconds before it is reused across diagnostic repeats. That setup cost must be
paid or amortized by an actual algorithm. It is not free because the present
fixed-state evaluator shares it. Early asymmetry fractions are 0.74--0.81 and
Ritz residual fractions 0.77--0.90; this rank-eight sketch does not accurately
resolve the full early operator. Closing this implementation at this budget
does not reject an accurately measured higher-rank geometry in general.

The image reference vectors remain dominated by sampling noise: their split
error estimates are comparable to their squared norms, including the full
continuation references. They cannot support fine guidance-bias rankings.
Consequently, improvements in decoded proxies are not relabeled as improved
Doob guidance. No expensive reference escalation or endpoint wave is queued
without a spectral-specific gain over the simpler controls.

## Exp-069: fresh-observation confirmation and closure

At `ccf61a6`, 24 fresh real-critic contexts complete all 576 probe and 384 proxy
conditions. Costs are 12976128 Q evaluations, 19464344 actor forwards and 216
Jacobian VJPs, with 90 MiB peak reserved memory. Larger antithetic references
resolve many small-error comparisons, while all low-ESS cases remain reported.

The selected Hopper rank-one proxy fails confirmation. Mean M32 continuation
MSE is 5.39962 versus isotropic 5.38995, a near-tied point estimate (0.18%
higher), and only two of four new observations improve. This small aggregate
difference does not establish significant harm. Frozen has error 28.96173 and zero guidance 8.11275.
Thus there is a substantial proxy-contraction benefit over frozen in these
states, but the distinct rank-one benefit is not confirmed.

HalfCheetah transfer has a favorable aggregate, rank one 29.77456 versus
isotropic 45.89736. This is concentrated in row26, whose early continuation
reference ESS is only 6.1/28.7 out of 32768 and whose split-reference error is
large. Only two of four observations improve. The selected confirmation task
cannot be replaced after seeing this transfer result, and no robust transfer
claim follows from it. Candidate references also remain unresolved at Hopper
row34/index9 (ESS 1.4); all such conditions are retained.

At M32, spectral paired probe MSE is also close to the matched random control:
Hopper candidate 11.90801 versus 11.84244 and continuation 4.84248 versus
4.92561; HalfCheetah candidate 9.48853 versus 9.39042 and continuation 60.33658
versus 60.03004. No consistent spectral advantage is established.

Decision: close the tested spectral-stratification and low-rank-response
settings at these budgets. Preserve the implementation, exact configs,
negative outcomes and visual evidence for reproduction. The open question is
whether inexpensive geometry can predict reward-relevant directions beyond
ordinary antithetic sampling and bulk contraction. Neither accurate toy
identities nor better denoiser reconstruction alone answers it. No simulator
returns or new practical endpoint improvement are claimed; terminal N4 from
exp-058 remains the accepted image baseline. No project job remains queued.

## Review of the closed pass at 8cfecfc

The supplied review preserves the original closure while narrowing its scope.
The [saved-record reanalysis](../../results/posterior_geometry/review-8cfecfc/report.md)
confirms the aggregate values. Hopper full-Jacobian reconstruction error is
0.0002037 versus isotropic 0.0016298, yet continuation guidance MSE is 6.10426
versus 5.38995. Improved mean reconstruction alone does not establish improved
conditional exponential-reward values or finite-budget guidance. HalfCheetah
excluding row26 has rank-one continuation MSE 3.96044 versus isotropic 2.93399.
This is a concentration sensitivity analysis; row26 remains in formal results.

The sampler applies guidance only at 0 < index <= guidance_end_index: Hopper
indices 1--8 and HalfCheetah 1--10. Thus only Hopper index5 of the locked 5/9/12
diagnostics lies in the active interval. At index5 isotropic/rank-one raw MSE
is 0.69553/5.69486, and unit-gamma transition-mean MSE is 0.28176/2.30701. The
strong discovery at index12 lies outside the active interval. Preserve all-time
mechanism aggregates but do not treat them as deployment-weighted results.
For arbitrary gamma, multiply unit-gamma displacement errors by gamma squared;
unsampled active indices and trajectory-level error remain unknown.

The toy fixes an isotropic Gaussian transition, then changes joint probes or
local denoiser proxies. It does not test the covariance-corrected VE transition
(delta I + delta squared times the log-density Hessian), its altered kernel
score, or Fisher-metric geodesics. The strongest paired spectral comparison is
top rank-two; bottom/extreme and other ranks largely used unpaired designs.

Real-model spectra refer to the symmetric part of the actual local Jacobian.
They are not exact posterior covariance spectra. In images, CFG scale4.5 forms
(1-s) J_negative + s J_positive, which need not be positive semidefinite even
for ideal conditional models. CFG indefiniteness does not explain asymmetry.
The early Ritz residuals0.77/0.90 describe a finite range sketch, not converged
extreme eigenvectors. The half-step finite-difference smoke covers index18
only. An early-geometry follow-up would first need an index10 step-size check.
No higher-rank image escalation is justified by the present evidence.

Each task in exp-069 has four new observations; three time states share each
observation's base path. Its256 repeats estimate fixed-context Monte Carlo
error, not256 independent observations. Each random basis was sampled once
per seed/context/budget and reused across repeats. A split-reference squared
difference divided by four is a fluctuation diagnostic, not a calibrated
confidence interval or a bound on self-normalized-reference bias, especially
at low ESS. Common-reference norm noise cancels from paired MSE differences;
reference uncertainty should be assessed on the comparison itself.

The old `squared_bias` field estimates the squared distance between a finite
repeat mean and a finite reference. It is not known true bias. Old per-repeat
vectors were not saved, so paired-difference intervals cannot be reconstructed
from those aggregates. Future runs save vectors and name this field
`estimated_squared_bias`. Old proxy streams seed+row+index collide across some
contexts, adding correlation without invalidating individual point estimates.
Future context streams use explicit SeedSequence coordinates.

The review motivates the [complement-pairing follow-up](../complement_pairing/PLAN.md).
That study first reproduces the top-aligned toy direction decomposition, then
tests whether four-point sign groups reduce full guidance MSE. The original
rank-one proxy and image-sketch settings remain closed.
