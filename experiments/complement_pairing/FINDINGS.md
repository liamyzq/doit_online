# Complement pairing: current evidence

The complement-cancellation mechanism is correct and produces large gains
with an exact reward direction in the toy. Exp-073 also finds useful candidate-
integrand gains with paid directions on real critics. None of the six feasible
shared-continuation task/method comparisons passes the locked discovery gate.
Hopper's strongest paid method improves the aggregate by20.72%, but only two
of four observations improve; all three feasible methods worsen HalfCheetah.
Close the tested settings without image, endpoint or simulator expansion.
Terminal N4 remains the practical image baseline.

The review's P0 corrections are recorded in the earlier geometry findings and
[recomputed review](../../results/posterior_geometry/review-8cfecfc/report.md).
Exp-071's nominal paired inference is superseded; exp-071-r2 is the authoritative
toy decision after the repeat-stream repair. The original failed smoke, its
FP32 diagnosis, all rerun costs and all raw observations remain preserved.

## Corrected interpretation of exp-064

Exp-070 at `9ca5f7f` passes all16 new sign-group, full-moment and oracle-coupling
checks. All12 replayed top-aligned M32 rows match the original total MSE within
1.11e-16. The Gaussian and mixture posterior geometry makes these particular
rewards functions of a single active coordinate. Mixture component probabilities
either depend on that same coordinate or cancel because the component scalar
reward laws coincide. This does not assert rank-one sufficiency for arbitrary
mixture rewards.

| Toy | Method | Full guidance MSE | Reward-direction MSE | Complement MSE |
|---|---|---:|---:|---:|
| Gaussian | Antithetic | 0.352628 | 0.046627 | 0.306001 |
| Gaussian | Random paired strata | 0.325315 | 0.045292 | 0.280023 |
| Gaussian | Spectral paired strata | 0.301321 | 0.002811 | 0.298510 |
| Mixture | Antithetic | 0.619810 | 0.086075 | 0.533736 |
| Mixture | Random paired strata | 0.586595 | 0.077204 | 0.509392 |
| Mixture | Spectral paired strata | 0.544322 | 0.004110 | 0.540212 |

Spectral strata reduce reward-direction error by93.97%/95.22% relative to
ordinary antithetic sampling. Complement error makes up99.07%/99.24% of what
remains. These are direction-specific effects; the original total-MSE gate
still fails. The review's new mechanism diagnosis is correct.

## Exp-071: fixed-budget exact toy

At `9ca5f7f`, all320 conditions complete on Orion with zero GPU tensor memory.
Costs include5,963,776 reward evaluations (262,144 of them direction pilots),
2,851,104 analytic component integrals and384 reference integrand evaluations.
Each condition has1024 repeats; two independent seeds are pooled below.
Per-repeat estimates and paired differences are preserved. The saved nominal
intervals are invalidated below; corrected inference comes from exp-071-r2.

The original M32 exact-direction point estimates improve by more than20%
against both antithetic and oracle paired-LHS controls in all four deterministic
conditions. Its nominal intervals assumed independent paired losses, which
the stream review disproved. Preserve these point estimates; use the corrected
exp-071-r2 run for intervals and the formal advance decision.

| Toy | Weight target | Antithetic MSE | Oracle paired LHS | Oracle four-point LHS |
|---|---|---:|---:|---:|
| Gaussian | Candidate denoiser | 0.349572 | 0.302405 | 0.015815 |
| Mixture | Candidate denoiser | 0.607041 | 0.538072 | 0.022647 |
| Gaussian | Analytic continuation | 0.084556 | 0.072852 | 0.006607 |
| Mixture | Analytic continuation | 0.316755 | 0.263955 | 0.023946 |

The complement contribution cancels to numerical precision. Four-point groups
pay for that cancellation with fewer active samples: the candidate oracle
Rao--Blackwell bound has MSE0.002881/0.004152, substantially below four-point
MSE0.015815/0.022647. This oracle is an explanatory bound, not an algorithm for
unknown real-model reward dependence.

Rotating the chosen direction by30degrees retains candidate MSE gains of
49.55%/45.83% over antithetic; analytic-continuation gains are47.80%/48.74%.
But the eight-query pilot captures only about one third of the squared true
direction and fails after paying its full budget: candidate MSE0.583995/1.058194
versus antithetic0.349572/0.607041. This new continuous-direction pilot, like the
earlier discrete four-query selector, has not established a useful toy method.

Independent future draws prevent exact samplewise complement cancellation.
M32 four-point MSE is0.517586/0.675950 versus antithetic0.606063/0.978450, gains
of14.60%/30.92%. Sharing future randomness within the respective two/four-point
groups improves the comparison: four-point0.119818/0.185254 versus shared-pair
antithetic0.348533/0.713738, gains65.62%/74.04%. Shared streams retain each
posterior draw's correct marginal. They do not make the self-normalized ratio
unbiased. All independent-noise outcomes remain reported.

Original decision: prepare the bounded real-critic direction study. The
repeat-stream correction now holds discovery until exp-071-r2 passes the gate. Preserve the failed paid pilot and noise sensitivity; neither
is hidden by the favorable oracle result. The
[complete analysis](../../results/complement_pairing/exp-071-complement-toy/analysis.json)
contains both budgets, every configured method and the exact gate decisions.

## Exp-072 validation failure and diagnosis

The first critic smoke at `4c21ae3` stops before discovery. Its directional
finite difference at epsilon step0.001 differs31.16% from autograd on Hopper.
Independent future rollout matches the preserved adapter exactly, and repeated
candidates with shared futures remain exactly equal. Failure costs are9 Q
evaluations,448 actor forwards and7 Jacobian/oracle VJP calls. Peak memory was
not captured on the old exception path; subsequent failures now retain it.

A bounded diagnostic uses the same contexts and compares absolute derivatives
across steps. The FP32 Q values have a large constant offset: Hopper log weights
are about2777.6, while the derivative is-0.18668; small perturbations differ by
only a few quantization increments and eventually produce identical values.
At step0.1, numeric/autograd differences are0.99% for Hopper and1.93% for
HalfCheetah. The diagnostic costs48 Q evaluations,62 actor forwards and20
Jacobian/oracle VJP calls, with46MiB peak reserved memory. Its full absolute
values are preserved in exp-072's `gradient-diagnostic.json`.

Exp-072-r2 changes the smoke's finite-difference step to0.1, retaining its4%
tolerance. It does not change the gradient oracle, finite-difference direction
pilot (radius0.25), estimator, budget or exp-073 conditions. The revised smoke passes as recorded below. This is a validation precision
failure, not evidence against complement pairing or a change to measured MSE.

## Repeat-stream correction before critic discovery

Exp-072-r2 completes with both gradient checks passing (0.99%/1.93%), exact
future-coupling checks and46MiB peak reserved memory. Its costs are15,208 Q
evaluations,94,680 actor forwards and20 Jacobian/oracle VJP calls. Its purpose
is execution validation; its small performance rows are not used for inference.

Main-agent review found a real statistical defect in common-random sampling.
Resetting a generator while changing an array's per-repeat sample count maps
some randomness from one method's repeat i to another method's repeat j. LHS
rank changes and future group sizes have the same issue. Individual methods
still have correct marginals and valid MSE point estimates, but paired losses
are dependent across repeat indices. Thus the nominal independent-repeat
standard errors and gate in exp-071 are invalid. Exp-073 was held before launch.

The correction draws fixed total-budget blocks for every repeat, including
normal probes, LHS permutations/jitter, posterior component uniforms and all
future Gaussian noises, then slices to each method's main sample count/rank.
This retains deliberate within-repeat common randomness and prevents cross-
repeat overlap. No integrand, direction method, reward budget or target changes.
Exp-074 tests six exact stream-alignment relations on multiple repeats. It
does not repeat settled posterior or gradient checks. After it passes,
exp-071-r2 reruns the same locked toy conditions; do not pool it with the old
run or treat it as independent generalization. Its corrected paired inference
must pass the original gate before exp-073 is launched.

## Exp-074 and exp-071-r2: corrected toy inference

At `0400f4e`, all six stream checks pass. The largest prefix/LHS discrepancies
are6.66e-16/8.88e-16; posterior draws and all future-noise relations agree
exactly. This detects the specific cross-repeat overlap that invalidated the
old intervals. It costs1920 toy reward evaluations and55296 synthetic normal
draws, with2MiB peak reserved GPU memory.

The corrected toy run completes all320 conditions in26.26seconds with zero
GPU allocation. Costs again total5,963,776 reward evaluations including262,144
pilot calls,2,851,104 analytic component integrals and384 reference evaluations.
Do not pool the rerun with exp-071: it replaces the paired-inference decision
under the same scientific conditions. Complete estimates and paired losses
remain available in the run directory.

| Toy | Target | Antithetic MSE | Oracle paired LHS | Oracle four-point LHS |
|---|---|---:|---:|---:|
| Gaussian | Candidate | 0.341770 | 0.298785 | 0.016956 |
| Mixture | Candidate | 0.612387 | 0.530842 | 0.023777 |
| Gaussian | Analytic continuation | 0.085773 | 0.073065 | 0.006590 |
| Mixture | Analytic continuation | 0.317784 | 0.266168 | 0.023800 |

Four-point LHS reduces candidate MSE95.04%/96.12% against antithetic and
94.32%/95.52% against oracle paired LHS. Analytic-continuation gains are
92.32%/92.51% and90.98%/91.06%, respectively. All eight paired Monte Carlo
intervals exclude zero in the favorable direction. These intervals condition
on the fixed analytic models; they do not measure model generalization.
All four deterministic conditions pass the original mechanism gate.

A30-degree direction error retains candidate gains47.52%/46.67% against
antithetic. The paid eight-query pilot still fails: candidate MSE is
0.583525/1.032286 versus0.341770/0.612387, with mean squared direction capture
only0.344/0.335. Independent future noise reduces oracle four-point gains
to14.04%/32.79%; sharing future randomness within each sign group yields
64.59%/73.27%. The complement cancellation mechanism is established in the
ideal-direction toy; acquisition and noisy continuation remain practical limits.

Decision: advance to the already locked exp-073 real-critic discovery after
the successful exp-072-r2 execution smoke. Use the corrected run's
[analysis](../../results/complement_pairing/exp-071-r2-complement-toy/analysis.json)
for this decision. The original exp-071 nominal gate remains superseded.

## Exp-073: real-critic discovery and closure

At `0400f4e`, all 24 contexts and 2,880 conditions complete on Orion. Each task
has four fresh observations, with indices3/5/7 inside its actual guidance
interval. Two MC seeds and256 repeats per condition measure fixed-context
sampling uncertainty, not additional independent observations. Primary results
use M32 and shared-future continuation. Every direction query reduces the
remaining main budget, all four weights and the full kernel score are retained,
and the table measures unit-gamma transition-mean MSE after the actual step gain.

| Task | Method | Displacement MSE | Gain vs antithetic | Gain vs random paired | Observations improved vs both |
|---|---|---:|---:|---:|---:|
| Hopper | Antithetic | 0.787592 | — | — | — |
| Hopper | Random paired LHS | 0.796474 | -1.13% | — | — |
| Hopper | Random four-point LHS | 0.655479 | 16.77% | 17.70% | 2/4 |
| Hopper | Full finite-difference four-point LHS | 0.624388 | 20.72% | 21.61% | 2/4 |
| Hopper | Four-query pilot four-point LHS | 0.649478 | 17.54% | 18.46% | 2/4 |
| HalfCheetah | Antithetic | 1.535894 | — | — | — |
| HalfCheetah | Random paired LHS | 1.598206 | -4.06% | — | — |
| HalfCheetah | Random four-point LHS | 2.222753 | -44.72% | -39.08% | 0/4 |
| HalfCheetah | Full finite-difference four-point LHS | 3.080517 | -100.57% | -92.75% | 0/4 |
| HalfCheetah | Four-query pilot four-point LHS | 2.421809 | -57.68% | -51.53% | 0/4 |

Hopper's full finite-difference method has a favorable aggregate difference of
-0.163205 against antithetic, with conditional MC95% interval
[-0.199785,-0.126624]. Replacing the pooled reference with each independent
block keeps the aggregate difference negative, ranging from-0.194795 to-0.120737.
The positive aggregate is real under these diagnostics. Its observation-level
MSE changes are-0.428898,-0.273683,+0.032249,+0.017514 for rows40/44/48/52.
Thus it fails the predeclared three-of-four observation requirement. The other
two feasible Hopper methods improve the same two observations and also miss
the20% aggregate threshold. All three HalfCheetah methods worsen every
observation against both simple controls, with unfavorable aggregate paired
intervals and reference-block differences.

### Accurate candidate directions do help the candidate target

Full coordinate finite differences closely recover the differentiable candidate
log-weight gradient: mean squared directional alignment is0.99722 in Hopper and
0.99979 in HalfCheetah. After paying6/12 direction queries, their four-point
candidate displacement MSE is0.094872/0.311008, compared with antithetic
0.142073/0.622378 and random paired LHS0.122657/0.602020. This is a33.22%/50.03%
gain against antithetic and22.65%/48.34% against random paired LHS. The gradient
oracle, which pays one reward query plus a backward pass, improves candidate
MSE38.61%/67.91%. These favorable results are retained despite the primary
continuation gate failure.

Right singular directions align better with the candidate gradient than the
largest symmetric-response eigenvector: squared alignment0.715/0.452 versus
0.562/0.279 for Hopper/HalfCheetah. Their four-point candidate gain against
antithetic is15.33%/25.95%, versus16.01%/-1.45% for symmetric directions.
Exact network sensitivity can explain some candidate behavior, but even an
almost exact candidate reward direction does not ensure useful continuation
pairing. In HalfCheetah, the candidate-gradient oracle's shared-continuation
MSE is2.418451,57.46% above antithetic. These data do not establish what an
accurate continuation-specific direction would achieve.

Independent future noise removes the Hopper benefit: full finite-difference
four-point MSE is1.531923 versus antithetic1.443404. It also worsens
HalfCheetah,4.706808 versus3.186842. Sharing future noises within ordinary
antithetic pairs already lowers the simple control to0.787592/1.535894.
The full independence/shared comparisons are retained, with the same marginal
continuation target. Four-point grouping spends fewer independent active draws
and future paths per reward budget; the observed tradeoff depends on the target.

### Cost and reference scope

The full finite-difference Hopper method spends6 direction queries and24 main
queries at nominal M32, leaving2 slots unused to preserve complete groups.
HalfCheetah spends12+20. The four-query pilot spends4+28, and random methods
spend0+32. For shared continuation, average actor forwards per call are223/193
for full finite differences,257 for the pilot and289 for either simple control.
These counts include the common transition forward. Every deployed estimator
also needs one common kernel VJP; the black-box directions add none. Symmetric
and SVD directions add a response forward and3/6 VJPs. The differentiable reward
oracle adds one reward/actor query and one joint backward pass. Measured batched
research times do not establish a standalone speedup.

All24 contexts retain eight independent8192-point reference blocks. Candidate
weight ESS ranges from11,402 to65,396. Hopper row40's continuation reference
is concentrated, with ESS9.77/11.02/6.06 despite65,536 samples. Row44 ranges
from108.39 to1511.55; all remaining Hopper references exceed15,474, and all
HalfCheetah references exceed10,407. No context is excluded. The favorable
Hopper difference survives the locked block replacements, but this does not
certify small self-normalized reference bias. Conditional MC intervals hold
the pooled reference fixed; block replacements are a sensitivity diagnostic.

Actual discovery cost is18,997,488 Q evaluations,114,549,088 actor forwards and
240 explicit Jacobian/oracle VJP calls. It takes1284.18seconds under shared-GPU
contention, with66.65/90MiB peak allocated/reserved memory. Costs include
reference simulations, diagnostic direction acquisition and all paid pilot
queries. Raw estimates, complete matrices, block moments and costs are retained;
the [locked analysis](../../results/complement_pairing/exp-073-complement-critic-discovery/analysis.json)
contains all budgets, integrands, controls and gate decisions.

Decision: zero of six feasible comparisons advances. No independent confirmation,
image run, endpoint evaluation or simulator return is launched. The budgeted
study establishes exact-direction complement cancellation and real candidate
benefits, while failing to establish a sufficiently broad, cost-accounted
continuation gain. It does not rule out other continuation directions or budgets.
All finite controllers have exited and no work remains queued.
