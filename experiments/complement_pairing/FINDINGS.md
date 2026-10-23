# Complement pairing: current evidence

Exp-070 verifies the supplied direction decomposition. Exp-071 has a strong
exact-direction MSE point-estimate signal, but its nominal paired intervals
are superseded by the repeat-stream correction below. Exp-071-r2 must establish
the advance gate before real-critic discovery. The paid eight-query random-subspace
pilot fails at the tested budgets. The next question is whether real,
low-dimensional models provide sufficiently accurate directions at low cost.
No real-model or endpoint success follows from the toy oracle.

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
Per-repeat estimates and paired differences are preserved. Intervals quantify
Monte Carlo uncertainty for the fixed toy, not population generalization.

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
pilot (radius0.25), estimator, budget or exp-073 conditions. Discovery remains
blocked until this corrected smoke completes. This is a validation precision
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
