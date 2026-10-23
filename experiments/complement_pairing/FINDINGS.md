# Complement pairing: current evidence

Exp-070 verifies the supplied direction decomposition, and exp-071 establishes
a strong exact-direction toy mechanism. The paid eight-query random-subspace
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

The M32 exact-direction four-point LHS passes all four locked deterministic
mechanism comparisons by more than20% against both antithetic and oracle
paired-LHS controls. Every paired difference interval is strictly negative.

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

Decision: advance the ideal-direction mechanism to the bounded real-critic
direction study. Preserve the failed paid pilot and noise sensitivity; neither
is hidden by the favorable oracle result. The
[complete analysis](../../results/complement_pairing/exp-071-complement-toy/analysis.json)
contains both budgets, every configured method and the exact gate decisions.
