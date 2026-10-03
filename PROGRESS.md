# Research state

## Current decision

The goal remains active. All supplied estimator families have been implemented,
tuned, and evaluated. Ordinary affine/exponential CV, projected RLS, spatial CV,
pair-contrast CV, SPIDER, and adaptive importance sampling are closed under the
tested image budgets. No image method is validated as a practical improvement.
Gradient EMA and direct full-future sampling remain for a broader endpoint
comparison. RL has not started; its required image-success condition is unmet.

The next wave, `exp-030`, contains ten locked groups on all eight prompts with
fresh discovery seeds41/53 (160 images). It covers tuned MC8/MC32, three gradient
EMA settings, and the best independent/paired full-future settings. Reserved
final seeds101/113/127/139/151/163/179/191 remain unused. Promising or marginal
candidates receive reserved-seed confirmation; clearly dominated ones can close.

## Validated implementation and baseline

The image engine freezes SD1.5 and the LAION aesthetic head, uses20 Euler steps
with9 positive-noise guided steps, and shares cached reward preprocessing across
methods. The16 focused checks in exp000/006/014/025 passed on Orion: Gaussian
moments, CV/IS identities, query/update ordering, sampler/VJP algebra, spatial
moments, coupled continuation noises, two-level moments/costs, and pair fitting.

The source image code already uses a fixed temperature, without batch standard
deviation normalization. Its Euler branch uses a VP endpoint formula and
correction scale. The study retains that as `source_vp` and uses a consistent
Euler endpoint as its primary convention; any convention change is not credited
to CV. The old reward callback also reloads CLIP, whereas all comparisons here
share one cached implementation. Source gamma is0.4; early gamma4/M8 smoke images
were visibly damaged, which prompted explicit baseline strength/temperature
calibration rather than treating a weak baseline as evidence for a new method.

FP16 VJP differences as small as0.00014 in norm can change later guided paths.
The first Bernese capture and same-seed calibration had identical first weights
and ratios but diverged later. Paired seeds align random draws, not bitwise
trajectories. Small endpoint effects need fresh-seed confirmation. Capture
timings include reference queries and are never generation-runtime benchmarks.

## Surrogate CV closure

Exp005 evaluated81 configurations on two discovery paths. Ordinary affine LMS
has no material ratio benefit. Exponential fits can lower pointwise error by
about75%, but the best ratio effect was only0.17--0.26% in post-warmup absolute
error. Random projected RLS worsened error and some settings had denominator
events. A historical constant CV is exactly redundant with paired samples.

Exp007's28 low-frequency DCT variants did not improve the discovery ratio error:
best47.680 versus MC8's47.484. Exp015 tests M1/M4/M16 and tau0.3/0.4/0.6. At M4/M16,
full CV effects stay below0.4%; projected/spatial models usually worsen error.
Some M1 memory estimates improve over reward-independent raw M1, but errors
remain thousands versus roughly82--108 for M4. This does not make M1 useful.

Independent exp016 VJP errors are53.418 for MC8,53.436 for frozen affine,53.893
for exponential LMS,53.471 for affine projected RLS,59.392 for exponential
projected RLS and54.205 for spatial exponential RLS. No candidate improves both
independent paths consistently. Ordinary CV is closed.

The targeted exp026 extension fits plus/minus reward or log-weight contrasts,
then fits the intercept separately. Across24 settings, its best normalized
ratio error is47.4803 versus47.4838 for MC8 (about0.007% lower), and pair-residual
error does not improve. Close this extension without further expansion. These
results concern useful finite-query image estimators, not impossibility of CV.

## SPIDER and proposal closure

Every initial SPIDER setting had denominator events. Exp012's12 larger-refresh
settings still had1--16 events each, while cost-matched MC had none. In stored
banks, current-minus-previous pair-contrast second moments are0.534--2.225 times
current-only for Bernese and0.663--15.389 for Formula1. Paying two queries per
fresh difference does not repay its sample cost even before accumulated error.

Independent exp017 has no denominator events, but normalized errors
89.448/82.318/174.710 versus matched MC27.421/24.893/19.632 for the three schedules.
Post-warmup absolute error is worse on each prompt. Close SPIDER under these
budgets; its failure is not solely a consequence of a corrupted discovery path.

All38 mixture AIS/CV-AIS settings in exp013 completed. Small scale1 tilts changed
normalized error by only0.91% (IS) or1.33% (CV-IS); larger tilts worsened it.
Independent exp024 reverses the aggregate effect: IS47.041 to47.213 and
CV-IS47.105 to47.256, improving lighthouse slightly but worsening Formula1.
Exp018 pure tilts also offer no useful gain. Close AIS and CV-AIS here.

Pure IS and CV-IS coincide at unclipped scale1 with complete antithetic pairs:
`p*u/q = H_u` and the paired mean epsilon equals the tilt`rho*b`, so analytic and
sampled surrogate moments cancel. This observed equality is correct algebra.

## EMA and fidelity evidence

Gradient EMA beta0.25 reduces actual VJP error from46.943 to8.011 in discovery
and from53.418 to8.981 independently. Its post-warmup absolute error is11.5%
(Formula1) and13.2% (lighthouse) of MC8. MC32 independent normalized error is18.029.
This is a biased control. Zero guidance itself has normalized error1, so noisy
reference MSE alone favors shrinkage and cannot establish useful reward steering.
The stronger mechanism result has not produced a clear endpoint benefit yet.

Paired full-future probes in exp008 share later noise only within antithetic
pairs and average two branches. Proxy/future weight-contrast correlations are
-0.149/0.111 early and0.354/0.558 late for Bernese/Formula1. Early mean-weight
ratios are63.4/203.4. These small probes do not resolve exact high-dimensional
bias, but motivate the bounded direct-future and two-level tests.

The initial endpoint wave has39 complete four-image conditions and3 failed
conditions. The bounded extension adds12 completed conditions. Selected screen
observations below each use only two prompts and two seeds; they are not success
claims. Queries exclude the one final evaluation; score evaluations count CFG
calls by candidate, consistently across methods.

| Setting | Reward | Alignment | Reward queries/image | Score evaluations/image |
| --- | ---: | ---: | ---: | ---: |
| MC8, tau0.6, gamma4 | 5.6115 | 0.2777 | 96 | 20 |
| MC32, tau0.6, gamma8 | 5.8927 | 0.2812 | 288 | 20 |
| Gradient EMA0.25, tau1, gamma8 | 5.6579 | 0.2734 | 96 | 20 |
| Gradient EMA0.25, tau0.6, gamma8 | 5.5339 | 0.2842 | 96 | 20 |
| Independent future M4, tau0.3, gamma1 | 5.6519 | 0.2797 | 40 | 236 |
| Paired future M8, tau0.6, gamma4 | 5.6157 | 0.2781 | 80 | 452 |
| Two-level fixed1, tau0.6, gamma1 | 5.5766 | 0.2788 | 176 | 236 |

The independent-future gain over MC8 is mixed across seeds and mainly driven by
one Formula1 image. Carry it into the broader comparison. The paired M8 method
has comparable screen reward to MC8 but many more score evaluations; include it
as the direct comparator rather than declaring a cost benefit from reward
queries alone. Actual runtime is measured under shared-GPU contention.

Close the two-level branch under the tested coefficients, temperatures and
budgets: fixed coefficients do not improve the reward/cost comparison; three
historical-mass settings fail with nonfinite VJP guidance, another has a
negative-denominator event and poor reward, and the higher-temperature setting
completes without gaining performance. Failed conditions retain partial images
without reporting their partial means as complete results. Old failures omit
exact costs within the failing image; future failures now save engine counters
and include step/moment diagnostics in the error message. Do not claim an exact
total research-query count from the old partial failure artifacts.
