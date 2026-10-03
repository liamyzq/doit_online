# Research state

## Current decision

The goal remains active. All supplied estimator families have been implemented,
tuned, and evaluated. Ordinary affine/exponential CV, projected RLS, spatial CV,
pair-contrast CV, SPIDER, and adaptive importance sampling are closed under the
tested image budgets. No image method is validated as a practical improvement.
Gradient EMA and direct full-future sampling remain for a broader endpoint
comparison. RL has not started; its required image-success condition is unmet.

Exp-030 completed all 160 images with no denominator events. Keep gradient EMA
beta 0.25 at tau 1 / gamma 8 and both direct-future settings for independent
confirmation. Their discovery gains do not establish success. Stronger guidance
at tau 0.6 failed on new prompts/seeds for both EMA and MC32. The moderate EMA
has only a small gain, but discovery image review finds it cleaner than the
high-reward EMA. Include it in a bounded confirmation follow-up, exp-032.

The active wave, `exp-031`, locks seven groups: unguided, both MC8 settings, selected
MC32, retained EMA, and independent/paired full-future sampling. Each gets eight
prompts and the reserved seeds 101/113/127/139/151/163/179/191, for 448 images.
The queue is active at commit 19c1bf7; its first guided outputs validate the new
timing fields. Exp-032 adds 64 images with the already tested moderate EMA
parameters after this finite queue exits, for 512 confirmation images in total.
No further image hyperparameter tuning follows these comparisons. Their
reward/cost, alignment and diversity evidence decides the RL gate; mixed final
results close under the evaluated budget.

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

## Broader endpoint results

Exp-030 evaluates eight prompts at seeds 41/53, 16 images per group. All 20 seed
shards completed successfully. The table reports group means; query counts omit
the single final evaluation, and seconds are per-image times under shared-GPU
contention. Two seeds do not support a precision claim.

| Locked setting | Reward | Alignment | Embedding distance | Queries | Score evaluations | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unguided | 6.1334 | 0.2654 | 0.1229 | 0 | 20 | 2.13 |
| MC8, tau 0.3, gamma 0.4 | 6.1945 | 0.2659 | 0.1220 | 96 | 20 | 24.16 |
| MC8, tau 0.6, gamma 4 | 6.1492 | 0.2763 | 0.1142 | 96 | 20 | 26.19 |
| MC32, tau 0.6, gamma 4 | 6.3942 | 0.2731 | 0.0890 | 288 | 20 | 60.85 |
| EMA 0.25, tau 1, gamma 8 | 6.3274 | 0.2754 | 0.0971 | 96 | 20 | 26.08 |
| Independent future M4, tau 0.3, gamma 1 | 6.3334 | 0.2688 | 0.1230 | 40 | 236 | 28.32 |
| Paired future M8, tau 0.6, gamma 4 | 6.4263 | 0.2694 | 0.1227 | 80 | 452 | 49.77 |

Both MC8 settings remain as comparators because the moderate setting has higher
reward and the strong setting higher alignment. MC32 gamma 8 averages only
6.0708 reward; select gamma 4 for confirmation. EMA at tau 0.6 / gamma 8 falls to
5.0380 reward. Moderate EMA averages 6.2228 reward and 0.2690 alignment, weaker
than the high-reward EMA on these two metrics. Close the tau 0.6 / gamma 8 setting.
The visual review below motivates keeping moderate EMA for exp-032.

The retained EMA gains 0.1329 reward over moderate MC8 at the same query budget,
but its two-seed embedding distance is lower. Independent future M4 gains
0.1389 with fewer reward queries and many more score evaluations. Paired future
M8 is slightly above MC32 on reward while using fewer reward queries and less
measured wall-clock, but alignment is lower. These are distinct tradeoffs that
require reserved-seed confirmation, not interchangeable success claims.

Grouped details are in `artifacts/exp-030-summary.json`, derived from the
original exp-030 grouping with `mc8-moderate` selected as the primary baseline.
The new per-image timing fields persist existing VJP/reward counters and sums
of step timers for exp-031. They do not recover missing historical VJP timings.

Visual review of all selected groups at discovery seed 53 finds a concrete EMA
failure: the lighthouse image has strong high-frequency texture and block-like
artifacts, although its aesthetic reward remains 6.348 versus unguided 6.360.
The paired-future M8 image is visibly cleaner at reward 6.92. MC8 strong also
damages the lighthouse and airplane images. Record this mismatch between reward
and visible quality when judging confirmation; an average reward gain alone
cannot override it. The raw lighthouse images and contact sheets are retained
under the exp-030 artifact directories.

Exp-032 is a bounded addition of the already tested moderate EMA configuration,
motivated by this discovery-image quality difference. It is selected after the
first ten exp-031 shards arrived; its parameters are not selected from those
partial confirmation means. Its new eight-run manifest remains separate from
the active exp-031 queue, and its wall-clock comparison will acknowledge the
different execution wave. No further EMA setting follows it.
