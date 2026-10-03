# Research state

## Current decision

The user reopened research after an external audit of `c5e4311`. The active
protocol is [docs/audit-followup.md](docs/audit-followup.md). Ordinary
high-dimensional CV/SPIDER/proposal grid expansion remains closed. The new
questions concern low-dimensional real-critic learnability, actual image
execution displacement, and categorical use of full-future candidate weights.
Exp-033/034/035 all completed at `25936fc`. With 96 queries, analytic online CV
has MSE/MC 0.00450 in dimension 6 but 0.99843 in dimension 16384; an externally
supplied exact surrogate works in both. The high-dimensional image references
remain too noisy to resolve direction: zero and quarter-shrinkage controls
outperform the larger estimates in reference MSE. These findings support the
audit's learnability and shrinkage concerns without changing the old endpoint
results.

Recovered KL shows future cafe failures with early displacement spikes, but a
failed EMA lighthouse has lower maximum KL than clean comparators. Magnitude
is not a sufficient failure explanation. Exp-036/037 passed the new execution
checks and two counter-checked image smokes. Exp-040 completed all 240 discovery
images: the common cap improves paired-future mean reward, but clear cafe,
lighthouse and dog-texture failures remain in the strong guidance families.
That cap repair is closed. Categorical resampling produces cleaner images with
small unresolved reward gains and lower alignment. Exp-043 locks both
categorical couplings, moderate MC8 with/without the cap, MC32 and base on eight
new reserved seeds (384 images), with no further parameter selection.

Hopper and HalfCheetah actor/critic pairs load without Gym/D4RL. Official medium-v2
datasets are downloaded for true observation states; a committed split separates
calibration, discovery, and confirmation by episode. Exp-038 completed 9248 Q
queries and passed both tasks' source-adapter identities with errors below 1e-6.
Exp-039 fixes temperatures at calibration median local Q standard deviations:
5.14282 for Hopper and 2.12394 for HalfCheetah. It next compares online versus
independently assisted surrogates on moving action paths with fixed observations.
Exp-041 tests the same realized future candidates at eight saved image states.
No simulator evaluation has run. The exp-040 and calibration queues have exited.

## Active computation checkpoint

The exp-039/041 nine-job queue was launched on Orion at `d6728de`. The last
retrieved critic log completed all Hopper discovery paths (16,384 rows), with
no estimator error. All four cafe common-candidate probes completed and passed
their expected four rows, 120 reward queries and 961/121 score-forward counts
for steps 11/18. Their small outputs are present locally. The remaining critic
and lighthouse results have not yet been retrieved.

As of 2026-10-03 18:30 UTC, bounded SSH reads time out, including a fresh
connection with multiplexing disabled. This prevents a new execution-status
observation; it is not a recorded experiment failure. Do not mark the remaining
runs completed or start duplicates. Exp-043 is committed and authorized after
the nine-job queue completes its counter checks, but its launch is not yet
confirmed. Keep the remote execution checkout at `d6728de` while that queue is
active, even if documentation and retrieved results advance locally or online.

## Initial study outcome

The initial study is complete. All supplied estimator families and targeted extensions
have been implemented, tuned, evaluated, and closed under the tested image
budgets. No new method established a usable image improvement. Strong guidance
EMA and paired future M8 have positive aggregate results but observed quality
failures; moderate EMA and independent future M4 remain unresolved at the final
budget. These decisions do not assert universal failure of the methods.

Exp-030 completed 160 discovery images. Exp-031 completed 448 confirmation
images at commit 19c1bf7, and exp-032 completed 64 moderate-EMA confirmation
images at commit 9941276. The combined final comparison contains 512 images
across eight groups, eight prompts, and reserved seeds
101/113/127/139/151/163/179/191. All 64 final jobs completed with no denominator
events or runtime failures; original images, embeddings, and terminal records
are retained. The first guided exp-031 outputs validated the new timing fields.

No project queue or experiment process remains on Orion. No further experiment
is scheduled. The required image-success condition for RL is unmet, so no RL
environment was installed and no RL evaluation was run. The stable outcome
report is [docs/findings.md](docs/findings.md); the sections below preserve how
the decisions were reached.

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
The stronger mechanism result did not establish a usable endpoint improvement
in the final comparisons below.

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

The independent-future gain over MC8 was mixed across seeds and mainly driven
by one Formula1 image, motivating its broader comparison. The paired M8 method
had comparable screen reward to MC8 but many more score evaluations; it was
retained as a direct comparator with actual runtime measured under shared-GPU
contention.

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
partial confirmation means. Its eight-run manifest remained separate from
exp-031 and ran after that queue exited. Its wall-clock comparison therefore
acknowledges the different execution wave. No further EMA setting followed it.

## Final reserved image evidence

Exp-031 completes 448 images, 64 per group, with no denominator events or
runtime failures. Seed-block intervals below condition on the fixed eight
prompts. The records are `artifacts/exp-031-summary.json` and
`artifacts/exp-031-vs-mc32.json`. Exp-032 completes another 64 images with no
denominator events or runtime failures. All eight groups are summarized in
`artifacts/exp-032-combined-summary.json`.

Moderate MC8 has reward 6.1100, alignment 0.27484, embedding distance 0.12390,
and mean time 25.31 s. MC32 has reward 6.3389, alignment 0.27755, distance
0.12859, and time 64.05 s. Strong MC8 is worse on reward (6.0196) despite higher
CLIP alignment (0.28300); its visual instability is consistent with discovery.

High-reward EMA improves over moderate MC8 by 0.1433 reward (95% interval
[0.0664, 0.2313]) at 24.84 s/image, with improved average alignment and no
embedding-distance loss. This positive metric result is real, but it does not
establish a usable image method: lighthouse seed 113 reproduces the severe
texture/block artifact observed at discovery seed 53. Other reserved lighthouse
images also show aggressive texture and banding. Close this strong setting for
visible quality failure; do not describe its reward effect as negative.

Independent future M4 has reward 6.1782, a delta of 0.0682 over moderate MC8
with interval [-0.0164, 0.1476]. It uses 40 rather than 96 reward queries, but
236 rather than 20 score evaluations and 26.38 rather than 25.31 seconds.
Its mean gain remains unresolved at the final budget. This lower-query tradeoff
does not validate an end-to-end improvement; close it under this budget.

Paired future M8 has reward 6.3361 versus MC32's 6.3389, with paired delta
-0.0028 and interval [-0.0558, 0.0487]. Alignment and embedding distance are
also close (0.27705 and 0.12863). Reward queries drop from 288 to 80, score
evaluations rise from 20 to 452, and measured time drops from 64.05 to 50.01 s.
The time components explain this tradeoff: reward evaluation takes 21.05 s
instead of 61.18 s, while future score evaluation adds about 26.16 s to oracle
time. These shared-GPU measurements are not dedicated-device benchmarks.

However, the aggregate conceals a material prompt failure. Cafe reward is
0.6898 below MC32, with deltas -1.6294 and -1.4724 at seeds 127 and 139. Raw
images at both seeds have severe saturation and structural corruption relative
to their MC32 comparators. Close this strong future setting for quality failure
despite its useful average cost result. The reserved cafe/lighthouse images and
EMA comparison sheets are retained as direct evidence. No strong-setting result
passes the image gate.

Moderate EMA finishes with reward 6.1322 versus moderate MC8's 6.1100, a paired
delta of 0.0222 with interval [-0.0094, 0.0486]. Alignment is 0.27434, a delta of
-0.00050 with interval [-0.00209, 0.00114]. Embedding distance is 0.13108, with
no evident diversity loss. Its reward gain remains unresolved, so close this
last setting under the final budget. It uses the same 96 reward queries and
20 score evaluations as MC8; its 22.56 s/image was measured in a different wave
and does not establish a speedup over MC8's 25.31 s/image.

All image candidates now have closure decisions. The conditional RL phase does
not proceed because none passes the image-success gate. The final report links
the raw quality counterexamples, quantitative comparisons, and limitations.
