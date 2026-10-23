# Research state

## Current decision

The audit-directed reward-direction round is complete on
`codex/doit-reward-directions`. See the [findings](experiments/reward_directions/FINDINGS.md)
and [locked protocol](experiments/reward_directions/PLAN.md).
All 17 formal jobs completed successfully; the new construction, adapter and
batch checks pass (31 + 10 + 26 checks). All finite controllers exited.

A single development ablation removes the split-agreement gate while retaining
ESS/zero-score fallback. Paid moments-four then reduces Gaussian shift/narrow
proxy MSE by 31.22%/34.70%, but real-task four-point gains do not transfer.
Fresh confirmation uses 32 observations per task from disjoint episodes.
HalfCheetah P64 random pairs confirms 4.984% lower displacement MSE; Hopper
F32 random four-point confirms 11.415%/11.611% against antithetic/random pairs.
The Hopper benefit is heterogeneous. HalfCheetah paid moments-pair has a
favorable 3.041% mean against random pairs but an interval including zero,
so it does not pass the locked confirmation rule. No further tuning follows.

Complete-action tests generate 20,480 actions using the preserved gamma and
best-of-8/best-of-4 policies. Neither confirmed construction establishes an improvement in the primary
selected mean Q. Hopper four-point improves raw mean Q before selection;
that secondary gain is retained. No speedup is established. The local MSE
results remain valid, but no final selected-action improvement or simulator
return gain is established. The conditional simulator extension does not proceed.

All actual pilot, reference, future, actor and VJP costs are recorded. Compact
artifacts and final action bundles are local; full raw probe trees remain at
the recorded Orion paths. The execution checkout remains at `41dccad`; later
commits record analysis. No worker or pending queue remains. These conclusions
apply to this round's fixed targets, M32/M64 budgets and frozen tasks.
Historical closures below retain their own scope.

The continuation-aware follow-up is complete at its stated budgets. Exp-075
passes 19 checks; exp-076 closes [paid per-future directions](experiments/future_directions/FINDINGS.md)
at its locked toy gate. The path-gradient oracle improves rotating-target M64
MSE by 54.98% in dimension6 but only 12.38% in dimension3, and black-box
acquisition costs eliminate practical gains. No real-critic extension follows.

The distinct [free future-noise coupling study](experiments/joint_future/FINDINGS.md)
passes 21 construction checks in exp-077. Exp-078 joint antithetic improves the
rotating dimension3 M32 toy by 25.68%/24.42% against shared antithetic/random
paired LHS, while the aligned control worsens. The predeclared simpler-pair
rule selects only joint antithetic and only Hopper. Exp-079 then passes all
new real-critic path and cost checks at `6e9db3c`.

Exp-080 completes 192 conditions across 12 fresh Hopper contexts. Primary M32
joint-antithetic displacement MSE is 3.126653 versus 2.393425/2.436307 for the
two controls: increases of 30.64%/28.34%. All four observations worsen, including
both observations with high reference ESS; conditional paired intervals and all
eight reference-block replacements favor the controls. M64 also worsens all
four observations. Concentrated references are retained and still limit claims
about exact error. Close this setting without confirmation or image expansion.
The toy mechanism is correct but does not transfer to these real contexts.

All raw vectors, matrices, reference blocks, costs and job events are recorded.
Every finite queue has exited and no new job is queued. Terminal N4 remains the
best confirmed practical image baseline. This closure concerns the tested paid
directions and joint sign coupling under named budgets; it does not rule out
other continuation-aware methods. The original exp-073 closure remains in force.

The complement-cancellation follow-up is complete on the geometry branch. See
[the protocol](experiments/complement_pairing/PLAN.md) and
[findings](experiments/complement_pairing/FINDINGS.md). Exp-070 reproduces the
review's diagnosis: spectral strata reduce active-direction error94--95%, while
over99% of the remaining error is in the complement. The old total-error gate
still fails. All16 construction/replay checks and six repeat-stream checks pass.

Exp-071-r2 establishes the exact-direction toy mechanism after repairing the
cross-repeat random overlap that invalidated exp-071's nominal paired intervals.
At M32, four-point LHS reduces candidate MSE95.04%/96.12% and analytic-continuation
MSE92.32%/92.51% against antithetic. Paired differences also favor it against
oracle paired LHS. The paid eight-query pilot fails; independent future noise
weakens the benefit. The original invalid inference and all rerun costs remain
recorded. Exp-072's FP32 derivative-check failure is diagnosed and the revised
smoke passes without changing the method.

Exp-073 completes24 real-critic contexts and2880 conditions at `0400f4e`, all
inside active guidance intervals and with acquisition costs charged. Full finite
differences almost exactly recover the candidate reward-gradient direction,
and their four-point candidate MSE improves33.22%/50.03% against antithetic in
Hopper/HalfCheetah. Primary shared-continuation results fail the locked gate:
Hopper's strongest paid method improves20.72% against antithetic and21.61%
against random paired LHS, but only2/4 observations improve. All three feasible
HalfCheetah methods worsen all four observations. Favorable Hopper aggregate
intervals and reference-block differences are retained; concentrated references
still limit bias claims. No setting proceeds to confirmation or images.

The study closes at its tested budgets. The correct complement-cancellation
mechanism and real candidate gains do not establish a broad continuation or
endpoint gain. Terminal N4 remains the practical image baseline. Raw per-repeat
vectors, matrices, reference blocks, failures and job events are recorded; all
finite queues have exited and no further run is queued.

The review also confirms a deployment mismatch in the older Hopper proxy
study: only index5 of its5/9/12 diagnostics is actively guided. At that index
isotropic continuation MSE is0.69553 versus rank-one5.69486. Full local Jacobians
improve reconstruction eightfold but worsen aggregate guidance MSE. The old
rank-one and image-sketch settings remain closed.

The posterior-geometry exploration on `codex/doit-posterior-geometry` is
complete at its stated budgets. Exp-063--069 implement and verify Gaussian
spectral stratification and low-rank denoiser-response proxies on analytic
toys, fixed SD1.5 states and authentic Hopper/HalfCheetah actors and critics.
See the [protocol](experiments/posterior_geometry/PLAN.md) and
[complete findings](experiments/posterior_geometry/FINDINGS.md).

All 40 analytic checks pass. Antithetic pairing supplies the dominant toy
variance reduction; top-two spectral strata never meet the 20% gain over both
antithetic and matched random controls in the 32 toy condition averages.
The four image contexts also fail that gate. Isotropic contraction explains
most proxy reconstruction improvement; rank-eight spectral information adds
little, costs 33 forwards and 16 VJPs per state, and poorly resolves the early
image operator. No image guidance-bias ranking is inferred from noise-dominated
reference vectors. All 128 saved discovery candidates were visually reviewed.

The rank-one Hopper proxy discovery signal fails fresh-observation confirmation:
M32 continuation error is 5.39962 versus isotropic 5.38995, with improvement
on only two of four new observations. HalfCheetah has a favorable aggregate
concentrated in a reference-limited observation and does not justify changing
the selected task after confirmation. Learned Jacobians are often strongly
nonsymmetric, limiting the posterior-covariance interpretation.

Close these tested settings without a new endpoint or simulator claim. The
open question is whether cheap geometry predicts reward-relevant variation
beyond antithetic sampling and bulk contraction. Terminal N4 from exp-058
remains the accepted practical image baseline. Every formal outcome and job
transition is recorded. All finite queues have exited; no new job is queued.
Long execution was handled by a low-cost agent with terminal/anomaly reports.

## Completed conditional-reward development

The user reopened method development on 2026-10-22. The new candidate calibrates
conditional terminal-reward residuals, using a single proxy query at sparse
checkpoints. Mean-only, Gaussian-moment, empirical-residual and direct
learned-value variants separate uncertainty from mean-bias correction and
existing value regression. Exp-059 first tests this mechanism on the analytic
targets with every calibration cost recorded. See
[the continuation protocol](experiments/uncertainty_smc/CONTINUATION.md).
Exp-059 completed all 76 conditions at `6f96363`. Empirical reward UC improves
held-out value prediction but fails to improve sparse-SMC endpoints over the
mean-only control. Gaussian reward closure is substantially worse on the
single-mode target. Even exact values do not ensure better finite-population
endpoints with this schedule. Exp-060 passed the real-critic execution smoke.
Exp-061 found no useful transfer from the cross-observation conditional reward
bank: on resolved HalfCheetah row18, empirical/point/future2 centered log-value
MSE is 5.0400/4.8749/0.2169. Hopper's continuation reference ESS is near one,
preventing fine ranking there. Close this bank at the tested budget. Exp-062
completed 144 further toy conditions and found that second-moment allocation
does not meet the required gain. More frequent resampling helps first-moment
values on the Gaussian toy, but direct value regression matches empirical UC.
The development pass is complete with both candidates closed at their tested
budgets; its gates do not justify image or simulator expansion. No job remains
queued. Reliable local mean/uncertainty prediction or proposal improvement is
still an open research question, not a scheduled continuation.
The completed exp-058 result below remains the accepted practical baseline;
this continuation has not established a new sampling improvement.

The user opened a new uncertainty/SMC/FORS study on 2026-10-21, with toy,
image, and RL experiments authorized. The active branch is now
`codex/doit-uncertainty-smc`; its protocol is
[experiments/uncertainty_smc/PLAN.md](experiments/uncertainty_smc/PLAN.md).
The completed study establishes a bounded image improvement from terminal
importance resampling N4/tau0.3, an existing baseline. The tested posterior
uncertainty, FORS and intermediate-particle combinations do not establish a
cost-controlled improvement. The older studies below retain their conclusions
at their tested budgets.

The first uncertainty waves are complete. Exp-047-r2 passed the analytic
identities; the first check failed from insufficient numerical quadrature, and
independent adaptive integration verified the formula before correction.
Exp-048/050 completed 260 toy conditions with 256 repeats each. Gaussian
uncertainty and corrected grouped candidates improve some fixed-particle
comparisons, but larger bootstrap populations win at matched toy transition
counts. Short branches do not repay their added cost here. Exp-049's FORS B=2
still has substantial clipping bias at about 55 guidance calls per output;
direct residual rejection is accurate at about two residual evaluations.
Do not scale FORS or claim a cost-controlled toy success from these results.

Exp-051 verified batched real actor transitions within 1.2e-7 and calibrated
actual finite-sampler residuals on independent real observation rows. Both
Hopper and HalfCheetah have early residual RMS about 1.36, including substantial
mean bias. Exp-052/053 therefore compare uncertainty against both point and
calibrated-mean controls, as well as larger bootstrap populations. The primary
metric is fixed-target projected CDF error with independent reference
uncertainty reported. This is an offline critic-distribution study; simulator
returns remain untested. The image evidence is recorded below.

Exp-055 completed 128 real-critic conditions at the terminal-calibrated
temperatures. On the three states with reliable references, larger bootstrap
populations beat every tested uncertainty/SMC combination at lower or comparable
actor cost and far fewer Q calls. Hopper row16 has reference ESS 11.21 out of
32,768 and split CDF MSE 0.01284, so its fine ranking is unresolved. Close these
configurations at this budget; neither uncertainty nor grouped proposals has
established a useful real-critic improvement. Complete costs are 10,506,240
actor and 6,881,280 Q evaluations, including references.

The next bounded experiment moved to images, where decoder/reward cost changes
the tradeoff. Exp-056 passed exact N1 equivalence and all cost checks (50 reward queries,
436 actor evaluations), with 5730 MiB peak reserved memory. Exp-057 locks seven
settings across eight prompts and two fresh seeds, 112 delivered images. Compare terminal importance resampling,
delayed point-potential SMC, and two-full-future uncertainty SMC against the
established MC8 and selected MC32 image settings. Intermediate resampling is
restricted to steps 10/14/18 to avoid the early proxy collapse observed in the
critic study. Terminal resampling is an established baseline, not a novelty
claim. Image reward gains require fresh seeds, alignment/diversity measures,
measured cost and visual quality review.

Exp-057 completed 112 images. Only terminal importance resampling N4 passes
the locked discovery criteria: reward 6.18440 versus MC8 6.03094, alignment
0.281381 versus 0.283229, and mean time 9.21s versus 27.63s. All 112 outputs
were reviewed; N4 has no distinct new quality failure and its measured
within-prompt embedding distance is not reduced. N8 and both sparse-SMC
combinations gain reward but exceed the 0.005 alignment-loss gate; future2
also exceeds MC8 time. Close those settings, preserving the positive reward
observations. Exp-058 confirmed only N4/tau0.3 against base, MC8 and selected
MC32 on eight new seeds (256 images).

Exp-058 completed all 32 jobs at `2c1416f`: 25,024 reward queries, 8,960 score
forwards and 1,152 VJPs. N4 meets the locked aggregate criterion, with reward
6.33180 versus MC8 6.15570, delta +0.17610 and 95% seed-block interval
[0.00911, 0.30592]. Mean alignment improves by 0.002491; embedding distance
increases 1.42%. N4 costs 4 reward queries, 80 score forwards and zero VJPs per
image, versus MC8's 97/20/9, with measured time 8.89 versus 31.30 seconds.
Shared-GPU timing is descriptive. N4's reward difference from selected MC32 is
unresolved: +0.00552, interval [-0.16793, 0.16153].

All 256 images were reviewed. N4 has no distinct new visual failure mode
relative to the controls, but inherits base-model defects, including a missing
lighthouse at seed59051 and malformed aircraft geometry. Labrador mean
alignment drops 0.01423 despite the favorable overall mean. All failures remain
included. This is a confirmed practical improvement on the eight fixed prompts
from a known importance-resampling baseline; novelty, new-prompt generalization
and simulator returns are unestablished. The goal is complete within that scope.
See [the findings](experiments/uncertainty_smc/FINDINGS.md) and
`results/uncertainty_smc/exp-058-analysis/summary.json`. The finite controller has
exited, all artifacts were retrieved, and this study has no further queued run.

Orion connectivity has recovered. Read-only inspection found no active project
queue or run, and the old checkout remains clean at `d6728de`. Exp-039 actually
failed with a nonfinite Gaussian moment ratio after 31,744 saved rows; its total
failed-run costs were not persisted. Five exp-041 probes completed, including
lighthouse seed211 step11; three other lighthouse probes were not observed
submitted. The recovered records and ledger corrections are under
`results/recovery/2026-10-21`. The new study uses a separate remote checkout.

### Previous exploration

The user resumed work with a separate Nebula exploration while Orion remains
unreachable. Branch `codex/doit-guidance-reliability` tests fixed-vector
sensitivity using [docs/guidance-reliability.md](docs/guidance-reliability.md);
that bounded exploration is now complete.
Orion monitoring is active again. Its existing exp-039/041 queue must be
recovered before any duplicate submission or continuation. The online-update
branch remains `codex/doit-correction`; its results and decisions below stand.

Nebula exp-044 confirms the proposed mechanism only on its known-reference
model: in dimension 512, a value RMS error of 0.01415 produces raw adjoint MSE
0.3180; half shrinkage gives 0.09245 and four-point smoothing 0.08242. With no
corruption the raw derivative is exact, while shrinkage introduces MSE 0.01286.
The numerical adjoint checks pass. This does not establish sensitivity error
in SD. The exp-045 single-state FP32 smoke passes its expected query, forward
and VJP counts and has a raw-direction finite-difference error below 1.5e-7.
The locked eight-state discovery probe then completed at `d85d1da`. All raw
directional finite differences agree within 3e-5 at their best step size. The
independent vectors are nearly orthogonal before the VJP, and smoothing does
not materially improve agreement. It costs about 5.1--5.5 s per operation versus
0.77--0.87 s for raw. Half shrinkage raises KL by about 35% early and 60% late
because it weakens linear/learned cancellation. Exp-046 therefore tests only a
raw-norm-matched half shrinkage and the larger predeclared K4 smoothing scale,
alongside MC8, a comparable-cost MC16 and an MC32 gamma-4 control at tau 0.3. Its
five groups, eight prompts and two fresh seeds total 80 FP32 images. No ideal
Jacobian error or endpoint improvement has been established by these diagnostics.
The earlier selected MC32 baseline used tau 0.6. The executed exp-046 configs
use tau 0.3 consistently across all five groups; their description of MC32 as
the previously selected setting was inaccurate. Treat it as a higher-budget
control at the common temperature, not a rerun of the selected baseline.

Exp-046 is now complete: all ten jobs at `6d1ceca` exited zero and produced
80 images with correct per-image cost counters and no denominator events.
Mean reward is 5.91513 for MC8, 5.93714 for MC16, 5.91560 for norm-matched
shrinkage and 5.90488 for K4 smoothing. Shrinkage's paired delta is +0.00046,
with opposite signs by seed; smoothing's is -0.01026, negative on both seeds.
Their alignment deltas are -0.00153 and -0.00109. Full visual review finds no
distinct new quality failure in either sensitivity variant, but no clear
improvement. Close both settings for lack of useful endpoint benefit at this
budget. No confirmation is launched and the reserved seeds remain unused.

The tau-0.3/gamma-4 MC32 control gains 0.09673 mean reward but has severe cafe
and Labrador saturation/structure failures at seed 541. Close that specific
control for quality; it is not evidence against the historical tau-0.6 baseline.
The complete wave uses 11,856 reward queries including final evaluations,
2,176 score forwards and 1,152 VJPs. Shared-device image times average
81.46/98.40/179.39/85.27/111.32 s for MC8/MC16/strong MC32/shrinkage/smoothing;
large timing differences by seed preclude a dedicated-device speed claim.
The Nebula finite queue has exited. The original Orion work remains pending
reconnection and recovery of exp-039/041. The last bounded SSH probe still
timed out. Scheduled follow-up `recover-orion-and-resume-doit` is active every
15 minutes in this chat, replacing the completed low-cost monitoring session.
It stays quiet on unchanged connection failure and resumes the original work
after recovering existing jobs and results.

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

## Orion recovery checkpoint

The exp-039/041 nine-job queue was launched on Orion at `d6728de`. The last
retrieved critic log completed all Hopper discovery paths (16,384 rows), with
no estimator error. All four cafe common-candidate probes completed and passed
their expected four rows, 120 reward queries and 961/121 score-forward counts
for steps 11/18. Their small outputs are present locally. The remaining critic
and lighthouse results have not yet been retrieved.

The connection outage began on 2026-10-03. Recovery on 2026-10-21 established
that the controller and jobs had exited. Exp-039 failed during HalfCheetah
evaluation with a nonfinite Gaussian moment ratio; Hopper completed all 16,384
rows and HalfCheetah saved 15,360 rows. Partial counts cannot establish the
whole failed-run cost. Lighthouse seed211 step11 completed with 120 reward
queries and 961 score forwards; the three other lighthouse jobs were not
observed submitted. The old terminal files for these two jobs are stale, so the
local ledger has explicit reconciliation events based on run records and
absent processes. Exp-043 remains unlaunched. The old execution checkout is
preserved while the new study proceeds separately.

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
