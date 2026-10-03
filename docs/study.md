# Study protocol

## Objective and scope

The user requests all correction families in the supplied proposal to be tried,
tuned, and extended where evidence supports a concrete improvement. Image
correctness comes first; successful image methods then scale to RL. There is
no promise of a positive scientific result. Every family receives a documented
success or closure decision after meaningful tuning and independent evaluation.

## Estimator target

At a fixed state let `Z = m + rho * epsilon`, with standard Gaussian epsilon,
and let `F(Z) = exp((reward(Z) - reward_offset) / tau)`. The offset and temperature
are fixed before drawing current samples. Estimate `H = E[F]`, `U = E[F epsilon]`,
then apply `J_mu^T (U / H) / sigma_transition`. The target is the infinite-sample
practical estimator; future-rollout bias is measured separately.

A control variate is frozen before current queries. Its analytic H and U are
added to fresh residual moments. Update its memory only after computing the
current estimate. Count denominator flooring, nonfinite values, and large
corrections as observations rather than silently replacing them with a baseline.

## Required method coverage

- Independent and antithetic fresh MC, including the original M=32 budget.
- A historical constant control variate, a moment-ratio EMA control, and an
  actual guidance EMA after the VJP. Ratio EMA is not labeled guidance EMA.
- Affine online control variates using normalized LMS and projected RLS.
- Exponential-affine control variates fitted to log weights.
- SPIDER-style coupled moment differences with periodic refresh. A difference
  costs two reward evaluations per noise sample.
- Exponential-surrogate proposals, with exact Gaussian or mixture `p/q`, both
  with and without the analytic control variate. Tune proposal scale, mixture
  mass, and a Gaussian KL radius; no unweighted shifted proposal is accepted.
- Reset-versus-persistent memory and frozen-versus-updated memory ablations.
- A bounded multifidelity/full-future correction if measured endpoint bias
  dominates. Full future score evaluations are included in cost accounting.

## Stages and decisions

### 1. Algebra and image correctness

Use known Gaussian affine/exponential moments to detect incorrect analytic
integrals or importance weights, exact-surrogate residual tests to detect wrong
update order, and comparison to the installed Euler scheduler to detect a
transition error. A failure is repaired before scientific runs. Smoke tests are
not evidence of reward improvement.

### 2. Frozen image states and tuning

Begin with the paper's Bernese prompt plus distinct object/scene prompts and
20 denoising steps. Capture states from an established baseline, keep fresh
queries separate from fitting history, and use independent large-sample
practical references. Compare ratio MSE, actual VJP guidance error, denominator
events, held-out residual error, and reward-query counts. Reference disagreement
sets the resolution of the comparison. Reuse raw reward banks for compatible
base-Gaussian methods; adaptive proposals require actual new oracle calls.

Initial tuning varies fresh query budgets 4/8/16, learning rates or ridge
regularization, memory coefficient, refresh interval, and proposal strength.
Warm-up and refresh count toward the total budget. Controls receive the same
query schedule. The initial stored-bank grid has 81 configurations at tau 0.3;
learned-method tuning starts at M=8, then retained variants are checked at M=4
and M=16 and other temperatures. M=1 is measured separately. Diagnostic
prediction error is distinct from the residual actually used after multiplying
the control-variate coefficient; proposal diagnostics include squared importance
weights in the residual second moment. Small M=1 is a separate ablation after warm-up, not a default.

Retain candidates only if an independent comparison supports lower ratio and
guidance error at the same query budget, or comparable error with fewer queries.
If regression error improves without guidance error improving, close that
configuration. Investigate at most a small number of targeted extensions per
observed failure, using new experiment IDs rather than an unrestricted grid.

### 3. Image rollouts

Use paired transition-noise seeds, separate estimator-noise streams, identical
backbone/reward preprocessing, and discovery/confirmation seeds split in advance.
Report aesthetic reward, CLIP text alignment, within-prompt embedding diversity,
query count, score forwards, VJP/update time, and total wall-clock. Include a
source-convention baseline separately from the consistent-Euler baseline.
Reserve fixed-state confirmation seeds 37 and 59, and final image confirmation
seeds 101, 113, 127, 139, 151, 163, 179, and 191. Image tuning may use seeds
11, 23, 41, and 53, never these reserved seeds.
Tune the fresh-MC baseline's gamma and temperature at each query budget on the
same discovery split used by learned methods. An endpoint gain over an
overguided M=8 baseline alone is insufficient; compare to the tuned baseline.
An image success requires an independent reward/cost improvement, supported by
the fixed-state mechanism, without an evident loss of diversity or alignment.
Marginal results receive a discriminating replication instead of being called
success. Clearly poor methods close under the evaluated budgets.

### 4. RL scaling and closure

Port retained estimators to the fixed-temperature D4RL action Gaussian, first
smoke the existing policy/critic checkpoints, then tune on discovery seeds and
confirm across locomotion tasks. Report normalized return, query cost, and
runtime against matched-budget DOIT. Preserve the original checkpoint and
dataset provenance. If no image method succeeds after all families and targeted
extensions are evaluated, close the image research line and do not claim RL
validation. The goal closes only when all required decisions and records exist.

## Sources for method context

- https://arxiv.org/abs/2602.16198
- https://arxiv.org/abs/1711.00123
- https://arxiv.org/abs/1807.01695
- https://arxiv.org/abs/2510.13504
- https://arxiv.org/abs/1505.01874
- https://arxiv.org/abs/2408.08252
- https://arxiv.org/abs/2601.21104

### Confirmation and endpoint wave (exp-014--023)

The focused fidelity checks detect wrong signs/scales in two-level H/U,
under-counted reward queries, and a coefficient fitted on its own correction
batch. A failure stops fidelity image runs until repaired. Previously settled
Gaussian, DCT, and continuation-pairing checks are not repeated.

Exp-015 tests representative affine/exponential LMS, projected exponential RLS,
and spatial exponential RLS at M1, M4, M16 and tau0.3/0.4/0.6, with MC and constant
controls. This is discovery-bank tuning, not independent confirmation. Exp-016
locks retained settings before independent VJP evaluation on lighthouse37 and
formula1-59; exp-017 repeats the SPIDER comparison there at three matched costs.
Exp-018 covers pure exponential-affine proposals with scale-zero RNG controls.

Exp-019--023 use Bernese and Formula1 seeds11/23 for a first endpoint screen.
MC8 and MC32 each receive gamma0.4/1/4 and tau0.3/0.6. Gradient EMA beta0.25/0.5
receives the same strength/temperature grid. Representative exponential CV and
ratio EMA receive gamma0.4/1. Direct future M4/M8 receives gamma0.4/1/4 with
warmup2M; a separate M4 condition removes shared continuation noise. Two-level
estimators use cheap M8, fine M4, warmup32/8, and coefficients fixed1 or
previous-mass times0.25/1 at gamma0.4/1. Every fine correction counts both its
proxy and full reward queries; every future denoiser evaluation is recorded.

The screen uses 168 images across 42 conditions. Choose candidates from paired
reward, alignment, visible quality, denominator behavior and actual runtime,
then expand to fresh prompts/discovery seeds before reserved-seed confirmation.
Do not select from final-confirmation seeds101/113/127/139/151/163/179/191.
A lower guidance MSE alone does not establish image success: a zero estimate has
normalized squared error1, whereas all discovery estimates still exceed that.

Exp-024 confirms the smallest nonzero mixture tilt on independent paths, using
8 repeats for IS and CV-IS and a scale-zero control for each. Discovery IS
scale1 reduced normalized ratio error by about0.9%, while larger scales worsened
it. This small effect requires consistent independent evidence before keeping
AIS; KL settings that never clip are not separate effective algorithms.

### Pair-contrast fitting extension (exp-025--026)

Pointwise residual reduction without pair-contrast reduction motivates one
additional CV extension: fit slopes directly to the plus/minus reward (or
log-weight) difference and fit the intercept separately. This removes the
moving Gaussian mean from the slope update's normalization. The surrogate stays
affine or exponential-affine, its Gaussian moments remain analytic, and fitting
still follows estimation. Exp-025 detects wrong pair ordering, difference
scaling and intercept handling using an exactly known function at a moved mean.
Exp-026 tests learning rates0.05/0.1/0.5/1 and coefficients0.25/0.5/1 on the same
discovery banks, with a paired MC8 control. A material gain requires independent
validation; a failure closes this extension without further parameter expansion.

### Bounded temperature/strength extension (exp-027--029)

The initial endpoint screen reaches gamma4 at the selected temperature for MC8,
MC32 and the highest-reward EMA setting. Exp-027 tests gamma8 at tau0.6 and1.0
for these three estimators, on the original discovery images only. This is one
bounded expansion before locking parameters for broader prompt/seed evaluation.
Exp-028 tests direct full-future M4/M8 at tau0.6 and gamma1/4, because the initial
tau0.3 strong-guidance failures resemble the temperature-sensitive proxy-MC8
failure. Exp-029 tests the two-level fixed1 and previous-mass-times0.25 variants
at tau0.6/gamma1. These12 conditions add48 discovery images, not confirmation
data. No further broad grid follows; use them to choose settings for the fresh
prompt/seed comparison.

### Broader endpoint comparison (exp-030)

Lock ten groups from the completed discovery screen and bounded extension:
unguided; MC8 moderate/strong; MC32 gamma4/gamma8 at tau0.6; three beta0.25
EMA settings covering the observed reward/alignment tradeoff; independent-future
M4; and paired-future M8 at tau0.6/gamma4. The exact source configs are recorded
in each new config. Evaluate all eight prompts with fresh discovery seeds41/53,
160 images total. Separate jobs by seed and interleave their order with a fixed
shuffle, so expensive methods do not all occupy one time interval or one GPU.

`analysis-exp-030-expanded.json` maps the actual run shards to comparison groups;
analysis rejects duplicate prompt/seed rows or mixed estimator/image settings.
The analysis preserves each row's source artifact directory. This broader
comparison selects settings for reserved-seed confirmation if any candidate is
promising or marginal. Clearly dominated methods can close here; two seeds do
not support a precision claim or a positive success declaration.

### Reserved image confirmation (exp-031)

Keep seven groups after the broader comparison: unguided; both MC8 settings
(moderate for discovery reward, strong for text alignment); MC32 at tau 0.6 and
gamma 4; guidance EMA beta 0.25 at tau 1 and gamma 8; independent future M4 at
tau 0.3 and gamma 1; and paired future M8 at tau 0.6 and gamma 4. Close the two
weaker EMA settings and MC32 gamma 8 after their broader results. The later
discovery-image review below reopens moderate EMA for one bounded confirmation;
the tau 0.6 / gamma 8 EMA remains closed. Preserve each
selected estimator and image config exactly; the new configs change only the
run identity, seed, purpose, and source-config reference.

Each group receives all eight prompts and the reserved seeds 101, 113, 127,
139, 151, 163, 179, and 191: 64 images per group and 448 total. Use one seed per
job, with the 56 jobs interleaved by a fixed shuffle. These are confirmation
results and must not be used for another image hyperparameter search. Evaluate
paired reward/alignment changes, variation across prompts and seeds, embedding
diversity, visible quality, actual query/score counts, and measured runtime.
Report 95% seed-block bootstrap intervals conditional on these eight prompts.

The primary analysis baseline is the MC8 setting with higher discovery reward;
also compare every candidate with the other MC8 setting and selected MC32.
Advance a method to RL only when this independent comparison supports a useful
reward/cost improvement without an evident alignment or diversity loss. A
mixed or unresolved final comparison is insufficient for advancing that method;
close it under this study's budget rather than claiming universal failure.

Persist the existing VJP and reward timer differences for each image in this
wave, plus sums of the recorded oracle, estimator, and memory-update timers.
The update timer is part of the estimator timer, and reward time is part of
oracle time. For future methods, oracle time also includes the future score
network evaluations. The reward timer excludes the single final evaluation;
total image time includes it. Earlier image records lack the VJP/reward timing
fields, so the grouped analysis reports these unrecorded fields as null. The
first confirmation results must contain finite nonnegative timing fields; a
failure would trigger a metadata repair, not a change to sampling parameters.

### Moderate EMA confirmation after discovery image review (exp-032)

The discovery image review finds visible artifacts in the higher-reward EMA
setting on lighthouse seed 53. The already tested moderate setting (beta 0.25,
tau 0.3, gamma 1) remains clean in the inspected lighthouse and airplane images.
Its smaller aggregate reward/alignment values therefore do not establish that
it is inferior when visible quality is included. Add this existing setting to
the reserved comparison without changing its parameters: eight seed shards,
64 images, and the same paired baseline artifacts from exp-031.

This bounded addition is chosen from discovery visual evidence after the first
ten exp-031 shards became available. Do not use partial confirmation means to
select its parameters or change the active exp-031 manifest. Run its separate
finite queue after exp-031 exits, and preserve this selection timing in the
research record. Interpret cross-wave wall-clock differences cautiously; the
query and score budgets are exactly matched to MC8. No further EMA setting is
added after this comparison.

## Final disposition

Exp-031 and exp-032 completed all 512 planned confirmation images. Strong EMA
and paired future M8 close for observed image quality failures despite positive
aggregate metrics; moderate EMA and independent future M4 close for unresolved
improvement at the final budget. All earlier families also have closure
decisions. The required image-success condition is unmet, so RL scaling does
not proceed. See [the final findings](findings.md) for the quantitative results,
quality evidence, limitations, and artifact locations.
