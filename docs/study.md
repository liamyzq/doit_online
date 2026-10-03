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
