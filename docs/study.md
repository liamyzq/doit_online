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
- A historical constant control variate and a guidance EMA control.
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
query schedule. Small M=1 is a separate ablation after warm-up, not a default.

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
