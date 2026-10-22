# Conditional reward uncertainty continuation

The user reopened development on 2026-10-22 after exp-058. Terminal N4 is the
accepted practical baseline. The new objective is to test a cheaper calibrated
continuation-value estimator, with explicit mean-only and known learned-value
controls. No novelty follows from putting an estimated potential inside SMC:
FK steering already describes point, many-future and learned intermediate values.

## Candidate and mechanism

For each calibration anchor a at a checkpoint, save the proxy reward r_a,
features f_a and independent full-continuation rewards R_aj. Save the anchor's
mean mu_a and population variance v_a. For a new particle with proxy r and
features f, select nearby standardized-feature anchors with normalized weights
w_a proportional to exp(-distance(f,f_a)^2/2). Use k=8 initially, with k=16 as
one bounded smoothing ablation. The distance is averaged over feature dimensions.

Predict m = r + sum_a w_a(mu_a-r_a) and v = sum_a w_a v_a. Keeping the within-anchor
variance separate avoids counting disagreement between anchor means as future
randomness. Compare three positive potentials:

- Mean only: log psi = m/tau.
- Gaussian reward UC: log psi = m/tau + v/(2*tau^2).
- Empirical reward UC: log psi = m/tau + log sum_aj (w_a/J)
  exp(sqrt(v)*(R_aj-mu_a)/(sqrt(v_a)*tau)). Deterministic anchors contribute zero
  standardized residual. This retains the calibration residual's non-Gaussian
  shape under an explicitly approximate conditional location/scale model.

Also compare direct local regression of exp(R_aj/tau), an existing learned-value
baseline. At inference every calibrated variant pays only the proxy query and
feature extraction; neighbourhood arithmetic and calibration costs remain
reported. This estimates a reward distribution, not the complete latent/action
posterior. State-dependent prediction and within-anchor uncertainty distinguish
it from the earlier pooled action-covariance experiment.

The base transition proposal stays unchanged. Initial potential is one; only
named sparse checkpoints update log weight by new minus cached log potential.
ESS-triggered resampling carries the cached value. The last increment uses the
actual terminal reward, regardless of intermediate tempering. Thus arbitrary
positive approximate potentials preserve the target's telescoping weights;
finite particle accuracy still needs evidence. No weight clipping is used.

## Exp-059: controlled toy mechanism and new algebra

Use the two existing analytic targets, independent calibration seed 59001,
128 calibration anchors and 64 full futures per anchor at reverse indices
5/3/1. Evaluate values on 256 new anchors with exact analytic continuation
values. Run endpoint discovery at seeds 60011/60017 with 256 populations per
condition. Compare point, mean-only, Gaussian, empirical, direct learned-value,
future2, an analytic-value diagnostic, and terminal populations N8/16/32/64.
Calibrated N8/N16 and k8/k16, plus intermediate strength 0.5 versus 1, form the
bounded first set. Terminal reward strength is fixed.

Before endpoint comparisons, check the new scalar moment formulas, a genuinely
deterministic future and additive reward offsets. Failure blocks this run's
comparisons and triggers a formula fix.
Do not repeat the established reverse-kernel and original SMC checks.

Primary endpoint measure is weighted-population CDF MSE. Centered log-value MSE
and normalized-potential KL diagnose why a method changes the particle result.
Every continuation transition and reward, calibration and diagnostic included,
is counted. Report online costs separately and the calibration amortization
needed for a total-cost claim. A lower fixed-N error is not sufficient.

An empirical/Gaussian UC candidate needs at least 20% lower endpoint error than
the mean-only control, and a favorable cost comparison with terminal sampling,
before a toy improvement claim. Mechanism evidence may motivate a bounded real
model check because decoder cost changes the tradeoff; label that distinction.
Confirmation will use new seeds and locked settings, never the discovery seeds.

## Planned real-model discrimination

First test conditional value prediction on frozen real critics using independent
calibration observations, new discovery observations, and independently sampled
future references. Retain point, earlier pooled Gaussian UC, mean-only, direct
learned-value and future2 controls. No simulator-return claim follows.

Image calibration will use independent seeds and charge all proxy, branch and
terminal scoring. Compare calibrated values with matched mean-only values and
Terminal N4/N8, while tuning only a bounded set of intermediate strengths and
checkpoints. Terminal tau remains fixed for comparisons of the same target.
Do not substitute a new endpoint reward or tune from confirmation. Record image
alignment, diversity, quality failures and runtime along with aesthetic reward.

Sources: [FK steering, section 3.3](https://arxiv.org/html/2501.06848v5#S3.SS3),
[TDS](https://arxiv.org/abs/2306.17775),
[TMPD](https://arxiv.org/abs/2310.06721). The conditional residual construction
is a candidate implementation whose novelty and performance are unestablished.

## Exp-059 result and next discrimination

All 76 conditions completed at `6f96363`, with 256 populations per condition.
The new moment checks passed; total costs are 2,604,032 transition samples and
838,912 reward evaluations, including 149,504/49,920 for calibration.
No Gaussian or empirical UC setting meets the endpoint improvement criterion.
For N8, empirical versus mean-only CDF MSE is 0.01331 versus 0.00963 on the
Gaussian target and 0.04768 versus 0.04726 on the mixture. N16 does not reverse
this conclusion. Gaussian reward closure fails particularly badly on the
Gaussian target: N16 error 0.04666 versus mean-only 0.00519.

There is a distinct value-prediction signal. At checkpoint 5, empirical versus
mean-only centered log-value MSE is 0.11605 versus 3.83134 (Gaussian) and
0.002835 versus 0.039963 (mixture). Direct value regression is stronger at the
early Gaussian checkpoint; empirical transport is stronger at later checkpoints.
Even the exact-value sparse-N8 diagnostic loses to point-N8 in these finite
populations, so accurate values alone do not solve the endpoint problem.
This is evidence for investigating the estimator, not for a sampling success.

The next experiment measures held-out conditional values on real critics.
It must distinguish transfer of the value-prediction improvement from error in
the continuation reference. Endpoint expansion is deferred until that result.
Full per-seed comparisons and all stage costs are in
`results/uncertainty_smc/exp-059-conditional-reward-toy/analysis.json`.

## Exp-060/061: conditional real-critic values

Reuse the validated 15-step actor and terminal-calibrated temperatures from
exp-054. At indices 5/9/12, features comprise raw observation, current action,
predicted clean action and its Q value. Each checkpoint has its own bank, with
feature standardization fitted only on calibration data. Predictions share the
same local neighbours for mean, Gaussian UC, empirical UC and direct regression.
The earlier pooled-action Gaussian and independent future2 are controls.

Exp-060 is a single-job execution smoke on both tasks: calibration rows 0/4,
four anchors per row and eight futures; diagnostic row18, four anchors and 32
reference futures. New feature-shape or nonfinite failures and incorrect cost
accounting block the larger run. Expected costs per task are 1,336 actor and
216 Q calls for calibration, then 2,656/444 for diagnostics. Measure peak memory
before setting concurrency. Do not infer prediction performance from this smoke.

After it passes, exp-061 calibrates on rows 0/2/4/6/8/10/12/14, with 16 anchors
per row and 64 independent futures. It evaluates rows18/22, with 64 anchors and
512 reference futures. Calibration, anchor propagation, reference and future2
have independent random streams where they share a starting state. A fixed
diagnostic feature/proxy evaluation is shared across the calibrated estimators;
the total diagnostic cost includes that evaluation once. The full expected
budget is 2,819,840 actor and 446,976 Q calls across both tasks. Prior pooled
action calibration is reused as a historical control, with its original cost
in exp-051; it is not a new data-matched fit.

Primary prediction measures are centered log-value MSE and normalized-value KL,
computed within each observation/checkpoint. Report independent half-reference
disagreement and per-anchor importance ESS before ranking close results. A
candidate warrants endpoint follow-up only if its errors improve materially
(target 20%) over mean-only and point on resolved conditions and its comparison
with direct learned values supports the extra construction. Accurate values
alone still do not establish better finite-particle endpoints, novelty or
simulator return. If the reference cannot distinguish the difference, report
it as unresolved rather than treating either estimate as exact.

Exp-060 passed all expected counts with 46 MiB peak reserved memory. Exp-061
completed all 12 conditions at `7f7a5a2`, exactly using the planned 2,819,840
actor and 446,976 Q calls, with 70 MiB peak reserved memory. The well-resolved
HalfCheetah row18 has mean centered log-value MSE 5.0400 for empirical UC,
5.4510 for mean-only, 4.8749 for point and 0.2169 for future2. The UC gain over
mean-only is small and it does not beat point. Its normalized-value KL is
1.6353 versus point 1.6695 and future2 0.0272. Direct cross-observation regression
is much worse on this row. This conditional feature bank does not supply a
useful replacement for real continuation sampling.

Hopper reference ESS medians are only 1.3--2.7 out of 512. Its fine ranking is
unresolved, not a reliable comparison against an exact oracle. HalfCheetah row22
also has appreciable reference uncertainty and provides no positive UC signal.
Close this cross-observation calibration at the tested budget; do not expand it
to images or simulator returns. See exp-061's `analysis.json` for all cases.

The next bounded mechanism question returns to toy: when the proposal remains
the base transition, does a conditional second moment allocate particles better
than a first-moment continuation value? This tests the inference from exp-059
that value accuracy alone is insufficient. It does not reuse the failed
cross-observation real-critic bank or imply a general rejection of UC.
