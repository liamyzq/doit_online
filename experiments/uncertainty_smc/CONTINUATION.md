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
