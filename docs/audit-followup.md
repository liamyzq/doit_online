# Research reopened after the external audit

The user requested continued research based on an audit of commit `c5e4311`.
The earlier [image findings](findings.md) remain valid for their tested methods
and budgets. This phase changes the research question and permits a bounded
low-dimensional real-critic study before an image method succeeds. It does not
reinterpret the first phase as a positive result.

The audit text was supplied by the user. Its linked ZIP was not attached here,
so exp-033 independently specifies the analytic mechanism experiment rather
than claiming to execute the auditor's unavailable script. The audit's own
numeric results are external claims until reproduced in this repository.

## Decisions and boundaries

Keep ordinary high-dimensional CV, random/DCT projection, pair-contrast fitting,
SPIDER, and adaptive-proposal grid expansion closed. Reopen three bounded
questions: whether low-dimensional real critic directions can be learned online;
whether actual execution displacement explains the image quality failures; and
whether using the same full-future information for categorical resampling is
more effective than estimating and executing a high-dimensional correction.

The image diagnostic seeds from the first phase are now seen data. New image
discovery seeds are 211 and 223, with 239 and 251 available for one bounded
extension. Final confirmation seeds are 307, 311, 331, 347, 359, 373, 389, and
401. Keep the eight named prompts fixed and report uncertainty conditional on
that set. Lock configurations before confirmation; do not tune on those seeds.
Record the real-critic observation and sampling splits before evaluating them,
after locating authentic observations and checkpoint dimensions.

## First wave: discriminate mechanisms before new endpoint grids

**Exp-033: finite-query learnability.** Use the existing estimator on
`F(z)=exp(0.3*z[0])`, with known ratio `0.3*e_1`, at dimensions 6, 64, 512, and
16384. Each of 64 repetitions has nine calls and 96 queries: 32 followed by
eight calls of eight, using complete antithetic pairs. Compare paired MC,
online exponential LMS with coefficient 1 or 0.5, and the externally supplied
exact surrogate. The primary statistic is post-warmup ratio MSE divided by MC
MSE, with paired repetition bootstrap intervals. No hyperparameter expansion
follows this illustrative experiment. A precise oracle with weak high-dimensional
online learning supports an information/learning limitation; oracle failure
invalidates the setup and must be repaired before interpreting learning.

This experiment also verifies the exact affine antithetic identity:
`H_CV=H_MC` and `U_CV=U_MC+rho*(I-C_hat)*b`. It reproduces the raw-M1 diagnostic
degeneracy, where the current-sample residual moment is zero despite nonzero
true error. The historical field `weighted_residual_moment` is renamed to
`in_sample_weighted_residual_moment` in new output and is explicitly not a
confidence or variance estimate. It must not drive adaptive refresh.

**Exp-034: recover execution dynamics.** Read all 512 existing final images'
saved step diagnostics. `correction_norm` was recorded after guidance EMA, so
the applied displacement is recoverable as `abs(gamma*gain)*correction_norm`.
Reconstruct the exact cached Euler schedule and report
`KL=displacement_norm^2/(2*noise_std^2)` for the conditional Gaussian mean shift.
Export per-image values and step scalars, including the known cafe/lighthouse
failures and their paired baselines. This uses no new reward queries. Elevated
KL is a possible mechanism, not a proof of poor quality; absent an association,
do not defend the method by continually adding damping parameters.

**Exp-035: reference and shrinkage diagnostics.** Reuse the two exp-009/010
capture paths and their frozen banks. Report absolute and relative ratio/VJP
errors, split-reference disagreement, estimate-reference direction agreement,
zero and quarter-shrinkage errors, and displacement errors at common gamma 0.4.
Use locked MC8, MC32, exponential LMS, and guidance EMA 0.25/0.5 settings with
the original eight repeats. These are diagnostic additions on existing data,
not new independent confirmation. Common reference noise does not automatically
reverse absolute-MSE rankings, but ratio bias and random normalization limit
their interpretation. No new image reward queries are required.

The analytic identity checks detect an incorrect mechanism implementation and
stop its interpretation if they fail. A remote syntax check detects errors in
the changed runner/evaluator paths and stops formal launch until repaired.
Previously passed Gaussian/CV/IS/sampler checks are not repeated.

## Next discriminating experiments

For low-dimensional CV, load the existing actor/critic checkpoints and real
observations. Keep the environment state fixed within each action-diffusion
rollout while the denoising mean and scale move. Compare a surrogate fitted from
an independent, explicitly counted pilot with online fitting from zero, fresh
MC, and zero/shrinkage controls. An offline good-fit control is diagnostic and
does not inherit the online method's query budget. Inspect action quality and
estimation error first; critic queries may be too cheap for a runtime gain.
Simulator return is not claimed from a fixed-state critic experiment.

For image stability, first use exp-034 to decide whether a small execution-KL
cap comparison is justified. Apply the same cap family to MC8, MC32, EMA, and
future methods and retain their unconstrained controls. This constrains deviation
from the base transition; it guarantees neither proximity to the tilted target
nor visual quality. Do not choose a method-specific cap from confirmation data.

For future information, compare M8 paired and independent continuation at equal
temperature, strength, query schedule, and candidate construction. Compare
Gaussian mean-shift execution with categorical sampling proportional to the
same candidates' endpoint weights. Use stochastic selection, not argmax, and
record its different transition kernel and absent VJP cost. Across full rollouts,
states necessarily diverge; a common-state comparison isolates the use of the
same realized candidates. Finite-candidate resampling is not an exact Doob
transition or a new method claim.

## Evidence and closure

Report average and prompt-specific reward, lower tails, alignment, diversity
proxies, actual reward/score/VJP costs, and systematic image review. CLIP metrics
and embedding distances cannot certify image quality. Record qualitative
failures separately from metric changes and do not infer failure rates from a
few selected examples. Each promising or unresolved discovery result gets at
most one locked independent comparison before its success or closure decision.

New runs record Python, Torch/CUDA, package versions, the GPU, explicit cached
model revisions, and image sigma schedules. Publish small per-image and step
records and reference-disagreement summaries alongside final findings; retain
large tensor banks and model weights on Orion. Use existing commit/config/run
provenance rather than adding content hashes.
