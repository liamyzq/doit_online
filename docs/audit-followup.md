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

## First diagnostic results and next locked wave

All exp-033/034/035 jobs completed at `25936fc`. Exp-033 finds online
exponential-CV MSE/MC ratios of 0.00450, 0.61691, 0.93695, and 0.99843 as
dimension increases from 6 to 64, 512, and 16384. At 16384 dimensions the 95%
interval is [0.99492, 1.00197]. The externally supplied exponential surrogate
is accurate to numerical precision in every dimension. This reproduces the
audit's qualitative learning limitation with this repository's explicit
learning rate, repetitions, and seeds; it does not claim to match its exact
unavailable experiment. All three new identity/diagnostic checks pass.

Exp-034 recovers 4032 guided steps across the 512 images. Median maximum step KL
is 4883 for moderate MC8, 39173 for MC32, 24691 for strong EMA, and 96830 for
future M8. Large final guided-step KL occurs in coherent images as well as
failed ones. Future cafe seed 127 has a step-11 KL of 36266 versus MC32's 2646;
seed 139 has 12528 versus 714. In contrast, the failed EMA lighthouse seed 113
has maximum KL 27093, below the clean future comparator's 123914. Displacement
size therefore cannot be a sufficient quality explanation. A single matched
cap is a discriminating experiment, not a presumed repair or grounds for an
unrestricted adaptive-damping search.

Exp-035 makes the reference limitation concrete. After warm-up, mean
split-reference squared distances are 81.25 for the ratio and 614.41 for VJP,
while zero-estimate squared errors are 20.50 and 158.19. Mean direction cosines
with these references are close to zero. EMA 0.25 lowers VJP squared error from
9203.45 to 1102.74, but zero guidance is still 158.19 and quarter-shrinking EMA
is 217.28. These data support strong variance shrinkage; they do not resolve
the true high-dimensional steering direction or replace endpoint evaluation.

**Exp-036/037 gate the changed execution code.** Three focused checks verify
the Euler gain/noise identity and KL bound, actual categorical selection without
additional Gaussian noise, and reuse of exactly the same future candidates,
weights, and queries. Two one-image Orion smokes then exercise capped MC8 and
categorical future M8. Check their counters and new diagnostics; stop and repair
on a failure. These smoke images are excluded from method selection.

**Exp-040 is a 240-image discovery comparison.** At seeds 211/223, run the eight
prompts for moderate MC8, strong MC8, MC32, strong EMA, paired future M8, and
independent future M8, each with no execution cap and with the same cap 8192.
This cap equals half the latent dimension: the applied displacement has at most
the transition noise's per-coordinate RMS. No second cap is selected now.
Add paired/independent categorical future M8 and an unguided reference, giving
15 groups. The M8 future comparisons share tau 0.6, candidate/warmup budgets
8/16, and the same continuation window; gradient methods use gamma 4, while
categorical execution has no gamma parameter. All groups save latents at steps
11 and 18 for a subsequent common-state information-use probe.

Categorical resampling follows the established idea in
[Soft Value-Based Decoding](https://arxiv.org/html/2408.08252v2), with sampled
future rewards here. The conditional finite candidate set is discrete; do not
report the Gaussian mean-shift KL as its kernel KL. Record selected candidate,
probability, candidate ESS, and VJP/query/score costs instead.

Official [D4RL dataset URLs](https://raw.githubusercontent.com/Farama-Foundation/D4RL/master/d4rl/infos.py)
provide genuine Hopper and HalfCheetah medium-v2 observations without simulator
installation. Their original actor/critic pairs load in the existing Torch
environment. The committed selection procedure takes one nonterminal observation
from each of 64 distinct complete episodes per task, shuffled with seed 41041:
16 calibration, 16 discovery, and 32 confirmation. The episode sets are disjoint,
observations remain unnormalized, and each selected row retains its dataset
index, episode bounds, paired recorded action, and source URL. No confirmation
critic outcome is used to select temperature or surrogate settings.

**Exp-038 calibrates the real-critic scale.** On the 16 calibration states per
task, run one unguided 15-step action path and query 32 antithetic actions at
each guided transition. Use the source eta/window (Hopper 0.6, indices 1--8;
HalfCheetah 0.2, indices 1--10), the source continuous VP schedule and actor,
and the minimum of its two q0 heads. Save local Q variation and recorded-action
Q before choosing one fixed temperature per task. This stage selects a scale,
not a method or an endpoint improvement. Four adapter identities per task
compare source epsilon, simulated endpoint, VJP, and corrected execution mean;
any discrepancy above 2e-4 stops the calibration. These checks detect a source
port error and require repair before the critic results can be interpreted.
No simulator return is measured. The source centers each current reward batch
and divides it by that batch's population standard deviation times its tau
(Hopper 0.4, HalfCheetah 0.5). This study instead fixes the calibrated temperature
so online memory and fresh estimates share a defined target. Source endpoint
strengths are 0.75 and 0.25 respectively; the adapter does not add action clipping.

**Exp-039 isolates low-dimensional estimation.** On eight discovery observations
per task and two unguided action paths per observation, compare MC8/MC32,
exponential LMS at learning rate 1, exponential RLS, frozen warm-up RLS, affine
RLS, an independently fitted local exponential surrogate (128 extra queries per
state), and a local critic-gradient exponential tangent (one extra Q query and
one critic backward pass). The last two are diagnostic assistance, not claims
at the online query budget. Online methods retain their original coordinate
anchor as means and scales move. The same complete current antithetic pairs are
used across methods; 16 current-query repetitions use independent noise, with
32 warm-up queries and then 8, or 32 throughout for MC32. There are no shared
samples between current, pilot and 8192-query practical-reference streams.
One Q query at the first guided mean supplies a fixed offset per path; its cost
is explicit and it does not fit a direction. Record denominator events, zero
and quarter-shrinkage error, reference split disagreement, ratio and VJP error,
and direction agreement. Bootstrap observations, preserving all path/repetition
pairings. A fixed-state critic gain is not a simulator-return claim.

**Exp-041 compares the same realized candidates.** Reuse eight saved discovery
states (cafe/lighthouse, seeds 211/223, steps 11/18 on paired-future free paths).
Each of two candidate repetitions uses M8 with both paired and independent
future continuation. Feed exactly each realized candidate/reward set into the
free gradient, cap-8192 gradient, and stochastic categorical kernels. Use the
same independent execution noise for the two Gaussian shifts and their base.
Reevaluate every resulting next state with two new shared continuation-noise
branches. Report the selected categorical outcome and the probability-weighted
mean over reevaluated candidates, alongside the uniform candidate mean. This
separates useful selection information from favorable noise in the selection
reward itself. Each state costs 120 image-reward evaluations, giving 960 total
and 4328 candidate-counted score forwards, including eight current-state
transitions. These conditional-state diagnostics do not replace full endpoint
confirmation or establish an exact Doob kernel.

## Completed discovery and locked confirmation

Exp-038 passed all source-adapter identities on both tasks (maximum observed
error 9.54e-7), with 288 guided-state records and 9248 Q queries. Local Q
variation changes considerably over the path: the median standard deviations
are 5.1428186893 for Hopper and 2.1239386797 for HalfCheetah, while their 90th
percentiles are 33.75 and 20.71. Fix each task's exp-039 temperature to its
calibration median. This gives a typical local log-weight change of order one
and deliberately exposes the larger early-step variation; it is not a
claim that one temperature matches the source's batch normalization at all
times. Discovery and confirmation observations remain unseen at this decision.

All 30 exp-040 jobs succeeded, producing 240 images with no denominator events.
The complete metric comparison is in `results/audit/exp-040-summary.json`.
The cap changes reward by +0.0474 for moderate MC8, +0.1873 for strong MC8,
-0.0872 for MC32, -0.0301 for strong EMA, +0.1253 for paired future M8, and
+0.0035 for independent future M8. These are discovery means from two seeds,
not confirmation intervals. Paired future M8 reaches 6.4447 with the cap,
compared with MC32's 6.4351 without it and 6.3479 with it.

All 240 images were reviewed in prompt contact sheets, with full-resolution
inspection of the main counterexamples. The cap preserves visible failures:
paired future cafe seed 223 remains strongly oversharpened and posterized
compared with capped MC32; capped EMA lighthouse seed 211 retains blocky sky
and harsh rock texture; capped independent future labrador seed 211 retains
strong artificial texture. Strong MC8 also retains conspicuous airplane/cafe
artifacts. These are concrete quality counterexamples, not a measured failure
rate. Published raw examples include
[paired-future cafe](../results/audit/quality/exp-040-future8-paired-k8192-cafe-s223.png),
[matched capped MC32 cafe](../results/audit/quality/exp-040-mc32-k8192-cafe-s223.png),
[capped EMA lighthouse](../results/audit/quality/exp-040-ema-strong-k8192-lighthouse-s211.png),
and [capped independent-future labrador](../results/audit/quality/exp-040-future8-independent-k8192-labrador-s211.png).
Close this common-cap repair for strong MC8, EMA and both gradient-future
couplings; do not keep reducing caps in search of a favorable result. The
same-candidate probe remains useful for interpreting information versus execution.

Categorical paired/independent futures give discovery mean rewards 6.0966 and
6.0417, versus 6.0279 for moderate MC8 and 5.9702 for base. Their alignment
means are lower (0.26925/0.27013 versus 0.27311 for moderate MC8), and their
images resemble the cleaner base outputs. Both are unresolved, not validated
improvements. Moderate MC8's shared-cap variant is also unresolved and costs
the same 96 guidance queries as its uncapped control.

**Exp-043 locks six groups on all eight reserved image seeds**: base, moderate
MC8 without/with cap 8192, MC32 without a cap, and paired/independent categorical
future M8. Every model, estimator and image parameter is copied unchanged from
its exp-040 config; only experiment metadata and seeds change. This is 48 jobs,
384 images. Compare paired reward intervals conditional on the fixed prompts,
alignment, within-prompt embedding distances, lower tails, actual costs and
image quality. No further image parameter selection follows this confirmation.
