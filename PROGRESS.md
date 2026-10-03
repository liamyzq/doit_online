# Research state

## Current state

The research goal is active. No correction method has been validated yet.
The repository has been initialized as DOIT Correction. Image
experiments will use frozen SD v1.5 and LAION aesthetic rewards on Orion before
any successful method is scaled to the existing D4RL/QGPO RL setup.

## What we learned

- `exp-000-checks` passed all 11 analytic and sampler checks on Orion.
- `exp-001-image-smoke` completed at 512 pixels with 7,298 MiB peak reserved
  memory and 21.94 seconds per image. It had no denominator events. However,
  consistent-Euler correction at gamma 4 and M=8 visibly oversaturated the
  image (reward 4.1543, CLIP alignment 0.20635). This prompted an M=32 strength comparison before fixed-state capture.
- All six `exp-004` calibration runs completed. At M=32, mean reward increased
  from 6.1129 (gamma 0) to 6.1485 (0.4), 6.2088 (1), and 6.4681 (4); gamma 4
  alignment was 0.3085 versus 0.3122 without guidance. Seed 11 at gamma 4 was
  visually coherent with M=32, unlike the M=8 smoke. Retain gamma 4 for the
  discovery capture to test this query-budget sensitivity. Two seeds do not
  establish an endpoint improvement; confirmation will include gamma 1 and 4.
- Confirmed by source inspection: the available image implementation already
  uses fixed temperature (reward centering, without division by batch standard
  deviation). The proposal's batch-normalization concern must be checked
  separately for RL; it is not present in this image source.
- Confirmed by source inspection: the Euler branch uses an Euler transition but
  a VP/DDIM endpoint formula and VP epsilon-correction scale. The new sampler
  will expose the source convention as a named comparison and use an explicitly
  consistent Euler parameterization for the primary study. This difference
  must not be credited to online variance reduction.
- The old image reward callback reloads CLIP on every call. All new methods will
  share one cached reward implementation and identical preprocessing.

## Open questions

1. Can a small, analytically integrable surrogate learn useful directions from
   one high-dimensional image rollout?
2. Does reduced moment/ratio error survive the score-network VJP and improve
   final reward without a loss of diversity or text alignment?
3. Is practical-endpoint bias larger than the Monte Carlo error being reduced?

## Current discovery observations

The first saved state (step 10) shows that even a 512-sample reference has large
direction noise. At tau 0.3, Bernese reference ratio squared norm is 25.55 and
one quarter of the independent half-reference squared difference is 25.06;
the corresponding Formula 1 values are 55.91 and 53.42. These are precision
diagnostics, not exact debiased norms for the nonlinear ratio. Use paired method
comparisons and the reference disagreement; do not claim the true direction is
resolved by these references.

For 32 first-transition candidates, proxy/full-future mean rewards at step 10
are 4.439/5.791 (Bernese) and 2.950/4.643 (Formula 1). Reward correlations are
0.203 and -0.240. These probes include independent later rollout noise, so they
do not by themselves establish that guidance bias dominates. A targeted extension
will share later noises within antithetic pairs and average two branches per
candidate to distinguish conditional-future variation from continuation noise.
Structured low-frequency latent projections will test whether a small spatial
model can learn useful information that a random projection misses.

Both captures completed with nine states and 7,201 total oracle queries each.
The Formula 1 gamma-4 path ended visibly corrupted (reward 3.2103, alignment
0.2521), although the Bernese path was coherent (6.2122, 0.2943). Confirmation
captures therefore use gamma 0.4 with fresh seeds 37 and 59, including a new
lighthouse prompt. Keep the unstable path as an observed failure condition.

The Bernese capture and earlier same-seed calibration have identical first-step
weights and ratio, but their first VJP norms differ by 0.00014; later trajectories
diverge and final rewards differ by 0.104. The FP16 guided trajectory is sensitive
to these small numerical changes. Paired seeds control random draws, not exact
bitwise trajectories; endpoint conclusions require multiple seeds and replication
when an effect is marginal. Capture wall-clock also includes reference/probe
queries and must never be used as generation runtime.

## First complete estimator comparison

All 81 `exp-005` configurations completed on the two discovery paths. Ordinary
affine LMS does not materially reduce ratio error. The best exponential LMS
reduces post-warmup squared error by only 0.17% on Bernese and 0.26% on Formula 1,
while its pointwise residual MSE falls by roughly 75%. Random-projection RLS
is worse; two exponential settings also floor their denominators. A constant
control is exactly redundant with paired antithetic samples, as expected.

Every SPIDER refresh-32 configuration had denominator events (2--65 among 288
state/repeat rows per configuration), making its mean error unusable. Try
refresh64/128 with matched MC costs before closing this branch. Ratio EMA with
beta0.25 reduces post-warmup error to 23.1% and 19.2% of M8 MC on the two paths,
but it is a biased control and has not passed VJP or endpoint validation.

The pointwise fit/error mismatch is consistent with learning the mean weight
while antithetic MC already cancels constant numerator contributions. The next
wave records residual differences within antithetic pairs, tests low spatial
frequency regressors, and makes actual importance-corrected oracle queries.

## Next experiments

1. Run focused DCT-moment and future-pairing checks; a failure stops the new
   extensions until repaired. Then evaluate 28 spatial regression configurations
   and 24 SPIDER/matched-MC configurations with larger refresh batches.
2. Measure actual VJP error for 14 representative estimators/controls and run
   38 live AIS/CV-AIS configurations across proposal scales, KL radii and mixture
   masses. Query counts and actual proposal norms determine the comparison.
3. Capture independent gamma0.4 states (lighthouse37 and formula1-59), and probe
   coupled full futures on the initial captures. Confirm retained methods on
   these states before endpoint image tuning and reserved-seed evaluation.
4. Scale only image successes to RL; no correction method is validated yet.

## Extensions and remaining decisions

`exp-006` passed the two focused checks for DCT Gaussian moments and shared
continuation noise within antithetic pairs. All 28 DCT variants in `exp-007`
failed to improve the discovery-bank ratio error: the best normalized squared
error was 47.680 versus MC8 at 47.484. Three settings had denominator events.
All 12 larger-refresh SPIDER variants in `exp-012` still had denominator events
(1--16 per variant); all 12 exactly cost-matched MC controls had none.
Independent moderate-strength paths will determine whether these failures depend
on the initially unstable discovery path.

Actual VJP evaluation in `exp-011` confirms essentially no improvement from
exponential CV: mean normalized squared error 46.872 versus MC8 46.943. Gradient
EMA beta0.25 gives 8.011, ratio EMA beta0.25 gives 22.778, and MC32 gives 15.788.
Gradient EMA is a promising biased control, not a validated method: zero
guidance itself has normalized error 1 by definition, so noisy-reference MSE
alone can favor shrinkage without useful reward steering. Endpoint comparisons
must tune MC strength and temperature independently and include an unguided
reference.

Paired full-future probes in `exp-008` share continuation noise only within
antithetic pairs and average two branches. At step10, proxy/future weight
contrast correlations are -0.149 (Bernese) and 0.111 (Formula1); at step16 they
are 0.354 and 0.558. At tau0.3 the future-to-proxy mean weight ratio ranges from
63.4/203.4 early to 2.68/1.11 late. These small probes do not resolve precise
high-dimensional guidance bias. They do show enough proxy/future discrepancy to
justify direct future and two-level moment experiments with explicit denoiser
costs. A historical mass coefficient is tested because proxy and future weights
have substantially different scales.

The next committed wave includes focused two-level algebra/query/history checks,
M1/M4/M16 and temperature comparisons on discovery banks, locked independent VJP
and SPIDER confirmation, pure Gaussian tilt controls, and a paired endpoint
sweep. Surrogate failure is not rescued by pointwise reward regression accuracy;
a useful method must improve guidance or endpoint outcomes at its actual cost.

All 38 live mixture AIS/CV-AIS settings in `exp-013` completed without denominator
events. At scale1/base-mass0.2, IS normalized ratio error is 51.215 versus its
RNG-matched zero-tilt control51.685 (0.91% lower); CV-IS is 51.276 versus51.965
(1.33% lower). The average proposal displacement is only0.0188 in standardized
noise coordinates. Larger displacements worsen error. Several KL settings give
identical results because they never bind. Exp-024 tests the small effect on
independent paths before a decision.

Stored-bank analysis explains why paying twice for SPIDER differences is poor
here. Across the eight post-warmup Bernese steps, the squared antithetic weight
difference for current-minus-previous is0.534--2.225 times the current-only
quantity; Formula1 ranges0.663--15.389. Even the best observed reduction does
not compensate for halving the fresh pair count, before including accumulated
historical error. The exact ratio error comparison remains the decision metric.

## Independent SPIDER closure and budget ablations

`exp-017` completed on the two independent gamma0.4 paths without denominator
events. Its normalized ratio errors are89.448/82.318/174.710 for warmup32-r3,
warmup64-r3 and warmup128-r2 (M16 difference-query budgets), compared with
27.421/24.893/19.632 for exactly matched MC. Post-warmup absolute error is worse
on each prompt separately. Close SPIDER under the tested schedules and budgets;
its failure is not explained solely by negative denominators or a corrupted
trajectory. Keep its implementation as a reproducible comparison.

Across M4/M16 and tau0.3/0.4/0.6 in `exp-015`, full affine/exponential CV changes
normalized error by less than0.4%; projected and spatial models usually worsen
it. At M1, some memory controls reduce the reward-independent raw-MC error,
but errors remain thousands versus roughly82--108 for M4. A raw-M1 improvement
does not make M1 useful. Random/spatial exponential RLS also has M1 denominator
events at tau0.3. The final CV decision still uses independent VJP confirmation.

`exp-016` independently confirms the ordinary CV failure: normalized VJP error
is53.418 for MC8,53.436 for frozen affine,53.893 for exponential LMS,53.471 for
affine projected RLS,59.392 for exponential projected RLS and54.205 for spatial
exponential RLS. No setting improves both paths consistently. Close the ordinary
LMS/RLS/spatial variants under this study's budgets. The pair-contrast fitting
extension is the only remaining surrogate branch.

Gradient EMA beta0.25 independently gives normalized VJP error8.981 versus
MC32's18.029. Post-warmup absolute error relative to MC8 is0.115 on Formula1 and
0.132 on lighthouse. This confirms error suppression, but the initial endpoint
screen has not shown a clear advantage over separately tuned MC. Its image
success decision remains open until expanded prompt/seed evaluation.

The endpoint screen encountered a real instability in
`exp-023-tuning-mf-mass-c025-g0.4`: the second image failed with nonfinite VJP
guidance, after one successful image. Preserve this as a failed configuration;
the first image cannot stand in for the four-image condition. The current log
does not identify the failing moment values. The next revision includes the
step and estimator diagnostics in this existing error message, so future
failures can be attributed without an extra diagnostic rerun.

## Live proposal and initial endpoint decisions

Independent `exp-024` closes the mixture AIS branch: normalized ratio error
changes from47.041 to47.213 for IS and47.105 to47.256 for CV-IS. Post-warmup
absolute error improves about2% on lighthouse but worsens about1.5% on Formula1.
The small discovery improvement did not reproduce consistently. `exp-018` also
shows no useful pure-tilt benefit. At unclipped scale1 with complete antithetic
pairs, pure IS and CV-IS are numerically identical: `p*u/q=H_u`, the paired mean
of base-coordinate epsilon is the tilt `rho*b`, and the analytic CV terms cancel
the sampled surrogate moments. This is expected algebra, not a separate gain.

The first endpoint wave has39 complete four-image conditions and3 failed image
conditions. Tuned MC8 (tau0.6/gamma4) averages reward5.6115 and alignment0.2777;
MC32 at the same parameters gives5.8147 and0.2779. Gradient EMA beta0.25 at
those parameters gives5.6344 but alignment0.2704. The strongest EMA reward
setting has no established quality/cost benefit. Ordinary exponential CV and
ratio EMA do not beat the tuned MC8 screen.

Direct paired-future M4/M8 at gamma1 gives reward5.4652/5.4491. The independent
continuation-noise M4 control gives5.6519, but its difference from proxy MC8 is
mixed across seeds and mainly driven by one Formula1 image. It needs a fresh
prompt/seed comparison. Fixed-coefficient two-level runs complete at reward
5.4844/5.4881 (gamma0.4/1), using176 reward queries plus236 score evaluations per
image. Three historical-mass settings fail with nonfinite guidance; the one
remaining setting has a denominator event and reward4.8162. No fidelity method
is validated yet. The predeclared bounded temperature/strength extension and
pair-contrast fit check/grid run next, followed by broader endpoint confirmation.

Failure records retain partial images without treating them as complete
conditions. Exact oracle counts inside the failing image were not saved by the
old failure path; do not claim a precise total research-query count from those
partial files. Future image failures now preserve the engine's query and score
counters, and include the failing step/moment diagnostics in the error text.
