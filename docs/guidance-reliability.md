# Guidance reliability exploration

The user authorized this separate branch and Nebula GPUs 0--4 on 2026-10-03
while Orion is unreachable. The online-update study remains on
`codex/doit-correction`; recover its already submitted jobs before resuming it.

The new question concerns the reliability of `J_mu.T @ v`. Distinguish an
incorrect numerical adjoint, a learned model's sensitivity relative to an
available teacher, noise in the reward-weighted vector, and continuation bias.
Only the first can be settled by finite differences on the same network. A
real SD model has no ideal Jacobian reference here. Better endpoint behavior
would support a useful biased control, not prove its derivative was wrong.

## Reused evidence

Exp-033 already identifies the high-dimensional online learning limitation.
Exp-035 shows that the existing image reference cannot resolve a true direction.
Do not repeat constant-weight antithetic identities or the old estimator grid.
Exp-041 and exp-043 already define the common-candidate continuation and
categorical controls; recover those on Orion instead of duplicating them.
The image reward already has fixed temperature. The new critic study also uses
offline calibration; batch reward standardization is not present in either.

## First bounded experiments

Exp-044 uses a known nonsymmetric smooth mean plus a small sinusoidal value
error. Its ideal adjoint is available analytically. Compare raw sensitivity,
half shrinkage of the learned component, and four-point Gaussian smoothing
with opposite perturbation pairs. Include a zero-corruption control. Report
MSE to the ideal adjoint and value error. This is a mechanism demonstration,
not evidence that SD has the same corruption. A fixed-vector central-difference
sweep checks the implementation; a failed identity blocks the real experiment.

Exp-045 uses unguided SD1.5 states at indices 11 and 18 for cafe and lighthouse,
seeds 521 and 523. FP32 keeps quantization from obscuring the directional check.
Use two independent complete-pair M32 candidate groups per state, fixed tau
0.3. Freeze each resulting vector across raw, learned-component shrinkage
(coefficients 0 and 0.5), and K4 smoothing at 0.1 and 0.5 times transition noise.
The larger perturbation is a finite-scale sensitivity control. Record its
typical norm as well as its per-coordinate scale. Never recompute reward weights
inside a perturbation. Use shared perturbations between the two vector groups.

Measure the finite-difference plateau along raw guidance and a random direction,
the independent-group cosine, the known-linear/learned cancellation, actual
Gaussian displacement KL at gamma 0.4, and every reward query, forward, VJP and
wall-clock cost. No large-M reference is labeled ground truth. A correct
adjoint closes numerical error as an explanation at these states. Large
cancellation warns that shrinkage can increase displacement. Improved agreement
alone does not establish useful guidance. A representative single-state smoke
must measure peak GPU memory before concurrency is chosen.

After these diagnostics, choose at most one shrinkage and one smoothing setting
for a small endpoint comparison with raw MC8 and MC32 at the same fixed reward
scale. Control actual displacement magnitude when interpreting shrinkage.
Count extra smoothing forwards/backwards and compare measured cost with using
more candidates. Reserve fresh seeds before confirmation; close this branch
under its tested budget if no useful effect survives quality and cost review.

## Sources and scope

[DOIT](https://arxiv.org/html/2602.16198v1) supplies the base-model Doob transform
and practical endpoint approximation. [SmoothGrad](https://arxiv.org/abs/1706.03825)
motivates averaging input gradients; its visualization results do not establish
a DOIT improvement. [Sobolev training](https://arxiv.org/abs/1706.04859) motivates
distinguishing value and derivative supervision, but no model training is
planned without an independent reference. [SVDD](https://arxiv.org/abs/2408.08252)
provides the existing derivative-free comparator's context.

## First execution results

Exp-044 completed at `c8d8fdd`, with 18 summary rows. Its nonsymmetric adjoint
identity has maximum error 1.11e-16 and the directional finite-difference sweep
reaches relative error 3.29e-9. Across dimensions 6/64/512, adding the specified
sinusoidal error raises raw adjoint MSE to 0.312/0.321/0.318 despite coordinate
value RMS error about 0.014. Half shrinkage lowers it to 0.090/0.092/0.092 and
K4 smoothing to 0.076/0.082/0.082. The clean control is exact for the raw method;
shrinkage instead creates MSE about 0.0129, and smoothing about 1e-8. This is
evidence for a possible mechanism and its tradeoff, not a fitted explanation
of the actual diffusion model.

The representative exp-045 smoke also completed at `c8d8fdd`: one state,
10 rows, 65 reward queries, 61 score forwards and 20 VJPs. Its minimum
raw-direction finite-difference relative error is 1.46e-7. Random projections
are much smaller and exhibit cancellation at tiny finite-difference steps;
their error improves at the larger step sizes. No numerical mismatch blocks
discovery. Peak reserved memory is 11,894 MiB.

The smoke's `vjp_seconds` does not measure complete smoothing wall time. A
follow-up instrumentation change times the full synchronized sensitivity call
and starts backward timing only after queued forwards complete. It also retains
probe query/forward counters if a later operation fails. This changes cost
recording, not the method or the successfully checked numerical result.

## Eight-state diagnosis and locked endpoint wave

Exp-045 discovery completed at `d85d1da`: eight states, 80 vector/method rows,
516 reward queries, 408 score forwards and 160 VJPs, with no runtime failures.
At every state, the best raw-direction finite-difference error is below 3e-5;
the random-direction counterpart is below 0.003. The fixed-network adjoint is
correct at the tested FP32 states. An unknown ideal model remains unidentified.

The independent candidate-vector cosine averages 0.0028 at index 11 and
-0.0016 at index 18. After the raw adjoint it is 0.0012 and -0.0185. Both
smoothing scales leave agreement near zero. This local evidence identifies
substantial uncertainty already in the vector supplied to the adjoint; it
does not prove that finite-sample guidance cannot improve endpoints.

Small smoothing changes the vector by 1.17% on average at index 11 and 0.014%
at index 18. Large smoothing changes it by 5.78% and 0.342%, respectively;
the largest individual change is 12.5%. Its mean KL ratio to raw is 1.005/1.000.
Full sensitivity time beyond the common transition forward is about 5.1--5.5 s
for K4 versus 0.77--0.87 s for raw. These are shared-GPU observations.

Half shrinkage raises mean KL by factors 1.355 and 1.605. The known-linear and
learned components oppose each other, with cosine from -0.983 to -0.786.
Consequently the endpoint shrinkage control rescales its modified vector to
the current raw vector's norm. It uses the raw VJP already required by the
formula, with no extra queries or backwards. This local norm match removes the
magnitude confound more directly than a separately tuned global gamma.

Exp-046 fixes eight prompts and seeds 541/547, five groups, 80 images total:

- Raw MC8, tau 0.3 and gamma 0.4: the existing moderate setting.
- Raw MC16 at the same tau/gamma: an approximate elapsed-cost comparator.
- Raw MC32, tau 0.3 and gamma 4: a higher-query, stronger-guidance control at
  the common temperature, generated in FP32 in the same environment.
- MC8 with half learned-component shrinkage and raw-norm matching, gamma 0.4.
- MC8 with K4 smoothing at delta 0.5 times transition noise, gamma 0.4.

All groups use FP32, disabled TF32, the same frozen models, complete opposite
candidate pairs, a 32-query first-step warm-up and fixed reward scale. Queries
per image are 96/160/288/96/96, excluding one common final evaluation.
Score-forward counts are 20/20/20/20/56 and VJP counts 9/9/9/9/36. MC16's
64 extra queries cost about 44 s at the measured 0.68 s/query, close to K4's
roughly 40 s additional sensitivity work over nine steps. Actual complete-image
time, not this estimate, determines the reported cost comparison.

The larger predeclared smoothing scale is retained because its early-state
change is measurable; the smaller scale's late-state effect is negligible.
Do not search additional scales or shrinkage coefficients in this wave.
Compare paired reward, alignment, embedding diversity and all prompt images.
An endpoint effect must be useful at its measured cost and free of material
quality failure before confirmation is warranted. Reserve seeds
557/563/569/571/577/587/593/599 for any confirmation; they are not used for
parameter selection. Close unsupported settings under this budget.

The exp-046 config purpose text incorrectly calls its MC32 group the previously
selected strong baseline. The selected exp-031/040/043 setting uses tau 0.6,
whereas the locked exp-046 execution uses tau 0.3. This discrepancy was found
while the queue was running, before endpoint analysis. Preserve the executed
configs and label the group by its actual parameters. It is a valid common-scale
control but cannot support a claim against the selected MC32 baseline. If a
sensitivity candidate warrants confirmation, include the actual selected
tau-0.6 MC32 setting there before making that comparison.

## Endpoint results and closure

All ten exp-046 jobs completed at `6d1ceca59bdee1c3064edf31e9b49625846173ba`,
producing 80 images. The controller exited and every run returned zero. All
per-image query/forward/VJP counters match the locked method settings, all
recorded rewards and times are finite, and no denominator event occurred.
The [paired summary](../results/reliability/exp-046-summary.json) and
[accounting](../results/reliability/exp-046-accounting.json) are derived from
complete saved runs. Two seeds support discovery decisions, not a seed-block
confidence interval or a broad equivalence claim.

| Method | Reward | Delta from MC8 | CLIP alignment | Embedding distance | Guidance queries | Score forwards / VJPs | Seconds/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MC8, gamma 0.4 | 5.91513 | 0 | 0.27059 | 0.11053 | 96 | 20 / 9 | 81.46 |
| MC16, gamma 0.4 | 5.93714 | +0.02201 | 0.27068 | 0.10619 | 160 | 20 / 9 | 98.40 |
| MC32, gamma 4, tau 0.3 | 6.01186 | +0.09673 | 0.26236 | 0.17656 | 288 | 20 / 9 | 179.39 |
| Half shrinkage, raw-norm matched | 5.91560 | +0.00046 | 0.26906 | 0.11269 | 96 | 20 / 9 | 85.27 |
| Four-point smoothing | 5.90488 | -0.01026 | 0.26950 | 0.11227 | 96 | 56 / 36 | 111.32 |

All groups use tau 0.3. Query counts include the first-step 32-query warm-up
and all subsequent guidance samples; each image has one additional common
final reward evaluation. The wave uses 11,856 total reward queries, 2,176
score forwards and 1,152 VJPs. Maximum reserved memory is 12,498 MiB.

Half shrinkage has seed-wise reward deltas -0.01698 and +0.01791. Its aggregate
delta +0.00046 is negligible in this discovery comparison, and alignment is
0.00153 lower than MC8. All prompt pairs were inspected: composition and detail
are close to MC8, with no distinct new quality failure and no clear visible
improvement. Close this coefficient with raw-norm matching for lack of a useful
endpoint gain under the tested settings. This does not assert exact equality
or rule out another scientifically distinct regime.

Four-point smoothing has negative reward deltas on both seeds (-0.00749 and
-0.01303), and its mean alignment is 0.00109 lower than MC8. The images remain
visually close to MC8. Smoothing adds 36 score forwards and 27 VJPs per image;
full measured sensitivity time rises from 6.94 to 41.74 seconds per image.
MC16 has higher mean reward and alignment with fewer forwards/backwards and
lower observed elapsed time, although it uses 64 additional reward queries.
Close the tested smoothing scale for lack of useful benefit at its cost.

Measured elapsed times reflect changing load on shared GPUs. For example,
MC8 averages 62.02 seconds at seed 541 and 100.91 at seed 547; MC32 averages
233.41 and 125.37. The mean times above are actual observed costs, not an
estimate of isolated-device speed or a precise causal timing ratio. Explicit
operation counts remain interpretable despite that variability.

The stronger MC32 common-temperature control increases aggregate reward but
has conspicuous quality failures. At seed 541, cafe and Labrador images have
severe saturation, dense artificial texture and damaged structure; airplane,
Formula 1 and mountain examples also show degraded texture. The seed-547 cafe
image becomes strongly outlined and posterized. Its aggregate embedding
distance must not be interpreted as a diversity improvement when corruption
changes the embedding. Close this tau-0.3/gamma-4 control for quality. It does
not revise the earlier selected tau-0.6 baseline's result.

The full visual review covers all eight prompt sheets, with both seeds and all
five methods in each:
[Bernese](../results/reliability/quality/exp-046-contact-bernese.jpg),
[Formula 1](../results/reliability/quality/exp-046-contact-formula1.jpg),
[airplane](../results/reliability/quality/exp-046-contact-airplane.jpg),
[Labrador](../results/reliability/quality/exp-046-contact-labrador.jpg),
[lighthouse](../results/reliability/quality/exp-046-contact-lighthouse.jpg),
[cafe](../results/reliability/quality/exp-046-contact-cafe.jpg),
[orchid](../results/reliability/quality/exp-046-contact-orchid.jpg), and
[mountain](../results/reliability/quality/exp-046-contact-mountain.jpg).
Unresized examples retain the [MC8 cafe](../results/reliability/quality/exp-046-mc8-cafe-s541.png)
and [strong MC32 cafe](../results/reliability/quality/exp-046-mc32-strong-cafe-s541.png),
plus [MC8 Labrador](../results/reliability/quality/exp-046-mc8-labrador-s541.png)
and [strong MC32 Labrador](../results/reliability/quality/exp-046-mc32-strong-labrador-s541.png).

Neither sensitivity variant warrants confirmation. All eight reserved seeds
remain unused, no further Nebula wave is queued, and the bounded exploration
is closed. The controlled-model mechanism is real, the fixed-network FP32
adjoint is correct at the tested states, and vector uncertainty is present
before applying it. These results do not identify an ideal SD Jacobian or
establish an endpoint advantage from smoothing or shrinkage. The original
online-update audit remains pending recovery of the already submitted Orion
experiments; it is not closed by this separate branch.
