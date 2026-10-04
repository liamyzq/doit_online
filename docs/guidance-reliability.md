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
scale. Include a simple global guidance reduction when interpreting shrinkage.
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
