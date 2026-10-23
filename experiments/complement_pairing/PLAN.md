# Complement cancellation under a fixed Gaussian kernel

The supplied review of `8cfecfc` reopens estimator development, while keeping
exp-063--069 closed at their locked budgets. The main question is whether
independent sign pairing of a reward-relevant subspace and its complement
reduces total guidance error enough to repay direction acquisition. The kernel,
kernel-score factor, reward temperature and outer sampler remain fixed.

For a basis U chosen before the main sample, draw a standard Gaussian active
coordinate a and independent complement b. Evaluate all four actual weights at
Ua+b, Ua-b, -Ua+b, -Ua-b. Every point has the original Gaussian marginal.
If the weight depends only on a, its complement numerator cancels exactly.
Otherwise the full complement contribution is still estimated. Numerator and
denominator target the original moments; their finite-sample ratio is biased.
Four-point groups use M/4 independent active draws, compared with M/2 for
ordinary antithetic pairs. No performance claim follows from the identity alone.

## P0: correct the record

Preserve original formal results. Reproduce only exp-064's top-aligned M32
conditions to add reward-direction and complement error, checking reconstructed
total MSE against the saved values. A discrepancy blocks the new decomposition
until its cause is resolved. Exp-069 gets separate active-guidance summaries and
unit-gamma transition-mean errors, along with its original all-time aggregate.
These reuse existing raw records and the preserved schedule without inference.

Describe true learned Jacobians and their symmetric parts precisely. CFG scale
4.5 can make even ideal conditional-response combinations indefinite; it does
not explain asymmetry. The image range sketch is not a converged extreme
eigensolver. No early-image finite-difference or larger-rank work is warranted
unless the later gate is met. The earlier tests neither change conditional
transition covariance nor implement Fisher-metric geodesics.

## P1: exp-070 and exp-071

Exp-070 checks the changed sign-group construction on Orion: fixed-index
Gaussian marginals, constant-weight cancellation, exact complement cancellation
for active-only weights, and retention of nonzero complement moments when the
reward depends on that space. Failures block exp-071 and require correcting
the construction. It also performs the bounded P0 replay.

Exp-071 locks the existing rotated Gaussian and mixture, dimension 12, M16/M32,
two fresh seeds, and 1024 repeats per fixed condition. Candidate denoising and
the analytic conditional expectation have independent exact quadrature targets.
One posterior terminal draw per candidate tests stochastic continuation, both
with independent future draws and shared draws within each pair/four-point
group. Every continuation marginal remains correct. No noisy weight is silently
replaced by its conditional mean.

Controls are ordinary antithetic, existing top-two paired strata, randomized
rank-two paired strata, and oracle rank-one paired strata. Compare rank-one
four-point designs with/without LHS, a random direction, a 30-degree direction
error and an independent eight-query central-difference pilot. The pilot fits
a continuous direction from four orthogonal random contrasts and uses candidate
rewards even when the target is continuation. All eight queries reduce M;
pilots never become main samples. The prior four-query spectral-axis selector
remains closed. An oracle Rao--Blackwell projection is labeled a mechanism
bound, never a deployable algorithm. Geometry and oracle direction knowledge
are explicit assumptions, not free measured procedures.

Primary metric is full guidance MSE. Save per-repeat vectors, reference vectors,
active/complement errors and paired loss-difference intervals. Repeats quantify
Monte Carlo uncertainty for these fixed analytic models, not generalization.
The P2 mechanism gate requires at M32 at least 20% aggregate MSE improvement
for oracle four-point LHS over both ordinary antithetic and oracle paired LHS,
on each model for candidate and analytic-continuation targets, with paired
95% Monte Carlo difference intervals below zero. Report independent/shared
future behavior, 30-degree sensitivity and pilot net gains separately. Passing
the ideal-direction gate permits a bounded low-dimensional direction study;
it does not assert that pilots or real models work.

## P2 and P3 gates

If P1 passes, lock a small real-critic discovery on fresh observations and
indices within the actual guidance interval. Compare symmetric eigenvectors,
right singular vectors of the full local response, differentiable reward
gradient oracle directions and a paid black-box direction estimate. Preserve
matrices, per-repeat estimates, multiple independent reference blocks and full
Q/actor/VJP costs. Direction streams are structured by task, observation, index,
budget and purpose; sharing is explicit for paired comparisons only. Include
unit-gamma displacement error and separately account for common kernel VJPs
and additional direction-acquisition derivatives. Simulator returns are outside
this mechanism study.

Only a fixed, feasible setting with at least 20% gain over simple cost-matched
controls, improvement on at least three of four fresh observations, and resolved
paired difference uncertainty proceeds to independent confirmation. Only a
confirmed setting can enter a bounded image test. Reference split statistics
are diagnostics, not calibrated bias bounds or confidence intervals. Use
reference block sensitivity of the actual difference to determine unresolved
comparisons, retaining all contexts. Do not rank using reference norm alone.

Terminal N4 remains the practical image baseline. Endpoint or simulator work
requires a separate favorable fixed-state, cost and independent confirmation
decision; it is not an automatic consequence of a toy mechanism.

## Locked P2 discovery after the P1 mechanism gate

Exp-071 point estimates meet the ideal-direction effect threshold on both
toys and both deterministic targets. A subsequent stream audit invalidated its
paired Monte Carlo intervals. Exp-074 and exp-071-r2 subsequently pass at
`0400f4e`, reestablishing the unchanged gate before discovery. Exp-072 checks
only the newly introduced candidate log-weight
gradient orientation and grouped future-noise implementation. Independent
future rollout must match the preserved adapter; identical candidates with
shared future randomness must remain identical. These checks distinguish a
time/index/coupling error from real estimator behavior. Failure blocks exp-073.
The smoke measures peak memory before discovery concurrency is set.

Exp-073 uses fresh observation rows40/44/48/52, indices3/5/7 (active for both
tasks), M16/M32, two seeds and256 repeats. Each context has eight independent
reference blocks of8192 paired probes. Later reference noises are shared within
ordinary pairs. Each point retains the original continuation marginal. Save
block moments, masses and ESS, complete response/kernel matrices, per-repeat
guidance and paired losses. The candidate target and independent/shared future
estimators are all reported. The primary advance target is shared-future
continuation at M32; candidate gains alone do not advance to images.

Fixed directions are measured once for scientific efficiency, with acquisition
cost explicitly charged to every hypothetical deployed guidance call. Full
coordinate finite differences use2d candidate Q/actor calls; the local reward
gradient oracle uses one Q/actor call and a joint backward pass. Main sample
count is the largest complete pair/four-point group that fits the remaining
query budget; unused slots are reported, not silently filled. Symmetric and
singular directions require the full response Jacobian here and expose that
extra derivative cost. A separate four-query random two-direction fit is drawn
independently per repeat. The shared kernel VJP is common to every algorithm.
Actual research counters include all diagnostic setup and references; per-call
costs describe deployment rather than falsely multiplying actual setup work.

Simple controls are ordinary antithetic and random rank-one paired LHS.
Oracle and finite-difference paired controls isolate the four-point tradeoff.
Random bases are fresh per repeat, shared explicitly between matched random
pair/four-point designs. Main/future random streams are explicitly coupled
across methods for paired loss comparisons and separated across contexts.

A task/method discovery can advance only if the mean unit-gamma displacement
MSE improves by at least20% against both simple controls, at least three of
four observations improve against both, and aggregate paired Monte Carlo
intervals exclude zero. The aggregate difference must keep its sign for all
eight independent reference-block replacements as well as the pooled reference.
This block sensitivity is not a calibrated confidence bound on ratio bias.
All conditions, including low-ESS references, remain in the aggregate.

Only the random-four, full-finite-difference-four and four-query-pilot-four
methods can pass the current inexpensive-direction gate. Exact matrix and
gradient oracles explain attainable direction quality and cost; they do not
constitute a cheap high-dimensional implementation. Discovery makes no speed
claim from batched repeats. If an eligible setting passes, lock at most one
setting on new observations and measure complete individual-call costs in
confirmation before any image expansion. No other hyperparameter sweep follows
a failed discovery under this budget.

## Execution

Use the existing geometry branch, Orion checkout and finite queue. Commit each
implementation and immutable config before its formal run. No simultaneous
sync with active/pending jobs. Low-cost agents execute and monitor established
work; main agent owns implementation, final review and scientific decisions.
Record every formal outcome and actual job transition in the project ledgers.

## Repeat alignment correction

The exp-071 common-RNG layout reused random positions across different repeat
indices when methods retained different counts or LHS ranks. Its point MSEs
remain valid, but its paired standard errors and nominal gate are superseded.
Exp-074 checks fixed-capacity repeat blocks for Gaussian probes, LHS fields,
posterior mixture draws and the actual continuation-noise function. Every
comparison uses its total query budget as the random block capacity before
slicing to the retained main count. Rerun the unchanged toy conditions as
exp-071-r2 after these checks; exp-073 has no prior run and retains its locked
configuration. The revised finite-difference smoke already passed and need not
be repeated for this change to random-array layout.

## Final locked decisions

Exp-074 passes all six repeat-stream checks, and exp-071-r2 passes the original
four deterministic M32 toy gates at `0400f4e`. Exp-072-r2 has already passed the
changed critic execution checks. Exp-073 completes the locked discovery, but
zero of its six feasible task/method comparisons passes: Hopper's best paid
setting clears the aggregate threshold on only two of four observations;
HalfCheetah's three feasible settings worsen all four. Close these settings
without P3, endpoint or simulator expansion. Preserve the favorable toy and
candidate-target findings and the reference limitations. No follow-up sweep
is added after observing the failed discovery gate.
