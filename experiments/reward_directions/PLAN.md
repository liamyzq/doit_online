# Reward-calibrated directions

The 2026-10-23 user audit proposes selecting a rank-one noise direction from
black-box reward moments, then using Gaussian LHS and sign groups to reduce
DOIT correction error. The original correction and outer Gaussian transition
remain the estimands. This study combines that proposal with the validated
four-point construction and common random numbers within future groups.

## Locked implementation

Use all eigenvectors of the actual transition's `J J.T`, plus 16 random unit
vectors from pool seed 81031. Directions live in output/noise coordinates.
M32/M64 spend 8/16 queries on ordinary antithetic pilots. Subtract each pilot's
empirical unweighted moments before scoring `a^2 + b^2/2`, multiplied by
`||J.T u||^2 / sigma^2`. Compare first-moment-only and no-J variants. Split
pilots by complete pairs. Enable the chosen direction only when the two
half-pilot directions have absolute inner product at least 0.8 and whole-pilot
ESS is at least P/4; otherwise use ordinary antithetic main probes. Zero scores
also fall back. These engineering thresholds are fixed for this first round.

Main RNG is independent of pilot RNG. Retain every original pilot in one
jointly normalized ratio for P/P and F/F. P/F uses only full-evaluated main
samples in its final ratio. P defaults to four-point LHS, F to paired LHS;
crossed layouts are ablations. Share future noise only inside each pair or
four-point group, with independent futures between groups. Structured common
seeds pair methods, while fixed-capacity random draws prevent cross-repeat
overlap when main sample counts differ. No reward gradients, clipping, changed
temperature, or anisotropic outer proposals are introduced.

## Checks and experiment sequence

Exp-081 checks the new centered moments, output-space spectral orientation,
known Gaussian tilt, adaptive main marginals, unnormalized numerator and
denominator expectations, and same-evaluator versus P/F ratio construction.
It also checks the Gaussian reference against the actual sixteen-step toy
process. A failure blocks formal comparisons until its cause is corrected.
Previously validated fixed-direction sign maps and old sampler adapters are
not rechecked unless changed.

Exp-082 uses a six-dimensional rotated anisotropic Gaussian and a two-component
mixture, each with a sixteen-step reverse-VE Euler Gaussian transition. Three
noise indices (2,7,12), four fixed states per index including a symmetric
center, and 512 complete pilot/main repeats cover M32/M64. Shift, symmetric
narrow, discontinuous two-interval, constant and fourth-Hermite rewards test
first-moment signals, second moments and blind spots. The in-pool control uses
the mixture eigenaxis; the off-pool control mixes that axis with a distant
spectral axis. Gaussian references use exact recursion for this discrete
process. Mixture references start with eight independent blocks of 4096
samples; the ideal posterior is never used as their full-continuation truth.
The fourth-Hermite moment-preservation claim applies only to the Gaussian
model; it is a non-Gaussian stress condition on the mixture.

Controls are full-budget antithetic, a random direction with the same layout,
the largest model direction, first-only selection, selection without J weights,
and a known direction mechanism control on toys. On Gaussian off-pool cases
the oracle is the route-specific terminal reward axis pulled through the
actual linear map. On mixture off-pool cases its nominal axis is only a
structural comparator, not an oracle optimum. Symmetric-center zero-guidance
conditions are diagnostic controls, not opportunities for MSE improvement.

Exp-083 will smoke the frozen Hopper and HalfCheetah actors and black-box
critics on Orion, recording memory and cost before concurrency is set.
Exp-084 development uses 16 observations per task, selected by seed83041 from
episodes disjoint from all previous 64 selections. Three active indices
(3,5,7) and 256 repeats per context use fixed tau, eta and guidance intervals.
An additional 32 fresh episodes are reserved for confirmation. Preserve the
old observations and model source. Reference evaluations do not select method
directions. Record all data and model provenance.

Primary metrics are displacement MSE after the actual timestep gain, with raw
guidance MSE, bias/variance, zero guidance, fallback frequency, pilot stability,
ESS and independent future group counts. Report P versus P, P versus F, and F
versus F separately. Save pilot/main eps, log rewards, final weights, selected
directions, J, pool, corrections, reference blocks and cost counters. Report
observation-level paired uncertainty, not MC repeats as independent data.

The fixed-state run artifacts store the kernel gain at unit gamma. Final
analysis scales displacement by the preserved task strengths (Hopper0.75,
HalfCheetah0.25), verified from the source before inspecting development
results. This constant rescaling changes absolute units but no within-task
ranking or interval sign. The study retains its already calibrated fixed tau
(Hopper0.031159553132296444, HalfCheetah0.07369107116716792). The source's
per-batch Q standardization is not part of this fixed-integral experiment.
See [source settings](SOURCE_NOTES.md) for endpoint and simulator provenance.

## Decisions and subsequent work

First establish whether oracle directions improve the matched estimator, then
whether small pilots retain that benefit. Toy failure does not by itself veto
the explicitly requested real-critic development comparison. After development,
lock the simplest supported variant for each route. Confirmation requires a
positive mean displacement improvement over both antithetic and its matched
random-direction control, an observation-block 95% interval excluding zero,
and the same sign under reference-block replacements. Report reference-limited
cases without removing them; increase reference budget only when its noise
prevents the locked comparison from being interpreted.

A proxy candidate can proceed to a bounded complete-action critic-score test
independently of its P-versus-F error. This test uses 128 actions per confirmation
observation with fixed action-selection policy, reporting mean/tail score,
diversity, time and all costs. Statistical gains do not establish computational
gains: J acquisition is charged, and controls may spend the measured equivalent
cost on more probes. Closed-loop simulator returns require a working isolated
environment and an endpoint result that justifies that next stage. Critic score
is never reported as environment return. Failed variants close only under the
named budgets and task settings.

All implementation/configuration is committed before formal runs. Use the
project finite queue in a separate Orion checkout/storage; preserve previous
execution checkouts. Record every completion, crash and invalid run in the
existing ledgers before interpretation.

## Bounded development follow-up: exp-085

The completed Gaussian in-pool records show useful P-route direction information
at M64: mean selected-axis capture0.7219 and exact-axis selection0.6278 for the
shift reward, but only0.1525 of repeats pass the split-agreement gate. Full-route
capture remains near the six-dimensional chance level. The real-critic pilot
methods mostly fall back, while simple structured controls can match or exceed
their error. This makes the gate a concrete unresolved mechanism on P.

Exp-085 removes only the split-agreement threshold, setting it to0. The ESS
threshold and zero-score fallback remain. Test the same four paid P variants
(first, no-J, moments pair, moments four) at the same M32/M64, states and seeds
on both in-pool toys and both16-observation development tasks. Reuse all
independent references and existing full-budget controls from their original
run artifacts; the runner rejects target/state/seed/budget differences. Compare
the new estimates with both controls and the original gate on paired repeats.
No new full-continuation tuning or threshold grid is introduced. Select between
these two development gate settings before touching the32-observation holdout.
Any apparent paid-direction gain must also compete with the cheaper random
paired control, even if the layout-matched random-four control is weaker.

## Locked confirmation: exp-086

Exp-085 completes successfully. Removing the agreement gate raises Gaussian
in-pool M64 shift/narrow moments-four proxy gains to31.22%/34.70% against
antithetic, versus27.82%/30.17% against random pairs. It does not make the
Hopper paid methods competitive. HalfCheetah M64 moments-pair is the only
paid real candidate with a favorable mean against the stronger random-pair
control:5.41% versus antithetic and1.52% versus random pairs, with the latter
observation interval still including zero. Its four-point counterpart worsens.

Lock one paid candidate: HalfCheetah P moments-pair at M64, agreement threshold0,
retaining ESS/zero-score fallback. Confirmation compares it with antithetic,
random pairs and random four-point, and requires the original positive-mean,
observation-interval and reference-block rules against both antithetic and
random pairs. Failure closes this paid setting at this budget.

Also confirm the useful free controls: Hopper P random pairs and F random
four-point at M32, and HalfCheetah P random pairs at M64. Hopper M32 is chosen
as the smaller development budget already supporting both routes. A random-pair
control is tested against antithetic; a random-four candidate must beat both
antithetic and random pairs. These are gains of the free sampling construction,
not evidence that reward-based direction selection helps.

Use all reserved rows16--47 (32 disjoint episodes per task), three active indices
3/5/7,256 repeats, and eight independent4096-sample blocks per reference.
Lock path/estimator/reference seeds86001/86011/86007 before running. No excluded
low-ESS observations, additional parameter search or reuse of these observations
for tuning follows. A confirmed proxy construction proceeds to the planned
complete-action evaluation; full-only success is reported at its own estimand.

## Complete-action implementation and exp-087 smoke

The endpoint sampler generates independent action states in batches and preserves
the original transition mean, timestep gain, guidance interval and gamma. Free
controls use one original kernel VJP per state; selected directions compute the
full small Jacobian and pay its recorded cost. The established estimator now
accepts either a single fixed-state pool/J or one pool/J per independent action.
Its scalar path retains the same operations and all pilot/main retention rules.

Exp-087 uses development row0 of both tasks. It checks batched transition means
and Jacobians against scalar contexts, per-state pilot scores/selections against
individual calls, single-VJP mapping against the Jacobian, and repeated-state
P/F estimates against the established scalar context. Any discrepancy above
2e-4 blocks endpoint runs. It also generates128 complete actions per method at the intended
endpoint batch size for all tested layouts and checks exact reward and actor-forward counts. This detects new
batch coupling, orientation, mapping and cost errors without repeating settled
fixed-state construction checks.

Complete-action records save every transition's probes, rewards, weights and
correction, plus trajectories, final actions and raw Q. Final selection groups
use preserved best-of8 for Hopper and best-of4 for HalfCheetah. Report both raw
and selected-action mean/lower-decile Q and mean pairwise action distance.
Timing distinguishes computation from artifact writing, and actual costs
include smoke/warm-up evaluations. Scores are frozen-critic endpoints, not
closed-loop returns.

The complete-action primary comparison is selected mean Q after the fixed
best-of policy, using paired observation bootstrap intervals. Raw mean Q, both
lower deciles and action diversity are retained to judge whether any mean gain
comes with a material quality or diversity loss. No parameter choice is made
from endpoint results. A simulator extension requires an endpoint gain that
survives its relevant controls and warrants installing the preserved environment.

## Confirmation outcomes and locked endpoints

Exp-086 confirms HalfCheetah M64 P random pairs against antithetic:4.984%
proxy MSE reduction,27/32 improving observations, paired95% interval
[-0.005790,-0.001080], and eight favorable reference-block signs. Moments-pair
improves7.873% against antithetic and3.041% against random pairs, but the latter
interval[-0.004518,0.000692] includes zero. It does not pass the locked paid-method
confirmation rule and receives no further tuning or endpoint extension.

Hopper M32 F random four-point confirms11.415%/11.611% reductions against
antithetic/random pairs, with intervals[-0.096653,-0.010026] and
[-0.099264,-0.008923] and eight favorable reference-block signs. Improvements
occur in16/32 and13/32 observations, respectively; the aggregate gain remains
heterogeneous. Hopper P random pairs improve0.688% against antithetic but its
interval includes zero, so that selected proxy setting does not advance.

Exp-088 tests complete actions for exactly these confirmed constructions:
Hopper F M32 with antithetic/random-pair/random-four controls, and HalfCheetah
P M64 with antithetic/random-pair controls. The full route retains its own
continuation estimand and actor costs. Use128 actions on each reserved
observation, fixed source gamma/selection, and path/estimator seeds88001/88011.
This endpoint test follows successful exp-087 batch/cost checks and its measured
128-action peak memory. The two route/task decisions are assessed separately;
there is no claim that reward-based selection itself is confirmed.

## Closure

All planned checks, development, bounded gate ablation, fresh confirmation and
conditional complete-action tests are complete. Neither exp-088 construction
establishes an improvement in the locked selected-mean-Q endpoint against its required controls.
Hopper's favorable raw mean-Q result is retained as a secondary endpoint.
No selected-mean threshold, action-selection rule or method is changed after
these results. The conditional simulator stage does not proceed. See the
final findings for effects, intervals, costs, limitations and artifact paths.
