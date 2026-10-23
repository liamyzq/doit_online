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
