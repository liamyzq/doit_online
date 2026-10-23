# Reward-calibrated directions: final findings

The audit-directed round is complete. Two free sampling constructions reduce
fixed-state correction MSE on fresh observations. Neither establishes a gain in the primary
complete-action score after the preserved best-of selection policy. Paid
reward-moment direction selection has a favorable but unresolved real-task
advantage over its stronger cheap control. Close this round at its stated
budgets without a simulator-return extension.

All 17 formal jobs completed successfully. Implementation checks pass: 31 new
construction/reference checks, 10 real-critic adapter/cost checks, and 26
complete-action batch checks. The [protocol](PLAN.md) records the single
bounded development gate ablation and the choices locked before confirmation.
No confirmation observation was used for parameter tuning.

## Fresh-observation correction error

Each task uses 32 reserved observations from episodes excluded from all earlier
studies. Every observation has three active noise indices and 256 complete
pilot/main repeats. Intervals resample observations after averaging their
three indices. Positive percentage reductions mean lower displacement MSE.
Absolute differences include the preserved task gamma, 0.75 for Hopper and
0.25 for HalfCheetah.

| Locked comparison | MSE reduction | 95% interval for MSE difference | Observations improving | Decision |
|---|---:|---|---:|---|
| HalfCheetah P64 random pairs vs antithetic | 4.984% | [-0.005790, -0.001080] | 27/32 | Confirmed |
| HalfCheetah P64 moments pair vs antithetic | 7.873% | [-0.009429, -0.001577] | 29/32 | Favorable, insufficient alone |
| HalfCheetah P64 moments pair vs random pairs | 3.041% | [-0.004518, 0.000692] | 22/32 | Not confirmed |
| Hopper P32 random pairs vs antithetic | 0.688% | [-0.018015, 0.004264] | 28/32 | Not confirmed |
| Hopper F32 random four-point vs antithetic | 11.415% | [-0.096653, -0.010026] | 16/32 | Confirmed, heterogeneous |
| Hopper F32 random four-point vs random pairs | 11.611% | [-0.099264, -0.008923] | 13/32 | Confirmed, heterogeneous |

All eight independent reference-block replacements favor both confirmed
constructions. They also favor the paid HalfCheetah pair, whose observation
interval still includes zero against random pairs. Its small favorable mean
remains unresolved at this budget; no additional direction or gate tuning
follows. The Hopper full benefit is an aggregate effect concentrated in part
of the observation set.

HalfCheetah random pairs also reduce error against the full target by 1.321%,
with interval [-0.009405, -0.000036]. The primary target remains its proxy.
P denotes fresh-candidate denoising with fixed task temperature; F denotes
continuation through the remaining base sampler. The preserved practical
source uses a cached-score proxy and per-batch Q standardization. This round
compares the audit's fixed integrals; [source settings](SOURCE_NOTES.md) record
that distinction and the unchanged transition/gamma/selection settings.

References remain finite. In confirmation, Hopper has 6/96 P and 10/96 F
contexts with pooled reference ESS below 10; HalfCheetah has none in either
route. All observations remain included. Block-sign stability supports the
reported pairwise comparisons but does not identify the exact infinite-sample
reference. Observation uncertainty is the unresolved limitation for the paid
HalfCheetah candidate.

## Complete actions

Exp-088 generates 128 complete actions per observation and method: 12,288
Hopper actions across three methods and 8,192 HalfCheetah actions across two.
The fixed selection policy chooses the best of 8 Hopper candidates or 4
HalfCheetah candidates. The predeclared primary endpoint is selected mean Q.
Intervals are paired observation bootstrap intervals. Q is the frozen critic's
score; environment returns were not measured.

| Primary endpoint comparison | Selected mean-Q difference | 95% interval |
|---|---:|---|
| Hopper F32 random four-point vs antithetic | -0.000175 | [-0.004999, 0.003005] |
| Hopper F32 random four-point vs random pairs | 0.000370 | [-0.002020, 0.002713] |
| HalfCheetah P64 random pairs vs antithetic | -0.000805 | [-0.008003, 0.005026] |

Hopper has a real favorable secondary result before selection: random four-point
raises raw mean Q by 0.007538 against antithetic, interval [0.002241, 0.014208],
and by 0.007245 against random pairs, interval [0.001683, 0.014480]. Selected
lower-decile Q improves by 0.004596 against antithetic, but its comparison with
random pairs remains unresolved. The primary best-of-8 mean does not improve
clearly against either control. Raw action diversity falls 0.003977 against
random pairs, about 1.38%; selected-action diversity differences are unresolved.
These secondary findings are retained without changing the primary endpoint.

HalfCheetah's raw mean-Q difference is -0.005693, interval
[-0.021191, 0.006176]. Selected lower-decile Q differs by -0.006359, interval
[-0.024411, 0.009831], and selected action diversity by -0.003559, interval
[-0.015605, 0.008143]. There is no established final score or diversity gain.

Within each task/route, methods use identical reward and actor budgets and one
original kernel VJP per guided action state. Hopper uses 1,052,672 reward and
10,022,912 actor point evaluations per method; HalfCheetah uses 2,625,536 and
2,682,880. Mean computation time per 128 actions is 3.559 seconds for Hopper
four-point versus 3.604/3.555 for antithetic/random pairs, and 1.728 seconds for
HalfCheetah random pairs versus 1.768 for antithetic. Paired time intervals
include zero. Artifact-writing time is recorded separately. Shared-GPU timing
does not establish a speedup.

The selected-mean endpoint criterion fails for both constructions. The planned
simulator extension therefore does not proceed. The confirmed local MSE gains
and Hopper raw-score gain remain valid findings at their own endpoints.

## Mechanism and the bounded gate ablation

The known-direction four-point mechanism works strongly. At M64, Gaussian
in-pool shift/narrow proxy MSE falls 91.02%/98.45% against antithetic. Full
continuation with group-shared futures improves 42.71%/34.03%. Mixture in-pool
proxy improvements are 92.95%/91.60%, and full improvements 50.15%/52.76%.
Off-pool Gaussian oracle shift/narrow improvements are 90.67%/97.31% for proxy
and 55.89%/30.09% for full. The mixture off-pool axis is a structural comparator,
not an exact oracle optimum. Constant and centered zero-guidance controls are
retained without ranking numerical roundoff as improvement.

Random four-point directions often increase error. For Gaussian in-pool M64
shift, proxy/full MSE rises 44.22%/69.07%. Direction information matters even
when every probe has the correct Gaussian marginal.

The original split-agreement gate suppresses useful proxy information. For
Gaussian M64 shift, the pilot selects the exact axis in 62.78% of repeats,
with mean squared alignment 0.7219, but enables it only 15.25% of the time.
Exp-085 removes only that gate and retains ESS and zero-score fallback. Paid
moments-four then improves Gaussian shift/narrow proxy MSE 31.22%/34.70% against
antithetic and 27.82%/30.17% against random pairs. Mixture improvements against
antithetic are 20.35%/7.61%.

The real-task four-point benefit does not transfer. All Hopper paid variants
remain worse than random pairs in development. HalfCheetah M32 moments-four
worsens 14.03% against antithetic, with all 16 observations worse; M64 worsens
0.81%. HalfCheetah M64 moments-pair is the sole paid candidate with a favorable
development mean against random pairs, which motivates its single reserved
confirmation. Its confirmation outcome is reported above.

Full-route pilot direction capture stays near chance on the Gaussian mechanism
conditions. P/F pays for a proxy pilot that cannot enter the full ratio and
typically worsens toy full error by roughly 30--40%. No full-route threshold
search follows. Negative findings apply to these tasks and M32/M64 acquisition
budgets; they do not establish impossibility for other direction estimators.

## Implementation, costs and artifacts

The original DOIT correction and outer transition are preserved. Main randomness
is independent of the pilot. Every same-evaluator pilot is retained in one
joint ratio; P/F excludes the proxy pilot from full weights. Futures are shared
only inside each pair/four-point group. No critic gradient is taken. All
acquisition, continuation, reference and smoke evaluations are counted.

The complete-action smoke passes all 26 new checks. Batched direction choices
and moment scores match the scalar calls exactly. The largest repeated-state
estimator discrepancy is 1.888e-4 relative to max(1, target magnitude), below
the predeclared 2e-4 tolerance. Exact reward and actor costs match all eight
smoke conditions. Peak GPU reservation is 90 MiB in smoke and 82 MiB for each
formal endpoint. Free endpoint methods pay one kernel VJP per action state;
fixed-state timing that amortizes J acquisition is never treated as deployment
efficiency for paid methods. VJP/actor counts use state/point evaluations, with
batch execution time measured directly.

The 17 jobs cover 16,944 estimator/endpoint conditions and 21,504 complete
actions including smoke. Real-model runs use 90,600,704 reward queries and
392,079,454 actor point evaluations. Toy and synthetic evaluations are recorded
separately in `results/reward_directions/analysis/run-costs.json`.

All top-level records, aggregate fixed-state estimates, logs and final action
bundles are local under `results/reward_directions/`. Analyses are in its
`analysis/` directory. Complete per-condition probe arrays, rewards, weights,
selected directions, Jacobians, group IDs and reference blocks remain on Orion
under `/home/mlw0719/doit_reward_directions_storage/runs/<experiment>/raw/`.
The complete exp-087 smoke raw tree is also local. Detailed result and job
ledgers record execution revisions and costs. All finite controllers exited;
no further job is queued. The remote execution checkout remains at `41dccad`;
later local commits contain records and interpretation.
