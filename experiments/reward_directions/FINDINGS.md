# Reward-calibrated direction experiments

Development is complete for exp-082/084. The single P-route gate ablation
exp-085 is running. The32-observation confirmation sets remain untouched.
These findings concern the fixed temperatures, query budgets and frozen
actors/critics in [the protocol](PLAN.md), not all possible direction estimators.

## What the checks establish

Exp-081 passes31 new construction, ratio-retention and discrete-reference
checks. Exp-083 passes the mixed pair/four-point future adapter and exact
reward/actor count checks on both real tasks. Main samples have fresh randomness;
all same-evaluator pilot samples enter one final ratio. Different sign groups
have independent futures. Gaussian toy truth is for the implemented16-step
process. Mixture truth uses eight independent4096-sample reference blocks.
The implementation is correct under these checks.

## Mechanism and acquisition cost

The known-direction four-point mechanism works. At M64, Gaussian in-pool
shift/narrow proxy displacement MSE falls91.02%/98.45% against antithetic.
The same directions reduce full-continuation MSE42.71%/34.03% when futures are
shared within each group. Mixture in-pool shift/narrow reductions are
92.95%/91.60% for proxy and50.15%/52.76% for full continuation. Off-pool Gaussian
oracle shift/narrow reductions remain90.67%/97.31% for proxy and55.89%/30.09%
for full continuation. The mixture off-pool axis is a structural comparator,
not an exact oracle optimum.

Random four-point directions often increase error, despite correct Gaussian
marginals. For example, Gaussian in-pool M64 shift proxy/full error rises
44.22%/69.07%. Correct direction information therefore matters; four-point
sampling alone is not a universal improvement.

Small pilots do not retain most of the known-direction benefit. Gaussian
in-pool M64 shift proxy moments select the exact axis in62.78% of repeats,
with mean squared alignment0.7219, but enable it only15.25% of the time.
The resulting four-point gain is6.04%, versus4.71% for free random paired LHS.
Full-continuation pilot alignment remains near chance and falls back about94%
of the time in this condition. The P/F route pays for a proxy pilot that
cannot enter the full ratio, and typically worsens full error by roughly30--40%
on these toys. Centered constant and zero-guidance cases are retained as
controls, without ranking numerical roundoff as an improvement.

This concrete proxy-gate diagnosis motivates exp-085: remove only the0.8
split-agreement threshold, retaining ESS and zero-score fallback. It reuses
independent references and original full-budget controls on the same development
states and seeds. No full-continuation threshold search is justified by its
near-chance direction information.

## Real development observations

Each task has16 observations from episodes excluded from every previous study,
three active noise indices per observation and256 complete pilot/main repeats.
Intervals below resample observations after averaging their three indices.
The table reports displacement MSE reduction against full-budget antithetic;
positive percentages mean lower error. Absolute MSE uses preserved deployment
gamma0.75 for Hopper and0.25 for HalfCheetah.

| Task and route | Method | M32 reduction | M64 reduction | Development interpretation |
|---|---|---:|---:|---|
| Hopper proxy | Random paired LHS |4.08%|4.40%|Intervals exclude zero;13/16 and14/16 observations improve|
| Hopper full | Random four-point LHS |12.94%|13.41%|Intervals exclude zero; only8/16 and7/16 observations improve|
| HalfCheetah proxy | Random paired LHS |2.21%|3.95%|M32 unresolved; M64 interval excludes zero and14/16 improve|
| HalfCheetah proxy | Gated moments four-point |-0.25%|3.80%|M64 improves over antithetic but does not beat the cheaper random pair|
| HalfCheetah full | Random four-point LHS |0.24%|-2.41%|Intervals include zero and15/16 observations worsen at both budgets|

All eight reference-block replacements retain the favorable sign for Hopper
proxy random pairs, Hopper full random four-point, and HalfCheetah M64 proxy
random pairs. Hopper full improvement is highly heterogeneous and needs fresh
confirmation. The HalfCheetah full M32 positive aggregate is driven by one
observation and supplies no useful broad evidence.

Paid moments do not currently establish a real-task advantage over the cheaper
structured controls. Hopper full moments pair improves only0.26%/0.63%, with
intervals crossing zero. Its moments four-point method is13.7--14.8% worse
than the free random-four comparator. HalfCheetah proxy M64 moments four-point
improves3.80%, while free random pairs improve3.95%.

Proxy-to-proxy and proxy-to-full results are distinct. Hopper M32 random-pair
proxy error improves4.08%, but error against the full target worsens1.77%
with an interval crossing zero. HalfCheetah M64 random pairs improve proxy
error3.95% and full-target error1.61%. Complete-action evaluation, if advanced,
will retain the fixed target and report critic score separately from returns.

Reference concentration remains material: Hopper has3/48 proxy and12/48 full
contexts with pooled ESS below10; HalfCheetah has0/48 and2/48. No observation
is removed. Reference-block stability supports the reported pairwise signs,
but does not prove that the pooled finite reference equals the exact target.

## Costs and artifacts

All six first-wave jobs completed at `0e4e9f3`. Real-task peak GPU reservations
were78/80MiB, and toy GPU allocation was zero. All pilots, proxy denoising,
remaining-future actor evaluations and reference costs are included in the
recorded counters. Actual fixed-state evaluation computes each context Jacobian
once; deployment would pay for it at every new state. No speedup is inferred
from that amortized fixed-state timing.

Small JSON records and aggregate estimates are under
`results/reward_directions/exp-082-*` and `exp-084-*`. Analyses are under
`results/reward_directions/analysis/`. Complete eps, rewards, weights, selected
directions, Jacobians, group IDs and per-condition estimates remain on Orion
under `/home/mlw0719/doit_reward_directions_storage/runs/<experiment>/raw/`.
The result ledger and job ledger retain exact revision, costs and terminal state.
No simulator return has been measured. The preserved Python3.12 environment
has no compatible Gym/D4RL/MuJoCo stack installed.
