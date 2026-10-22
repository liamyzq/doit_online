# Uncertainty, local correction, and particle correction

Opened on 2026-10-21 from `6f6bfc7`, on branch
`codex/doit-uncertainty-smc`. The user supplied a new derivation and authorized
toy, image, and RL experiments, including evidence-driven combinations. This
authorization supersedes the earlier image-before-RL gate for this exploration.
The earlier studies and their negative results remain valid at their budgets.

## Objective and decision

Find a reproducible improvement at a stated inference budget. A toy mechanism
result is a prerequisite, not evidence of real-model endpoint success. A useful
candidate must improve a prespecified distribution or task metric on fresh
seeds, survive a comparable-cost baseline, and avoid the quality failures
already observed in the image study. We will distinguish a known-method
baseline from a potentially new combination.

The supplied plan separates three errors: approximating the continuation value,
sampling its local tilted kernel, and correcting the complete path distribution.
The first implementation prioritizes uncertainty-aware values and explicit
proposal-density SMC. FORS is evaluated separately with an accurate oracle
before paying its rejection cost on a neural model.

## Fixed target and implementation contract

The target is the distribution of the implemented base sampler tilted by
`exp(beta * terminal_reward)`, with beta fixed across each comparison. SMC
initial weights include the initial potential. Incremental log weights are
`log P - log Q + log psi_new - log psi_old`; the final potential is the actual
terminal reward. Random potential realizations travel with their particles.
Random pilots determine a proposal before an independent propagation draw.
Final output is a weighted sample, never a maximum-reward selection.

Gaussian shift proposals retain an explicit density. Common deterministic
terminal maps receive no Gaussian density correction and no displacement.
Importance weights are not clipped. Finite particle counts do not imply exact
sampling. FORS clipping defines a modified local target and must be measured.

## First experiments

Exp-047 checks the newly implemented identities on Orion: Gaussian conditional
moments and tilted values against analytic formulas; complete path weight
telescoping; density ratios against direct normalized densities; exact Doob
incremental weights; and the FORS Poisson identity. A failure blocks formal
comparisons and leads to fixing the corresponding algebra or state bookkeeping.

Exp-048 uses one-dimensional Gaussian and two-mode Gaussian-mixture forward
noising models with exact reverse kernels and analytic reward-tilted endpoint
distributions. The Gaussian case follows the plan's Gaussian-kernel setting.
The mixture case deliberately uses its exact mixture reverse kernel and its
explicit density, so mode errors are not confounded with a discretized score
model. Compare point, damped point, Gaussian uncertainty, exact mixture, short
branch plus residual Gaussian uncertainty, and full-future values, individually
and with SMC. Report endpoint CDF error, mode mass error, ESS, ancestry, and all
value, reward, and transition evaluations. Oracle access is labeled explicitly.
Use discovery seeds 47011/47017 and reserve 48017/48023/48029/48043 for a locked
confirmation. The first favorable combination needs at least 20% lower primary
CDF mean squared error on confirmation at a comparable measured cost before a
toy success claim.

Exp-049 isolates local nonlinear sampling error using an exact quadratic value.
Compare a Gaussian mean shift, analytic tilted sampling, direct residual
rejection, and clipped FORS at B=0.25/1/2. Measure distribution error, acceptance,
clipping, and guidance calls per accepted sample. This decides whether FORS is
an efficient component for further work; its Poisson identity alone is not a
success criterion.

After these results, lock a bounded real-critic or image experiment using the
existing frozen models and evaluator. Prefer the low-dimensional real critic
for validating uncertainty and explicit density correction. Simulator returns
require a separately validated simulator environment. Reuse real observation
splits, unnormalized observations, and the source action convention. Image
experiments retain prompt alignment, diversity, lower tails, and visual review
alongside aesthetic reward; costs include all pilot, branch, reward, decoder,
score, and backward operations. Existing selected MC32 uses tau=0.6.

## Compute and records

Use a separate Orion checkout and storage for this study, the existing isolated
project Python environment, and the shared finite queue launcher. Physical GPUs
0--4 remain the allowed set. Run one smoke before choosing concurrency, reserve
at least 2048 MiB beyond the measured job allowance, and preserve all older
checkouts and outputs. Commit code and configs before formal runs. Every formal
outcome goes to `experiments/results.tsv`; job transitions go to
`jobs/jobs.jsonl`. Small outputs live in `results/uncertainty_smc/`.

Sources: the user's supplied derivation; [FORS](https://arxiv.org/abs/2602.01338),
[TMPD](https://arxiv.org/abs/2310.06721),
[TDS](https://arxiv.org/abs/2306.17775), and
[DOIT](https://arxiv.org/abs/2602.16198). Gaussian moment correction and twisted
SMC are established baselines, not new contributions by themselves.

## Discovery decision and grouped candidate extension

Exp-047-r2 passed all eight checks after resolving quadrature error in the first
check run. Exp-048 completed 116 conditions. At N=16, Gaussian uncertainty
reduced Gaussian-toy CDF MSE from 0.004953 to 0.004019; the mixture improvement
was smaller, and no stable benefit appeared at N=8. Short branching and full
futures did not repay their larger costs in this discovery. Exp-049 found that
FORS B=2 still had terminal variance about 0.53 versus the exact 0.364 and cost
about 55 guidance calls/output. Direct residual rejection was accurate at about
two residual evaluations/output. Do not scale FORS on the present evidence.

Exp-050 tests a grouped candidate particle update. Given a parent and M base
proposals z_j with positive values v_j, select j with probability v_j/sum(v).
Its incremental weight is mean(v)/psi_parent. Conditional on the candidate
pool, the weighted expectation of f(z_j) is
`sum_j v_j*f(z_j)/(M*psi_parent)`. Averaging over the base candidates gives
the required `integral P(z|x)*psi_child(z)*f(z)/psi_parent`. This is an
extended-space particle correction; no Gaussian density is assigned to the
selected candidate. Store its realized child potential for the next denominator.

This combination avoids fitting and then perturbing a Gaussian shift from a
small pilot pool. Compare point, Gaussian, exact mixture, branch2, and future2
values at M=4/8, N=4/8/16, including bootstrap particle counts matched to the
number of base transition samples. The extension is motivated by observed
pilot cost and finite-pool noise; it is not a claim that candidate selection
alone is new. A finite-pool weight identity is added before formal execution.

## Real-critic discovery contract

Exp-050's grouped update reduces fixed-N error, but bootstrap with more
particles wins at the same number of toy transition samples. This is not an
efficiency success. Gaussian uncertainty still helps some fixed-N comparisons;
branching remains expensive. Continue to real models, whose actor and reward
costs differ from analytic arithmetic, with explicit larger-bootstrap controls.

Exp-051 calibrated finite-sampler residuals using only real observation rows
0/4/8/12, 128 trajectories per row per task. Both new batched adapters match
the validated scalar transition within 1.2e-7. The 15,372 actor evaluations
include this check. Peak reserved memory is 46 MiB. Early residual RMS is about
1.36 on both tasks and includes substantial mean bias; therefore covariance
must be compared to a calibrated-mean-only ablation.

Exp-052 is a four-repeat execution smoke with a small reference, not research
evidence. It detects batched particle shape errors, invalid cached-potential
resampling, nonfinite weights, and actual memory use in the changed paths;
failure blocks exp-053. Exp-053 then locks observation rows 16/20, seeds
53011/53017, 128 independent particle populations per condition, and 32,768
independent base reference paths per state. These are distribution experiments
on frozen real actors/critics, not simulator return evaluations.

The target is the source actor's finite Gaussian chain tilted by the actual
terminal twin-Q minimum at the previously calibrated task temperature.
Observations and terminal actions remain raw. There is no final best-of-N
selection. Primary error is mean squared weighted empirical CDF discrepancy
along coordinate axes and four fixed random projections, at nine base-reference
quantiles per projection. Report split-reference disagreement and ESS to expose
reference uncertainty. Report pooled error separately from per-population MSE.
The Q mean is a secondary target moment; maximizing it is not this objective.

Compare bootstrap N=8/16/32/64, point/calibrated-point/uncertainty SMC N=8/16,
grouped candidates, one conditional Gaussian pilot proposal, and the bounded
branch2-plus-uncertainty combination. Gaussian values average exponential
rewards over two antithetic residual draws. Initial and cached intermediate
potentials remain included; all terminal values are the actual Q/tau. Count
every reference, calibration, pilot, branch, reward, and actor evaluation.
Assess the calibration cost separately and amortized, never silently discard it.

A candidate advances only if the observed CDF benefit is materially larger
than reference disagreement and survives the relevant mean-only and measured
cost controls. Confirmation will use new observation rows from 32 onward and
new seeds, with a locked method and a target of at least 20% lower primary
error at a comparable stated cost. No simulator or image success follows from
an offline critic-distribution result.

Exp-052 passed all 32 execution conditions with exact expected cost counts and
48 MiB peak reserved memory. At the historical proxy temperatures its 512-path
reference ESS was 509--511: the terminal target was almost the base sampler.
Exp-053 therefore remains unlaunched. Exp-054 measured terminal Q variation on
calibration rows only, at a cost of 61,440 actor and 4,096 Q evaluations, and
fixed tau to half the median within-state terminal Q standard deviation.
The resulting temperatures are 0.0311595531 (Hopper) and 0.0736910712
(HalfCheetah). This defines a new, explicitly named target. Exp-055 retains the
same locked methods, discovery rows, seeds, repetitions, and reference budget
at those temperatures. No method uses discovery rewards to select tau. A
reference that is too concentrated to resolve the candidate differences will
be reported as unresolved rather than used to claim an improvement.
