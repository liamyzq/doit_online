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
