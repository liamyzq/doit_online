# DOIT Correction

Same-rollout variance reduction for training-free diffusion guidance. This research
repository studies online control variates, coupled recursive Monte Carlo, and
importance-corrected adaptive proposals for DOIT's Gaussian moment estimator.

The primary question is whether information from earlier steps reduces current
correction error and improves generated samples at the same actual reward-query
budget. Fitting error alone is not a success criterion.

The active branch is `codex/doit-posterior-geometry`. The latest study tests
Gaussian probe stratification and denoiser-response proxies; see its
[protocol](experiments/posterior_geometry/PLAN.md) and [current state](PROGRESS.md).
It uses a new Orion checkout and retains the previous study as a baseline.

The preceding branch is `codex/doit-uncertainty-smc`. The user opened a separate
study of posterior uncertainty, local FORS sampling and strict particle
correction, with toy, image and RL experiments authorized. Start with its
[protocol](experiments/uncertainty_smc/PLAN.md),
[current findings](experiments/uncertainty_smc/FINDINGS.md), and
[research state](PROGRESS.md). Orion connectivity has recovered; prior jobs
and their crash/completion records were recovered before new execution.

The new toy and frozen real-critic experiments found no cost-controlled gain
from the tested uncertainty combinations. Terminal importance resampling N4
passes the locked image confirmation on eight fresh seeds and eight fixed
prompts: reward +0.17610 versus MC8 (95% seed-block interval [0.00911, 0.30592]),
4 instead of 97 reward queries, and 8.89 instead of 31.30 measured seconds per
image. It uses 80 instead of 20 score forwards and no VJPs. Mean alignment and
embedding distance do not decrease; individual prompt failures remain.
This is an established baseline, not a novel uncertainty method. Shared-GPU
timings and fixed-prompt intervals limit the claim; see the findings for details.
The study uses separate Orion checkout/storage and preserves the older studies.
Small results live under `results/uncertainty_smc/`.

The subsequent [method-development continuation](experiments/uncertainty_smc/CONTINUATION.md)
implemented conditional reward residual calibration and second-moment particle
allocation. Exp-059--062 completed 220 toy conditions, a critic smoke and 12
real-value conditions. Empirical UC improves some toy value estimates, but
cross-observation critic prediction and the second-moment allocation variant
do not establish a new sampling improvement. More frequent resampling helps a
limited toy comparison and is matched by direct learned-value regression.
Both candidates are closed at the tested budgets; no new image or simulator
run is queued. The terminal-N4 baseline remains the best validated result.

Earlier research reopened after an external audit of `c5e4311`; its
[audit follow-up protocol](docs/audit-followup.md) remains available. The
previous `codex/doit-guidance-reliability` branch ran the authorized
[guidance reliability exploration](docs/guidance-reliability.md) on Nebula
during the Orion outage. Those historical findings remain valid.

The bounded Nebula exploration is complete: the fixed-vector FP32 adjoint
checks pass, but neither norm-matched shrinkage nor four-point smoothing shows
a useful endpoint gain in 80 discovery images. Their mean reward differences
from MC8 are +0.00046 and -0.01026; smoothing adds 27 VJPs per image. Both
settings are closed under this budget, with no confirmation wave. The
[report](docs/guidance-reliability.md#endpoint-results-and-closure) contains
the costs, paired results, all prompt comparison sheets, and limitations.

The initial image study is complete. All supplied families and targeted extensions have
documented closure decisions. None established a usable image improvement;
strong EMA and paired future M8 have favorable aggregate results but
visible quality failures. The final comparison contains 512 images, and all
16 focused numerical checks passed on Orion. The image-success condition for
RL scaling was not met, so no RL evaluation was run.

Start with [the final findings](docs/findings.md),
[PROGRESS.md](PROGRESS.md), [the study protocol](docs/study.md), and
[COMPUTE.md](COMPUTE.md). Experimental outcomes are in `experiments/results.tsv`;
job events are in `jobs/jobs.jsonl`. Large artifacts live on the worker recorded
in each result. The prior guidance-reliability branch's small outputs are in `results/reliability/`.
The final aggregate JSON files and report image evidence are included in
[`results/final/`](results/final/) for viewing directly on GitHub.
Commands for running a committed configuration and analyzing paired image
groups are in [the reproduction guide](docs/reproduction.md).
The [estimator note](docs/estimators.md) explains the implemented moment
identities, the biased EMA controls, and the costs of future sampling.

The user authorized physical GPUs 0--4 on Orion and Nebula, including concurrent
use with other users when free memory permits. The latest study explicitly
authorizes RL experiments independently of image success; simulator returns
still require a validated isolated simulator environment.

## Sources

- Zhu et al. (2026), [Training-Free Adaptation of Diffusion Models via Doob's h-Transform](https://arxiv.org/abs/2602.16198).
- [Published RL implementation](https://github.com/liamyzq/Doob_training_free_adaptation).
- The user's supplied research proposal, summarized in `docs/study.md`.
- Source locations and implementation differences are recorded in `docs/sources.md`.

This project fits small online statistical models; the diffusion backbone and
reward model stay frozen. It does not claim that a ratio of unbiased moment
estimates is itself unbiased.
