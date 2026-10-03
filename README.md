# DOIT Correction

Same-rollout variance reduction for training-free diffusion guidance. This research
repository studies online control variates, coupled recursive Monte Carlo, and
importance-corrected adaptive proposals for DOIT's Gaussian moment estimator.

The primary question is whether information from earlier steps reduces current
correction error and improves generated samples at the same actual reward-query
budget. Fitting error alone is not a success criterion.

Research has reopened following an external audit of `c5e4311`. See the
[audit follow-up protocol](docs/audit-followup.md) and current
[research state](PROGRESS.md) for the active questions and new seed split.

The initial image study is complete. All supplied families and targeted extensions have
documented closure decisions. None established a usable image improvement;
strong EMA and paired future M8 have favorable aggregate results but
visible quality failures. The final comparison contains 512 images, and all
16 focused numerical checks passed on Orion. The image-success condition for
RL scaling was not met, so no RL evaluation was run.

Start with [the final findings](docs/findings.md),
[PROGRESS.md](PROGRESS.md), [the study protocol](docs/study.md), and
[COMPUTE.md](COMPUTE.md). Experimental outcomes are in `experiments/results.tsv`;
job events are in `jobs/jobs.jsonl`. Large artifacts live on Orion.
The final aggregate JSON files and report image evidence are included in
[`results/final/`](results/final/) for viewing directly on GitHub.
Commands for running a committed configuration and analyzing paired image
groups are in [the reproduction guide](docs/reproduction.md).
The [estimator note](docs/estimators.md) explains the implemented moment
identities, the biased EMA controls, and the costs of future sampling.

The user authorized Orion GPUs 0--4, including concurrent use with other users
when free memory permits. Image correctness and effectiveness precede RL scaling.

## Sources

- Zhu et al. (2026), [Training-Free Adaptation of Diffusion Models via Doob's h-Transform](https://arxiv.org/abs/2602.16198).
- [Published RL implementation](https://github.com/liamyzq/Doob_training_free_adaptation).
- The user's supplied research proposal, summarized in `docs/study.md`.
- Source locations and implementation differences are recorded in `docs/sources.md`.

This project fits small online statistical models; the diffusion backbone and
reward model stay frozen. It does not claim that a ratio of unbiased moment
estimates is itself unbiased.
