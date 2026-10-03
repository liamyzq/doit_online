# DOIT Correction

Same-rollout variance reduction for training-free diffusion guidance. This research
repository studies online control variates, coupled recursive Monte Carlo, and
importance-corrected adaptive proposals for DOIT's Gaussian moment estimator.

The primary question is whether information from earlier steps reduces current
correction error and improves generated samples at the same actual reward-query
budget. Fitting error alone is not a success criterion.

Start with [PROGRESS.md](PROGRESS.md), [the study protocol](docs/study.md), and
[COMPUTE.md](COMPUTE.md). Experimental outcomes are in `experiments/results.tsv`;
job events are in `jobs/jobs.jsonl`. Large artifacts live on Orion.
Commands for running a committed configuration and analyzing paired image
groups are in [the reproduction guide](docs/reproduction.md).

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
