# Research instructions

Use English for files and detailed English commit messages explaining what
changed and why. Commit implementation and experiment configs before formal
runs. Run representative checks on Orion. Before a check, identify the specific
failure it detects and how that would change the next action.

Use only physical GPUs 0--4 on Orion or the user-authorized Nebula exploration
worker. Stacking with other users is authorized according to
free memory. Never stop or modify other users' jobs. Measure peak memory in a
smoke run before setting concurrency. Use the project launcher and isolated
environment; preserve the old DOIT repositories.

Read `COMPUTE.md`, `PROGRESS.md`, and the relevant experiment config before work.
Record every formal result, crash, and invalid run in `experiments/results.tsv`
and job state changes in `jobs/jobs.jsonl`. Preserve actual query costs,
including warm-up, refresh, both SPIDER terms, and future-simulation evaluations.

The main agent owns scientific decisions and result-defining implementation.
Delegate established execution, bounded inspections, and healthy job monitoring.

Keep scope focused on this research: no speculative hardening, compatibility
frameworks, feature flags, or content hashing. Do not repeat settled checks.
Say plainly when a result is correct or a method fails. Do not equate negative
results under a named budget with a universal impossibility result.
