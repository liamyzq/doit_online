# Reproducing runs

See [Compute](../COMPUTE.md) for Orion access, environment details, memory
guidance, and the run policy. This page records the commands for reproducing
committed runs; it does not duplicate the machine inventory.

## Run one config on Orion

Commit the code and config first, then sync the branch as described below. In
an Orion shell (`ssh orion`), create a one-item queue containing the committed
config path and a unique run ID:

```bash
cat > /tmp/one-run.json <<'EOF'
[{"id":"my-run-id","config":"configs/my-config.json"}]
EOF
cd /home/mlw0719/doit_correction
source scripts/compute/orion/env.sh
.venv/bin/python scripts/compute/orion/queue.py \
  --queue /tmp/one-run.json --memory-mib 11000
```

The queue takes a JSON list of items with `id` and `config` fields. It records
the checkout commit at queue start, selects among GPUs 0–4 using free memory,
and finishes after the finite queue completes. For 512-pixel image jobs,
[Compute](../COMPUTE.md) recommends `11000` MiB per job. The queue reserves
2048 MiB per GPU by default and allows two of this project's jobs per GPU.
Existing run output directories are refused.

## Sync committed code

From the local checkout, pass the name of the branch containing the committed
code and configs:

```bash
scripts/compute/orion/sync.sh <branch>
```

Sync only between finite queues so pending jobs keep using the revision recorded
at launch. The script bundles that local branch, transfers it through the
`orion` SSH alias, and fast-forwards the corresponding remote branch. It refuses
to sync when tracked remote changes are present.

## Group image analysis

The grouped analysis manifest names a baseline and maps each comparison group to
one or more run artifact directories. For example,
[analysis-exp-030-expanded.json](../configs/analysis-exp-030-expanded.json)
uses several paired runs in each group. Ensure the manifest paths point to the
run directories containing `run.json`, `config.json`, `images.json`, and the
saved embeddings, then run from the repository root:

```bash
python scripts/analysis/images.py \
  --manifest configs/analysis-exp-030-expanded.json \
  --output artifacts/analysis-exp-030-expanded.json
```

Within each group, all runs must share estimator and image settings; comparison
groups must contain the same prompt/seed pairs. The script reports group means,
paired differences from the baseline, and seed-block intervals when there are at
least four seeds. The intervals condition on the listed prompts and do not
estimate uncertainty over a wider prompt population.

## Nebula sensitivity exploration

Use the isolated Nebula checkout and environment described in
[COMPUTE.md](../COMPUTE.md). The committed endpoint wave executed at `6d1ceca`:

```bash
cd /home/mlw0719/doit_reliability
source scripts/compute/nebula/env.sh
.venv/bin/python scripts/compute/orion/queue.py \
  --queue configs/queue-sensitivity-endpoints.json \
  --memory-mib 13000 --max-per-gpu 1
```

Use new experiment IDs and output directories for an intentional reproduction;
the recorded runs are complete. Do not relaunch them as recovery work.
`scripts/compute/nebula/sync.sh` syncs the committed exploration branch only
between queues. Exp-044 and exp-045 configs specify the preceding controlled
model and frozen-vector diagnostics.

After retrieving complete exp-046 run folders, including images and embeddings,
run the following from the local repository root:

```bash
python scripts/analysis/images.py \
  --manifest configs/analysis-exp-046-sensitivity-endpoints.json \
  --output artifacts/exp-046-summary.json
python scripts/analysis/sensitivity_gallery.py
```

The report retains the exact executed tau-0.3 MC32 configuration; it is not
the historical selected tau-0.6 baseline. Both discovery seeds are shown in
every comparison sheet. Summary JSON and image evidence are copied under
`results/reliability/` for review without access to a worker.

## Run records and artifacts

Each run directory is stored under
`$DOIT_STORAGE/runs/<experiment-id>` and contains the config, run metadata,
metrics, and mode-specific outputs. `run.json` records the commit, config path,
PID, start time, status, and elapsed time. Queue stdout is stored at
`$DOIT_STORAGE/artifacts/<experiment-id>.log`; queue state changes are appended
to `$DOIT_STORAGE/jobs/jobs.jsonl`. Record completed results and their artifact
directories in [experiments/results.tsv](../experiments/results.tsv).
