# Compute

## Authorized machine

- SSH alias: `orion`, user `mlw0719`; it connects through Quest.
- GPUs: physical IDs 0--4, five NVIDIA RTX A6000 cards with 49,140 MiB each.
- Concurrent use with other users is explicitly authorized, subject to available
  memory. Do not kill, pause, change priority, or otherwise alter their jobs.
- Initial free-memory snapshot: approximately 12, 33, 30, 30, 12 GiB across IDs
  0--4. Utilization was 90--100%; refresh memory readings before launching.
- Remote checkout: `/home/mlw0719/doit_correction`.
- Artifacts and model cache: `/home/mlw0719/doit_correction_storage`.
- Python: `/home/mlw0719/doit_correction/.venv/bin/python`.
- HF_HOME: `/home/mlw0719/doit_correction_storage/huggingface`.

## Environment

The isolated Python 3.12.9 venv uses base PyTorch 2.6.0+cu124 through system site
packages. Installed in the venv: torchvision 0.21.0+cu124, numpy 1.26.4,
diffusers 0.32.2, transformers 4.49.0, accelerate 1.4.0,
huggingface_hub 0.36.0, safetensors 0.8.0, Pillow 12.3.0, scipy 1.17.1.
Imports and `pip check` passed. Base packages were not modified.

SD v1.5 snapshot: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
CLIP ViT-L/14 snapshot: `32bd64288804d66eefd0ccbe215aa642df71cc41`.
Both are in the project's Hugging Face hub cache. The aesthetic head is
`/home/mlw0719/doit_correction_storage/models/sa_0_4_vit_l_14_linear.pth`.
Setup and download logs are in storage as `setup.log` and `download.log`.

## Launch policy

Use `scripts/compute/orion/env.sh` and `queue.py`. Run one image smoke first and
measure peak allocated/reserved memory. Set the per-job reservation from that
measurement, with a margin, and retain at least 2 GiB of free memory per device.
The queue handles only this project's jobs. Each queue is finite. Refresh
memory readings while waiting; high utilization is expected on this shared
machine. Report wall-clock under this contention rather than treating it as a
dedicated-device benchmark.

Code and configs must be committed before formal launch. Sync committed code
using `scripts/compute/orion/sync.sh`. Preserve config path, commit, seed, GPU,
PID, and artifact directory in job and result records.
Sync only between finite queues; pending jobs must not import code from a later
revision than the queue's recorded commit. The queue writes its event ledger
to storage `jobs/jobs.jsonl`; pull those events into this repository's ledger
when recording each completed wave.

No image or RL training is authorized on the Quest login nodes. Quest and
Nebula are reference-source locations for this project, not experiment workers.
