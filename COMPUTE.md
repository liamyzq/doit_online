# Compute

## Reward-calibrated directions

The 2026-10-23 round uses branch `codex/doit-reward-directions`, Orion checkout
`/home/mlw0719/doit_reward_directions`, and storage
`/home/mlw0719/doit_reward_directions_storage`. Use
`scripts/compute/reward_directions/{env.sh,sync.sh}` and the existing finite
queue. The isolated environment and model cache are symlinks to the preserved
original study environment/cache. No old execution checkout was changed.
Exp-081 completed31 checks at `16c00b5` with0/0MiB peak allocated/reserved GPU
memory; its finite controller exited and records were retrieved. It used a
512MiB allowance plus2048MiB reserve on physical GPU4. Exp-083 passed the new
mixed-group adapter and cost checks on both tasks at `f73ed2e`, measuring
42.31/48MiB peak allocated/reserved memory. Retain4096MiB plus2048MiB reserve
and at most one project worker per physical GPU0--4 for development. The
separate CPU toys allocated no GPU tensors in exp-081. All exp-083 raw
artifacts were retrieved and its controller exited before the next sync.
The new observation split excludes all 64 episodes per task used by previous
studies and writes 16 development plus 32 confirmation observations per task
under the new storage's `data/critic-states/`. Selection seed is83041; models,
temperatures, eta and active guidance intervals remain fixed.

All six exp-082/084 jobs completed at `0e4e9f3`; the finite controller exited.
Real-task peak reserved memory was78/80MiB; toys used0MiB. Small records,
aggregate estimates and logs are local; approximately13GB of raw per-condition
NPZ artifacts remain in the recorded Orion run directories. Exp-085 retains
the4096MiB allowance plus2048MiB reserve and one worker per physical GPU.
It reuses exp-082/084 references without repeating their evaluation costs.
All four exp-085 jobs completed at `98d4b25` and controller1148466 exited.
Their compact records, estimates and logs are local; raw NPZ remains remote.
Both real jobs reserved76MiB peak. Exp-086 uses the same memory allowance
and concurrency, with the committed fresh-observation confirmation configs.
Both exp-086 jobs completed at `74a99b8`, with78MiB peak reserved memory,
and controller1662717 exited. Compact artifacts, aggregate estimates and logs
are local; raw records remain remote. Exp-087 measured the128-action batch
and checked the new complete-action path before exp-088 launched.
Exp-087 passed all26 new batch checks at `41dccad`, with69.516/90MiB peak
allocated/reserved memory at128 actions. Full smoke artifacts and logs are local;
controller2483750 exited. Both exp-088 endpoints completed at `41dccad`,
with82MiB peak reserved memory, using4096MiB plus2048MiB reserve and one
worker per physical GPU. Controller2567660 exited. All compact records and
96 Hopper/64 HalfCheetah final action bundles are local; step NPZs remain
remote. Neither selected-mean endpoint establishes a gain, so no simulator
extension or new worker is queued. The execution checkout remains at41dccad;
later local commits record the results and analysis.

## Posterior geometry exploration

The new branch `codex/doit-posterior-geometry` uses separate Orion checkout
`/home/mlw0719/doit_geometry` and storage `/home/mlw0719/doit_geometry_storage`,
with `scripts/compute/geometry/{env.sh,sync.sh}` and the existing finite queue.
Share the preserved isolated environment and model cache by symlink. Preserve
earlier execution checkouts. Exp-063/064 use CPU analytic code without GPU
tensors. Exp-065/066 critic smoke/discovery peaks were 46/88 MiB reserved.
The exp-067 512-pixel FP32 image smoke passed its finite-difference check with
16,288 MiB peak reserved memory. Use an 18,336 MiB allowance for exp-068,
physical GPUs 0--4, one project job per device and 2,048 MiB free reserve.
Exp-068 completed at `a48c54f` with 16,284 MiB peak reserved memory and all
PNG/vector artifacts retrieved. Exp-069 completed at `ccf61a6` with 90 MiB
peak reserved memory under a 4096 MiB allowance plus 2048 MiB reserve. All
geometry finite controllers exited; no worker or pending job remains. The
remote execution checkout stays at `ccf61a6`; later local commits record the
analysis. No simulator environment was installed or return evaluation run.

## Complement cancellation follow-up

Exp-070/071 completed at `9ca5f7f` with zero GPU tensor allocation; all16
construction/replay checks passed. Exp-072 at `4c21ae3` failed its FP32
finite-difference validation and its complete failure record is retained.
Exp-072-r2 at `8a4cd08` passed gradient and coupling checks, with40.77/46MiB
peak allocated/reserved memory. Critic runs used a4096MiB allowance plus2048MiB
reserve, with one project job per physical GPU0--4.

After a common-RNG repeat-alignment correction at `0400f4e`, exp-074 passed
all six focused stream checks (maximum discrepancy8.88e-16,2MiB reserved).
Exp-071-r2 completed320 conditions with5,963,776 reward evaluations and zero
GPU allocation; the corrected toy gate passed. Exp-073 then completed24
contexts and2880 conditions, with18,997,488 reward queries,114,549,088 actor
forwards and240 Jacobian/oracle VJP calls. It used66.65/90MiB peak allocated/
reserved memory and1284.18seconds under shared-GPU contention.

All output trees and logs are retrieved, including raw estimates, complete
matrices and reference blocks. The remote execution checkout remains at
`0400f4e`; later commits record analysis. All controllers have exited and no
worker or pending queue remains. Zero of six feasible critic comparisons passes
the locked advance gate, so no confirmation, image or simulator job follows.
No standalone runtime or simulator-return claim is made.

## Continuation-conditioned directions

Exp-075/076 completed on Orion at `8ac4337`, with their finite controllers
exited and full output trees and logs retrieved. Exp-075 passed all 19
conditional Gaussian, joint-moment, complement-contribution, full-score and
repeat-alignment checks; it used 524,288 toy reward evaluations and 4,096
reference quadrature evaluations, with zero GPU tensor allocation. Exp-076
completed all 128 exact toy conditions. It used 16,711,680 main reward
evaluations, 7,798,784 direction-acquisition reward evaluations and 1,605,632
oracle derivative evaluations (24,510,464 total reward evaluations), with zero
GPU tensor allocation. The main-agent review closed the paid-direction design
at its locked toy gate; no real-critic, image or simulator job follows it.

## Joint future-noise coupling

Exp-077/078 completed on Orion at `ea193ab`, with both finite controllers
exited and full output trees and logs retrieved. Exp-077 passed all 21 sign-map,
exact-moment, constant-weight cancellation and repeat-alignment checks; it used
655,360 toy reward evaluations and zero GPU tensor allocation. Exp-078 completed
all 80 toy conditions with 15,728,640 reward evaluations and no direction
acquisition queries; peak allocated/reserved memory was 0/0MiB. Exp-079 then
passed its Hopper row25/index3 smoke at `6e9db3c`: all future-map, constant-
guidance and preserved-rollout comparisons had zero discrepancy. It used 3,328
reward queries, 37,317 actor forwards and 3 kernel-Jacobian VJPs; peak
allocated/reserved memory was 33.16/40MiB. Its complete output tree and log are
retrieved. Exp-080 then completed all 192 conditions across 12 Hopper contexts
at `6e9db3c`: 3,145,728 reward queries, 28,311,596 actor forwards and 36
kernel-Jacobian VJPs. Kernel Jacobian time was not separately instrumented.
Peak allocated/reserved memory was 57.50/78MiB, with 158.19seconds reported
elapsed time. The full raw output tree and log are retrieved. The main-agent
review closes joint antithetic at the M32/M64 discovery gate. All finite
controllers have exited; no worker or pending queue remains. No confirmation,
image or simulator run follows. The execution checkout remains at `6e9db3c`;
later commits contain the analysis and research records.

## Current uncertainty study

Recovery on 2026-10-21 found both Orion and Nebula reachable with no active
project queue or run. The preserved Orion checkout was clean at `d6728de`,
Nebula at `6d1ceca`. The old Orion
exp-039 crash and five completed exp-041 probes have been recovered; see
`results/recovery/2026-10-21/recovery.json`. Preserve those execution checkouts.

New branch `codex/doit-uncertainty-smc` uses Orion checkout
`/home/mlw0719/doit_uncertainty` and storage
`/home/mlw0719/doit_uncertainty_storage`, via
`scripts/compute/uncertainty/{env.sh,sync.sh}` and the shared finite queue.
Its `.venv` points to the already isolated project environment in
`/home/mlw0719/doit_correction/.venv`; no existing package is changed. The new
storage shares the existing Hugging Face cache by symlink. A fresh smoke
determines memory allowance before concurrent new runs. The user's latest
authorization includes bounded RL experiments independently of image success;
simulator execution still requires a working isolated simulator environment.

The new batched critic smoke exp-052 passed all 32 conditions with 48 MiB
peak reserved memory. Subsequent critic-only queues retain a conservative
4096 MiB allowance plus 2048 MiB free reserve and at most one job per GPU.
Exp-056 image particle execution passed N1 equivalence and exact cost checks,
with 5730 MiB peak reserved memory. Image discovery also includes the existing
MC8/MC32 paths (previously 7298 MiB), so retain an 11000 MiB job allowance plus
2048 MiB reserve and at most one job per physical GPU 0--4.

The new image confirmation exp-058 completed all 32 jobs and 256 images on
2026-10-22 at `2c1416f`. Peak reserved memory was 7300 MiB. Its finite controller
exited, all images/embeddings and records were retrieved, and no further run is
queued in this study. The execution checkout remains at the run revision;
later local commits contain analysis and documentation. Terminal N4 passed the
bounded confirmation criterion documented in `experiments/uncertainty_smc/FINDINGS.md`.

The continuation's CPU toy exp-059 completed at `6f96363`, with all 76 conditions
and new moment checks successful. Its finite controller exited and records were
retrieved. Exp-060/061 completed at `7f7a5a2`; their reserved-memory peaks were
46/70 MiB. Both finite controllers exited and all records were retrieved. No
continuation job is active. Retain 4096 MiB plus 2048 MiB reserve and one job
per GPU for this critic path.

Exp-062 also completed its 144 CPU toy conditions and new moment checks at
`2437de2`, with zero GPU allocation. Its finite controller exited, all records
were retrieved, and no new job is queued. The continuation closes at its stated
budgets with no image/simulator expansion.

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

Use `scripts/compute/orion/env.sh` and `queue.py`. The first 512-pixel image smoke
measured 6,240 MiB allocated and 7,298 MiB reserved. Use 11,000 MiB per image job
for subsequent waves, allowing extra stored samples and VJP work. Retain at
least 2 GiB of free memory per device and allow at most two jobs per GPU.
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

No image or RL training is authorized on the Quest login nodes. Quest remains
a reference-source location. The user subsequently authorized Nebula as an
experiment worker for the separate guidance-reliability exploration below.

## Nebula exploration worker

The user authorized physical GPUs 0--4 and stacking by free memory on
2026-10-03. SSH alias `nebula` connects as `mlw0719`. These are RTX A6000
49,140 MiB devices, driver 580.178.04. The initial free-memory snapshot was
20,848 / 16,635 / 16,695 / 15,620 / 11,387 MiB at high utilization.

Use branch `codex/doit-guidance-reliability`, remote checkout
`/home/mlw0719/doit_reliability`, storage `/home/mlw0719/doit_reliability_storage`,
and a new isolated `.venv`. The Orion package pins, imports, and `pip check`
passed. Both pinned model snapshots are downloaded. Existing
source `/home/mlw0719/doob_h_transform/training_free/RCGDM` at `52887f3` and
its `.venv_reject` are preserved. That environment has newer incompatible
package provenance and is not the study environment.

Source `scripts/compute/nebula/env.sh`, then run the shared finite queue
`scripts/compute/orion/queue.py`; it reads `DOIT_REPO`, `DOIT_STORAGE` and
`DOIT_MACHINE`. Sync committed code with `scripts/compute/nebula/sync.sh` only
between finite queues. The exp-045 FP32 smoke used 11,752 MiB allocated and
11,894 MiB reserved. Reserve 13,000 MiB per FP32 image/probe job plus 2048 MiB
free memory, with at most one project job per device for the next wave.
The aesthetic head is copied from
`/home/mlw0719/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth` into the new storage.
Pinned SD and CLIP snapshots are available in its own HF cache. Setup and
acquisition records are retained in `results/reliability/environment` and the
exp-044 result folder.

The exp-046 finite queue completed all ten jobs and 80 images at `6d1ceca`.
The controller exited; no further Nebula run is queued. Maximum reserved
memory was 12,498 MiB, within the 13,000 MiB job allowance. Both sensitivity
settings are closed at discovery, so no confirmation or larger concurrency
experiment is planned. All PNGs and embeddings were retrieved
locally; small records and selected visual evidence are in `results/reliability`.

A low-cost agent polled Orion read-only during the Nebula exploration. Its
last bounded 15-second probe still timed out and the agent has stopped.
The active scheduled follow-up `recover-orion-and-resume-doit` now checks
every 15 minutes in the existing chat. It stays quiet while connectivity is
unchanged and resumes recovery when Orion is reachable; the local desktop
application must remain running. Upon reconnection recover the existing
exp-039/041 jobs and outputs before resuming the online-update study. Do not
sync the Nebula branch onto Orion's active execution checkout or launch
duplicate jobs there.

## Initial study execution state

Exp-031 and exp-032 finished all 64 final jobs, producing 512 images. Both queue
controllers exited, their terminal records and job events were retrieved, and
no project queue or experiment process remained on Orion. No further run is
queued. The image study closed without a method meeting the RL gate; no RL
environment was installed and no RL job was launched.

## Audit-directed continuation

The user reopened research after the audit of `c5e4311`. Orion GPUs 0--4 and the
existing stacking policy remain authorized. Read-only inspection found no
remaining project queue or run process. The first continuation wave is defined
in `configs/queue-audit-diagnostics.json`; its state belongs in the job ledger
and PROGRESS. The new protocol permits low-dimensional real-critic mechanism
experiments before image success. It does not imply a working simulator stack.

Exp-033/034/035 completed at `25936fc` with no runtime failures. The new package
`h5py==3.14.0` was then installed with `--no-deps` in the project venv; existing
packages were preserved. Installation output is in storage `setup-h5py.log`.
Official Hopper/HalfCheetah medium-v2 HDF5 datasets are in `data/d4rl/` below
storage, with source URLs and byte sizes in `acquisition.json`. They each have
one million observations and 2186/1000 complete episode boundaries respectively.
Use `scripts/data/prepare_critic_states.py --config configs/critic-observation-split.json`
for the committed 64-observation episode-disjoint split. Do not replace these
real states with random vectors. Model construction through `diffusion_SDE.model`
works without Gym; the normal `search.ddim` module imports Gym/D4RL and is not
used for the bounded mechanism loader.

On 2026-10-03, exp-036/037, all 30 exp-040 jobs, and exp-038 calibration completed
successfully. The next nine-job exp-039/041 queue was launched at `d6728de`, controller
PID 1206101, with log `controllers/critic-common-candidates.log` below storage.
Later shared-device free memory became too low for broad stacking under the
11,000 MiB plus 2048 MiB reserve policy, so the queue waited between submissions.
At 18:30 UTC, bounded SSH reads were timing out even with multiplexing disabled.
No terminal failure was observed through that connection delay. Recover status
and existing outputs before any restart or duplicate submission; `PROGRESS.md`
records the last retrieved experiment milestones. The user subsequently asked
to publish retrieved records and pause until they verify Orion. Monitoring and
the automatic exp-043 follow-up launch are stopped; the already submitted
remote queue's current state remains unconfirmed.

## RL source preparation

The complete public evaluation copy is
`/home/mlw0719/doob_transform/test/Doob_training_free_adaptation`, revision
`0bc6f95`. It has actor/critic checkpoint pairs for all nine locomotion tasks.
Only base and `masklaw_plot_20260720` are registered Conda environments; neither
has the Gym/D4RL/MuJoCo Python stack. No RL environment is validated yet.
Existing `~/.mujoco/mujoco210` contains MuJoCo2.1 shared objects, whose direct
system links resolve. An old build log failed on incompatible generated C
pointer types; that is not evidence of a currently working mujoco-py build.

The source's online evaluation does not load offline datasets. It builds a
Gym/D4RL environment, loads `behavior_ckpt600.pth` and `critic_ckpt100.pth`, and
passes raw observations to the policy. State normalization must not be invented
when reusing these checkpoints. `search/ddim.py:11-32` constructs the model;
`search/doob_search.py:289-320` constructs action candidates; `utils.py:30-58`
runs episodes and obtains D4RL normalized scores. A new isolated RL environment
and a representative checkpoint/simulator smoke remain required if image
results justify scaling. Do not start RL evaluation before that decision.

The RL source has no dependency lockfile or installation recipe. Its setup.py
declares package metadata only. No Python 3.10 interpreter or cached Python
3.10 Conda package was found in the existing registered environments and cache
roots. The old doob-search environment named in a build log no longer exists.

All nine source task configs use positive DDIM eta (0.2--0.8). The guided
transitions have positive variance; see `search/doob_search.py:115-129` and
`:270-286`. Each guided step currently evaluates the score twice, once for the
base step and once with gradients for the VJP. The source has no full remaining
future simulation inside the lookahead estimator; adding the image future
method would require that loop and its score costs, with the fixed state passed
to the critic for each final action.

Upstream dependency metadata supports using Python 3.10 for this legacy stack:
[D4RL setup.py](https://raw.githubusercontent.com/Farama-Foundation/D4RL/master/setup.py)
declares Python `>=3.7,<3.11` and Gym `<0.24.0`, so the existing Python 3.12 image
environment is outside that range. The
[mujoco-py README](https://github.com/openai/mujoco-py/blob/master/README.md)
specifies MuJoCo 2.1.0 and `mujoco-py>=2.1,<2.2`, matching the existing binary
directory. Its requirements do not cap Cython, but the official
[mujoco-py issue 773](https://github.com/openai/mujoco-py/issues/773) and
[D4RL issue 231](https://github.com/Farama-Foundation/D4RL/issues/231) document
Cython 3 callback compilation failures. These failures justify trying Cython
below 3 with Python 3.10, but the combination has not been validated on Orion.
These sources were inspected without installing or validating an RL environment.
