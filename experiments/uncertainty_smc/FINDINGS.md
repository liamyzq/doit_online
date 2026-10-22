# Uncertainty and particle correction: findings

Terminal importance resampling with four base trajectories and tau=0.3 meets
the locked image criterion on eight fresh seeds across eight fixed prompts.
Compared with MC8, delivered-image aesthetic reward improves by 0.17610
(95% seed-block interval [0.00911, 0.30592]), with higher mean text alignment,
no measured embedding-distance reduction, and 4 rather than 97 reward queries
per delivered image. This is a useful project baseline, not a new uncertainty
method. The tested uncertainty, FORS and intermediate-SMC combinations do not
establish an improvement at their stated budgets.

This study evaluates the supplied posterior-uncertainty, local FORS, and strict
SMC proposals in a separate experiment area. The target in the distribution
experiments is the implemented base sampler tilted by the actual terminal
reward. Implementation identities pass; finite-query improvement is a separate
empirical question. Every formal run, including the first failed quadrature
check, is recorded in `experiments/results.tsv`.

## Method-development continuation

The later exp-059--062 pass develops two additional candidates: conditional
reward residual calibration for inexpensive positive potentials, and using the
conditional second moment to allocate particles under the unchanged base
proposal. The [continuation report](CONTINUATION.md) records their formulas,
existing-method controls, locked decisions, complete costs and negative results.

Across 220 toy conditions, empirical calibration improves some value estimates
but has no established endpoint advantage over the relevant controls. The
second-moment variant does not meet its advance criterion. More frequent
resampling improves the Gaussian N16 first-moment case by 21.4% against point,
but direct learned-value regression performs as well and the mixture benefit
is small. This is a limited schedule/value-estimation signal, not a distinct
new UC result.

The real-critic continuation covers 12 observation/checkpoint conditions.
On the reference-resolved HalfCheetah row18, empirical/point/future2 centered
log-value MSE is 5.0400/4.8749/0.2169. Hopper's near-one reference ESS prevents
fine ranking. Both new candidates are closed at the tested budgets; the image
and simulator expansion gates were not met. This does not invalidate UC in
general or establish novelty. All formal outcomes remain in the ledgers.

## Analytic models

Exp-047-r2 passes all eight initial identities, with maximum absolute error
3.55e-15. The original mixture-tower check failed because 64-node quadrature
was inadequate; independent adaptive integration verified the formula before
increasing resolution. Exp-050 also verifies the grouped-candidate weight
identity to 2.22e-16. These are correctness results, not performance results.

Exp-048/050 complete 260 toy conditions, each with 256 repeats. Gaussian
uncertainty helps some comparisons at a fixed particle count. For example,
N=16 Gaussian-toy CDF MSE decreases from 0.004953 to 0.004019. Grouped proposal
selection with uncertainty reaches 0.002946 on that toy, versus 0.005238 for
its point-potential counterpart. However, bootstrap N=128 at the same grouped
transition budget reaches 0.001011. On the mixture toy the corresponding
uncertainty/point/bootstrap values are 0.007231/0.009708/0.001481. Neither the
uncertainty nor grouped combination establishes a cost-controlled success.

Exp-049 gives FORS an exact local quadratic oracle, avoiding learned-oracle
error as a confound. At B=2 its variance remains about 0.53 versus target 0.364,
with about 55 guidance evaluations per output. Direct residual rejection is
accurate at about two residual evaluations per output. Close the tested FORS
setting for clipping error and cost; this does not assert universal failure.

## Frozen real RL actors and critics

These are offline distribution experiments at real observation states, not
environment-return evaluations. Both Hopper and HalfCheetah adapters reproduce
the validated actor transition to within 1.2e-7. Calibration states and discovery
states are episode-disjoint. Calibration residuals include mean bias, so
uncertainty is compared with both uncorrected and mean-corrected point values.

The historical proxy temperature made the terminal reward tilt nearly flat.
Exp-054 therefore defines a new target temperature using calibration states
only: half the median within-state terminal-Q standard deviation. This gives
0.0311595531 for Hopper and 0.0736910712 for HalfCheetah. No discovery reward
was used to tune these temperatures.

Exp-055 completes 128 conditions comparing bootstrap N=8/16/32/64, point,
calibrated-point and Gaussian uncertainty SMC, grouped candidates, a conditional
Gaussian pilot proposal, and a branch-plus-uncertainty combination. All costs
include references: 10,506,240 actor evaluations and 6,881,280 Q evaluations.

| State | Bootstrap N16 CDF MSE | Point SMC N8 | Gaussian SMC N8 | Reference ESS / 32768 |
| --- | ---: | ---: | ---: | ---: |
| Hopper row20 | 0.014437 | 0.031616 | 0.035849 | 18941 |
| HalfCheetah row16 | 0.042677 | 0.172357 | 0.171930 | 3352 |
| HalfCheetah row20 | 0.016188 | 0.111033 | 0.122684 | 18060 |

These three methods each use 240 actor evaluations per population; Q costs are
16, 128 and 248 respectively. Larger bootstrap populations also beat the other
combinations at their relevant costs. Hopper row16 has reference ESS 11.21 and
split-reference CDF MSE 0.01284, so precise ranking there is unresolved. It is
not used to claim either a gain or a finely measured failure.

The poor proxy values induce early particle collapse: N16 point-SMC initial ESS
fractions are about 0.07--0.12. Adding uncertain proxy draws does not reliably
repair that collapse. Close these tested critic configurations at this budget.
A successful probability correction does not prevent finite-particle failure
when intermediate potentials are poor.

## Image experiment

Exp-056 verifies a separate sparse image particle sampler: N1 has exactly the
same reward and text alignment as the original unguided sampler. All expected
counts match (50 reward queries, 436 score evaluations across five outputs).
Peak reserved memory is 5730 MiB.

Exp-057 compares seven locked settings across eight prompts and fresh seeds
57011/57017: base, historical MC8, selected MC32, terminal importance resampling
N4/N8, sparse point SMC N4, and sparse two-full-future SMC N4. Intermediate
potentials occur only at indices 10/14/18. Random potentials are cached and
carried through resampling; the final potential is the true image reward.
Terminal importance resampling is an established baseline, not a novelty claim.
The output is drawn by normalized weight, never selected by maximum reward.

Exp-057 completes all 112 outputs with 7,088 reward queries, 9,856 actor
forwards and 288 VJPs. The complete contact sheets were visually reviewed.
Reported per-image seconds follow the existing generation timer after prompt
setup: they include all candidate continuations, guidance, decoder and scoring
work. Model loading and prompt preparation remain in each run's full elapsed
record. Timings were observed on shared GPUs, not dedicated hardware.

| Method | Reward | Alignment | Seconds/image | Reward queries | Actor forwards |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC8 | 6.03094 | 0.283229 | 27.63 | 97 | 20 |
| Selected MC32 | 6.21638 | 0.282337 | 73.85 | 289 | 20 |
| Terminal N4 | 6.18440 | 0.281381 | 9.21 | 4 | 80 |
| Terminal N8 | 6.40737 | 0.276569 | 20.83 | 8 | 160 |
| Sparse point SMC N4 | 6.42012 | 0.275298 | 9.34 | 16 | 92 |
| Sparse future2 SMC N4 | 6.35862 | 0.276052 | 31.80 | 28 | 224 |

Only terminal N4 passes the predeclared discovery gate: +0.15347 reward versus
MC8, alignment loss 0.00185, less measured time, no diversity loss and no distinct new
image-quality failure. The other particle settings lose more than 0.005
alignment; full-future SMC also costs more time than MC8. Their positive reward
results are real, but they fail the joint criterion. Close these settings at
this budget. No confidence interval is inferred from the two discovery seeds.

## Independent image confirmation

Exp-058 completes all 32 jobs at commit `2c1416f`, with 256 delivered images:
64 each for base, MC8, selected MC32 and terminal N4. The eight reserved seeds
are 59011/59017/59023/59029/59039/59051/59057/59063. No parameter was retuned
from confirmation. Total costs are 25,024 reward queries, 8,960 score forwards
and 1,152 VJPs; peak reserved memory is 7,300 MiB. No runtime failures occurred.

| Method | Reward | Alignment | Embedding distance | Seconds/image | Reward queries | Score forwards | VJPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 6.09413 | 0.270451 | 0.121282 | 2.31 | 1 | 20 | 0 |
| MC8 | 6.15570 | 0.271300 | 0.119259 | 31.30 | 97 | 20 | 9 |
| Selected MC32 | 6.32629 | 0.273077 | 0.119912 | 75.81 | 289 | 20 | 9 |
| Terminal N4 | 6.33180 | 0.273791 | 0.120957 | 8.89 | 4 | 80 | 0 |

Terminal N4 passes the predeclared aggregate confirmation thresholds: reward
gain 0.17610 exceeds 0.15, its 95% seed-block interval [0.00911, 0.30592] is
positive, mean alignment increases by 0.002491, and embedding distance increases
by 1.42%. Measured generation time is lower on every seed. These timings come
from a shared machine and are descriptive; they do not establish a 3.52-fold
dedicated-device speedup. The exact query saving is 95.88%, while score forwards
increase fourfold and the nine VJPs disappear. Counts include terminal scoring
and all unselected candidates. The weighted-pool reward diagnostic, 6.39267,
is not substituted for the actual delivered-image reward, 6.33180.

Against selected MC32, the reward difference is only +0.00552, with interval
[-0.16793, 0.16153]. This comparison does not establish superiority or statistical
equivalence. N4 attains a similar observed mean with substantially fewer reward
queries and less measured time. MC32 also has visible corruption at lighthouse
seed59029 and airplane seed59063, so its mean score alone is not a quality claim.

All 256 outputs were visually reviewed on 16 contact sheets. N4 has no distinct
new visual failure mode relative to the base and MC8 controls, and does not
reproduce the strong-guidance texture/block collapse from earlier experiments.
It still produces incorrect individual images: lighthouse seed59051 omits the
lighthouse, airplane seed59051 has malformed geometry and a close crop, and
labrador seed59029 adds clothing. The base/control outputs also exhibit missing
subjects, anatomical errors and synthetic lettering. These outputs remain in
the evaluation, without filtering or regeneration. See the
[visual review](../../results/uncertainty_smc/exp-058-analysis/visual-review.md),
[seed59029 comparison](../../results/uncertainty_smc/quality/confirmation-s59029-2.png)
and [seed59051 comparison](../../results/uncertainty_smc/quality/confirmation-s59051-2.png).
The comparative quality criterion is met; this is not a claim that every output
is faithful or artifact-free.

The aggregate benefit is uneven. Descriptive N4-minus-MC8 prompt means are:

| Prompt | Reward difference | Alignment difference |
| --- | ---: | ---: |
| Airplane | +0.54780 | +0.015896 |
| Bernese | +0.11386 | +0.003230 |
| Cafe | -0.03935 | -0.003749 |
| Formula1 | +0.39524 | +0.004630 |
| Labrador | +0.24680 | -0.014231 |
| Lighthouse | +0.00480 | +0.007182 |
| Mountain | +0.13259 | +0.005740 |
| Orchid | +0.00706 | +0.001230 |

In particular, Labrador loses alignment despite the favorable aggregate, and
seed59051 has reward delta -0.33034 averaged over prompts. The interval lower
bound is close to zero and conditions on the eight fixed prompts; it does not
quantify generalization to new prompts. No simulator-return improvement is
claimed. These limitations bound the confirmed result rather than changing
the locked aggregate selection rule.

The retained method generates four independent base trajectories, evaluates
their final images once each, and draws one image with probability proportional
to `exp(reward / 0.3)`. It is finite-pool importance resampling, not exact sampling
from the terminal-tilted distribution. The completed goal is a reproducible
practical improvement over this project's MC8 image setting. Novelty and broad
task generalization remain unestablished. No further experiment is queued.

## Reproduction and records

Configurations are under `configs/uncertainty_smc/`, implementation under
`doit_correction/uncertainty_*.py`, and the analysis command is
`scripts/analysis/uncertainty_images.py`. Read `COMPUTE.md` before remote use.
Use `scripts/compute/uncertainty/sync.sh` only between finite queues; the remote
checkout is `/home/mlw0719/doit_uncertainty` and storage is
`/home/mlw0719/doit_uncertainty_storage`. The recorded run commit, config, PID,
physical GPU, model revisions and environment are in each run's JSON records.

For the discovery summary, after retrieving each complete run folder including
PNGs and embeddings into `results/uncertainty_smc/<experiment-id>`, run:

```sh
python scripts/analysis/uncertainty_images.py \
  --queue configs/uncertainty_smc/queue-image-discovery.json \
  --output results/uncertainty_smc/exp-057-analysis
```

For confirmation, use `queue-image-confirmation.json`, output
`results/uncertainty_smc/exp-058-analysis`, and add `--confirmation`.
The analysis requires NumPy and Pillow. Discovery used an isolated local
Python 3.9 environment with NumPy 2.0.2 and Pillow 11.3.0; this environment
does not alter the frozen remote model environment. PNG outputs are retained
locally and in remote storage; small metadata and embeddings are committed.
The analysis requires the entire committed queue and all configured images,
so a partial retrieval cannot silently become a complete comparison.
