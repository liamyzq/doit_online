# DOIT Correction: final findings

The requested image study is complete. Every supplied correction family and the
targeted extensions received a tuning and closure decision. No new method
established a usable image improvement under the evaluated budgets, so the
condition for scaling to RL was not met. No RL environment was installed and no
RL evaluation was run. All project experiment processes have finished.

Several results are positive in a narrower sense: guidance EMA reduces measured
VJP error; strong EMA improves average aesthetic reward at the MC8 query budget;
and direct future M8 nearly matches MC32's average reward with fewer reward
queries and lower measured time. The latter two settings also produce concrete
image quality failures. Those failures prevent a success claim based on means.

## Scope and confirmation

The engine uses frozen SD1.5 at 512 pixels, a frozen LAION aesthetic head,
20 Euler steps, and nine guided transitions with positive noise. Fresh MC
baselines receive their own temperature and guidance-strength tuning. The
primary sampler uses a consistent Euler endpoint; changing the source's VP
convention or caching its reward model is not counted as a CV benefit.

The final comparison has eight groups with eight prompts and eight reserved
seeds each: **512 images**. The prompts cover Bernese dog, Formula1, airplane,
Labrador, lighthouse, cafe, orchid, and mountain. Seeds are 101, 113, 127, 139,
151, 163, 179, and 191. All 64 final jobs completed with zero denominator events
and no runtime failures. The 16 focused numerical checks also passed on Orion.
Earlier two-level experiments include three recorded runtime failures.

Seven groups were locked after the 160-image discovery comparison. Moderate EMA
was added from discovery-image quality evidence after the first ten exp-031
shards arrived; its parameters were already tested and were not selected from
partial confirmation means. It ran in a separate wave. This selection timing
and the fixed parameters are preserved in the [study protocol](study.md).

## Final image results

Each row averages 64 images. Alignment is CLIP text-image similarity; distance
is within-prompt cosine distance between image embeddings, a diversity proxy.
Queries exclude the single final evaluation. Score evaluations count CFG
denoiser calls by candidate. Seconds are measured per image on shared GPUs.

| Setting | Reward | Alignment | Distance | Queries | Score evaluations | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unguided | 6.0410 | 0.27457 | 0.12504 | 0 | 20 | 1.91 |
| MC8, tau 0.3, gamma 0.4 | 6.1100 | 0.27484 | 0.12390 | 96 | 20 | 25.31 |
| MC8, tau 0.6, gamma 4 | 6.0196 | 0.28300 | 0.13439 | 96 | 20 | 25.33 |
| MC32, tau 0.6, gamma 4 | 6.3389 | 0.27755 | 0.12859 | 288 | 20 | 64.05 |
| Guidance EMA 0.25, tau 1, gamma 8 | 6.2533 | 0.28038 | 0.12719 | 96 | 20 | 24.84 |
| Independent future M4, tau 0.3, gamma 1 | 6.1782 | 0.27784 | 0.12722 | 40 | 236 | 26.38 |
| Paired future M8, tau 0.6, gamma 4 | 6.3361 | 0.27705 | 0.12863 | 80 | 452 | 50.01 |
| Guidance EMA 0.25, tau 0.3, gamma 1 | 6.1322 | 0.27434 | 0.13108 | 96 | 20 | 22.56* |

*Moderate EMA ran in a different wave. Its lower observed time does not establish
a speedup over MC8; their query and score budgets are identical.

Paired reward deltas below use moderate MC8 as the baseline. The 95% intervals
resample whole seed blocks and condition on these eight prompts; they are not
intervals for a general prompt population.

| Candidate | Reward delta | 95% interval | Decision |
| --- | ---: | --- | --- |
| MC32 | +0.2289 | [0.2099, 0.2477] | Keep as the higher-budget baseline |
| Strong guidance EMA | +0.1433 | [0.0664, 0.2313] | Close: visible quality failure |
| Independent future M4 | +0.0682 | [-0.0164, 0.1476] | Close: unresolved improvement |
| Paired future M8 | +0.2261 | [0.1740, 0.2783] | Close: visible quality failure |
| Moderate guidance EMA | +0.0222 | [-0.0094, 0.0486] | Close: unresolved improvement |

Moderate EMA has no evident embedding-distance loss, but its reward improvement
remains small and unresolved. Its alignment delta is -0.00050 with interval
[-0.00209, 0.00114]. Independent future M4 reduces reward queries from 96 to 40,
but adds 216 score evaluations and takes 26.38 rather than 25.31 seconds.
Neither result establishes an end-to-end improvement at the final budget.

Against MC32, paired future M8 has reward delta -0.0028 with interval
[-0.0558, 0.0487], and alignment delta -0.00051 with interval
[-0.00329, 0.00254]. This does not prove equivalence. It reduces reward queries
by 72.2% and measured image time by 21.9%, while increasing score evaluations
from 20 to 452. Reward time falls from 61.18 to 21.05 seconds; future denoising
adds about 26.16 seconds inside oracle time. VJP time is about one second for
both. The recorded timers overlap: reward is part of oracle time and memory
update is part of estimator time, so they must not be summed as a partition.

## Quality failures that change the decision

Strong guidance EMA's positive reward effect survives reserved-seed evaluation,
but the lighthouse at seed 113 reproduces the severe grain, block artifacts,
and posterization seen at discovery seed 53. The reward and average alignment
metrics do not exclude this failure. The raw
[strong EMA image](../results/final/exp-031-confirm-gema-reward-s113/lighthouse/113/image.png)
and the same-seed
[moderate MC8 image](../results/final/exp-031-confirm-mc8-moderate-s113/lighthouse/113/image.png)
are retained. The eight reserved lighthouse examples were inspected in the
[first](../results/final/exp-031-ema-lighthouse-quality-1.png) and
[second](../results/final/exp-031-ema-lighthouse-quality-2.png) comparison sheets.

Paired future M8 has higher mean reward than MC32 on six of eight prompts, but
its cafe mean is 0.6898 lower. At cafe seeds 127 and 139, it produces severe
saturation and structural corruption; the MC32 images remain coherent. The
paired reward deficits are 1.6294 and 1.4724. These examples explain why its
favorable aggregate cost result is insufficient.

| Cafe seed | Paired future M8 | MC32 |
| --- | --- | --- |
| 127 | [Raw image](../results/final/exp-031-confirm-future8-paired-s127/cafe/127/image.png) | [Raw image](../results/final/exp-031-confirm-mc32-g4-s127/cafe/127/image.png) |
| 139 | [Raw image](../results/final/exp-031-confirm-future8-paired-s139/cafe/139/image.png) | [Raw image](../results/final/exp-031-confirm-mc32-g4-s139/cafe/139/image.png) |

These are direct qualitative observations, not a blinded quality score or an
estimated failure rate. The raw images are unmodified. An average reward gain
does not make these observed failures disappear.

## Coverage and closure of the other families

| Family and evolution | Evidence | Closure reason |
| --- | --- | --- |
| Affine/exponential online CV; normalized LMS, random projected RLS, reset/frozen memory, M1/M4/M8/M16 and temperature tuning | exp-005/015/016 and endpoint controls | Strong pointwise fits do not consistently improve ratio or VJP guidance error on independent paths. |
| Low-frequency spatial/DCT surrogate | exp-006/007/015/016 | No useful ratio benefit; independent guidance error worsens. |
| Direct antithetic pair-contrast fitting | exp-025/026, 24 tuned variants | Best ratio error 47.4803 versus MC8 47.4838, only 0.007% lower; pair residuals do not improve. |
| Historical constant and moment-ratio EMA controls | exp-005/015/021 | A constant CV cancels exactly with antithetic samples; ratio EMA offers no useful endpoint advantage. |
| SPIDER coupled differences with periodic and larger refresh | exp-005/012/017 | Initial denominator failures; independent errors 89.45/82.32/174.71 versus matched MC 27.42/24.89/19.63. Two-query differences do not repay their cost. |
| Exact-weight Gaussian/mixture adaptive proposals, with and without CV; scale, mass, and KL tuning | exp-013/018/024 | Small discovery gains reverse independently: IS 47.041 to 47.213 and CV-IS 47.105 to 47.256. |
| Two-level proxy/full-future corrections; fixed or historical coefficients and temperature extension | exp-014/023/029 | No useful reward/cost gain; three historical-coefficient settings fail with nonfinite guidance, and another has a denominator event. |
| Guidance EMA and independent/paired direct future sampling | exp-011/016/020/022/027/028/030/031/032 | Final decisions and measured tradeoffs are above. |

Guidance EMA's independent normalized VJP error falls from MC8's 53.418 to
8.981; MC32 is 18.029. This is useful mechanism evidence for biased smoothing,
but zero guidance itself has normalized squared error 1. Noisy reference MSE
can favor shrinkage and cannot establish successful reward steering.

The equality of pure IS and CV-IS at an unclipped scale-one exponential tilt
with complete antithetic pairs is correct analytic cancellation. It is not an
implementation failure. The [estimator note](estimators.md) records the
identities, update order, and cost conventions.

## Limits and retained records

These closure decisions apply to the evaluated SD1.5/aesthetic-reward setting,
budgets, and targeted extensions. They do not prove that control variates or
future corrections cannot work elsewhere. The prompt set is small; 512-sample
fixed-state references remain noisy in high dimensions, and FP16 VJP changes
can alter later paths even with paired random seeds. Shared-GPU times are not
dedicated-device benchmarks. Three older failed two-level runs lack exact
within-failing-image costs, so the record does not claim an exact total research
query count.

[PROGRESS.md](../PROGRESS.md) preserves the experiment sequence and interpretation;
[results.tsv](../experiments/results.tsv) records formal outcomes and decisions;
[jobs.jsonl](../jobs/jobs.jsonl) records execution events. Final aggregation uses
[analysis-exp-032-combined.json](../configs/analysis-exp-032-combined.json), with
the output in [exp-032-combined-summary.json](../results/final/exp-032-combined-summary.json).
The separate MC32 comparison is
[exp-031-vs-mc32.json](../results/final/exp-031-vs-mc32.json).

Original images, embeddings, and small result files are available in local
`artifacts/`; complete run artifacts, including stored tensor banks, remain at
`orion:/home/mlw0719/doit_correction_storage/runs/`. Final aggregate JSON files and the image evidence linked above are included
in `results/final/` for online reading. Full run artifacts are excluded from Git. The [reproduction guide](reproduction.md) and [compute record](../COMPUTE.md)
document how to use them. The code remains available for reuse, with no active
experiment queue and no scheduled follow-up.
