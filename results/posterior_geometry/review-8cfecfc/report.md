# Exp-069 saved-artifact review

Aggregates are equal means over saved per-seed proxy rows. `mse` is relative to that proxy’s own fixed-reference guidance; `total_mse_to_continuation` compares the estimate with the independent continuation reference. `denoiser_mse` is the candidate clean-action reconstruction error; continuation has no point reconstruction and is omitted.

## Task aggregates

| Task | Proxy | Own-reference MSE | Continuation MSE | Denoiser reconstruction MSE | Reference bias² to continuation |
|---|---|---:|---:|---:|---:|
| hopper-medium-v2 | center_isotropic | 2.58847 | 5.38995 | 0.00162978 | 4.43177 |
| hopper-medium-v2 | geometry_1 | 2.31439 | 5.39962 | 0.000561901 | 3.84222 |
| hopper-medium-v2 | full_linear | 2.86686 | 6.10426 | 0.000203718 | 4.63646 |
| hopper-medium-v2 | continuation | 4.9683 | 4.9683 | — | 0 |
| halfcheetah-medium-v2 | center_isotropic | 1.73882 | 45.8974 | 0.000308731 | 43.8424 |
| halfcheetah-medium-v2 | geometry_1 | 7.92773 | 29.7746 | 0.000194828 | 20.2204 |
| halfcheetah-medium-v2 | full_linear | 9.73375 | 30.2738 | 9.87583e-06 | 18.6592 |
| halfcheetah-medium-v2 | continuation | 61.4814 | 61.4814 | — | 0 |

## Hopper per-index MSE

| Index | Proxy | Own-reference MSE | Continuation MSE |
|---:|---|---:|---:|
| 5 | center_isotropic | 0.186789 | 0.695529 |
| 5 | geometry_1 | 1.57595 | 5.69486 |
| 5 | full_linear | 1.59628 | 5.92451 |
| 5 | continuation | 3.24325 | 3.24325 |
| 9 | center_isotropic | 0.603088 | 1.42559 |
| 9 | geometry_1 | 1.19534 | 3.57092 |
| 9 | full_linear | 1.39951 | 3.5157 |
| 9 | continuation | 3.91873 | 3.91873 |
| 12 | center_isotropic | 6.97553 | 14.0487 |
| 12 | geometry_1 | 4.17187 | 6.93309 |
| 12 | full_linear | 5.60478 | 8.87257 |
| 12 | continuation | 7.74292 | 7.74292 |

## Geometry-1 relative MSE gain vs isotropic by observation

Gain = `1 - geometry_1 / center_isotropic`; positive values favor geometry-1.

| Task | Row | Own-reference MSE gain | Continuation MSE gain | Denoiser MSE gain |
|---|---:|---:|---:|---:|
| hopper-medium-v2 | 26 | -1.0959 | -0.7152 | 0.6783 |
| hopper-medium-v2 | 30 | -0.1150 | -0.2058 | 0.4510 |
| hopper-medium-v2 | 34 | 0.5187 | 0.1607 | 0.6470 |
| hopper-medium-v2 | 38 | 0.2200 | 0.1767 | 0.7240 |
| halfcheetah-medium-v2 | 26 | -3.9833 | 0.3866 | 0.4194 |
| halfcheetah-medium-v2 | 30 | -0.5911 | -1.4456 | 0.5591 |
| halfcheetah-medium-v2 | 34 | -3.7867 | 0.0619 | 0.2503 |
| halfcheetah-medium-v2 | 38 | 0.1964 | -0.2171 | 0.3999 |

## HalfCheetah excluding row 26

| Proxy | Own-reference MSE | Continuation MSE | Denoiser reconstruction MSE |
|---|---:|---:|---:|
| center_isotropic | 0.33665 | 2.93399 | 0.000336915 |
| geometry_1 | 0.694443 | 3.96044 | 0.000216385 |
| full_linear | 1.38715 | 6.12488 | 1.1718e-05 |
| continuation | 3.78595 | 3.78595 | — |

## Active guidance indices and unit-gamma correction gains

The source runner uses `0 < index <= guidance_end_index`. Gains use the saved timestep pairs and VP schedule from the preserved Orion `diffusion_SDE/schedule.py`: `alpha(t)=exp(-0.25 t²(20−0.1)−0.5t·0.1)`, `sigma=sqrt(1−alpha²)`. Applying `CriticEngine.transition` (critic.py:102–117), `noise²=eta² sigma_t²/sigma_s² (1−alpha_s²/alpha_t²)`, `a=alpha_t/alpha_s`, `b=sqrt(sigma_t²−noise²)−a sigma_s`, and `correction_gain=−b sigma_s`. This is the coefficient multiplying gamma times guidance in the transition-mean displacement; no model/reward inference was done.

| Task | Active guidance indices | Diagnostic indices | Index | Unit-gamma correction gain |
|---|---|---|---:|---:|
| hopper-medium-v2 | 1–8 | [5, 9, 12] | 5 | 0.6364775345 |
| hopper-medium-v2 | 1–8 | [5, 9, 12] | 9 | 0.3576065525 |
| hopper-medium-v2 | 1–8 | [5, 9, 12] | 12 | 0.1678484736 |
| halfcheetah-medium-v2 | 1–10 | [5, 9, 12] | 5 | 0.5399202457 |
| halfcheetah-medium-v2 | 1–10 | [5, 9, 12] | 9 | 0.2972462703 |
| halfcheetah-medium-v2 | 1–10 | [5, 9, 12] | 12 | 0.1429402661 |

The source schedule was read over SSH from `/home/mlw0719/doob_transform/test/Doob_training_free_adaptation/diffusion_SDE/schedule.py`; it specifies beta0=0.1 and beta1=20.0. All aggregate metrics use only the saved exp-069 JSON artifacts.

## Active diagnostic intervals only

Errors below average only sampled indices where the configured sampler applies guidance. Unit-gamma displacement MSE multiplies each raw guidance error by its own squared correction gain before averaging. Multiply by gamma squared for any fixed nonunit guidance strength. These sampled indices do not represent the entire active interval.

| Task | Proxy | Raw guidance MSE | Unit-gamma displacement MSE | Independent observations | Fixed contexts |
|---|---|---:|---:|---:|---:|
| hopper-medium-v2 | frozen | 24.740104 | 10.022307 | 4 | 4 |
| hopper-medium-v2 | center_isotropic | 0.695529 | 0.281761 | 4 | 4 |
| hopper-medium-v2 | geometry_1 | 5.694859 | 2.307008 | 4 | 4 |
| hopper-medium-v2 | full_linear | 5.924512 | 2.400041 | 4 | 4 |
| hopper-medium-v2 | continuation | 3.243254 | 1.313854 | 4 | 4 |
| halfcheetah-medium-v2 | frozen | 274.419829 | 46.627561 | 4 | 8 |
| halfcheetah-medium-v2 | center_isotropic | 37.861512 | 7.482497 | 4 | 8 |
| halfcheetah-medium-v2 | geometry_1 | 26.138144 | 5.153873 | 4 | 8 |
| halfcheetah-medium-v2 | full_linear | 26.540301 | 4.924228 | 4 | 8 |
| halfcheetah-medium-v2 | continuation | 57.580373 | 11.394901 | 4 | 8 |
