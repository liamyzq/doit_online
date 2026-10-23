# Preserved RL source settings

Read-only inspection on2026-10-23 found the original source at
`/home/mlw0719/doob_transform/test/Doob_training_free_adaptation`,
commit`0bc6f956`. No source or environment was modified.

`exp_doob.py:19-20,35-48` and `search/configs.py:15,30-45` configure Hopper
medium-v2 with eta0.6, gamma0.75, normalized-Q tau0.4, guided indices1--8,
and final best-of8 selection. HalfCheetah medium-v2 uses eta0.2, gamma0.25,
normalized-Q tau0.5, guided indices1--10, and best-of4. Both use32 antithetic
inner probes and15 diffusion transitions.

`search/doob_search.py:132-179` implements the cached-score proxy
`(next_candidate - sigma_t * epsilon_s) / alpha_t`, reusing the current
noise prediction across candidates. Its source weights standardize each
batch's Q values before applying tau. This study instead retains the fixed
calibrated temperatures already used in the repository, as required to compare
one fixed kernel integral. Candidate-denoiser and full-continuation routes do
not claim exact reproduction of the source's practical proxy or adaptive
reward normalization.

`search/doob_search.py:185-197,266` computes the VJP through the transition mean
and changes the noise prediction by `-gamma * sigma_s * guidance`. The baseline
critic guidance coefficient is zero (`search/ddim.py:45`). The actual
transition displacement coefficient is therefore gamma times the kernel gain
recorded by this repository. The fixed-state artifacts retain unit-gamma
values; final physical displacement comparisons apply the task constants.

`search/ddim.py:180-195` selects the largest-Q final candidate. No action clamp
is applied in this path; `clip_x0=True` is unused and solver thresholding
defaults false (`diffusion_SDE/model.py:242`,
`diffusion_SDE/dpm_solver_pytorch.py:354`). `utils.py:43-46` sends that action
directly to the environment. The sampler uses15 first-order, uniformly spaced
steps from1 to0.001 (`search/ddim.py:54-69`, `search/doob_search.py:47-52`).

The isolated study environment uses Python3.12.9 and Torch2.6.0, with no Gym,
D4RL, mujoco-py, mujoco, glfw, Cython or dm-control package installed. Simulator
execution would require a separate compatible environment; this inspection
did not install packages or initialize a simulator. Critic evaluation is
already functional and does not require those dependencies.
