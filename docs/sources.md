# Source provenance

The original directories are read-only references; no old experiment is modified.

## Images

`nebula:/home/mlw0719/doob_h_transform/training_free/RCGDM`, Git HEAD `52887f3`,
origin `https://github.com/liamyzq/doob_transform.git`. The working tree contains
uncommitted reward, runner, and comparison code. Inspected files include
`doob_sd_pipeline.py`, `test_generation.py`, `utils.py`, `reward_utils.py`, and
`doob_sd_pipeline_w_comparison.py`.

The reward is the Hugging Face `openai/clip-vit-large-patch14` encoder, normalized
image features, and LAION `sa_0_4_vit_l_14_linear.pth`. The source converts decoded
images to 8-bit PIL before the standard CLIP processor. Preserve this convention.
The base model is `sd-legacy/stable-diffusion-v1-5`, classifier-free scale 4.5.
The original runner's `doob_guidance_scale` defaults to 0.4 at
`test_generation.py:67`; this is separate from classifier-free guidance.

The source Euler transition is at lines 382--396, endpoint proxy at 520--523,
fixed-temperature weights at 541--546, and VJP/epsilon correction at 548--562.
The new implementation writes these equations explicitly and caches reward
models. Its primary Euler endpoint is `Y - sigma_next * epsilon_prediction`,
with transition noise scale `sigma_up`; the score correction changes epsilon by
`-gamma * sigma_current * correction`. The source-convention comparison retains
the VP endpoint/scaling separately. All estimator comparisons share one choice.

The image source carries an MIT license, copyright (c) 2023 Hacky Huang. Its
notice is retained in `LICENSE` for reused prompt/preprocessing conventions.

## RL

`orion:/home/mlw0719/doob_transform/test/Doob_training_free_adaptation`, commit
`0bc6f95`, origin `https://github.com/liamyzq/Doob_training_free_adaptation.git`.
It contains pretrained actor/critic checkpoints and nine tasks with five result
rows each. Original tuning data remain in
`orion:/home/mlw0719/doob_transform/Doob_search/locomotion`.

The Gaussian action transition is in `search/doob_search.py:122--150`; the
endpoint proxy and critic queries are at 162--171. At 173--179 the source divides
each state's current candidate rewards by their sample standard deviation before
the temperature softmax. A fixed-temperature port must make that change explicit.
The VJP is at 178--194 and the epsilon correction at 263--270.

Each environment state defines its own `F(a) = exp(Q(s,a) / tau)`. The source
batches several states; online statistics must be state-specific within each
action-generation rollout. HalfCheetah and Walker2d actions have six dimensions;
Hopper has three. `search/ddim.py:18--26` reads dimensions from Gym spaces.
The reward uses the smaller output of twin three-layer Q networks in
`diffusion_SDE/model.py:73--85`.

`pipeline.py:15--20` resolves checkpoint pairs under
`models_rl/{dataset}0large_actor/behavior_ckpt600.pth` and `critic_ckpt100.pth`.
The inspected HalfCheetah medium expert and Walker2d medium replay pairs exist.
The README lists PyTorch, MuJoCo and D4RL without version pins. No RL execution
has been validated for the correction repo yet.
