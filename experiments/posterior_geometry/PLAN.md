# Posterior geometry for DOIT probes and denoiser proxies

Opened on 2026-10-22 from `7dda3d6` on `codex/doit-posterior-geometry`.
The supplied latest plan is the posterior-sensitivity note pasted in this chat.
This is a new study; earlier uncertainty/SMC conclusions remain unchanged.

## Research question and prior evidence

Can a small amount of local denoiser geometry reduce the error of a fixed DOIT
expectation per actual reward query and per elapsed second? Two mechanisms are
tested separately: Gaussian stratification changes dependence between probes
without changing their Gaussian marginals; a local response proxy changes the
integrand and may reduce the frozen-score approximation bias.

The initial online-CV/SPIDER search and exp-044--046 sensitivity smoothing did
not establish a useful image improvement. Exp-047--062 uncertainty potentials
and particle allocations likewise failed their cost comparisons. Terminal
importance resampling N4/tau0.3 remains the accepted practical image baseline
(exp-058). No new method is credited with that established baseline's gain.
Antithetic sampling is essential: it cancels the constant-weight numerator
exactly, and may explain most of the useful variance reduction on its own.

## Mathematical contract

For a fixed isotropic Gaussian transition `Y = mu(x) + sigma * epsilon`, retain
the original kernel score `J_mu(x).T @ epsilon / sigma`. For an orthonormal
direction matrix U, draw `epsilon = b + U(a - U.T b)`, where a uses randomized
Gaussian Latin hypercube strata and b is independent standard Gaussian. Every
probe has the original marginal law. Antithetic variants pair each complete
probe with its negative. Geometry-only rotation of iid normals is a null
control, not a proposed variance reduction method. No noise covariance
rescaling, importance weights, or outer-drift preconditioning is introduced.

The second candidate uses `D(mu) + B_hat (Y-mu)`, with
`B_hat = b_perp I + U diag(b_i-b_perp) U.T`. A center-only isotropic control
separates moving the proxy's center from learning directional response.
Full candidate denoising and full continuation are separate references.
Every proxy has its own integration target; reduced error against that target
does not establish reduced bias against the continuation expectation.
The kernel-score estimate with a parent-dependent proxy is not claimed to be
the full derivative of a new surrogate expectation.

For an exact VE posterior, `J_D = Sigma/t`; for learned networks this is only
an interpretation. Report symmetry and approximation error before describing
a sketch as covariance. Retain background response rather than zeroing the
discarded Hessian. Spectral choices use algebraic eigenvalues, with the
largest-magnitude negative direction distinguished from the largest positive.

## Bounded first pass

1. Exp-063 validates the new Gaussian stratification construction, analytic
   posterior moments, local response, and toy references on Orion. A failure
   blocks the affected comparison and triggers a formula or implementation fix.
2. Exp-064 uses a rotated anisotropic Gaussian and a two-component mixture,
   with rewards aligned to largest, smallest, middle and random directions.
   Compare iid, antithetic, matched random/spectral strata, and independent
   reward-pilot direction selection. Use M=8/16/32/64 and ranks 1/2/4/8 where
   applicable. Split squared bias and variance relative to each proxy's own
   reference, then measure bias relative to the continuation integrand.
3. Exp-065/066 establish execution and fixed-state comparisons on the preserved
   Hopper and HalfCheetah actors/critics with authentic held-out observations.
   Charge pilots, geometry, reference continuations and all Q evaluations.
   Use independently split references; unresolved reference error prevents
   fine ranking. These are RL mechanism experiments, not simulator returns.
4. Exp-067/068 run a representative FP32 image smoke and a bounded fixed-state
   study on the preserved SD1.5 model. Measure candidate reconstruction error,
   reward prediction error, ratio variability and complete computation costs.
   Include full candidate denoising and a small full-future comparison even if
   toy gains are absent, so real-model failure modes are observed directly.
5. Advance a method to a separate fresh-seed image endpoint comparison only if
   it gives at least 20% estimator MSE improvement over both antithetic and a
   matched random-direction control, or at least 20% proxy-bias reduction over
   frozen and center-only controls, without an unmeasured reference dominating
   the difference. Budget comparisons include geometry and pilots; a query-only
   gain is explicitly distinguished from a time gain. Endpoint evaluation must
   include base, historical MC8 and terminal N4. Any claimed practical gain
   requires fresh-seed confirmation and visual review. Simulator expansion
   requires a favorable RL mechanism result and a working isolated environment.

Configurations lock exact seeds, dimensions and sample counts before each
formal wave. No failed state, prompt or seed is removed. Discovery is not a
confirmation claim. Negative results close only the tested budget and method.

## Execution and reporting

Use new Orion checkout `/home/mlw0719/doit_geometry` and storage
`/home/mlw0719/doit_geometry_storage`, sharing only the already isolated Python
environment and model cache. Preserve all earlier checkouts. Commit code and
configurations before formal queues. Use GPUs 0--4, at most one new job per GPU
and 2048 MiB free reserve; choose allowances from representative smoke peaks.
CPU analytic work allocates no GPU tensors. The existing finite queue controls
execution. Record all formal completions, crashes and invalid runs in the
experiment ledger and observed state changes in the job ledger.

The main agent owns algorithms, comparisons and interpretation. A low-cost
agent monitors established jobs, retrieves their known outputs and reports
only completion, failure or a decision-relevant anomaly. It stays quiet during
healthy waits. Main-agent wakeups are reserved for research decisions.

The source formulation is [DOIT](https://arxiv.org/html/2602.16198v1), especially
its kernel-score estimator and Practical frozen-score approximation. Existing
denoiser-covariance estimation is exemplified by
[Free Hunch](https://arxiv.org/html/2410.11149v2). The Gaussian strata, antithetic
construction and low-rank response alone are not novelty claims.
