# Continuation-conditioned directions

The user reopened research after the exp-073 closure and started a new goal.
Exp-073 remains closed under its original criteria. Its paid directions closely
matched the candidate gradient and improved candidate guidance, but the main
continuation gate failed. This study distinguishes a mismatched candidate
sensitivity direction from the cost and path-count penalty of four-point groups.

## Estimator and target

Write the original continuation weight as W(epsilon, Z), where epsilon is the
Gaussian one-step probe and Z contains all later independent Gaussian noises.
For each future block, draw Z first and obtain a rank-one direction U(Z) from
that fixed continuation at the kernel center. Then draw independent main Gaussian
active and complement noises. A two-point block is {Ua+b, -Ua-b}; a four-point
block is {Ua+b, Ua-b, -Ua+b, -Ua-b}. All points evaluate the actual continuation
with their block's Z, all weights and full kernel scores are retained.

Conditional on Z and any independent direction pilot, each indexed epsilon
still has N(0,I) law. Thus the joint marginal (epsilon,Z) and the target moments
are unchanged even though direction and future noise are related. A randomized
LHS active coordinate is independent of Z. The ratio retains finite-sample bias;
block count grows with total budget, so this is not a fixed single-future ratio.
Main probe randomness never enters direction selection.

The new per-future cost matters. Exact pathwise gradients pay one terminal
reward/continuation and a backward pass per block. Full finite differences pay
2d complete continuations per block. A random two-direction central difference
pays four. These costs reduce the number of main pairs or quadruples. The
candidate finite-difference comparator pays its original 2d candidate calls once
per estimate. Report reward queries, actor forwards, backward work and unused
budget slots separately. Oracle derivatives explain potential; only black-box
finite-difference methods are eligible for the current practical gate.

## Exp-075/076: exact controlled toy

The toy has epsilon~N(0,I_d), a fair future sign S and independent Z~N(0,1).
Let u_S=cos(theta)e1+S sin(theta)e2 and
W=exp[-(u_S^T epsilon - b - a Z)^2/(2 w^2)]. Its exact weighted moment is
b cos(theta)e1/(w^2+1+a^2). Set b=1, a=0.7, w=0.7. Dimensions3/6 match the real
critics; theta0/60 degrees isolates future-direction rotation without changing
the scalar weight law. The candidate direction is e1.

Exp-075 checks the newly conditional construction: pointwise Gaussian moments
and epsilon/future independence under future-dependent bases, exact complement
cancellation when the block direction is correct, retention of nonzero moments
with a wrong direction, analytic first moments against a large independent
sample, and fixed-repeat random alignment across different block counts. A
failure blocks exp-076 and requires correcting the changed construction.

Exp-076 uses total budgets32/64, seeds76011/76017 and4096 repeats. Controls are
shared-future ordinary antithetic and random rank-one paired LHS. Compare the
paid candidate-direction quadruple, paid path-gradient pair/quadruple, paid full
coordinate path finite-difference quadruple and paid four-query path pilot
pair/quadruple. Save every per-repeat estimate, paired differences, exact
reference, moment diagnostics and all direction/main query counts. Known toy
analytic gradients are explicitly oracle derivatives with a counted value call.

The mechanism gate at M64 requires the paid path-gradient quadruple to improve
rotating-target MSE by20% over both simple controls in each dimension, with
paired MC95% intervals below zero. Report every failed paid black-box method and
the M32 boundary. Passing permits a bounded real-critic direction study, not an
image claim. If the oracle gate fails, close this design at these budgets.

## Conditional real-critic study

After the toy gate, validate only the changed explicit-noise differentiable
continuation and per-future direction selection on Orion. Compare its forward
result to the preserved rollout with identical noises, and compare the complete
path gradient to resolved finite differences. Do not repeat settled kernel,
Gaussian-stratum or previous candidate-gradient checks. Measure memory before
setting concurrency and retain all query costs incurred by checks.

Lock discovery rows41/45/49/53 and reserve42/46/50/54 for independent
confirmation. A bounded config inventory finds neither set in any prior formal
critic configuration. Use indices3/5/7, the same preserved actor/critic models,
reward temperatures and transition kernel. M64 is the primary budget because
per-future acquisition leaves only a few independent paths at M32; M32 is a
predeclared secondary boundary. Both simple controls receive the same budgets.
Use two seeds,256 repeats and eight independent8192-point reference blocks per
context. Main/future/pilot streams use structured coordinates and fixed total-
budget repeat blocks. Preserve all matrices, directions or sufficient alignment
summaries, raw estimates and reference block moments.

The primary metric is unit-gamma transition-mean MSE for the actual continuation
integrand. To advance, a feasible path finite-difference method at M64 must beat
both simple controls by20% in a task, improve the same three of four observations,
have favorable aggregate conditional paired intervals, and retain the difference
sign for all eight reference-block replacements. Retain every context, including
low-ESS references. Block sensitivity does not certify ratio-reference bias.
At most one setting advances to the reserved observations; its complete
individual-call cost is measured before any bounded image transfer. No
candidate-only result, derivative oracle, or favorable post-hoc slice advances.
No simulator or endpoint run follows without a separate favorable decision.

## Execution and closure

Main owns design, implementation and decisions. Low-cost agents execute exact
committed queues and monitor quietly to terminal state or anomaly. Use the
existing Orion geometry checkout and isolated environment, physical GPUs0--4,
and the established launcher. Commit code/configs before each formal run; never
sync with active or pending workers. Append every outcome and actual job event
to the project ledgers. Close this bounded design if its mechanism or practical
gate fails, preserving favorable and negative evidence. A new idea then requires
a separately stated hypothesis rather than a parameter sweep of the failed one.
