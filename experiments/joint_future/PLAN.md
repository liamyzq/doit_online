# Coupling current and future Gaussian noises

This is a separate hypothesis after exp-076 closes paid per-future directions.
Exp-073 showed that sharing future noise within ordinary antithetic pairs can
reduce continuation error. Exp-076 shows that acquiring a direction for every
future can spend most of a small budget. Test whether sign coupling of the
future itself reduces error without any direction or pilot reward queries.

Let epsilon be the one-step Gaussian probe and Z all later independent Gaussian
noises. Every point keeps the original joint marginal. Compare:

- Independent-future antithetic: (epsilon,Z1), (-epsilon,Z2).
- Shared-future antithetic: (epsilon,Z), (-epsilon,Z).
- Random rank-one paired LHS with shared future, the existing simple control.
- Joint antithetic: (epsilon,Z), (-epsilon,-Z).
- Cartesian four-point: (epsilon,Z), (-epsilon,Z), (epsilon,-Z), (-epsilon,-Z).

The full score factor uses epsilon in every case; later noises are not scored.
No main or future sample is chosen using its reward. Gaussian sign symmetry
preserves the target joint moments. The self-normalized ratio remains biased
at finite M. The Cartesian method has M/4 independent main draws versus M/2
for the paired methods. Future forward and reward counts remain exactly M per
step/terminal evaluation; there is no free direction oracle or hidden pilot.

## Exp-077 and exp-078

Use the already verified exact rotating-direction joint toy from exp-076,
dimensions3/6, angles0/60, b=1,a=0.7,w=0.7. Exp-077 checks the new sign maps,
constant-weight cancellation, exact target numerator/denominator moments under
each coupling, and within-repeat common-field alignment with different group
sizes. A failure blocks formal comparison and requires fixing the coupling.
Do not repeat the previous toy reference quadrature or conditional-basis checks.

Exp-078 compares the five listed methods at M32/M64, seeds78011/78017 and4096
repeats. M32 is primary; M64 describes scaling, not an alternative gate selected
after results. Preserve per-repeat vectors, paired differences, numerator/mass
summaries and actual costs. No context, seed or method is dropped.

A new coupling at M32 can advance for a given dimension if it improves the
60-degree rotating target by20% against both shared-future antithetic and
random paired LHS, with paired MC95% intervals below zero. The aligned target
is a reported control, not an additional selection condition. Eligibility is
per dimension because the physical tasks have different action dimensions;
this is a new gate for a new hypothesis and does not revise exp-076. If both
couplings pass, select joint antithetic for its simpler pair structure. Advance
at most one coupling, on the corresponding real task(s). No passing coupling
closes this hypothesis at the tested budgets.

## Conditional real-critic and image phases

Before discovery, a bounded Orion smoke checks that explicit future sign maps
retain the preserved rollout for the shared/independent controls and that
constant weights cancel the full score. Measure peak memory. The actor,
transition schedule, temperature and kernel score are unchanged; path gradients
are not implemented. Use the existing isolated environment and launcher.

Reserve fresh discovery observations41/45/49/53 and confirmation42/46/50/54.
Use indices3/5/7, two seeds,256 repeats, M32 primary/M64 secondary and eight
independent8192-point reference blocks per context. The independent reference
uses the original shared-future antithetic law. All couplings have its same
marginal target. Main and future random arrays use common total-budget repeat
blocks so paired losses remain independent across repeats. Record all costs,
raw estimates, context kernel Jacobians and reference block moments.

At M32, the selected coupling must improve average unit-gamma displacement
MSE20% against both simple controls, improve the same three of four observations,
have conditional paired MC intervals below zero, and preserve the favorable
aggregate difference for all eight reference-block replacements. Keep low-ESS
contexts; block sensitivity does not bound ratio-reference bias. A passing
setting receives one reserved-observation confirmation with complete individual-
call timing. Only a confirmed setting advances to a bounded image fixed-state
study. No endpoint or simulator run follows without a separately favorable
decision. A failed practical gate closes this setting without a new sweep.

Main owns scientific implementation and interpretation. Commit each code/config
before execution; delegate established Orion queues and quiet monitoring.
Record every outcome and actual job event. Preserve exp-073 and exp-076 closure
records and terminal N4 as the practical image baseline.
