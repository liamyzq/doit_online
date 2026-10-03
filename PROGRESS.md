# Research state

## Current state

The research goal is active. No correction method has been validated yet.
The repository has been initialized as DOIT Correction. Image
experiments will use frozen SD v1.5 and LAION aesthetic rewards on Orion before
any successful method is scaled to the existing D4RL/QGPO RL setup.

## What we learned

- `exp-000-checks` passed all 11 analytic and sampler checks on Orion.
- `exp-001-image-smoke` completed at 512 pixels with 7,298 MiB peak reserved
  memory and 21.94 seconds per image. It had no denominator events. However,
  consistent-Euler correction at gamma 4 and M=8 visibly oversaturated the
  image (reward 4.1543, CLIP alignment 0.20635). This prompted an M=32 strength comparison before fixed-state capture.
- All six `exp-004` calibration runs completed. At M=32, mean reward increased
  from 6.1129 (gamma 0) to 6.1485 (0.4), 6.2088 (1), and 6.4681 (4); gamma 4
  alignment was 0.3085 versus 0.3122 without guidance. Seed 11 at gamma 4 was
  visually coherent with M=32, unlike the M=8 smoke. Retain gamma 4 for the
  discovery capture to test this query-budget sensitivity. Two seeds do not
  establish an endpoint improvement; confirmation will include gamma 1 and 4.
- Confirmed by source inspection: the available image implementation already
  uses fixed temperature (reward centering, without division by batch standard
  deviation). The proposal's batch-normalization concern must be checked
  separately for RL; it is not present in this image source.
- Confirmed by source inspection: the Euler branch uses an Euler transition but
  a VP/DDIM endpoint formula and VP epsilon-correction scale. The new sampler
  will expose the source convention as a named comparison and use an explicitly
  consistent Euler parameterization for the primary study. This difference
  must not be credited to online variance reduction.
- The old image reward callback reloads CLIP on every call. All new methods will
  share one cached reward implementation and identical preprocessing.

## Open questions

1. Can a small, analytically integrable surrogate learn useful directions from
   one high-dimensional image rollout?
2. Does reduced moment/ratio error survive the score-network VJP and improve
   final reward without a loss of diversity or text alignment?
3. Is practical-endpoint bias larger than the Monte Carlo error being reduced?

## Next experiments

1. Capture nine guided states per M32/gamma4 trajectory for Bernese seed 23 and
   Formula 1 seed 11. Use 128-sample tuning banks and independent 512-sample
   references, plus 32 coupled full-future probes at steps 10 and 16.
2. Compare all proposed families and controls on frozen image states, including
   held-out references and true query costs; tune only on discovery seeds.
3. Evaluate retained methods on independent image rollouts. Evolve candidates
   in response to measured failures, then scale successful methods to RL.
