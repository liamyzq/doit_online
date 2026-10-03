# Research state

## Current state

The research goal is active. No correction method has been validated yet.
The local empty repository is being initialized as DOIT Correction. Image
experiments will use frozen SD v1.5 and LAION aesthetic rewards on Orion before
any successful method is scaled to the existing D4RL/QGPO RL setup.

## What we learned

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

1. Validate moment identities, importance weights, estimator update order, and
   Euler sampling; smoke one image and measure memory and runtime.
2. Compare all proposed families and controls on frozen image states, including
   held-out references and true query costs; tune only on discovery seeds.
3. Evaluate retained methods on independent image rollouts. Evolve candidates
   in response to measured failures, then scale successful methods to RL.
