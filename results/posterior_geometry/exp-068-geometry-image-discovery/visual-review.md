# Fixed-state proxy visual review

All 128 saved discovery candidates were reviewed in four comparison sheets.
Rows are frozen, center identity, center isotropic, spectral ranks 2/4/8,
random-subspace response, and individual candidate denoising. Each column
uses the same Gaussian candidate across proxies. These are intermediate-state
proxy decodes, not delivered images or independent endpoint evaluations.

At index 10, frozen and center-identity decodes are heavily corrupted by
high-frequency colored texture in both prompts. Isotropic contraction removes
most of that visible corruption. Spectral ranks and the random response look
very similar to the isotropic result; the individual denoiser has additional
candidate-specific changes in the cafe facade and lighthouse coastline.
The contracted proxies remain soft and do not reproduce all of those changes.

At index 18, all proxies depict the same recognizable cafe or lighthouse scene,
with smaller differences in window details, cliff texture and lighting. The
spectral variants have no distinct visible improvement over the isotropic or
random controls in this saved subset. This observation does not establish
endpoint quality, prompt fidelity across seeds, or restored future uncertainty.

- [Cafe, index 10](../quality/cafe-s68011-step10.jpg)
- [Cafe, index 18](../quality/cafe-s68011-step18.jpg)
- [Lighthouse, index 10](../quality/lighthouse-s68011-step10.jpg)
- [Lighthouse, index 18](../quality/lighthouse-s68011-step18.jpg)

Raw PNGs and reference-vector NPZ files remain in the local run directory and
the matching Orion storage directory. Contact sheets are committed. No saved
candidate was removed or regenerated after inspection.
