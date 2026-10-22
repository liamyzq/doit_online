# Exp-058 visual review

All 256 delivered images were reviewed at the end of the complete queue,
using both four-prompt contact sheets for every reserved seed:
59011, 59017, 59023, 59029, 59039, 59051, 59057 and 59063.
Columns are selected MC32, MC8, terminal N4 and base. No output was excluded,
replaced, regenerated or used to retune a parameter. The airplane and lighthouse
N4 outputs at seed59051 were also inspected at their full 512-pixel resolution.

Terminal N4 has no distinct new visual failure mode relative to the controls.
It does not show the strong-guidance texture and block collapse that disqualified
previous project candidates. The comparative quality criterion is met, while
base-model semantic and geometry failures remain. This is a qualitative judgment
about this complete batch, not an artifact-free guarantee or a measured human
preference score.

Concrete limitations retained in the comparison include:

- Terminal N4 lighthouse seed59051 omits the requested lighthouse.
- Terminal N4 airplane seed59051 has malformed, closely cropped aircraft geometry.
- Terminal N4 Labrador seed59029 adds a blue garment and has low prompt alignment.
- Labrador seed59011 has duplicated-tail anatomy across all four methods.
- Base and guided airplane seed59057 show a cropped wing rather than a complete
  aircraft; other base/control images contain synthetic lettering and anatomy errors.
- Selected MC32 lighthouse seed59029 has strong colored high-frequency corruption,
  and its airplane seed59063 has conspicuous text/texture corruption.

The first three are real N4 limitations, not removed as outliers. The aggregate
alignment gain does not imply a gain for every prompt: Labrador loses 0.014231
alignment on average. N4 seed59051 also loses 0.33034 mean reward against MC8.

Four complete comparison sheets are retained in Git so the report exposes both
favorable and unfavorable cases, including these examples:

- [Seed59029, airplane/Bernese/cafe/Formula1](../quality/confirmation-s59029-1.png)
- [Seed59029, Labrador/lighthouse/mountain/orchid](../quality/confirmation-s59029-2.png)
- [Seed59051, airplane/Bernese/cafe/Formula1](../quality/confirmation-s59051-1.png)
- [Seed59051, Labrador/lighthouse/mountain/orchid](../quality/confirmation-s59051-2.png)

All 16 review sheets remain beside this file locally and can be regenerated with
the recorded analysis command. All original images and embeddings are retained
locally in their run folders and on Orion under the recorded storage paths.
The scientific conclusion uses the complete 64-image group means and paired
seed-block intervals, not a selection of favorable pictures.
