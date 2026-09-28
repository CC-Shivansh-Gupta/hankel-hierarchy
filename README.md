# hankel-hierarchy

Does the Hankel singular value spectrum of a learned world model's latent dynamics define the
abstraction level for hierarchical planning?

This repository is preregistered. Read [`PREREGISTRATION.md`](PREREGISTRATION.md) first.
Every numeric parameter is in [`config/prereg.yaml`](config/prereg.yaml).

**Commit history is part of the protocol.**
1. The preregistration commit contains only the hypothesis, the protocol and the frozen
   parameters. No trained checkpoint had been evaluated when it was made.
2. The diagnostic commit adds the analysis code, the results and the figures for Gates A–C.
   The preregistration files underneath it are unchanged.
3. The planning experiments (D–G) stay sealed unless all three gates pass.

Negative results are kept and reported.

**Protocol 2** (a new hypothesis, not a revision) is preregistered in [`prereg-2/`](prereg-2/PREREGISTRATION.md),
frozen at `c2474be`. Its held-out verdict and result package are in [`RESULTS2.md`](RESULTS2.md), and its deviations
are in [`prereg-2/DEVIATIONS.md`](prereg-2/DEVIATIONS.md).
