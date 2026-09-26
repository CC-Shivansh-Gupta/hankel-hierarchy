# Review and claim boundaries

The diagnostic result is commit `f0e8c94c2a111b9bb8d30f8857cd36ba3824ac98`, and it is preserved unchanged. This file
adds context and revises nothing in it. `RESULTS.md`, `PREREGISTRATION.md` and `config/prereg.yaml` are untouched.

**Review.** Ryan Gomez (The Bu1LD) reviewed the frozen preregistration (`0173614`), the pinned TD-MPC2 code, the
deviation log and the diagnostic record on 26 Sep 2026. His conclusion: **the negative verdict stands.**

## Why the rejection does not depend on the contestable choices

- **The random-initialisation controls never enter the verdict.** Gate A fails before the control comparison is
  reached, because none of the 17 analysed models has a qualifying gap at H = 64. The largest eligible log-drop is
  0.70 decades, and the frozen threshold is 1.0. No change to the control construction could rescue Gate A.
- **The D2 exclusion does not carry the result.** Its acceptance bar was fixed before the converted checkpoint first
  ran, and the model was excluded before any spectrum of it existed. Gates A and C fail across all other models.
- **The strongest form of the result is a conjunction.** There is no preregistered spectral knee, and local-to-pooled
  balanced subspaces are 60–71° apart. Both numerical routes (D1) agree on every gate.

## Claim boundaries

These are limitations on what the result means. None is a protocol error, and none may be changed after the fact.

1. **R-full is an anti-degenerate architecture control, not a literal random agent.** The pinned constructor
   zero-initialises the final Q layer, which would make C ≡ 0. R-full re-draws that layer (§7), so it is a documented
   deviation from the literal constructor.
2. **R-dyn is a rejection-only stress control.** It pairs random dynamics with the *trained* Q map at *trained*
   anchor coordinates, so the latent semantics of the two parts were not learned jointly. It can make the test
   harder to pass. It supports no positive or mechanistic claim.
3. **The output map is a frozen surrogate.** Q̄, the mean over all five decoded heads (§5), is not TD-MPC2's own
   evaluation path, which decodes the ensemble and then takes the min or average of two randomly subsampled heads.
   Q̄ was frozen before any outcome existed. The result is a statement about this output map, and Q̄ will not be
   changed post hoc.

## What follows

Experiments D–G stay sealed. The exploratory observation in `RESULTS.md` is that the learned latent dynamics are locally
expansive and the pooled Gramians anchor-dominated. It may be pursued only as a **new, separately preregistered
hypothesis**, and never as a repair of this one. The rejection above stands underneath any successor.
