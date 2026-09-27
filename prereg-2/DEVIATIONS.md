# Deviations and notes: protocol 2

Logged under prereg-2/PREREGISTRATION.md §11. The freeze commit is `c2474be59c90df62c40762a3a6d438269d3a1657`, and its
files are never edited.

## E1: stale status label (editorial; no change to content), 27 Sep 2026

At the freeze commit, the header of `PREREGISTRATION.md` still reads "**Status:** DRAFT, not frozen", and the first
comment line of `config/prereg.yaml` still reads "DRAFT until the freeze commit". The label was not updated before
committing.

The same header sentence defines the freeze point, "frozen at the first commit that contains this file", which is
`c2474be`. So which text binds is unambiguous. **Both files are frozen as committed.** Every threshold, task, seed
and rule in them is binding. The label is left as it is, because editing a freeze commit, even cosmetically, is what
the freeze exists to prevent.

When this note was written, no model had been run and no §5 statistic had been computed on any model.

## C1: how the §8 synthetic cases were built and judged (clarification; no verdict threshold involved), 27 Sep 2026

§8 names five validation cases but gives no construction details or tolerances. Writing them exposed two facts about
single-output systems, and both are recorded here. They affect only how the code is *validated*, never any §5 or §6
threshold.

1. **Distinct decay rates.** With one scalar output, modes that share an eigenvalue are observable along a single
   direction only. So every planted k-dimensional block (S1, S3) uses k distinct rates, or the block is not
   identifiable at all.
2. **"Recovers" and "aligns with" (S1) are judged relatively, with margins.** A scalar output makes the Hankel spectrum
   decay steeply, so a 16-dimensional *balanced* subspace is not exactly a 16-dimensional *modal* block. S1 therefore
   checks each of the following:
   - the energy-pooled subspace is ≥ 30° closer to the planted V than to the bulk U;
   - after growth normalisation, the pooled subspace lies within 20° of the bulk anchors' own balanced subspaces and
     ≥ 45° from the planted episode's anchors;
   - that pooled subspace is ≥ 10° closer to U than to V;
   - θ̂_stab ≤ 30° and ≥ 15° below the protocol-1 pooling.
3. **S5 uses long-memory random Jacobians** (A_t = 0.97 I + i.i.d. noise; B_t and C_t i.i.d.). With short-memory A_t
   there is no shared effective window, and so no artefact to exhibit.

Code: `tests/test_h2_synthetic.py`, with 8/8 passing before any model was run.
