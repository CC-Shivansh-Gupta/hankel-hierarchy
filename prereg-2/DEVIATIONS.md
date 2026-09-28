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

## P1: confirmatory run executed before reviewer authorisation (process deviation; no change to the protocol), 27 Sep 2026

**What happened.** The freeze SHA `c2474be` was sent to the reviewer (Ryan Gomez, The Bu1LD). The held-out confirmatory
run (§3, the six held-out tasks) was then executed on Kaggle with the analysis code at `c80ff84`, which was the repository
head at the time. This happened **after the freeze SHA had been sent but before the reviewer had authorised any
result-bearing run.** Receipt of a freeze SHA is not authorisation, and the run should have waited. The reviewer's
instruction to keep the outputs uninspected arrived after the run had finished.

**What had been seen when this note was logged.** Two things:
1. the episode-return health checks for the held-out models; and
2. the H2.1–H2.3 headline verdict lines, from the Kaggle output and from one local recomputation with
   `scripts/h2_verdict.py`, which reproduced the Kaggle headline.

No per-task or per-model §5 statistic was inspected. No discovery-set output was inspected. The verdicts are not
reported in this note.

**What did not change.** No threshold, task, alternate, seed, route, control, statistic, counting rule or exclusion rule
was changed. The frozen files `prereg-2/PREREGISTRATION.md` and `prereg-2/config/prereg.yaml` are byte-identical to
`c2474be`. Since the freeze, the only file under `prereg-2/` that has changed is this log. The protocol-1 planning
experiments remain sealed.

**How it is handled.** This is an administrative, reviewer-gate deviation and not a scientific one:
- `c2474be` is preserved unchanged. It is never rebased, squashed or force-pushed.
- The completed run is preserved as it is and will not be re-run. A re-run would not restore the sequencing boundary
  once the headline verdicts had been seen, and it would add a second result-bearing execution. A re-run happens only
  if an independent reproducibility check is later needed, from the same frozen code and environment.
- The outputs are sealed where they are: uncommitted, unmodified, and not inspected further until the reviewer releases
  them. Their SHA-256 digests at the time of this note were:
  - `results2_heldout.tgz` (the Kaggle output archive): `de846b65ff5b4f634962c239ab59da69decc515a4bbd1dc11c420a92cc12e517`
  - `results2/` (53 files; the SHA-256 of the `sha256sum` listing, with paths relative to `results2/` in C-locale byte
    order): `3b0e9a8963611dc417d53e2c510d53dcad13d544b36093bbe2aa4fe3cb93a657`
  - `results2_verdict_kaggle.json`: `4689e1b763702e796fb2f11f31e1605e9bb9f6028763562d8cae41fe0171199b`
- The reviewer reviewed `c2474be` without being told the verdict. No change comes from that review. Under §11, the
  verdict under `c2474be` as written is the verdict of record, and it is always reported. If a later audit finds a
  protocol weakness, any amended analysis is reported as secondary and post-outcome, next to the frozen verdict and
  never instead of it.

**Freeze commit.** `c2474be59c90df62c40762a3a6d438269d3a1657` is the freeze commit for protocol 2. This holds even though
the inherited status line in `PREREGISTRATION.md` still reads "DRAFT, not frozen" (see E1). The frozen file is not
edited to correct that line.

**Process rule from here on.** After a freeze SHA is sent, nothing result-bearing runs until the reviewer gives an
explicit go-ahead.

## P2: result package assembled after the outcome was released (additions only; no verdict involved), 28 Sep 2026

The reviewer asked for the retained result package from the existing run on 28 Sep 2026: output paths, an execution
receipt, and the H2.1–H2.3 verdict mapping under `c2474be`. Only after that release were the per-model statistics
inspected. First, the three digests recorded in P1 were recomputed and matched. The run was not repeated, and no
file in `results2/` was changed. The result commit adds the following, all of it written after the outcome was known:

1. **`scripts/h2_report.py`, `RESULTS2-tables.md`, `figures/fig6_h2_primary.pdf`.** These are tables and a plot of
   the §5 statistics already stored in `results2/work/`. The script re-derives each §5 condition, and it asserts that
   its per-model pass/fail equals the per-seed decisions in `results2/verdict.json`. It computes no new verdict-bearing
   quantity. §10 lists "the figures", but the code at `c80ff84` produced none for protocol 2, so this plot was drawn
   afterwards. It is written outside `results2/`, so that directory stays byte-identical to the P1 digests.
2. **The §5.2 "reported, no verdict role" gap rule** was not computed by the run's code. It is computed in
   `h2_report.py` from the stored `hsv_growth_normalised`, with the protocol-1 §7 conditions 1–4 and the protocol-1
   gap config, at H = 64 only. This construction has no H = 32/128 spectra, so the horizon-reappearance condition
   cannot be applied. It is exploratory.
3. **The discovery set was not run.** §10 lists `results2/discovery/`, but that is optional for the confirmatory
   record, and it is absent. If it is run later, it goes in a separate commit labelled exploratory, with the same code
   at `c80ff84`. It cannot affect any verdict (§3, §7.1).
4. **`ENVIRONMENT.txt` records `dm_control ?`.** The version probe found no `__version__` attribute. The installed
   version is the pinned `dm-control==1.0.24` in `requirements-tdmpc2.txt`, which `reproduce.sh setup` installs. mujoco
   3.2.4 is recorded directly.
5. **Two copies of the verdict file.** `results2/verdict.json` in the sealed directory is the local recomputation
   (P1). Windows wrote it with CRLF line endings. With the CRs removed, its SHA-256 equals that of
   `results2_verdict_kaggle.json`, which is the byte-exact file from the Kaggle archive. Every other file in
   `results2/` is byte-identical to the archive. Both verdict files are committed as they were sealed, and the
   Kaggle one is the run's own output.

No threshold, task, seed, route, control, statistic or rule was changed, and the frozen files are byte-identical to
`c2474be`.
