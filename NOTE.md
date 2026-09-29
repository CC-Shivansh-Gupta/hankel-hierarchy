# Two preregistered negative results on Hankel structure in learned world-model dynamics

**Technical closeout note.** Shivansh Gupta (IIT Bombay). Bu1LD research track, reviewed by Ryan Gomez. 29 Sep 2026.

This note closes the track. It restates both studies exactly as they were frozen and run. It adds no analysis, runs
nothing new and revises no verdict. Every number below is taken from files committed at the SHAs listed in §4. The
confirmatory record (§2–§6) is kept separate from exploratory material (§7). The proposals in §9 are proposals only:
nothing in them is frozen, and nothing in them has been run.

## 1. The original question

Does the Hankel singular value (HSV) spectrum of a learned world model's latent dynamics define the abstraction level for
hierarchical planning? The idea imports balanced truncation (Moore, 1981). The high-level state would be the latent
directions that are jointly controllable by actions and observable in predicted value. The directions below a spectral
knee would be left to a reactive low-level controller.

The model class is TD-MPC2 (Hansen, Su & Wang, ICLR 2024), single-task, state-based DMControl checkpoints at
`model_size=5`. The code is pinned at `nicklashansen/tdmpc2@e9f5932`, and the checkpoints at Hugging Face revision
`8fb2a82e`, with every file SHA-256-pinned. The shared construction is:
- Jacobians A = ∂f/∂z and B = ∂f/∂a of the latent dynamics, and C = ∂Q̄/∂z of the mean of the five decoded Q heads.
  All are taken at encoded real states along 10 evaluation episodes, in float64.
- A 448-dimensional SimNorm tangent projection.
- 250 anchors per model.
- Finite-horizon time-varying Gramians W_c and W_o.

## 2. Protocol 1: is there a spectral knee?

**Frozen** at `0173614f918c0828b0f978378f1c0c2807be9990` (`PREREGISTRATION.md`, `config/prereg.yaml`).
**Result** at `f0e8c94c2a111b9bb8d30f8857cd36ba3824ac98` (`RESULTS.md`). **Review** at `383eb5b` (`REVIEW.md`).

**Hypothesis and gates.** There are three gates, each judged on 6 tasks × 3 seeds:
- **A, a gap exists.** A pooled-HSV log-drop d_k ≥ 1 decade at 2 ≤ k ≤ H/2 is required. It must also be at least 3× the
  local median drop, reappear within ±25% at the robustness horizons, and not be reproduced by the R-full or R-dyn
  random-initialisation controls.
- **B, not PCA.** The model collapses to PCA if ρ_S > 0.9 or θ_pca < 15°.
- **C, stable.** θ_stab ≤ 30°, where θ_stab is the mean angle between the per-anchor and pooled balanced subspaces.

The primary horizon is H = 64, with 32 and 128 as robustness horizons. **Decision rule:** a gate passes if at least 4
of 6 tasks pass it, and a task passes if at least 2 of 3 seeds pass. The hypothesis survives only if A, B and C all
pass.

**Tasks.** walker-walk, cheetah-run, hopper-hop, quadruped-run, humanoid-walk and finger-spin, with 18 checkpoints.

**Result: REJECTED.**

| Task | Seeds analysed | Largest eligible log-drop (decades; need ≥ 1) | θ_stab (°; need ≤ 30) | ρ_S (collapse > 0.9) | θ_pca (°; collapse < 15) | A | B | C |
|---|---|---|---|---|---|---|---|---|
| walker-walk | 3 | 0.18–0.40 | 60.3–62.6 | 0.46–0.66 | 79.7–80.5 | ✗ | ✓ | ✗ |
| cheetah-run | 3 | 0.35–0.70 | 61.1–67.5 | 0.28–0.58 | 80.1–81.9 | ✗ | ✓ | ✗ |
| hopper-hop | 3 | 0.23–0.48 | 62.2–66.0 | 0.26–0.56 | 80.5–81.1 | ✗ | ✓ | ✗ |
| quadruped-run | 3 | 0.22–0.34 | 70.9–71.0 | 0.24–0.40 | 79.7–80.6 | ✗ | ✓ | ✗ |
| humanoid-walk | 2 (2-of-2) | 0.36–0.50 | 60.0–60.5 | 0.54–0.55 | 77.0–78.8 | ✗ | ✓ | ✗ |
| finger-spin | 3 | 0.28–0.66 | 62.9–70.9 | 0.65–0.73 | 79.9–81.2 | ✗ | ✓ | ✗ |
| **Tasks passing** | | | | | | **0/6** | 6/6 | **0/6** |

No model has a qualifying gap, so Gate A fails before the random-initialisation controls are consulted. The balanced
ordering is not PCA (B passes), but it is not stable along trajectories either (C fails). Values are from the
verdict-of-record route (§5). Gates B and C use the default k = 16, because A failed.

**Protocol-1 deviations** (`DEVIATIONS.md`):
- **D1** added the square-root factor route as a second numerical route. It was written before any checkpoint was
  loaded.
- **D2** was written after the run had loaded checkpoints but before any spectrum had been inspected. It set a
  pre-fixed return bar for key-renaming the pre-release `humanoid-walk-3.pt`. The converted model scored a mean return
  of 1.3 against a bar of 714, so it was excluded, and humanoid-walk was judged 2-of-2.
- **C1–C11** are clarifications of the frozen text. C11 moved the simulator pins to dm-control 1.0.24 and mujoco 3.2.4
  for NumPy 2.

## 3. Protocol 2: is the null explained by transient growth?

**Frozen** at `c2474be59c90df62c40762a3a6d438269d3a1657` (`prereg-2/PREREGISTRATION.md`, `prereg-2/config/prereg.yaml`).
`c2474be` is the freeze commit even though the status line it inherits reads "DRAFT, not frozen". That line is logged
as E1 and was not edited. **Code** at `c80ff84d577645a6dadcf54e12c06e7bf8669674`. **Result** at
`00a361640cdcbca0bdd728ddb6d9dcd544a3bbbd` (`RESULTS2.md`, `RESULTS2-tables.md`).

**Motivation, and why it was a new hypothesis.** Protocol 1's exploratory diagnostics showed locally expansive learned
dynamics. That suggested the pooled, energy-averaged Gramians might be dominated by a few fast-growing anchors.
Protocol 2 tested that suggestion as three separate hypotheses, with no disjunctive rescue:

| | Hypothesis | Model criterion (all conditions required) |
|---|---|---|
| H2.1 | Growth dominance (H = 64) | κ_c, κ_o ≥ 0.5 (top-decile energy share); ρ_c, ρ_o ≥ 0.7 (Spearman of energy against local growth); θ_bulk − θ_top ≥ 15°; κ > the paired R-full κ |
| H2.2 | Growth-normalised stability (H = 64, k = 16) | θ̂_stab ≤ 30°, **and** ≤ the paired R-full θ̂_stab − 15°. Here the dynamics are discounted by the model's median growth rate and the anchors are pooled with equal energy |
| H2.3 | Trajectory-local structure (H = 16, k = 4) | θ_near ≤ 30° (anchors 40 steps apart, so the windows do not overlap); Δ = θ_far − θ_near ≥ 15°; Δ ≥ the paired R-full Δ + 10° |

**Evidence boundary.** Verdicts come only from six held-out tasks that no analysis had touched: reacher-hard,
finger-turn-hard, hopper-stand, walker-run, quadruped-walk and humanoid-run. That is 17 checkpoints. humanoid-run-3 is
in the pre-release layout, found by a key-only screen *before* the freeze, so humanoid-run is judged 2-of-2 on seeds
1–2. The 17 protocol-1 models are a **discovery set with no verdict role**. **Decision rule:** the same 2-of-3 and
4-of-6 counting as protocol 1, applied to each hypothesis separately. **Control:** R-full only.

**Result: H2.1, H2.2 and H2.3 all REJECTED.**

| Task | Seeds | κ (H2.1) | θ̂_stab ° (H2.2) | Δ ° (H2.3) | Seeds meeting H2.1 / H2.2 / H2.3 |
|---|---|---|---|---|---|
| reacher-hard | 3 | 0.19–0.20 | 57.3–58.7 | −1.2 to 0.4 | 0 / 0 / 0 |
| finger-turn-hard | 3 | 0.61–1.00 | 60.8–63.7 | −2.0 to 4.2 | 0 / 0 / 0 |
| hopper-stand | 3 | 0.20–0.28 | 38.6–58.1 | 0.2 to 1.2 | 0 / 0 / 0 |
| walker-run | 3 | 0.78–0.84 | 59.2–60.2 | −4.2 to −3.7 | 0 / 0 / 0 |
| quadruped-walk | 3 | 0.18–0.22 | 47.3–47.8 | −1.2 to 0.8 | 0 / 0 / 0 |
| humanoid-run | 2 (2-of-2) | 0.60–0.80 | 56.5–59.6 | 13.3 to 13.9 | 0 / 0 / 0 |
| **Tasks supporting (need ≥ 4 of 6)** | | | | | **0 / 0 / 0** |

**Which conditions failed (17 models).**
- **H2.1:** κ > R-full held for 17/17, but the subspace-domination condition held for only 2/17, and no model met all
  six conditions.
- **H2.2:** θ̂_stab ≤ 30° held for 0/17, and the R-full margin for 0/17.
- **H2.3:** θ_near ≤ 30° held for 4/17, Δ ≥ 15° for 0/17, and the R-full margin for 0/17.

Every per-model value and every condition mark is in `RESULTS2-tables.md`.

## 4. Freeze, code and result record, with retained digests

| Stage | Commit | Contents |
|---|---|---|
| P1 freeze | `0173614f918c0828b0f978378f1c0c2807be9990` | Protocol 1 and its parameters |
| P1 code | `7fa0c6b`, `5bd91b9` (pipeline, validated on synthetic systems before any checkpoint was loaded); `7a85331`, `62d39a7` (C11); `9163ef2`, `dfb5c3b` (D2) | Pipeline, simulator pins and the D2 conversion |
| P1 result | `f0e8c94c2a111b9bb8d30f8857cd36ba3824ac98` | `results/` (git tree `d46392214196bc5fa2d3f787fa1ee24713e2f458`), `RESULTS.md`, figures 1–5 |
| P1 review | `383eb5b639990fea761b883dfaba77934c8ea1ff` | `REVIEW.md`: claim boundaries; the verdict stands |
| P2 freeze | `c2474be59c90df62c40762a3a6d438269d3a1657` | Protocol 2 and its parameters |
| P2 editorial | `e61a848` | E1: the stale DRAFT label |
| P2 code | `c80ff84d577645a6dadcf54e12c06e7bf8669674` | `scripts/h2.py`, `collect_h2.py`, `h2_verdict.py`; synthetic S1–S5, 8/8 passing |
| P2 deviation | `f19f8ac6c3affac822908cebf45ff67b9ea52a11` | P1: the sequencing deviation (§5) |
| P2 result | `00a361640cdcbca0bdd728ddb6d9dcd544a3bbbd` | `results2/` (git tree `9d2ac61c4d2cc59972e5befd5c39f18b4f41dc05`), `RESULTS2.md`, P2 |

**Retained digests (SHA-256).** The two archives are kept locally and are not committed, under the `*.tgz` ignore rule.
- `results_final.tgz`, the protocol-1 run archive: `11c298be4f6e200f00267485d6c1612aa23abbb3f3516a9366cd817dbc70e3a2`
- `results2_heldout.tgz`, the protocol-2 run archive: `de846b65ff5b4f634962c239ab59da69decc515a4bbd1dc11c420a92cc12e517`
- The `results2/` manifest: `3b0e9a8963611dc417d53e2c510d53dcad13d544b36093bbe2aa4fe3cb93a657`
- `results2_verdict_kaggle.json`: `4689e1b763702e796fb2f11f31e1605e9bb9f6028763562d8cae41fe0171199b`

**What is unchanged at the commit containing this note:**
- The four frozen files are byte-identical to their freeze commits.
- `results/` and `RESULTS.md` are unchanged since `f0e8c94`.
- `results2/` and `RESULTS2.md` are unchanged since `00a3616`.

## 5. Process deviation, and how it was handled

**What happened.** The protocol-2 confirmatory run was executed on 27 Sep 2026, after the freeze SHA had been sent to
the reviewer but **before** the reviewer authorised any result-bearing run.

**What was seen, and when.** Before the deviation was logged, only the episode returns and the H2.1–H2.3 headline
verdict lines had been seen. No per-task or per-model statistic had been seen. The reviewer then reviewed `c2474be`
without being told the verdict.

**How it was handled:**
- **Logged** as an administrative, reviewer-gate deviation (P1, `f19f8ac`), not a scientific one.
- **Not re-run.** A re-run cannot restore the sequencing boundary once headline verdicts have been seen, and it would add
  a second result-bearing execution.
- **Sealed.** The outputs were sealed by SHA-256 and left uncommitted until the reviewer released them. The digests were
  recomputed on release and matched. The outputs were then committed unchanged.
- **Nothing changed.** No threshold, task, seed, route, control or rule was changed at any point.
- **Verdict of record.** The verdict under `c2474be` is the verdict of record. Any later amended analysis would be
  secondary and post-outcome.

**What was added after the release (P2).** Tables, a figure and one exploratory gap-rule computation were added. None
is verdict-bearing.

**The execution receipt has one weak point.** The code that ran is identified from timestamps and commit order:
`c80ff84` was the head of `main` for the whole run. The Kaggle console log was not kept, so no printed SHA confirms it.

**Protocol 1.** D2 is the only entry written after checkpoints had been loaded, and it records what had been seen at
that time.

## 6. What the agreement between numerical routes does and does not establish

Two computations of the same mathematical objects are run on every model:
- **The formed-Gramian route:** sum W_c and W_o, eigendecompose, then take the SVD.
- **The square-root factor route:** build the factors from the Jacobian products, pool them by incremental QR, then take
  the SVD.

With a scalar output, cond(W_o) reaches about 1e18, which is beyond float64.
- **Protocol 1:** the formed-Gramian route (§6) was the verdict of record under §14, with the factor route as the D1
  check.
- **Protocol 2:** the factor route decided, and the formed-Gramian route was the check.

**What it establishes.**
- In both protocols the two routes give the same verdict on every gate, hypothesis and task:
  - protocol 1: `routes_agree: true`, with the largest eligible log-drop per model agreeing to within 1e-14 decades;
  - protocol 2: no flagged disagreement, and 0/6 on both routes for all three hypotheses.
- So the rejections are not an artefact of float64 precision loss in one implementation. In particular, precision loss
  did not erase a real knee: the largest drop found anywhere is 0.70 decades, against a 1-decade threshold, on both
  routes.

**What it does not establish.**
1. **The routes are not independent evidence about the model.** Both use the same rollouts, anchors, autograd Jacobians,
   tangent projection, output map Q̄ and controls. Agreement says nothing about whether those shared inputs are the
   right objects. In particular:
   - whether a first-order product of 64 Jacobians represents the nonlinear model at all (never tested; see §9, Q1);
   - whether Q̄ is the right output map;
   - whether a finite-horizon time-varying Gramian is the right lens.
2. **It is not a replication.** Both routes come from the same code, the same run and the same machine. Neither protocol
   was reproduced on independent hardware.
3. **Agreement on verdicts is not agreement on every statistic.**
   - In protocol 2, the undiscounted pooled angles on hopper-stand differ by up to 14.3° between routes, where the
     median ‖Φ(τ+64, τ)‖₂ is about 3–5e7. One condition-level result depends on the route: hopper-stand-1's
     θ_bulk − θ_top is 17.8° on the factor route and 6.3° on the Gramian route. That model fails H2.1 on κ under both.
   - In protocol 1, θ_stab differs by up to 3.9°.
   - These differences sit far from every decision threshold that mattered. Still, they show that the statistics which
     depend most on growth are the least numerically robust.

## 7. Exploratory material (outside the confirmatory record)

This section is kept separate as it stands in the result files. It has no verdict role, and none of it qualifies any
rejection.
- **Protocol 1** (`RESULTS.md`, exploratory section): training concentrates the spectrum smoothly relative to the
  controls. The learned latent dynamics are locally expansive: the spectral radius exceeds 1 at 71–100% of steps.
- **Protocol 2** (`RESULTS2.md`, exploratory section):
  - the Gramian route;
  - the §5.2 ablations;
  - the protocol-1 gap rule applied to the growth-normalised spectra, where no model has a gap;
  - the observation that R-full is strongly contractive (γ 0.55–0.82 in 16 of 17 controls), while the trained models
    have γ 0.92–1.32.
- **Discovery set:** never run under protocol 2. It remains non-confirmatory, and it is not run as part of this
  closeout.

## 8. Claim boundary, limitations and conclusion

**Claim boundary.** Every statement here is limited to:
- TD-MPC2 single-task, state-based DMControl checkpoints at the pinned code and revision;
- 12 tasks in total (6 per protocol), with 3 released seeds each, except where a seed was excluded;
- the latent linearisation at encoded real states along evaluation-mode trajectories;
- the 448-dimensional tangent coordinates;
- the frozen output map Q̄, the mean of all five decoded Q heads. This is a surrogate for TD-MPC2's own two-random-head
  min/average evaluation path (`REVIEW.md`, boundary 3);
- the horizons, k and pooling constructions as frozen.

R-full is an anti-degenerate architecture control, not a literal random agent. R-dyn (protocol 1 only) is a
rejection-only control with no mechanistic reading. Nothing here addresses planning performance, pixels, distractors,
other architectures or other output maps. Protocol-1 experiments D–G, and every planning, pixel or distractor
experiment, remain **sealed**.

**Limitations.**
1. **Linearisation validity** over 64 steps (and 16 steps for H2.3) was assumed and never measured. Where the median
   growth over one window reaches 1e5–1e8 (walker-run, hopper-stand and humanoid-run; up to 2e8 in protocol 1),
   first-order validity over the full window is doubtful. That limits what the negative results say
   about the nonlinear model. It does not change them.
2. **One model family and one output map.** The DreamerV3 secondary model was dropped before the protocol-1 freeze,
   because no official DMControl checkpoint exists.
3. **R-full's growth profile.** It is contractive where the trained models are expansive. That makes it a lenient
   comparator for H2.1's κ condition and a demanding one for the angle tests of H2.2 and H2.3.
4. **Numerical robustness.** The growth-dominated statistics are the least numerically robust (§6.3).
5. **Execution receipt.** The protocol-2 code identity rests on timestamps and commit order, not on a printed SHA (§5).
6. **Sample size.** Two to three seeds per task and six tasks per protocol. This is enough for the preregistered
   counting rules, but not for estimating effect sizes.

**Conclusion.** The preregistered tests rule out a specific set of claims, for this model class and construction:
- The pooled Hankel spectrum of TD-MPC2's value-observed latent dynamics has no preregistered knee (P1-A).
- Its balanced subspace is not stable along trajectories (P1-C).
- The instability is not explained as dominance by fast-growing anchors in the preregistered sense (H2.1).
- It is not removed by growth-discounting with equal-energy pooling (H2.2).
- It is not replaced by a trajectory-local structure that beats an untrained control (H2.3).

The balanced ordering does differ from variance ordering (P1-B). But under both preregistered constructions, and on
held-out models for the second, it gives no stable abstraction to plan over. The original hypothesis, that this
spectrum sets the level of abstraction for hierarchical planning, is therefore not supported at the diagnostic stage.
Its planning predictions were never tested, and they stay sealed.

## 9. Possible successor questions (proposals only: nothing frozen, nothing run)

Each of these would need its own preregistration and its own evidence boundary before any result-bearing run. None
tests Hankel structure again, and none can change a verdict above.

**Fresh evidence boundary for any successor.** Released TD-MPC2 checkpoints on DMControl tasks that no analysis has
touched. The 35 checkpoints loaded in either protocol and their controls are excluded from any confirmatory role. That is 18
in protocol 1, counting the excluded humanoid-walk-3, and 17 in protocol 2. Candidates whose files are already hashed,
with no forward pass run on any of them, are walker-stand, cheetah-jump, reacher-easy, finger-turn-easy and
hopper-hop-backwards (`prereg-2/config/prereg.yaml`, `alternate_sha256`). humanoid-stand is excluded, because its
state-dict keys were inspected.

**Q1. Over what horizon is first-order linearisation of TD-MPC2's latent dynamics predictive of the model's own
nonlinear response?**
- *Motivation:* both protocols rest on Jacobian products over 16–64 steps, and neither measured whether those products
  describe the network (limitation 1).
- *Why it is not a rewrite:* it tests the validity of the lens, not the presence of any structure, and no outcome can
  produce a Hankel-structure claim.
- *A falsifiable form:* inject preregistered perturbation sizes δ at anchors. Compare Φ(τ+h, τ)δ with the difference
  between the model's nonlinear rollouts, and define the validity horizon h\* as the first h at which the relative error
  exceeds a frozen tolerance.
- *Cost:* forward passes only, with the existing pipeline and fresh checkpoints.
- *Priority:* first, because its answer bounds how any linear-systems analysis of these models, including the two
  above, should be read.

**Q2. Does the finite-time growth of the learned latent dynamics track the finite-time growth of the true simulator
along the same trajectories?**
- *Motivation:* trained latent dynamics were expansive where untrained ones were contractive (an exploratory
  observation, §7). It is unknown whether that growth is inherited from the physics or produced by the model.
- *Why it is not a rewrite:* it asks where the growth comes from, not whether it hides a subspace.
- *A falsifiable form:* compare per-anchor latent growth ‖Φ(τ+h, τ)‖₂ with the simulator's finite-difference state
  Jacobian products over the same windows and actions. Use a frozen rank-correlation threshold and fresh tasks.
- *Cost:* moderate.
- *What it must guard against:* latent and physical state spaces differ in dimension and scale, so only scale-free
  comparisons are admissible.

**Q3. When during training does the latent dynamics become expansive, and does that coincide with the rise in episode
return?**
- *Motivation:* it is the same exploratory contrast, trained against initialised, read as a question about learning.
- *Why it is not a rewrite:* it is about training dynamics, not Hankel structure.
- *A falsifiable form:* track a frozen growth statistic across training checkpoints on fresh tasks, against return.
- *Cost:* high. Released checkpoints are final-only, so this needs training runs and compute not currently available.
  Listed for completeness, and lowest priority.
