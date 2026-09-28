# Results — protocol 2 (transient growth and Hankel structure), held-out confirmatory run

**Verdict under the freeze commit `c2474be59c90df62c40762a3a6d438269d3a1657`: H2.1, H2.2 and H2.3 are all REJECTED**
(prereg-2/PREREGISTRATION.md §6). This is the verdict of record. No threshold, task, seed or rule was changed to reach
it, and none will be changed after it (§6.1, §11).

| Hypothesis | Primary statistic | Tasks supporting (§6: need ≥ 4 of 6) | Models meeting the full criterion | Verdict |
|---|---|---|---|---|
| H2.1 growth dominance | κ = min(κ_c, κ_o) | **0 of 6** | 0 of 17 | **REJECTED** |
| H2.2 growth-normalised stability | θ̂_stab | **0 of 6** | 0 of 17 | **REJECTED** |
| H2.3 trajectory-conditioned structure | Δ = θ_far − θ_near | **0 of 6** | 0 of 17 | **REJECTED** |

**`c2474be` is the protocol-2 freeze commit**, even though the status line it inherits in `PREREGISTRATION.md` reads
"DRAFT, not frozen" (prereg-2/DEVIATIONS.md E1). The frozen files are byte-identical to `c2474be` at this commit.

**Process record.** This run was executed after the freeze SHA was sent to the reviewer but before the reviewer
authorised it. That is logged as a process deviation, P1 (`f19f8ac`). At that point only the episode returns and the
headline verdict lines had been seen. The run was not repeated, and its outputs are the ones sealed by digest in P1.
The per-model statistics were inspected only after the reviewer released the package (P2).

## Result package

| What | Path |
|---|---|
| Frozen protocol and parameters | `prereg-2/PREREGISTRATION.md`, `prereg-2/config/prereg.yaml` (at `c2474be`) |
| Deviations: E1 (status label), C1 (synthetic cases), P1 (sequencing), P2 (this package) | `prereg-2/DEVIATIONS.md` |
| Verdict-bearing output: per-hypothesis verdicts, per-task and per-seed decisions, route check | `results2_verdict_kaggle.json` (byte-exact from the run). `results2/verdict.json` is the identical local recomputation with CRLF line endings (P2 item 5) |
| Per-model §5 statistics, 17 trained and 17 paired R-full, both numerical routes, per-anchor values | `results2/work/{task}_s{seed}.json`, `results2/work/{task}_rfull{1000,1001,1002}.json` |
| Per-model receipts: checkpoint SHA-256, the 10 episode returns, wall time | `results2/diagnostics/{task}_s{seed}.json` |
| Run environment | `results2/ENVIRONMENT.txt` |
| All tables below, generated from the files above | `RESULTS2-tables.md` (`python -m scripts.h2_report`) |
| Figure: the primary statistics against thresholds and R-full | `figures/fig6_h2_primary.pdf` |
| Code that produced the outputs | `c80ff84`: `scripts/h2.py`, `scripts/collect_h2.py`, `scripts/h2_verdict.py`, `kaggle/h2_kaggle.ipynb` |

## Execution receipt

- **Code.** `c80ff84d577645a6dadcf54e12c06e7bf8669674`. The Kaggle notebook clones `main` and runs
  `reproduce.sh setup → test → h2-smoke → h2-heldout (per task) → h2-verdict`. The earliest output in the archive is
  dated 01:08 UTC on 27 Sep 2026, 31 minutes after `c80ff84` was committed (00:37 UTC). The last is dated 06:11 UTC.
  The next commit, `f19f8ac`, was made at 14:06 UTC. So `c80ff84` was the head of `main`, and the only commit
  containing the protocol-2 code, for the whole run. The Kaggle console log was not kept, so this is established from
  the timestamps and commit order rather than from a printed SHA.
- **Hardware and stack.** Kaggle, Tesla T4, torch 2.10.0+cu128, numpy 2.0.2, mujoco 3.2.4 and dm-control 1.0.24 (the
  pinned version; the probe printed `?`, see P2), tensordict 0.8.3.
- **Models.** All 17 frozen held-out checkpoints loaded. Each file's SHA-256 was checked against the frozen config at
  download (`reproduce.sh`) and is recorded again in its diagnostics file. All 17 match. There were no load failures,
  no exclusions and no substitutions. `humanoid-run` is judged 2-of-2 on seeds 1–2, as frozen.
- **Controls.** The paired R-full controls are torch seeds 1000/1001/1002 for trained seeds 1/2/3 (humanoid-run:
  1000/1001).
- **Behaviour.** The mean returns over 10 episodes are 961–986 (reacher-hard, finger-turn-hard, quadruped-walk),
  866–965 (hopper-stand), 883–891 (walker-run) and 669–700 (humanoid-run). Two hopper-stand seeds have one low episode
  each (sd 289 and 164). Protocol 2 has no return gate, and none was applied.
- **Time.** Trained-model collection took 5.33 h, at 17–23 min per model.
- **Recomputation.** `python -m scripts.h2_verdict` run locally on the retained `results2/` reproduces the Kaggle
  verdict file exactly, apart from line endings.
- **Integrity.** `results2_heldout.tgz`, the `results2/` manifest and the Kaggle verdict file match the SHA-256 digests
  sealed in P1, and they are committed here unchanged. Every file in `results2/` except `verdict.json` is
  byte-identical to the archive. The archive itself (868 kB) is not committed, under the existing `*.tgz` ignore
  rule. Its digest is in P1.

## What failed, specifically (confirmatory)

Every condition for every model is in `RESULTS2-tables.md`. No model met the full criterion of any hypothesis, so no
task reached the 2-of-3 (or 2-of-2) seed count.

**H2.1.** One condition held everywhere, and one almost nowhere:
- **κ > R-full κ** holds for all 17 models.
- **θ_bulk − θ_top ≥ 15°**, the subspace-domination condition, holds for only 2 of 17: finger-turn-hard-2 at 28.4° and
  hopper-stand-1 at 17.8°. Both fail other conditions.
- On walker-run (all 3 seeds) and humanoid-run (both seeds), energy concentration and growth ranking hold, with
  κ = 0.60–0.84 and every ρ ≥ 0.71. The top-decile anchors still sit no closer to the pooled subspace than the bulk
  does (θ_bulk − θ_top = 0.3–0.9°).
- reacher-hard and quadruped-walk fail concentration and growth ranking outright.

**H2.2.** θ̂_stab ranges from 38.6° to 63.7°, against the 30° threshold, and no model is 15° more stable than its
R-full control. The margin runs from −19.4° to +5.4°. On hopper-stand and finger-turn-hard the untrained control is
*more* stable. So under the frozen construction, removing the growth does not reveal a stable pooled subspace.

**H2.3.** Δ ranges from −4.2° to +13.9°, against 15°. θ_near ≤ 30° holds for 4 of 17 models, and on those Δ is only
0.2–2.8°. The largest contrasts, on humanoid-run (13.9° and 13.3°), are *smaller* than the paired R-full contrasts (17.7° and
15.4°). So they fail the control margin as well as the threshold.

**Numerical routes.** The Gramian route gives the same verdicts (0/6 for all three), and `h2_verdict.py` flagged no
disagreement at hypothesis or task level.

## EXPLORATORY (§7.1: reported and labelled; no verdict role, no claim)

These are observations from the confirmatory data and the check route. None of them qualifies, softens or reinterprets
the three rejections above. Any hypothesis drawn from them would need its own preregistration and fresh held-out models.

1. **Route agreement is weakest where growth is largest.** On the growth-normalised and H = 16 statistics the
   routes agree to within 1.3°. On the *undiscounted* pooled statistics of hopper-stand-1 and hopper-stand-2, where the median
   ‖Φ(τ+64, τ)‖₂ is about 3–5e7, they differ by up to 14.3° (θ_top). This is the conditioning problem that protocol-1
   D1 anticipated, and it is why the factor route decides. One condition-level result depends on the route:
   hopper-stand-1's θ_bulk − θ_top is 17.8° on the factor route and 6.3° on the Gramian route. That model fails H2.1 on
   κ_c and κ_o under either route.
2. **§5.2 ablations.** Growth discounting alone changes θ from the protocol-1 pooling by −9.1° to +4.8°, and
   equal-energy pooling alone by −9.0° to +7.0°. Together they never bring any model below 38°.
3. **The §5.2 gap rule on the growth-normalised spectra.** No model has a qualifying gap at H = 64. The largest log-drop
   in 2 ≤ k ≤ 32 is 0.08–0.51 decades, against protocol 1's 1-decade condition.
4. **The R-full comparator is strongly contractive.** Its γ_m is 0.55–0.82 in 16 of 17 controls (hopper-stand
   seed 1001: 1.09), and its median ‖Φ(τ+64, τ)‖₂ falls as low as 1e-17. The trained models have γ_m 0.92–1.32. So
   R-full differs from the trained models in growth by construction. That makes it a lenient comparator for H2.1's κ
   condition, which all 17 models passed, and a demanding one for H2.2 and H2.3, where R-full's angles are often
   already as low as the trained ones.
5. **Discovery set.** Not run (P2). `results2/discovery/` does not exist.

## What this licenses (§7)

§7's "all rejected" case applies. Under all three preregistered constructions, these held-out TD-MPC2 models show no
stable Hankel structure (H2.2) and no trajectory-local structure (H2.3), and their pooled Gramians are not dominated,
in the H2.1 sense, by growth anchors. This strengthens protocol 1's negative (`f0e8c94`, `RESULTS.md`, `REVIEW.md`).
The claim boundary is §2 of the protocol and `REVIEW.md`: TD-MPC2 single-task state-based checkpoints, the all-five-head
Q̄ surrogate output map, and R-full as an architecture control only. Protocol-1 experiments D–G and every planning,
pixel or distractor experiment stay **sealed**.
