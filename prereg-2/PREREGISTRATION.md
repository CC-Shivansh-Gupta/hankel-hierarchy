# Preregistration 2 — Transient Growth and Hankel Structure in Learned World Models

**Protocol version:** 2 (a new hypothesis; not a revision of protocol 1)
**Author:** Shivansh Gupta (IIT Bombay)
**Track:** The Bu1LD research wave, reviewed by Ryan Gomez
**Status:** DRAFT, not frozen. It is frozen at the first commit that contains this file. No H2 statistic (§5) has been
computed on any model, held-out or discovery, at that commit. Every numeric parameter is in `prereg-2/config/prereg.yaml`.

---

## 0. Relation to protocol 1

Protocol 1 (`PREREGISTRATION.md`, frozen at `0173614`) tested whether the Hankel spectrum of TD-MPC2's learned
latent dynamics has a knee that defines a hierarchy. It was **rejected**: Gate A passed on 0 of 6 tasks and Gate C on
0 of 6 (`f0e8c94`, `RESULTS.md`, `REVIEW.md`). **That rejection stands and nothing here revisits it.**

This protocol tests a different hypothesis, suggested by an *exploratory* observation from protocol 1: the learned
dynamics are locally expansive, and the pooled Gramians look dominated by a few fast-growing anchors. Because the 17
protocol-1 models produced that observation, they are the **discovery set** and carry **no verdict role** here. Every
verdict comes from **held-out models that no analysis has touched** (§3).

## 1. Hypotheses

Each hypothesis has its own verdict. No combination of them is a disjunctive rescue ("H2.2 or H2.3"). A failure in one
is reported as a failure, whatever the others show.

- **H2.1 (transient-growth dominance).** In learned TD-MPC2 latent dynamics, the finite-horizon Gramians pooled over a
  trajectory are dominated by a small fraction of anchors, and that dominance is explained by local finite-horizon
  growth. It exceeds what the architecture alone produces.
- **H2.2 (growth-normalised structure).** Remove the growth, by discounting the dynamics by the model's typical growth
  rate and pooling anchors with equal energy. The pooled balanced subspace is then stable across anchors, and more
  stable than in a random-initialisation control.
- **H2.3 (trajectory-conditioned structure).** The balanced subspace is local rather than global. It agrees between
  nearby anchors whose windows do not overlap, disagrees between distant anchors, and this near/far contrast exceeds
  the contrast in a random-initialisation control.

## 2. Exact claim boundary

**What this protocol can establish.** Whether H2.1–H2.3 hold for **TD-MPC2 single-task, state-based DMControl
checkpoints** on six held-out tasks, with the output map, linearisation and coordinates of protocol 1 (§4).

**What it cannot establish:**
- that hierarchy helps planning. Protocol 1's experiments D–G stay sealed, and no result here unseals them;
- anything about other architectures, pixels or distractors;
- anything beyond the protocol-1 output map. The all-five-head Q̄ remains a frozen surrogate for TD-MPC2's own
  evaluation path (`REVIEW.md`, boundary 3);
- any positive mechanistic claim from R-dyn. Only R-full is used as a control here (`REVIEW.md`, boundaries 1–2).

## 3. Models

**Held-out set: the rule, and the list it produces.**
- **Rule.** Take one standard DMControl task for each of the five discovery domains other than cheetah (walker, hopper,
  quadruped, humanoid, finger), using a task *not* in the discovery set. Add `reacher-hard` in place of cheetah, whose
  only standard task, cheetah-run, is in the discovery set. That gives action dimensions from 2 to 21, as in protocol 1.
  Every released seed (1, 2, 3) of each task is used, and there is no seed selection.
- **List.** `reacher-hard` (2), `finger-turn-hard` (2), `hopper-stand` (4), `walker-run` (6), `quadruped-walk` (12),
  seeds 1–3 each, and `humanoid-run` (21), **seeds 1–2 only**: 17 checkpoints. They come from the same Hugging Face
  revision as protocol 1 (`8fb2a82e…`), with every SHA-256 pinned in the config.

**The layout screen and its outcome.** This screen was done **before this freeze**, and its per-file record is in the
config. Protocol 1 met a checkpoint stored in a pre-release layout (D2). Each candidate file was downloaded,
hash-checked, and had only its **state-dict keys** inspected with PyTorch's `weights_only` loader: no forward pass and
no code execution. 17 of the 18 are in the current layout. `humanoid-run-3.pt` is pre-release.
- The only domain-matched alternate, `humanoid-stand`, was screened next. Its seed 3 is also pre-release.
- So the seed-3 checkpoints of *all three* humanoid tasks are pre-release, counting protocol 1's `humanoid-walk-3`.
- No humanoid task has three loadable seeds. Substituting another domain would break the selection rule and amount to
  searching over tasks.
- So `humanoid-run` is kept with its two loadable seeds and is **judged 2-of-2** (§6), which is strictly harder to
  pass. No conversion is attempted, because protocol 1's conversion of such a file failed its behavioural check.

**The list as frozen is final.**
- **After the freeze** no task, seed or checkpoint may be added, swapped or dropped for any reason except a load
  failure. A load failure means exclusion under the protocol-1 D2 rule, with the task judged 2-of-2, and never
  substitution.
- The key inspection shows file format only. It cannot reveal any §5 statistic.

**Discovery set (exploratory only).** The 17 protocol-1 models are analysed with the same code and reported in a
separate, labelled section. No H2 statistic is computed on them before the held-out verdict is written.

## 4. Inherited definitions (unchanged from protocol 1)

- **Rollouts:** 10 episodes, env seed = torch seed = 0…9, `agent.act(eval_mode=True)` (protocol 1, C7).
- **Linearisation:** float64 copy, z_t = enc(o_t), A = ∂f/∂z, B = ∂f/∂a, C = ∂Q̄/∂z with a_t held fixed (C8).
- **Coordinates:** 448-dimensional SimNorm tangent coordinates (§4 of protocol 1).
- **Anchors:** τ ∈ {128, 138, …, 368}, 25 per episode and 250 per model.
- **Environment:** dm-control 1.0.24 and mujoco 3.2.4 (protocol 1, C11), recorded in `ENVIRONMENT.txt`.
- **Numerics:** HSVs and balanced bases come from square-root factors (protocol 1, D1). This is the **primary** route
  here, validated in advance. The form-then-eigendecompose route is computed as a check, and any disagreement between
  the two routes is reported.
- **Principal angles:** mean principal angle in degrees between equal-dimension subspaces.
- **Seeds:** trained seeds 1, 2, 3 per task (humanoid-run: 1, 2; §3). Environment seeds 0–9, with torch seed equal to env seed. R-full torch
  seeds 1000/1001/1002, paired with trained seeds 1/2/3.
- **Numerical routes:** the factor route decides every verdict. The Gramian route is computed alongside, and any gate
  on which the two disagree is flagged in `results2/verdict.json`. A disagreement is reported. It never changes the
  factor-route verdict.

Let W_c(τ; H) and W_o(τ; H) be the per-anchor finite-horizon Gramians of protocol 1 §6, and S(τ; H, k) the per-anchor
balanced subspace of dimension k.

## 5. Statistics and constructions

**The transient-growth quantity.** At anchor τ, forward growth is g_f(τ) = ‖Φ(τ+64, τ)‖₂ and backward growth is
g_b(τ) = ‖Φ(τ, τ−64)‖₂, the largest amplification the learned linearised dynamics apply over one Gramian window.
Every H2 construction below is built from these two quantities.

**Primary statistics, one per hypothesis:**

| | Primary statistic | Required conditions (all must also hold) |
|---|---|---|
| H2.1 | κ = min(κ_c, κ_o), the energy share of the top-decile anchors | ρ_c, ρ_o ≥ 0.7; θ_bulk − θ_top ≥ 15°; κ > R-full's κ |
| H2.2 | θ̂_stab, the growth-normalised stability angle | θ̂_stab ≤ R-full's θ̂_stab − 15° |
| H2.3 | Δ = θ_far − θ_near, the local-vs-global contrast | θ_near ≤ 30°; Δ ≥ R-full's Δ + 10° |

### 5.1 H2.1: growth dominance (H = 64)

For each model and each Gramian g ∈ {c, o}:

- **Energy concentration.** e_g(τ) = tr W_g(τ; 64). Let κ_g be the share of Σ_τ e_g(τ) contributed by the top 10% of
  anchors ranked by e_g. A uniform spread gives κ = 0.10.
- **Growth attribution.** ρ_o = Spearman(e_o(τ), ‖Φ(τ+64, τ)‖₂) and ρ_c = Spearman(e_c(τ), ‖Φ(τ, τ−64)‖₂), taken
  over anchors.
- **Subspace domination.** P is the pooled balanced subspace (protocol 1 §6, energy-averaged) with k = 16. θ_top is
  the mean angle to P over the top-decile anchors ranked by e_c·e_o, and θ_bulk the mean angle over the bottom half.

**A model shows growth dominance** if **all** of the following hold: κ_c ≥ 0.5, κ_o ≥ 0.5, ρ_c ≥ 0.7, ρ_o ≥ 0.7,
θ_bulk − θ_top ≥ 15°, **and** min(κ_c, κ_o) exceeds the same statistic for the model's paired R-full control.

### 5.2 H2.2: growth-normalised pooled structure (H = 64, k = 16)

- **Growth rate:** γ_m = exp( median_τ log ‖Φ(τ+64, τ)‖₂ / 64 ), one per model.
- **Discounted dynamics:** Â_t = Ã_t / γ_m. B̃ and C̃ are unchanged.
- **Per-anchor Gramians:** Ŵ_g(τ) are the protocol-1 Gramians with Â in place of Ã.
- **Equal-energy pooling:** W̄_g^eq = mean_τ Ŵ_g(τ) / tr Ŵ_g(τ).
- **Stability:** θ̂_stab is the mean over anchors of the angle between Ŝ(τ; 64, 16) and the pooled Ŝ^eq(16), exactly
  as in protocol 1 §9 but for this construction.

**A model is stable under growth normalisation** if θ̂_stab ≤ 30° **and** θ̂_stab is at least 15° below the same
statistic for its paired R-full control.

**Reported, no verdict role:**
- the protocol-1 §7 gap rule applied to the HSVs of (W̄_c^eq, W̄_o^eq);
- the two single-change ablations (discount only; equal-energy only), so it is visible which ingredient does the work.

### 5.3 H2.3: trajectory-conditioned local structure (H = 16, k = 4)

At H = 16 an anchor's full window spans 2H = 32 steps. So anchors 40 apart have **non-overlapping** windows.
Overlapping windows would share Jacobians and make "local agreement" partly an artefact (checked in §8, case S5).
Rank W_o ≤ H gives k ≤ 8, and k = 4 is fixed.

- θ_near is the mean angle between S(τ) and S(τ+40), over same-episode anchor pairs.
- θ_far is the mean angle between S(τ) and S(τ′) with |τ − τ′| ≥ 200, over same-episode pairs.
- The contrast is Δ = θ_far − θ_near.

**A model shows trajectory-conditioned structure** if θ_near ≤ 30°, Δ ≥ 15°, **and** Δ exceeds its paired R-full
control's Δ by at least 10°. The control term matters: a network whose Jacobians barely depend on state has
θ_near ≈ θ_far, and so Δ ≈ 0, whatever its θ_near.

### 5.4 Controls

R-full only: the protocol-1 construction with torch seeds 1000/1001/1002, paired with trained seeds 1/2/3 (C10; humanoid-run uses 1000/1001), and
evaluated at the paired seed's anchor states and actions. Every control statistic uses the same construction as the
trained statistic it is compared with. R-dyn is not used (`REVIEW.md`, boundary 2).

## 6. Decision rules

For each hypothesis separately: **a task supports it** if at least 2 of its 3 seeds meet the model criterion.
**The hypothesis is supported** if at least 4 of the 6 held-out tasks support it. Otherwise it is **rejected**.
`humanoid-run` has two seeds (§3), so it supports a hypothesis only if **both** seeds meet the criterion (2-of-2).
The same applies to any task that loses a seed to a post-freeze load failure.

Each of H2.1, H2.2 and H2.3 is reported as SUPPORTED or REJECTED, with every statistic, whatever the outcome.
There is **no overall "structure exists" verdict** that could be satisfied by any one part.

### 6.1 Failure criteria

- **H2.1 is REJECTED** if fewer than 4 of 6 held-out tasks have ≥ 2 of 3 seeds meeting every H2.1 condition. It is
  rejected in particular if the top decile holds < 50% of the energy, if the energy is not growth-ranked (ρ < 0.7),
  or if an untrained R-full network is just as concentrated.
- **H2.2 is REJECTED** under the same counting, if growth normalisation leaves θ̂_stab > 30° or fails to beat R-full by
  15°. Rejection here means transient growth was *not* hiding stable structure.
- **H2.3 is REJECTED** under the same counting, if nearby non-overlapping anchors disagree (θ_near > 30°), if near and
  far do not differ (Δ < 15°), or if R-full shows the same contrast.
- **Invariants.** Every statistic is computed and reported whatever the verdict. A rejected hypothesis is preserved
  as rejected, and no threshold, horizon, k, task, seed, construction or control is changed to produce a pass. A
  variant would be a new hypothesis with its own preregistration.

## 7. What each outcome licenses

- **H2.1 supported:** the protocol-1 spectra were growth-dominated, which is a mechanistic account of that null. It is
  still a claim about the pooled construction only.
- **H2.2 supported:** a stable, growth-normalised Hankel subspace exists in these models. That licenses a *new*,
  separately preregistered planning study. It does not unseal protocol-1 experiments D–G.
- **H2.3 supported:** the structure is trajectory-local rather than global. That motivates state-conditioned
  abstractions, again only under a new preregistration.
- **All rejected:** no stable Hankel structure in these models under any of the three constructions. This strengthens
  protocol 1's negative.

### 7.1 What stays exploratory, and what stays sealed

**Exploratory (reported and labelled; never verdict-bearing):**
- every statistic on the 17 discovery models;
- the protocol-1 §7 gap rule under the H2.2 construction;
- the two single-change H2.2 ablations;
- the Gramian numerical route;
- any horizon, k or pooling other than those in §5.

**Sealed:**
- protocol-1 experiments D–G (planning over truncated subspaces);
- any planning, pixel or distractor experiment.

None of these is unsealed by any outcome here. Each would need its own preregistration.

## 8. Synthetic validation (before any model is touched)

The code for §5 is validated only on synthetic linear time-varying systems:

- **S1, planted dominance.** A stable LTV sequence in which 10% of anchors carry an expansive segment along a planted
  subspace. §5.1 must flag dominance. §5.2 must recover the bulk's subspace, and the undiscounted pooled subspace must
  align with the planted one.
- **S2, no dominance.** A stationary stable LTI system. κ must be ≈ 0.10, and no dominance may be flagged.
- **S3, rotating subspace.** A dominant subspace that rotates slowly along the trajectory. §5.3 must find θ_near small,
  θ_far large and Δ ≥ 15°. The pooled θ_stab must be large.
- **S4, state-independent.** A constant LTI system. It must give Δ ≈ 0, so §5.3 must not pass.
- **S5, overlap artefact.** Independent random Jacobians at every step. With overlapping windows (Δτ = 10, H = 64)
  the near angle must come out spuriously small. With the §5.3 non-overlapping design it must not.

## 9. Permitted before the freeze

**Permitted before the freeze:**
- reading code;
- downloading, hashing and **inspecting state-dict keys** of held-out checkpoints (§3);
- writing and synthetically validating the analysis code;
- writing this document.

**Not permitted:**
- any forward pass of a held-out checkpoint;
- computing any §5 statistic on any trained model, held-out or discovery.

## 10. Artifacts

The result commit sits on top of the freeze commit, with the freeze files byte-identical. It contains the code, the
synthetic tests, `results2/` (per-model statistics, per-hypothesis verdicts in `results2/verdict.json`, and discovery
results in `results2/discovery/`, labelled exploratory), the figures and `ENVIRONMENT.txt`.

## 11. Deviations

Any departure after the freeze is logged in `prereg-2/DEVIATIONS.md` with its date and reason. A deviation never
changes a verdict, and the verdict under this protocol as written is always reported.
