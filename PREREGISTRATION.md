# Preregistration — The Hierarchy Is in the Spectrum

**Protocol version:** 1
**Author:** Shivansh Gupta (IIT Bombay)
**Track:** The Bu1LD research wave, reviewed by Ryan Gomez
**Status:** FROZEN at the commit that first contains this file. No spectrum, Gramian, Jacobian
or latent statistic of any trained checkpoint has been computed at that commit. The companion
file `config/prereg.yaml` holds every numeric parameter below in machine-readable form. The
verdict script in the diagnostic commit reads its thresholds from that file and from nowhere
else.

---

## 1. Hypothesis

In a learned latent world model, the right abstraction level for hierarchical planning is set
by the **Hankel singular value (HSV) spectrum** of the learned latent dynamics, with the
**value head** as the output map. The high-level state is made of the directions that are
jointly controllable (actions can steer them) and observable (they change predicted value). The
directions below a spectral gap belong to a reactive low-level controller.

Three properties must hold before the hypothesis is worth any planning experiment. This
protocol tests exactly those three:

- **A. A gap exists.** The HSV spectrum has a knee: a drop that stands out from the local decay
  rate, is stable across seeds and horizons, and does not appear in randomly initialised
  networks.
- **B. It is not PCA.** The balanced ordering and subspace are measurably different from
  variance ranking and the principal-component subspace of the same latent states.
- **C. It is stable.** The balanced subspace computed at individual states along trajectories
  agrees with the subspace pooled over all of them.

## 2. Exact claim boundary

**What this protocol can establish.** It can show whether a Hankel-spectral structure with the
three properties above exists in **TD-MPC2 single-task, state-based DMControl checkpoints**,
with the Q-ensemble as output map, over Gramian horizons H ∈ {32, 64, 128}.

**What it cannot establish, and what will not be claimed from it:**
- that planning over the truncated subspace improves performance. That is the costly
  prediction, and it is sealed (§11);
- anything about bound-driven replanning, pixel observations, distractors, or other
  architectures.

**If all three gates pass,** the sealed experiments in §11 may be unsealed under the conditions
given there. Passing licenses no claim that hierarchy helps planning.

**If any gate fails,** the hypothesis as stated is rejected for this model class and reported
as a negative result. It will not be rescued by changing thresholds, horizons, the output-map
definition, the task set, the checkpoint set, or the model family. Exploratory analyses (§10.4)
may be reported but are labelled as such and cannot change a verdict. A narrower variant, such
as one limited to distractor environments or to another architecture, would be a new hypothesis
with its own new preregistration, and will be described that way.

**Relation to the original submission.** The submission said the hypothesis was void if "no
model" showed a gap. This protocol replaces that with a majority rule across tasks and seeds,
a knee criterion, and random-initialisation controls (§7, §10). Every change makes the test
**harder to pass**. None makes it easier.

## 3. Models, checkpoints, environments

**Primary model family.** TD-MPC2 (Hansen, Su & Wang, ICLR 2024), released single-task,
state-based checkpoints.
- Code: `https://github.com/nicklashansen/tdmpc2` at the commit pinned in
  `config/prereg.yaml → models.code_commit`.
- Checkpoints: the source URL and a SHA-256 hash for every file are pinned in
  `config/prereg.yaml → models.checkpoint_source / checkpoint_sha256`. A checkpoint whose hash
  does not match is not analysed.

**Tasks (6), chosen to span action dimensionality:** `walker-walk`, `cheetah-run`, `hopper-hop`,
`quadruped-run`, `humanoid-walk`, `finger-spin`.
**Seeds:** the three released seeds per task, giving 18 trained models.

**Architecture facts this protocol depends on.** Each was checked against the pinned code before
the freeze. The checkpoints load with `model_size=5`, which sets latent dimension n₀ = 512. SimNorm
is applied to the latent in groups of 8, giving 64 groups. Both the state encoder and the dynamics
MLP end in Linear → LayerNorm → SimNorm. The Q-ensemble has 5 heads, and each predicts a 101-bin
distribution over symlog values on bins evenly spaced in [−10, 10]. Q-head dropout
(p = 0.01, first layer) is active only in training mode. DMControl episodes are 500 agent steps
(1000 simulator steps at action repeat 2) with no early termination.

**Secondary (exploratory, no verdict role).** DreamerV3, analysing only the deterministic GRU
state. This model is **dropped**. No official public DMControl checkpoint was found before the
freeze. The DreamerV3 repository publishes none, and the only DMControl checkpoints on Hugging
Face are third-party pixel-based ablations with modified encoders.

**Environment record.** Python, PyTorch, MuJoCo and dm_control versions are recorded in
`ENVIRONMENT.txt` in the diagnostic commit. All linear algebra runs in float64.

## 4. Linearisation rule

**Trajectories.** For each model: 10 episodes with environment seeds 0–9. The torch seed equals
the environment seed. Actions come from the agent in its released evaluation mode (MPPI
planning), without modification.

**Linearisation points.** Jacobians are evaluated at **encoded real states** z_t = enc(o_t) and
at the actions a_t actually taken, at every agent step. Imagined rollouts are not used. The
one-step latent consistency error ‖f(z_t, a_t) − enc(o_{t+1})‖ is reported for context only.

**Anchors.** τ ∈ {128, 138, 148, …} ≤ T − 128, where T is the episode length in agent steps.
The margin of 128 equals the largest horizon, so **every horizon uses the same anchor set**.
With T = 500 this gives τ ∈ {128, 138, …, 368}: 25 anchors per episode and 250 per model.

**Tangent coordinates.** SimNorm confines each group of 8 latent coordinates to a simplex, so
64 directions (the per-group sum directions) are structurally degenerate. Let U ∈ ℝ^{512×448}
be an orthonormal basis for the complement of those 64 directions. Every Jacobian is expressed
in these coordinates: Ã = UᵀAU, B̃ = UᵀB, C̃ = CU, so n = 448. This is hygiene. It removes exact
structural zeros from condition numbers and dimension counts. It is not a guard against a false
gap, because those zeros sit far outside the gap-search range.

## 5. A, B, C definitions

With z' = f(z, a) the TD-MPC2 latent dynamics:

- **A_t = ∂f/∂z** at (z_t, a_t) — n₀ × n₀, then projected to n × n.
- **B_t = ∂f/∂a** at (z_t, a_t) — n₀ × m, with m the action dimension (actions in [−1, 1]).
- **C_t = ∂Q̄/∂z** at (z_t, a_t) — 1 × n₀, with a_t **held fixed** (partial derivative), where

  Q̄(z, a) = (1/5) Σ_{i=1}^{5} symexp( Σ_b softmax(ℓ_i(z, a))_b · bin_b ),

  ℓ_i being the 101-bin logits of Q-head i.

Every Jacobian is taken with the network in evaluation mode, so dropout is off, and C uses the
online Q-ensemble parameters (`_Qs`), not the target copy.

**Why the value head.** The output map decides what "observable" means. The decoder or state
reconstruction would score high-variance but irrelevant directions as important. The value head
scores directions by their effect on predicted return, which is the quantity a high-level
planner optimises. One output map is primary. The alternatives in §10.4 are exploratory only.

**Scale invariance.** HSVs are invariant to changes of state coordinates. Scaling C or B
multiplies every HSV by the same factor. Every criterion below uses **ratios** of HSVs, so
reward and value units cancel.

## 6. Finite-horizon Gramian construction

**Why finite-horizon.** Infinite-horizon Gramians exist only if the learned transition is
Schur-stable, and nothing guarantees that. Time-limited Gramians are always defined.

**State-transition matrix.** Φ(t, s) = Ã_{t−1} ⋯ Ã_s for t > s, and Φ(s, s) = I.

**At anchor τ, over horizon H:**

- Controllability (reachability into z_τ from the previous H steps):
  W_c(τ) = Σ_{s=τ−H}^{τ−1} Φ(τ, s+1) B̃_s B̃_sᵀ Φ(τ, s+1)ᵀ
- Observability (effect of z_τ on value over the next H steps):
  W_o(τ) = Σ_{s=τ}^{τ+H−1} Φ(s, τ)ᵀ C̃_sᵀ C̃_s Φ(s, τ)

**Pooled Gramians per model** (the primary object): W̄_c = mean over τ of W_c(τ), and W̄_o
likewise. Averaging Gramians is an energy average, as with empirical Gramians. HSVs are never
averaged across different bases.

**HSVs and balancing (square-root method).** Factor W̄_c = L_c L_cᵀ and W̄_o = L_o L_oᵀ by symmetric
eigendecomposition, clipping negative round-off eigenvalues to zero. Take the SVD
L_oᵀ L_c = U_h Σ V_hᵀ. Then σ = diag(Σ). The retained directions are
V_k = L_c V_h[:, :k] Σ_k^{−1/2}, and the retained subspace is S_bal(k) = range(V_k).

**Numerical floor.** Any HSV below 10⁻⁶ · σ₁ is treated as numerically zero.

**Horizons.** Primary H = 64. Robustness H ∈ {32, 128}.

**Why not TD-MPC2's native planning horizon (3).** With a scalar output, rank W_o(τ) ≤ H. At
H = 3, at most 3 HSVs per anchor are non-zero, so a "gap" at k ≤ 3 would be guaranteed by
construction and would test nothing. The hypothesis concerns a **high-level** planner, which by
definition plans over a longer horizon than the low-level MPC. So the Gramian horizon is set to
candidate high-level horizons, not to TD-MPC2's low-level one. The same rank bound, rank
W_o(τ) ≤ H, is why the gap search below stops at k ≤ H/2.

**Reported, no verdict role:** the spectral radius ρ(Ã_t) at every linearisation (distribution
per model), and ‖Φ(τ + H, τ)‖₂ per anchor, so that any instability or finite-horizon growth is
visible rather than hidden.

## 7. Spectral-gap definition

Let σ₁ ≥ σ₂ ≥ … be the pooled HSVs at horizon H. Define the log-drops

  d_k = log₁₀ σ_k − log₁₀ σ_{k+1}.

**A valid gap at k requires all of:**
1. 2 ≤ k ≤ H/2. A single dominant mode (k = 1) is expected structurally with a scalar output,
   and does not count as a hierarchy.
2. σ_{k+1} ≥ 10⁻⁶ · σ₁. A drop into numerical noise is not a gap.
3. **d_k ≥ 1.** At least an order of magnitude.
4. **d_k ≥ 3 × median{ d_j : |j − k| ≤ 8, j ≠ k, 1 ≤ j ≤ H/2 }.** The drop must stand out from
   the local decay rate.

Condition 4 carries the most weight. Hankel spectra of systems with few inputs and outputs are
known to decay fast even when nothing interesting is happening. A smooth exponential decay with
consecutive ratios above 10 would satisfy condition 3 at every k. A knee must be a *departure*
from the decay rate, not the decay rate itself.

If several k qualify, **k\*** is the one with the largest d_k.

**A model has a gap** if a valid gap exists at the primary H = 64 and a valid gap reappears
within ±25% of k\* at H = 128, and also at H = 32 whenever k\* ≤ 16.

**A task passes Gate A** if:
- at least 2 of its 3 seeds have a gap, with each k\* within ±25% of those seeds' median k\*,
  **and**
- in **each** of the two random-initialisation control families below, fewer than 2 of 3
  controls show a valid gap within ±25% of that median.

**Random-initialisation controls.** There are two families. Each family uses torch seeds 1000, 1001
and 1002, and all its Jacobians are evaluated at the **trained** model's anchor states and actions.

- **R-full (architecture).** Every weight comes from the `WorldModel` constructor at the pinned
  commit. The one exception is the final linear layer of each Q-head, which is re-drawn from the
  same truncated normal (std 0.02, zero bias) as every other linear layer.
- **R-dyn (value head alone).** The dynamics MLP comes from the constructor, as in R-full. C comes
  from the **trained** Q-ensemble.

**The trained gap fails the control** if 2 or more of the 3 controls in **either** family show a
valid gap within ±25% of the trained median.

*Why the exception, and why two families.* The pinned code zero-initialises the final Q-layer
weight, and its bias is initialised to zero. At initialisation, then, every Q-head outputs
all-zero logits, Q̄ is constant, C ≡ 0, W̄_o = 0, and every HSV is zero. Built as literally
specified, the control could never show a gap, so it could never reject one. R-full fixes that.
R-dyn separates a gap produced by learned dynamics, which is what §1 claims, from one the trained
value head would produce over any dynamics. Both families can only make the test harder to pass.

## 8. PCA and subspace controls

The same anchor states z_τ are used for both controls, in the same tangent coordinates.

**Rank-based test (Spearman).** For each of the 512 original latent coordinates j:
- Hankel importance: h_j = √( (U W̄_c Uᵀ)_jj · (U W̄_o Uᵀ)_jj );
- variance: v_j = Var_τ(z_{τ,j}).

The statistic is ρ_S = Spearman(h, v) over j = 1…512.

**Subspace test (principal angles).** S_pca(k) is spanned by the top-k eigenvectors of the
anchor-state covariance in tangent coordinates. The statistic θ_pca is the mean principal angle,
in degrees, between S_bal(k) and S_pca(k).

**k per task.** If the task passed Gate A, k is the median k\* across its passing seeds.
Otherwise k = 16.

**A model collapses to PCA** if **ρ_S > 0.9 or θ_pca < 15°**. Agreement under either test is
enough.

**Reported for reference, no verdict role:** θ between PCA subspaces of different seeds, as a
noise floor, and the expected angle between random k-dimensional subspaces in ℝ⁴⁴⁸.

## 9. Trajectory and subspace stability test

For each anchor τ, apply the square-root method of §6 to W_c(τ) and W_o(τ) at H = 64, giving
S_bal^{(τ)}(k) with the same k as in §8. Because rank W_o(τ) ≤ H and k ≤ H/2, this is always
well-defined.

Let θ_stab be the mean over anchors of the mean principal angle between S_bal^{(τ)}(k) and the
pooled S_bal(k). **A model is stable if θ_stab ≤ 30°.**

## 10. Kill criteria

### 10.1 Gate verdicts (6 tasks × 3 seeds)

- **Gate A — gap.** PASS if at least 4 of 6 tasks pass (§7).
- **Gate B — not PCA.** A task shows divergence if at least 2 of its 3 seeds do **not** collapse
  (§8). PASS if at least 4 of 6 tasks show divergence.
- **Gate C — stability.** A task is stable if at least 2 of 3 seeds are stable (§9). PASS if at
  least 4 of 6 tasks are stable.

### 10.2 Overall

The hypothesis **survives the diagnostic stage if and only if A, B and C all pass.** Any failure
means the hypothesis as stated is **rejected** for this model class (§2), and the sealed
experiments are not run under it.

### 10.3 Invariants

- All gates are computed and reported whatever the earlier gates show.
- Every result is committed and preserved, including flat spectra, unstable subspaces, and
  collapse to PCA.
- The verdict comes mechanically from `scripts/verdict.py` applied to `config/prereg.yaml`, and
  is written to `results/verdict.json`.

### 10.4 Exploratory analyses (reported, labelled, cannot change any verdict)

- C from the reward head instead of the Q-ensemble.
- C as the total derivative through the policy prior, instead of the partial derivative.
- Horizons outside {32, 64, 128}.
- The DreamerV3 secondary model.

## 11. Sealed downstream experiments

Not run, not prototyped on trained models, and not partially executed until the unsealing
conditions below are met. The kill criteria are fixed here. Implementation detail goes into an
addendum that may add detail or tighten criteria, never relax them.

- **D. Retained-dimension planning sweep (the costly prediction).** A hierarchical planner is
  given the top-k balanced dimensions, with k swept well past k\*. The prediction is that
  performance peaks near k\* and degrades after it. **Kill: monotone improvement in k.**
- **E. Trivial-mechanism controls.** Matched-size random truncation, and truncation in reversed
  Hankel order. **Kill: matched-size random truncation performs comparably to balanced
  truncation.**
- **F. Bound-driven replanning.** Replan when the accumulated truncation bound
  2 Σ_{i>k} σ_i crosses tolerance, compared against fixed-rate replanning. Kill criteria go in
  the addendum and must include the case where the bound is too loose to inform the rate (for
  example, loose by more than 10×).
- **G. Distracting Control Suite.** The discriminating experiment. It needs pixel-based TD-MPC2
  models trained with natural-video distractors, which do not exist as released checkpoints and
  so need compute.

**Unsealing conditions:**
1. Gates A, B and C all pass.
2. The diagnostic commit is pushed with every artifact in §12.
3. Ryan Gomez has been sent the diagnostic commit.
4. `PREREGISTRATION-ADDENDUM-1.md` is committed before any sealed experiment runs.

## 12. Artifacts the diagnostic commit must produce

The diagnostic commit sits on top of the preregistration commit. The preregistration files must
be byte-identical in it. It must contain:

**Code**
- `scripts/collect_anchors.py` — rollouts, anchor states, actions, per-step Jacobians
- `scripts/gramians.py` — tangent projection, LTV finite-horizon Gramians, square-root HSVs and balancing
- `scripts/gate_a_spectrum.py`, `scripts/gate_b_pca.py`, `scripts/gate_c_stability.py`
- `scripts/verdict.py` — reads only `config/prereg.yaml` and `results/`, writes `results/verdict.json`
- `scripts/make_figures.py`
- `reproduce.sh` — one entry point that regenerates everything below from the pinned checkpoints
- `tests/test_gramians_synthetic.py` — see §13

**Results**
- `results/hsv/{task}_s{seed}_H{H}.npz` — HSVs, d_k, and k\* where found
- `results/hsv/{task}_rfull{seed}_H{H}.npz` and `results/hsv/{task}_rdyn{seed}_H{H}.npz` — the
  two random-initialisation control families
- `results/pca/{task}_s{seed}.json` — ρ_S, θ_pca, and k
- `results/stability/{task}_s{seed}.json` — per-anchor angles and θ_stab
- `results/diagnostics/{task}_s{seed}.json` — spectral-radius distribution, ‖Φ‖₂ per anchor, one-step consistency error
- `results/verdict.json` — per-task and per-gate verdicts and the overall verdict
- `ENVIRONMENT.txt` — full package versions, OS, and CPU/GPU used

**Figures**
- `figures/fig1_hsv_spectra.pdf` — pooled spectra, 6 tasks × 3 seeds, random-init controls overlaid
- `figures/fig2_log_drops.pdf` — d_k with the local-median criterion drawn
- `figures/fig3_spearman.pdf` — h against v per task
- `figures/fig4_principal_angles.pdf` — θ_pca with the 15° line and reference angles
- `figures/fig5_stability.pdf` — θ_stab distribution with the 30° line

## 13. Permitted before the freeze, and between freeze and diagnostics

**Permitted before this commit:**
- reading the TD-MPC2 code;
- downloading and hashing checkpoints;
- writing this document.

**Not permitted before this commit:** any forward pass of a trained checkpoint, including
rollouts, Jacobians, Gramians, or latent statistics.

**Between the freeze and the diagnostic run,** analysis code is validated **only on synthetic
linear systems**:
1. Random stable LTI systems: HSVs must match a reference implementation to 10⁻⁸ relative.
2. A system with a planted knee: the §7 rule must detect it at the planted k.
3. A system with smooth exponential HSV decay, with consecutive ratios above 10: the §7 rule
   must find **no** gap. This test confirms the knee criterion does what condition 4 claims.

## 14. Deviations

Any departure from this document after the freeze is logged in `DEVIATIONS.md`, with the date
and the reason. A deviation can never change a verdict. If one occurs, the verdict under this
original protocol is always reported alongside any other result.
