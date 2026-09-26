# Deviations and clarifications

Logged under PREREGISTRATION.md §14. The preregistration commit is
`0173614f918c0828b0f978378f1c0c2807be9990`. Its files are never edited. Every change is recorded here instead.

Every entry below was written on **26 Sep 2026**. At that time no checkpoint had been downloaded, loaded or run.
The analysis code had been run only on the synthetic systems of §13.

---

## D1: HSVs computed from square-root factors, not from formed Gramians (numerical method)

**What §6 says.** Form W̄_c and W̄_o, then factor each one by symmetric eigendecomposition. Take the SVD of L_oᵀ L_c.

**What was found.** §13 case 1 requires HSVs to match a reference to 1e-8 relative. The reference is a 40-digit
mpmath computation. With a scalar output, W_o reaches condition number ~1e18 on the synthetic systems, which is
beyond float64. Forming W_o and eigendecomposing it therefore loses relative accuracy in the smallest HSVs:

| route | max rel. error, σ ≥ 1e-6 σ₁ (the floor) | max rel. error, σ ≥ 1e-4 σ₁ |
|---|---|---|
| §6 as written (form Gramians, eigendecompose) | 3.2e-7 | 3.0e-10 |
| factor route (D1) | 1.8e-12 | 5.2e-14 |

The measurement covers 4 random stable systems (n = 12–30, m = 1–6, p = 1) at H ∈ {32, 64, 128}. Only
HSVs the §7 rule can read are compared: above the floor, with k ≤ H/2 + 1.

**Change.** The factors are built directly from the Jacobians. The columns of R_τ are Φ(τ,s+1)B_s, and the rows of O_τ are C_sΦ(s,τ).
The per-anchor factors are pooled by incremental QR, and the SVD is taken of L_oᵀ L_c. This gives the same W̄_c, W̄_o
and HSVs mathematically, but the condition number is never squared. Code: `scripts/gramians.py` (`controllability_factor`,
`observability_factor`, `PooledFactor`, `balance_factors`).

**Effect on verdicts.** None is expected. An error of 3.2e-7 relative is 1.4e-7 decades, and the §7 thresholds are whole
decades. Both routes are nevertheless computed on every model, and `results/verdict.json` carries both verdicts. Under
§14 the verdict from the §6 route as written is the **verdict of record**. If the two routes disagree anywhere,
the disagreement is reported next to it.

---

## Clarifications (the text allowed more than one reading, and one was fixed before any data)

None of these changes a number in `config/prereg.yaml`. Where the readings differ in strictness, the stricter one was chosen.

**C1. Log-drops into the floor (§6, §7).** HSVs below 1e-6 σ₁ are set to exactly zero. A drop into zero, or from zero
to zero, gets d = +∞. Such a neighbour therefore *raises* the local median in condition 4, instead of being skipped,
which would lower it. *(Stricter.)*

**C2. What a random-initialisation control must show (§7).** A control "shows a valid gap" if it has one at the
primary horizon H = 64. It does **not** also need the multi-horizon reappearance that a trained model needs. So controls
reproduce the trained gap more easily. *(Stricter.)*

**C3. Seed agreement (§7).** Take the seeds that have a gap and their median m. The *agreeing* seeds are those with
k\* within ±25% of m, and at least 2 must agree. The task's k (used by §7 controls and §8/§9) is the median over the
agreeing seeds. *(Reading of "each k\* within ±25% of those seeds' median k\*".)*

**C4. Horizon reappearance (§7).** "A valid gap reappears within ±25% of k\*" is read literally. *Some* valid gap at the
robustness horizon must lie within ±25% of the primary k\*. It does not have to be the largest drop at that horizon. *(Literal reading.)*

**C5. "Within ±25%"** is inclusive: |k − k_ref| ≤ 0.25 · k_ref.

**C6. Half-integer task k (§8).** With two agreeing seeds, the median k\* can be a half-integer. k is then rounded
half-up. *(Neutral.)*

**C7. "Released evaluation mode" (§4).** Actions come from `agent.act(obs, t0, eval_mode=True)`, the call the pinned
code's own evaluation loops use (`trainer/online_trainer.py`, `trainer/offline_trainer.py`). The pinned `evaluate.py`
omits `eval_mode=True`, so it adds exploration noise to every action. That is not an evaluation mode.
*(Reading of the text.)*

**C8. Linearisation in float64 (§3, §5).** Rollouts run in the released float32 agent, unmodified. Linearisation uses a
float64 copy of the same weights. Each z_t is re-encoded in float64 from the observation the float32 agent saw, and
A, B and C are taken by autograd at (z_t, a_t). C is one backward pass through Q̄, and A and B come from
`torch.func.jacrev` of the dynamics. *(Follows "all linear algebra runs in float64".)*

**C9. Engineering smoke test before the diagnostic run (§13).** `collect_anchors.py --smoke` runs the real
TD-MPC2 code path on a **random-initialisation network at torch seed 999**. That seed is not one of the control seeds,
and no checkpoint is loaded. It checks Jacobians against finite differences, float32 against float64, rollout
determinism, array shapes, and that the constructor's Q head gives C ≡ 0. It computes **no Gramian, HSV or spectrum**.
§13's rule is about validating the *analysis*. This test validates the plumbing and cannot inform any choice.

**C10. Which anchor states a control uses (§7).** There are 3 controls per family per task and 3 trained seeds, each
with its own anchor states. They are paired: torch seed 1000 with trained seed 1, 1001 with seed 2, and 1002 with
seed 3. Each control is evaluated at its paired seed's anchor states and actions. *(Neutral.)*
