# Results — diagnostic stage (Gates A–C)

**Verdict: the hypothesis is REJECTED for this model class** (PREREGISTRATION.md §10.2). It is not rescued by any
change of threshold, horizon, output map, task set or model family (§2).

| Gate | Rule (§10.1) | Tasks passing | Verdict |
|---|---|---|---|
| A: a gap exists | ≥ 4 of 6 | **0 of 6** | FAIL |
| B: not PCA | ≥ 4 of 6 | 6 of 6 | pass |
| C: stable | ≥ 4 of 6 | **0 of 6** | FAIL |

- **Mechanical:** `results/verdict.json` comes from `python -m scripts.verdict`, which reads only `config/prereg.yaml`
  and `results/`. Recomputed locally from the raw per-model files, it is identical to the one produced on the run machine.
- **Numerics (D1):** both routes give the same verdict on every task (`routes_agree: true`). The §6 route is the verdict of record.
- **Models:** 17 of 18. `humanoid-walk-3.pt` needed the D2 layout conversion. Its converted model scored a mean return
  of 1.3 against the pre-fixed bar of 714 (0.8 × 893), so it was **excluded** before any spectrum of it existed. Record:
  `results/diagnostics/humanoid-walk_s3_conversion.json`. humanoid-walk was judged 2-of-2 on seeds 1 and 2.
- **Behaviour:** every analysed model's mean return lies within the spread of TD-MPC2's published final returns
  (for example walker-walk 984 vs 977–983, finger-spin 985–989 vs 990–991). So the simulator version change (C11) did
  not visibly alter the agents.

## What failed, specifically

**Gate A.** No trained model has a valid gap at H = 64. That means no seed on any task, so seed agreement and the
random-initialisation controls were never reached. Every model fails **condition 3** (a drop of at least one decade): the
largest log-drop over 2 ≤ k ≤ 32 in any of the 17 models is 0.70 decades (cheetah-run seed 1). Condition 4
(standing out from the local decay) *is* met in places, with ratios up to 4.9× the local median, but only by drops too
small to count. See `figures/fig1_hsv_spectra.pdf` and `figures/fig2_log_drops.pdf`.

**Gate C.** θ_stab ranges from 60.0° to 71.0° across all 17 models, against the 30° threshold. The subspace balanced at
each anchor is far from the pooled one. See `figures/fig5_stability.pdf`.

**Gate B passes.** ρ_S ranges from 0.24 to 0.73 (the collapse line is > 0.9), and θ_pca from 77° to 82° (the collapse
line is < 15°). The balanced ordering is not variance ordering. Because Gate A failed, B and C were evaluated at the
default k = 16 (§8).

## Exploratory observations (§10.4: reported and labelled; they change no verdict)

1. **Learning concentrates the spectrum, smoothly.** At k = 32 the trained models have σ₃₂/σ₁ ≈ 6e-6 to 2e-2. The
   random-initialisation controls have ≈ 1e-2 to 1.4e-1, and all 448 of their HSVs sit above the floor, against
   45–276 for the trained models. So training produces a steeply decaying Hankel spectrum, but not one with a knee.
2. **The learned latent dynamics are locally expansive.** This is the §6 "reported, no verdict role" diagnostic. The
   spectral radius of Ã_t exceeds 1 at 71–100% of steps (medians 1.10–1.54). ‖Φ(τ+64, τ)‖₂ has medians from 2e2 to 2e8,
   and max/median ratios up to 7.8e3 within a single model. The pooled Gramians are energy averages, so they are
   dominated by the fastest-growing anchors. A plausible reading, **not tested here**, is that the spectrum mostly
   reflects local growth directions that change from anchor to anchor. That fits both the smooth decay and the Gate C
   angles. Testing it would be a new hypothesis with its own preregistration (§2).

## What this licenses

Nothing about hierarchical planning. The sealed experiments D–G (§11) stay sealed under this hypothesis. Any narrower
variant (for example, Gramians restricted to contractive segments, or a shorter horizon) is a new hypothesis and needs a
new preregistration. It would be described that way, not as a rescue of this one.
