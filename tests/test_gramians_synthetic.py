"""§13 synthetic validation. These are the only systems the analysis code may be run on before
the diagnostic run.

  1. random_stable_lti          HSVs match an independent reference to hsv_rel_tol
  2. planted_knee               the §7 rule finds the knee at the planted k
  3. smooth_exponential_decay   ratios > 10 everywhere, and the §7 rule finds NO gap

Run with `python -m pytest tests` or `python tests/test_gramians_synthetic.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mpmath as mp
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from scripts import gramians as G
from scripts import gate_a_spectrum as GA
from scripts.prereg import load_cfg

CFG = load_cfg()
TOL = CFG["synthetic_validation"]["hsv_rel_tol"]
FLOOR = CFG["gramians"]["numerical_floor_rel"]
H0 = CFG["gramians"]["primary_horizon"]
HORIZONS = [H0] + CFG["gramians"]["robustness_horizons"]


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def random_stable(n, rho, rng):
	A = rng.standard_normal((n, n))
	return A * (rho / max(abs(np.linalg.eigvals(A))))


def lti_stack(A, B, C, H):
	"""Constant Jacobians over a trajectory long enough for one anchor at τ = H."""
	T = 2 * H + 1
	return (np.repeat(A[None], T, 0), np.repeat(B[None], T, 0), np.repeat(C[None], T, 0), H)


def hankel_reference_hsv(A, B, C, H):
	"""Independent reference: singular values of the explicit block-Hankel matrix O_H R_H,
	where R_H = [B, AB, …, A^{H-1}B] and O_H = [C; CA; …; CA^{H-1}]. This is the definition
	of a finite-horizon Hankel singular value, and it forms no Gramian at all."""
	n = A.shape[0]
	R, O, P = [], [], np.eye(n)
	for _ in range(H):
		R.append(P @ B)
		O.append(C @ P)
		P = A @ P
	return np.linalg.svd(np.vstack(O) @ np.hstack(R), compute_uv=False)


def system_with_hsv(target, rng, H_design=64, cond=10.0):
	"""A diagonal system whose horizon-H_design HSVs equal `target`, hidden behind a random
	change of basis with condition number `cond`.

	A = diag(a), B = diag(b), C = diag(c) gives W_c = diag(b² s), W_o = diag(c² s) with
	s_i = Σ_{j<H} a_i^{2j}, so HSV_i = |b_i c_i| s_i. Pick a_i, then b_i = c_i = sqrt(σ_i / s_i).
	"""
	n = len(target)
	a = rng.uniform(0.3, 0.8, n)
	s = (1 - a ** (2 * H_design)) / (1 - a ** 2)
	g = np.sqrt(np.asarray(target) / s)
	Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
	Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
	T = Q1 @ np.diag(np.geomspace(1.0, cond, n)) @ Q2
	Ti = np.linalg.inv(T)
	return T @ np.diag(a) @ Ti, T @ np.diag(g), np.diag(g) @ Ti


def pipeline_hsv(A, B, C, H, method="factor"):
	"""The analysis path at one anchor. 'factor' is the D1 route. 'gramian' is §6 as written:
	form W_c and W_o by recursion, then balance by symmetric eigendecomposition."""
	As, Bs, Cs, tau = lti_stack(A, B, C, H)
	if method == "gramian":
		Wc = G.controllability_gramian(As, Bs, tau, H)
		Wo = G.observability_gramian(As, Cs, tau, H)
		return G.balance(Wc, Wo)[0]
	R = G.controllability_factor(As, Bs, tau, H)
	O = G.observability_factor(As, Cs, tau, H)
	return G.balance_factors(R, O.T)[0][:A.shape[0]]


def mp_reference_hsv(A, B, C, H, dps=40):
	"""Independent reference at 40 significant digits: finite-horizon Gramians by direct
	power sums in mpmath, W_c = Lc Lcᵀ by mp Cholesky, and HSV² = eig(Lcᵀ W_o Lc)."""
	with mp.workdps(dps):
		A_, B_, C_ = mp.matrix(A.tolist()), mp.matrix(B.tolist()), mp.matrix(C.tolist())
		n = A.shape[0]
		Wc, Wo, P = mp.zeros(n, n), mp.zeros(n, n), mp.eye(n)
		for _ in range(H):
			PB, CP = P * B_, C_ * P
			Wc += PB * PB.T
			Wo += CP.T * CP
			P = A_ * P
		Lc = mp.cholesky(Wc)
		ev = mp.eigsy(Lc.T * Wo * Lc, eigvals_only=True)
		return np.sort(np.array([float(mp.sqrt(max(e, 0))) for e in ev]))[::-1]


def knee_spectrum(n, k, slope, extra):
	"""log10 σ_j = -slope (j-1), with an extra drop of `extra` decades after j = k."""
	j = np.arange(1, n + 1)
	return 10.0 ** (-slope * (j - 1) - extra * (j > k))


# ---------------------------------------------------------------------------
# §4 tangent coordinates
# ---------------------------------------------------------------------------

def test_tangent_basis():
	U = G.tangent_basis(512, 8)
	assert U.shape == (512, 448)
	assert np.allclose(U.T @ U, np.eye(448), atol=1e-12)
	ones = np.kron(np.eye(64), np.ones((8, 1)))            # the 64 group-sum directions
	assert np.abs(U.T @ ones).max() < 1e-12
	assert np.linalg.matrix_rank(np.hstack([U, ones])) == 512


# ---------------------------------------------------------------------------
# §6 Gramians
# ---------------------------------------------------------------------------

def test_ltv_recursion_matches_definition():
	"""Recursions against the literal §6 sums, with explicit Φ products and time-varying A."""
	rng = np.random.default_rng(0)
	n, m, p, T, H, tau = 7, 2, 1, 40, 12, 15
	A = np.stack([random_stable(n, 0.95, rng) for _ in range(T)])
	B = rng.standard_normal((T, n, m))
	C = rng.standard_normal((T, p, n))

	def Phi(t, s):
		P = np.eye(n)
		for i in range(s, t):
			P = A[i] @ P
		return P

	Wc = sum(Phi(tau, s + 1) @ B[s] @ B[s].T @ Phi(tau, s + 1).T for s in range(tau - H, tau))
	Wo = sum(Phi(s, tau).T @ C[s].T @ C[s] @ Phi(s, tau) for s in range(tau, tau + H))
	assert np.allclose(G.controllability_gramian(A, B, tau, H), Wc, rtol=1e-12, atol=1e-12)
	assert np.allclose(G.observability_gramian(A, C, tau, H), Wo, rtol=1e-12, atol=1e-12)


def test_lti_gramians_match_lyapunov_closed_form():
	"""For LTI, W_c(H) = W_∞ − A^H W_∞ A^Hᵀ, with W_∞ from a Lyapunov solve."""
	rng = np.random.default_rng(1)
	n, m, p, H = 16, 3, 1, 64
	A, B, C = random_stable(n, 0.97, rng), rng.standard_normal((n, m)), rng.standard_normal((p, n))
	As, Bs, Cs, tau = lti_stack(A, B, C, H)
	AH = np.linalg.matrix_power(A, H)
	Wc_inf = solve_discrete_lyapunov(A, B @ B.T)
	Wo_inf = solve_discrete_lyapunov(A.T, C.T @ C)
	Wc_ref, Wo_ref = Wc_inf - AH @ Wc_inf @ AH.T, Wo_inf - AH.T @ Wo_inf @ AH
	rel = lambda X, Y: np.linalg.norm(X - Y) / np.linalg.norm(Y)
	assert rel(G.controllability_gramian(As, Bs, tau, H), Wc_ref) < 1e-10
	assert rel(G.observability_gramian(As, Cs, tau, H), Wo_ref) < 1e-10


def test_factors_reproduce_gramians_ltv():
	"""R Rᵀ = W_c(τ) and Oᵀ O = W_o(τ) on a time-varying system, and the pooled factor
	reproduces the mean Gramian."""
	rng = np.random.default_rng(6)
	n, m, p, T, H = 9, 3, 1, 90, 16
	A = np.stack([random_stable(n, 0.95, rng) for _ in range(T)])
	B, C = rng.standard_normal((T, n, m)), rng.standard_normal((T, p, n))
	anchors = G.anchor_set(T, margin=H, stride=7)
	pc, po = G.PooledFactor(n), G.PooledFactor(n)
	for t in anchors:
		R, O = G.controllability_factor(A, B, t, H), G.observability_factor(A, C, t, H)
		assert np.allclose(R @ R.T, G.controllability_gramian(A, B, t, H), rtol=1e-12, atol=1e-12)
		assert np.allclose(O.T @ O, G.observability_gramian(A, C, t, H), rtol=1e-12, atol=1e-12)
		pc.add(R)
		po.add(O.T)
	Wcs, Wos = G.anchor_gramians(A, B, C, anchors, H)
	Lc, Lo = pc.factor(), po.factor()
	assert np.allclose(Lc @ Lc.T, G.pool(Wcs), rtol=1e-11, atol=1e-12)
	assert np.allclose(Lo @ Lo.T, G.pool(Wos), rtol=1e-11, atol=1e-12)
	h1 = G.balance_factors(Lc, Lo)[0][:n]
	h2 = G.balance(G.pool(Wcs), G.pool(Wos))[0]
	keep = h1 >= 1e-4 * h1[0]
	assert np.max(np.abs(h1[keep] - h2[keep]) / h1[keep]) < 1e-8


def test_pooling_is_anchor_mean():
	rng = np.random.default_rng(2)
	n, m, T, H = 6, 2, 60, 8
	A = np.stack([random_stable(n, 0.9, rng) for _ in range(T)])
	B, C = rng.standard_normal((T, n, m)), rng.standard_normal((T, 1, n))
	anchors = G.anchor_set(T, margin=H, stride=5)
	Wcs, Wos = G.anchor_gramians(A, B, C, anchors, H)
	assert np.allclose(G.pool(Wcs), sum(Wcs) / len(Wcs))
	assert anchors[0] == H and anchors[-1] <= T - H


def test_anchor_set_matches_prereg():
	a = G.anchor_set(500, CFG["anchors"]["margin"], CFG["anchors"]["stride"])
	assert a[0] == 128 and a[-1] == 368 and len(a) == CFG["anchors"]["per_episode"] == 25


# ---------------------------------------------------------------------------
# Case 1: random stable LTI, HSVs to hsv_rel_tol
# ---------------------------------------------------------------------------

CASE1 = [(12, 2, 0.9), (20, 4, 0.95), (30, 6, 0.8), (16, 1, 0.97)]
_CASE1_REF = {}


def case1_errors(method):
	"""Max relative HSV error against the mp reference per (system, H), over HSVs above the
	floor with k ≤ H/2 + 1 (every σ the §7 rule can read). Also over σ ≥ 1e-4 σ₁ alone."""
	out = []
	for seed, (n, m, rho) in enumerate(CASE1):
		rng = np.random.default_rng(100 + seed)
		A, B, C = random_stable(n, rho, rng), rng.standard_normal((n, m)), rng.standard_normal((1, n))
		for H in HORIZONS:
			if (seed, H) not in _CASE1_REF:
				_CASE1_REF[seed, H] = mp_reference_hsv(A, B, C, H)
			ref = _CASE1_REF[seed, H]
			got = pipeline_hsv(A, B, C, H, method)[:n]
			keep = ref >= FLOOR * ref[0]
			keep[H // 2 + 1:] = False
			assert keep.sum() >= 2
			err = np.abs(got - ref) / np.where(ref > 0, ref, 1.0)
			strong = keep & (ref >= 1e-4 * ref[0])
			out.append((seed, H, err[keep].max(), err[strong].max()))
	return out


def test_case1_mp_reference_agrees_with_float64_hankel():
	"""Sanity check on the reference itself: the explicit Hankel SVD agrees where float64 can."""
	rng = np.random.default_rng(100)
	n, m, rho = CASE1[0]
	A, B, C = random_stable(n, rho, rng), rng.standard_normal((n, m)), rng.standard_normal((1, n))
	ref, hk = mp_reference_hsv(A, B, C, H0), hankel_reference_hsv(A, B, C, H0)[:n]
	keep = ref >= 1e-3 * ref[0]
	assert np.max(np.abs(ref[keep] - hk[keep]) / ref[keep]) < 1e-11


def test_case1_random_stable_lti_hsv():
	"""§13 case 1 as specified. The factor route (D1), which every HSV the rules read goes
	through, must match to hsv_rel_tol everywhere above the floor."""
	for seed, H, err, _ in case1_errors("factor"):
		assert err < TOL, (seed, H, err)


def test_case1_gramian_route_accuracy_is_characterised():
	"""The §6 route as written, kept for the verdict of record (§14). It must meet
	hsv_rel_tol for σ ≥ 1e-4 σ₁ and stay within 1e-5 relative down to the floor. That is
	4e-6 decades, far below any §7 threshold. Measured numbers go in DEVIATIONS.md D1."""
	for seed, H, err, err_strong in case1_errors("gramian"):
		assert err_strong < TOL, (seed, H, err_strong)
		assert err < 1e-5, (seed, H, err)


def test_case1_balanced_realisation():
	"""Tᵀ W_o T = Σ and T⁻¹ W_c T⁻ᵀ = Σ on a full-rank system."""
	rng = np.random.default_rng(3)
	n = 8
	A, B, C = random_stable(n, 0.8, rng), rng.standard_normal((n, n)), rng.standard_normal((n, n))
	As, Bs, Cs, tau = lti_stack(A, B, C, H0)
	Wc, Wo = G.controllability_gramian(As, Bs, tau, H0), G.observability_gramian(As, Cs, tau, H0)
	hsv, T = G.balance(Wc, Wo)
	Ti = np.linalg.inv(T)
	assert np.allclose(T.T @ Wo @ T, np.diag(hsv), atol=1e-9 * hsv[0])
	assert np.allclose(Ti @ Wc @ Ti.T, np.diag(hsv), atol=1e-9 * hsv[0])
	hsv2, T2 = G.balance_factors(G.controllability_factor(As, Bs, tau, H0),
								 G.observability_factor(As, Cs, tau, H0).T)
	assert np.allclose(hsv2[:n], hsv, rtol=1e-10)
	assert np.allclose(T2[:, :n].T @ Wo @ T2[:, :n], np.diag(hsv2[:n]), atol=1e-9 * hsv[0])


# ---------------------------------------------------------------------------
# Case 2: planted knee
# ---------------------------------------------------------------------------

def test_case2_planted_knee_found_at_planted_k():
	"""Through the full pipeline: system → Gramians at every horizon → HSVs → §7 model rule."""
	for k_plant, slope in [(3, 0.2), (6, 0.25), (12, 0.15), (20, 0.12)]:
		rng = np.random.default_rng(200 + k_plant)
		A, B, C = system_with_hsv(knee_spectrum(40, k_plant, slope, 1.5), rng)
		hsv_by_H = {H: pipeline_hsv(A, B, C, H) for H in HORIZONS}
		assert GA.valid_gaps(hsv_by_H[H0], H0, CFG["gap"], FLOOR) == \
			{k_plant: GA.log_drops(hsv_by_H[H0], FLOOR)[k_plant]}
		assert GA.model_gap(hsv_by_H, CFG) == k_plant


def test_case2_knee_too_small_is_rejected():
	"""0.9 decades on a 0.1 slope stands out locally but fails condition 3 (d ≥ 1)."""
	s = knee_spectrum(40, 8, 0.1, 0.8)
	assert GA.valid_gaps(s, H0, CFG["gap"], FLOOR) == {}


def test_case2_dominant_first_mode_is_not_a_gap():
	"""k = 1 is excluded by condition 1, however large the drop."""
	s = knee_spectrum(40, 1, 0.1, 3.0)
	assert GA.log_drops(s, FLOOR)[1] > 3
	assert GA.valid_gaps(s, H0, CFG["gap"], FLOOR) == {}


def test_case2_gap_beyond_H_over_2_is_ignored():
	s = knee_spectrum(60, 33, 0.05, 2.0)
	assert GA.valid_gaps(s, 64, CFG["gap"], FLOOR) == {}


def test_case2_drop_into_floor_is_not_a_gap():
	"""Condition 2: σ_{k+1} must sit above 1e-6 σ₁."""
	s = knee_spectrum(40, 5, 0.1, 7.0)
	assert GA.valid_gaps(s, H0, CFG["gap"], FLOOR) == {}


def test_case2_knee_must_reappear_across_horizons():
	flat = knee_spectrum(40, 40, 0.2, 0.0)
	knee = knee_spectrum(40, 10, 0.2, 1.5)
	assert GA.model_gap({64: knee, 128: knee, 32: knee}, CFG) == 10
	assert GA.model_gap({64: knee, 128: flat, 32: knee}, CFG) is None
	assert GA.model_gap({64: knee, 128: knee, 32: flat}, CFG) is None       # k* ≤ 16 needs H=32
	assert GA.model_gap({64: knee, 128: knee_spectrum(40, 12, 0.2, 1.5), 32: knee}, CFG) == 10
	assert GA.model_gap({64: knee, 128: knee_spectrum(40, 13, 0.2, 1.5), 32: knee}, CFG) is None
	late = knee_spectrum(60, 20, 0.1, 1.5)
	assert GA.model_gap({64: late, 128: late, 32: flat}, CFG) == 20          # k* > 16 skips H=32


# ---------------------------------------------------------------------------
# Case 3: smooth exponential decay, ratios > 10, NO gap
# ---------------------------------------------------------------------------

def test_case3_smooth_decay_has_no_gap():
	"""The test that condition 4 exists for. Condition 3 alone would fire at every k."""
	for ratio in [10.5, 15.0, 30.0]:
		s = ratio ** -np.arange(40.0)
		d = GA.log_drops(s, FLOOR)
		above = [k for k in range(2, 33) if s[k] >= FLOOR * s[0]]
		assert above and all(d[k] >= 1.0 for k in above)      # cond 3 is met wherever it can be
		assert GA.valid_gaps(s, H0, CFG["gap"], FLOOR) == {}


def test_case3_condition4_alone_rejects_smooth_decay():
	"""With the floor at 1e-6 and ratios > 10, only about 5 HSVs survive, and clarification C1
	(floored neighbours count as +inf) could be doing the rejecting. Switching the floor off
	isolates condition 4: 31 finite log-drops, all ≥ 1, and still no gap."""
	for ratio in [10.5, 15.0]:
		s = ratio ** -np.arange(40.0)
		d = GA.log_drops(s, 1e-300)
		assert all(np.isfinite(d[k]) and d[k] >= 1.0 for k in range(1, 33))
		assert GA.valid_gaps(s, H0, CFG["gap"], 1e-300) == {}


def test_case3_smooth_decay_through_pipeline():
	"""Same, but the spectrum comes out of a system, so it carries numerical noise."""
	rng = np.random.default_rng(300)
	A, B, C = system_with_hsv(12.0 ** -np.arange(10.0), rng, cond=3.0)
	hsv_by_H = {H: pipeline_hsv(A, B, C, H) for H in HORIZONS}
	for H in HORIZONS:
		assert GA.valid_gaps(hsv_by_H[H], H, CFG["gap"], FLOOR) == {}
	assert GA.model_gap(hsv_by_H, CFG) is None


def test_case3_jittered_decay_has_no_gap():
	"""Log-drops drawn from [1.05, 1.4] decades: all ≥ 1, no single one ≥ 3x its neighbours."""
	rng = np.random.default_rng(301)
	for _ in range(200):
		s = 10.0 ** -np.concatenate([[0.0], np.cumsum(rng.uniform(1.05, 1.4, 39))])
		assert GA.valid_gaps(s, H0, CFG["gap"], FLOOR) == {}


# ---------------------------------------------------------------------------
# Gate A task logic
# ---------------------------------------------------------------------------

def test_task_gate_a():
	flat = knee_spectrum(40, 40, 0.2, 0.0)
	knee6 = knee_spectrum(40, 6, 0.2, 1.5)
	quiet = {"r_full": [flat] * 3, "r_dyn": [flat] * 3}
	ok, k, _ = GA.task_gate_a([6, 6, 7], quiet, CFG)
	assert ok and k == 6
	ok, _, d = GA.task_gate_a([6, 7, None], quiet, CFG)
	assert ok
	ok, _, d = GA.task_gate_a([6, None, None], quiet, CFG)
	assert not ok and d["reason"] == "too few seeds with a gap"
	ok, _, d = GA.task_gate_a([4, 8, None], quiet, CFG)
	assert not ok and d["reason"] == "seeds disagree on k*"
	# one reproducing control in each family is tolerated; two in either family is not
	ok, _, _ = GA.task_gate_a([6, 6, 6], {"r_full": [knee6, flat, flat], "r_dyn": [knee6, flat, flat]}, CFG)
	assert ok
	for fam in ("r_full", "r_dyn"):
		ctl = {"r_full": [flat] * 3, "r_dyn": [flat] * 3}
		ctl[fam] = [knee6, knee6, flat]
		ok, _, d = GA.task_gate_a([6, 6, 6], ctl, CFG)
		assert not ok and d["control_reproductions"][fam] == 2


def test_zero_output_control_is_degenerate():
	"""The trap behind the R-full exception (§7): C ≡ 0 gives W_o = 0 and all HSVs zero, so the
	control could never show a gap."""
	rng = np.random.default_rng(4)
	n, m = 10, 3
	A, B = random_stable(n, 0.9, rng), rng.standard_normal((n, m))
	hsv = pipeline_hsv(A, B, np.zeros((1, n)), H0)
	assert np.all(hsv == 0.0)


# ---------------------------------------------------------------------------
# §8/§9 subspace helper
# ---------------------------------------------------------------------------

def test_principal_angles():
	rng = np.random.default_rng(5)
	X = rng.standard_normal((20, 4))
	assert G.mean_principal_angle_deg(X, X @ rng.standard_normal((4, 4))) < 1e-5
	E = np.eye(20)
	assert abs(G.mean_principal_angle_deg(E[:, :3], E[:, 3:6]) - 90.0) < 1e-9
	th = np.radians(30.0)
	Y = np.cos(th) * E[:, :1] + np.sin(th) * E[:, 1:2]
	assert abs(G.mean_principal_angle_deg(E[:, :1], Y) - 30.0) < 1e-6


if __name__ == "__main__":
	tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	failed = 0
	for name, fn in tests:
		try:
			fn()
			print(f"PASS  {name}")
		except Exception as e:
			failed += 1
			print(f"FAIL  {name}: {type(e).__name__}: {e}")
	print(f"\n{len(tests) - failed}/{len(tests)} passed")
	sys.exit(1 if failed else 0)
