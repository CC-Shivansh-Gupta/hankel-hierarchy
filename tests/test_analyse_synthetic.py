"""The per-model accumulator and gates B/C, checked on synthetic systems only (§13).

Run with `python -m pytest tests` or `python tests/test_analyse_synthetic.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts import gramians as G
from scripts import gate_b_pca as GB
from scripts import gate_c_stability as GC
from scripts.analyse import ModelAccumulator
from scripts.prereg import load_cfg

CFG = load_cfg()


def ltv_system(n, m, p, T, rho, seed):
	rng = np.random.default_rng(seed)
	A = []
	for _ in range(T):
		X = rng.standard_normal((n, n))
		A.append(X * (rho / max(abs(np.linalg.eigvals(X)))))
	return np.stack(A), rng.standard_normal((T, n, m)), rng.standard_normal((T, p, n))


def run_accumulator(episodes, Hs, H0, anchors, k_keep=4):
	n = episodes[0][0].shape[1]
	acc = ModelAccumulator(n, Hs, H0, k_keep=k_keep)
	calls = []
	for A, B, C in episodes:
		def jac(t, A=A, B=B, C=C):
			calls.append(t)
			return A[t], B[t], C[t]
		acc.add_episode(jac, anchors)
	return acc.finalize(), calls


def test_accumulator_matches_direct_computation():
	Hs, H0, T = [4, 8, 16], 8, 70
	eps = [ltv_system(10, 2, 1, T, 0.97, s) for s in (0, 1)]
	anchors = G.anchor_set(T, margin=16, stride=5)
	out, calls = run_accumulator(eps, Hs, H0, anchors)
	per_ep = len(set(range(0, anchors[-1] + 16)))
	assert len(calls) == 2 * per_ep, "each Jacobian must be computed exactly once per episode"
	for H in Hs:
		Wcs = [G.controllability_gramian(A, B, t, H) for A, B, C in eps for t in anchors]
		Wos = [G.observability_gramian(A, C, t, H) for A, B, C in eps for t in anchors]
		hsv_ref, T_ref = G.balance(G.pool(Wcs), G.pool(Wos))
		keep = hsv_ref >= 1e-6 * hsv_ref[0]
		for route in ("gramian", "factor"):
			got = out[route]["hsv"][H][:len(hsv_ref)]
			assert np.max(np.abs(got[keep] - hsv_ref[keep]) / hsv_ref[keep]) < 1e-7, (route, H)
		if H == H0:
			for route in ("gramian", "factor"):
				assert G.mean_principal_angle_deg(out[route]["basis"][:, :3], T_ref[:, :3]) < 1e-4
	# per-anchor bases at H0, in anchor order
	A, B, C = eps[1]
	t = anchors[2]
	_, Tb = G.balance(G.controllability_gramian(A, B, t, H0), G.observability_gramian(A, C, t, H0))
	idx = len(anchors) + 2
	for route in ("gramian", "factor"):
		assert G.mean_principal_angle_deg(out[route]["anchor_bases"][idx][:, :3].astype(float), Tb[:, :3]) < 1e-2


def test_transition_norm():
	A, B, C = ltv_system(8, 1, 1, 40, 1.05, 3)
	out, _ = run_accumulator([(A, B, C)], [4, 8], 8, [8, 20])
	Phi = np.eye(8)
	for j in range(20, 28):
		Phi = A[j] @ Phi
	assert abs(out["diagnostics"]["phi_norm"][8][1] - np.linalg.norm(Phi, 2)) < 1e-8 * np.linalg.norm(Phi, 2)


# ---------------------------------------------------------------------------
# Gate B
# ---------------------------------------------------------------------------

def test_gate_b_gramian_diag_routes_agree():
	rng = np.random.default_rng(10)
	U = G.tangent_basis(64, 8)
	L = rng.standard_normal((56, 20))
	assert np.allclose(GB.gramian_diag_original(U, L=L), GB.gramian_diag_original(U, W=L @ L.T))


def test_gate_b_collapse_rules():
	rng = np.random.default_rng(11)
	n, k = 40, 4
	S = np.linalg.qr(rng.standard_normal((n, k)))[0]
	h = rng.random(64)
	# identical subspaces and identical rankings: collapse by both tests
	r = GB.model_gate_b(h, h, S, S, CFG)
	assert r["collapse"] and r["rho_s"] > 0.99 and r["theta_pca_deg"] < 1e-5
	# orthogonal subspaces, unrelated rankings: no collapse
	E = np.eye(n)
	r = GB.model_gate_b(h, rng.random(64), E[:, :k], E[:, k:2 * k], CFG)
	assert not r["collapse"] and abs(r["theta_pca_deg"] - 90) < 1e-9
	# either test alone is enough
	assert GB.model_gate_b(h, h, E[:, :k], E[:, k:2 * k], CFG)["collapse"]
	assert GB.model_gate_b(h, rng.random(64), S, S, CFG)["collapse"]


def test_gate_b_pca_basis_finds_planted_directions():
	rng = np.random.default_rng(12)
	n, N = 30, 2000
	Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
	Z = (rng.standard_normal((N, n)) * np.r_[10.0, 8.0, 6.0, np.ones(n - 3) * 0.1]) @ Q.T
	assert G.mean_principal_angle_deg(GB.pca_basis(Z, 3), Q[:, :3]) < 1.0


def test_gate_b_task_rules():
	c, nc = {"collapse": True}, {"collapse": False}
	assert GB.task_divergence([nc, nc, c], CFG)
	assert not GB.task_divergence([nc, c, c], CFG)
	assert GB.task_k(True, 6.5, CFG) == 7 and GB.task_k(True, 6.0, CFG) == 6
	assert GB.task_k(False, None, CFG) == 16


def test_random_subspace_reference_is_sane():
	ref = GB.random_subspace_angle_reference(448, 16, samples=20)
	assert 80 < ref < 90


# ---------------------------------------------------------------------------
# Gate C
# ---------------------------------------------------------------------------

def test_gate_c_stability():
	rng = np.random.default_rng(13)
	n, k = 30, 3
	P = rng.standard_normal((n, 5))
	same = [P + 1e-6 * rng.standard_normal((n, 5)) for _ in range(10)]
	assert GC.model_gate_c(same, P, k, CFG)["stable"]
	rand = [rng.standard_normal((n, 5)) for _ in range(10)]
	r = GC.model_gate_c(rand, P, k, CFG)
	assert not r["stable"] and r["theta_stab_deg"] > 30
	assert GC.task_stable([{"stable": True}, {"stable": True}, {"stable": False}], CFG)
	assert not GC.task_stable([{"stable": True}, {"stable": False}, {"stable": False}], CFG)


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
