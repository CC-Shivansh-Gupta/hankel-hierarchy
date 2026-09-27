"""prereg-2 §8 synthetic validation, cases S1–S5, plus the §5–§6 decision logic.
Synthetic LTV systems only; no model is involved.

Run with `python -m pytest tests` or `python tests/test_h2_synthetic.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.linalg import expm

from scripts import h2 as H2
from scripts.gramians import anchor_set, mean_principal_angle_deg
from scripts.prereg import ROOT, load_cfg

CFG = load_cfg(ROOT / "prereg-2" / "config" / "prereg.yaml")
T = 500
ANCHORS = anchor_set(T, 128, 10)


def run(episodes, n):
	"""episodes: list of (A (T,n,n), B (T,n,m), C (T,1,n)). Two passes, then the statistics."""
	acc = H2.H2Accumulator(n, CFG)
	jacs = [lambda t, e=e: (e[0][t], e[1][t], e[2][t]) for e in episodes]
	for i, j in enumerate(jacs):
		acc.pass1_episode(j, ANCHORS, i)
	acc.set_gamma()
	for i, j in enumerate(jacs):
		acc.pass2_episode(j, ANCHORS, i)
	return acc, acc.finalize()


def lti_episode(A, B, C, rng, noise):
	n = A.shape[0]
	As = A[None] + noise * rng.standard_normal((T, n, n)) / np.sqrt(n)
	return As, np.repeat(B[None], T, 0), np.repeat(C[None], T, 0)


# ---------------------------------------------------------------------------
# S1 planted dominance
# ---------------------------------------------------------------------------

def s1_system(seed=0):
	"""n = 48 in three 16-dim blocks U | V | rest, hidden behind a random rotation Q.
	Bulk episodes 1–9 are stable, with U slow (rate a_e from 0.90 to 0.99, varying by episode) and V fast.
	Episode 0, 10% of anchors, is expansive along V (rate 1.12)."""
	rng = np.random.default_rng(seed)
	n, m = 48, 3
	Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
	U, V = Q[:, :16], Q[:, 16:32]
	B = Q @ np.vstack([rng.standard_normal((16, m)), 0.5 * rng.standard_normal((16, m)), 0.1 * rng.standard_normal((16, m))])
	C = np.hstack([rng.standard_normal((1, 16)), 0.5 * rng.standard_normal((1, 16)), 0.1 * rng.standard_normal((1, 16))]) @ Q.T
	eps = []
	for e in range(10):
		# rates are spread within each block: with one output, a repeated eigenvalue is observable
		# along a single direction only, so a 16-dim block needs 16 distinct rates to be identifiable
		spread = np.linspace(0.0, 0.06, 16)
		if e == 0:
			d = np.r_[0.90 - spread, 1.14 - spread, np.linspace(0.35, 0.2, 16)]
		else:
			d = np.r_[np.linspace(0.92, 0.99, 9)[e - 1] - spread, 0.6 - spread, np.linspace(0.35, 0.2, 16)]
		eps.append(lti_episode(Q @ np.diag(d) @ Q.T, B, C, rng, noise=0.001))
	return eps, n, U, V


def test_s1_planted_dominance():
	eps, n, U, V = s1_system()
	acc, s = run(eps, n)
	a = s["routes"]["factor"]
	# H2.1: dominance is flagged
	assert s["kappa_c"] >= 0.5 and s["kappa_o"] >= 0.5, (s["kappa_c"], s["kappa_o"])
	assert s["rho_c"] >= 0.7 and s["rho_o"] >= 0.7, (s["rho_c"], s["rho_o"])
	assert a["theta_bulk"] - a["theta_top"] >= 15, (a["theta_bulk"], a["theta_top"])
	# With one scalar output the Hankel spectrum decays steeply, so a 16-dim *balanced* subspace is
	# not exactly a 16-dim *modal* block: some weak U modes rank below some V or rest modes.
	# "Aligns with" and "recovers" are therefore judged relatively, with 30° and 15° margins.
	P = acc.pool_energy.basis_and_hsv(16)["factor"][0]
	Shat = acc.pool_disc_equal.basis_and_hsv(16)["factor"][0]
	ang = mean_principal_angle_deg
	# the undiscounted, energy-averaged pooled subspace follows the planted V (the protocol-1 failure mode)
	assert ang(P, U) - ang(P, V) >= 30, (ang(P, V), ang(P, U))
	# H2.2: growth normalisation hands the pooled subspace to the bulk. It must sit close to the
	# bulk anchors' own (discounted) balanced subspaces, which are the identifiable objects, and far
	# from the planted episode's anchors (indices 0-24). It must also have moved toward U, away from V.
	th_hat = np.array(a["per_anchor_theta_hat"])
	assert th_hat[25:].mean() <= 20, th_hat[25:].mean()
	assert th_hat[:25].mean() >= 45, th_hat[:25].mean()
	assert ang(Shat, V) - ang(Shat, U) >= 10, (ang(Shat, U), ang(Shat, V))
	assert a["theta_hat_stab"] <= 30, a["theta_hat_stab"]
	assert a["theta_hat_stab"] <= a["theta_stab_protocol1_pooling"] - 15
	# both routes agree on the statistics that decide
	g = s["routes"]["gramian"]
	assert abs(g["theta_hat_stab"] - a["theta_hat_stab"]) < 1 and abs(g["theta_top"] - a["theta_top"]) < 1


# ---------------------------------------------------------------------------
# S2 no dominance
# ---------------------------------------------------------------------------

def test_s2_no_dominance():
	rng = np.random.default_rng(1)
	n, m = 24, 2
	A = rng.standard_normal((n, n))
	A *= 0.9 / max(abs(np.linalg.eigvals(A)))
	B, C = rng.standard_normal((n, m)), rng.standard_normal((1, n))
	eps = [lti_episode(A, B, C, rng, noise=0.001) for _ in range(4)]
	_, s = run(eps, n)
	assert 0.09 < s["kappa"] < 0.2, s["kappa"]
	assert not H2.model_criteria(s, s, CFG)["H2.1"]


# ---------------------------------------------------------------------------
# S3 rotating subspace, S4 state-independent
# ---------------------------------------------------------------------------

def rotating_system(omega, seed=2, episodes=4):
	"""A 4-dim dominant subspace rotating at rate omega (radians per step) inside n = 24."""
	rng = np.random.default_rng(seed)
	n, m = 24, 4
	K = np.zeros((n, n))
	for i in range(4):
		K[i + 4, i], K[i, i + 4] = 1.0, -1.0
	# distinct rates: with one output, repeated eigenvalues are not separately observable
	D = np.diag(np.r_[[0.95, 0.93, 0.91, 0.89], np.linspace(0.6, 0.3, n - 4)])
	B0 = np.vstack([rng.standard_normal((4, m)), 0.1 * rng.standard_normal((n - 4, m))])
	C0 = np.hstack([rng.standard_normal((1, 4)), 0.1 * rng.standard_normal((1, n - 4))])
	R = np.stack([expm(omega * t * K) for t in range(T)])
	A = np.einsum("tij,jk,tlk->til", R, D, R)
	B = np.einsum("tij,jk->tik", R, B0)
	C = np.einsum("ij,tkj->tik", C0, R)
	return [(A, B, C)] * episodes, n


def test_s3_rotating_subspace():
	eps, n = rotating_system(omega=(np.pi / 2) / 260)
	_, s = run(eps, n)
	_, ctl = run(*rotating_system(omega=0.0))          # S4, state-independent: the control
	a = s["routes"]["factor"]
	assert a["theta_near"] <= 30 and a["contrast"] >= 15, (a["theta_near"], a["contrast"])
	assert a["theta_far"] >= 45
	assert a["theta_stab_h16_pooled"] >= a["theta_near"] + 15      # structure is local, not global
	assert H2.model_criteria(s, ctl, CFG)["H2.3"]


def test_s4_state_independent_does_not_pass():
	eps, n = rotating_system(omega=0.0)
	_, s = run(eps, n)
	a = s["routes"]["factor"]
	assert a["theta_near"] < 1 and abs(a["contrast"]) < 1, (a["theta_near"], a["contrast"])
	assert not H2.model_criteria(s, s, CFG)["H2.3"]


# ---------------------------------------------------------------------------
# S5 overlap artefact
# ---------------------------------------------------------------------------

def test_s5_overlap_artefact():
	"""Independent random Jacobians at every step, with long memory (A_t = 0.97 I + small i.i.d. noise;
	B_t and C_t i.i.d.), so there is no persistent structure at all. Windows 10 apart at H = 64 share
	most of their effective steps, so they agree spuriously. The §5.3 design (H = 16, 40 apart, no
	overlap) must show no contrast."""
	rng = np.random.default_rng(5)
	n, m = 40, 2
	eps = []
	for _ in range(3):
		A = 0.97 * np.eye(n)[None] + 0.01 * rng.standard_normal((T, n, n)) / np.sqrt(n)
		eps.append((A, rng.standard_normal((T, n, m)), rng.standard_normal((T, 1, n))))
	acc, s = run(eps, n)
	S64, ids = acc.S64["factor"], acc.anchor_ids
	ov_near = [mean_principal_angle_deg(S64[i], S64[i + 1]) for i in range(len(ids) - 1)
			   if ids[i][0] == ids[i + 1][0] and ids[i + 1][1] - ids[i][1] == 10]
	ov_far = [mean_principal_angle_deg(S64[i], S64[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))
			  if ids[i][0] == ids[j][0] and ids[j][1] - ids[i][1] >= 200]
	assert np.mean(ov_far) - np.mean(ov_near) >= 15, (np.mean(ov_near), np.mean(ov_far))   # the artefact
	a = s["routes"]["factor"]
	assert a["contrast"] < 15, a["contrast"]                                               # the design avoids it
	assert not H2.model_criteria(s, s, CFG)["H2.3"]


# ---------------------------------------------------------------------------
# §5–§6 criteria and decisions
# ---------------------------------------------------------------------------

def fake(kc=0.6, ko=0.6, rc=0.8, ro=0.8, top=10, bulk=40, th=20, near=10, far=40):
	r = {"theta_top": top, "theta_bulk": bulk, "theta_hat_stab": th, "theta_near": near,
		 "theta_far": far, "contrast": far - near}
	return {"kappa_c": kc, "kappa_o": ko, "kappa": min(kc, ko), "rho_c": rc, "rho_o": ro,
			"routes": {"factor": r, "gramian": r}}


def test_model_criteria_thresholds():
	ctl = fake(kc=0.2, ko=0.2, th=60, near=10, far=15)
	assert H2.model_criteria(fake(), ctl, CFG) == {"H2.1": True, "H2.2": True, "H2.3": True}
	assert not H2.model_criteria(fake(ko=0.49), ctl, CFG)["H2.1"]
	assert not H2.model_criteria(fake(rc=0.69), ctl, CFG)["H2.1"]
	assert not H2.model_criteria(fake(top=30, bulk=44), ctl, CFG)["H2.1"]
	assert not H2.model_criteria(fake(), fake(kc=0.7, ko=0.7), CFG)["H2.1"]      # control just as concentrated
	assert not H2.model_criteria(fake(th=31), ctl, CFG)["H2.2"]
	assert not H2.model_criteria(fake(th=20), fake(th=34), CFG)["H2.2"]           # beats control by < 15°
	assert not H2.model_criteria(fake(near=31, far=60), ctl, CFG)["H2.3"]
	assert not H2.model_criteria(fake(near=10, far=24), ctl, CFG)["H2.3"]         # contrast < 15
	assert not H2.model_criteria(fake(near=10, far=30), fake(near=10, far=21), CFG)["H2.3"]   # margin < 10


def test_decide_counts_and_two_of_two():
	good, bad, ctl = fake(), fake(kc=0.1, ko=0.1, th=80, near=50, far=50), fake(kc=0.1, ko=0.1, th=80, near=10, far=10)
	tasks = {f"t{i}": [(good, ctl), (good, ctl), (bad, ctl)] for i in range(4)}
	tasks["t4"] = [(bad, ctl)] * 3
	tasks["t5"] = [(good, ctl), (bad, ctl)]                  # a 2-seed task needs 2-of-2
	d = H2.decide(tasks, CFG)
	assert d["hypotheses"]["H2.1"] == {"tasks_supporting": 4, "verdict": "SUPPORTED"}
	assert not d["tasks"]["t5"]["H2.1"]
	tasks["t0"] = [(good, ctl), (bad, ctl), (bad, ctl)]
	assert H2.decide(tasks, CFG)["hypotheses"]["H2.2"]["verdict"] == "REJECTED"
	tasks["t0"] = [(good, ctl)]                              # a task left with one seed cannot support
	assert not H2.decide(tasks, CFG)["tasks"]["t0"]["H2.1"]


def test_h2_verdict_end_to_end():
	"""h2_verdict on fabricated result files: 2-of-2 for humanoid-run, recorded exclusions, and a
	missing file with no record must raise."""
	import json, tempfile
	from scripts import h2_verdict as HV
	good, bad = fake(), fake(kc=0.1, ko=0.1, th=80, near=50, far=50)
	ctl = fake(kc=0.1, ko=0.1, th=80, near=10, far=10)
	with tempfile.TemporaryDirectory() as tmp:
		base = Path(tmp)
		(base / "work").mkdir()
		(base / "diagnostics").mkdir()
		for task, seeds in CFG["models"]["seeds_by_task"].items():
			for s in seeds:
				ok = task != "humanoid-run" or s == 1        # humanoid-run: only 1 of its 2 seeds passes
				(base / "work" / f"{task}_s{s}.json").write_text(json.dumps(good if ok else bad))
				(base / "work" / f"{task}_rfull{HV.paired_ts(s)}.json").write_text(json.dumps(ctl))
		per_task, excl = HV.load_set(base, CFG["models"]["seeds_by_task"])
		out = HV.summarise(per_task, excl, "test")
		assert out["verdicts"] == {"H2.1": "SUPPORTED", "H2.2": "SUPPORTED", "H2.3": "SUPPORTED"}
		assert out["tasks_supporting"]["H2.1"] == 5 and not out["tasks"]["humanoid-run"]["H2.1"]
		assert out["route_disagreements"] == []
		(base / "work" / "walker-run_s2.json").unlink()
		try:
			HV.load_set(base, CFG["models"]["seeds_by_task"])
			raise AssertionError("missing file without a record must raise")
		except FileNotFoundError:
			pass
		(base / "diagnostics" / "walker-run_s2_load_failure.json").write_text("{}")
		per_task, excl = HV.load_set(base, CFG["models"]["seeds_by_task"])
		assert excl == ["walker-run-2"] and len(per_task["walker-run"]) == 2


if __name__ == "__main__":
	tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	failed = 0
	for name, fn in tests:
		try:
			fn()
			print(f"PASS  {name}", flush=True)
		except Exception as e:
			failed += 1
			print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)
	print(f"\n{len(tests) - failed}/{len(tests)} passed")
	sys.exit(1 if failed else 0)
