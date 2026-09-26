"""verdict.py end to end on fabricated results/work files with known answers. No model is involved.

Run with `python -m pytest tests` or `python tests/test_verdict_synthetic.py`.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scripts import verdict as V
from scripts.gramians import tangent_basis
from scripts.prereg import load_cfg

CFG = load_cfg()
U = tangent_basis(512, 8)
N_ANCH = 40


def spectrum(knee_k=None, n=64, slope=0.1, extra=1.5):
	j = np.arange(1, n + 1)
	return 10.0 ** (-slope * (j - 1) - (extra * (j > knee_k) if knee_k else 0))


def trained_file(path, rng, knee_k, unstable=False, pca_like=False):
	"""A trained-model file. The knee is at knee_k at every horizon, or there is none.
	The balanced basis is either unrelated to the PCA directions or equal to them (pca_like), and
	the per-anchor bases either track the pooled basis or are random (unstable)."""
	n = 448
	basis = np.linalg.qr(rng.standard_normal((n, 32)))[0]
	scales = np.r_[np.linspace(10, 5, 32), np.full(n - 32, 0.05)]
	if pca_like:        # leading variance directions = the balanced basis
		pca_dirs = np.linalg.qr(np.hstack([basis, rng.standard_normal((n, n - 32))]))[0]
	else:               # leading variance directions orthogonal to the balanced basis
		P = rng.standard_normal((n, n))
		P -= basis @ (basis.T @ P)
		pca_dirs = np.linalg.qr(P)[0]
	Zt = (rng.standard_normal((N_ANCH * 8, n)) * scales) @ pca_dirs.T
	Z = Zt @ U.T + 1.0 / 8
	if pca_like:
		diag_c = diag_o = Z.var(axis=0) ** 1.0
	else:
		diag_c, diag_o = rng.random(512), rng.random(512)
	if unstable:
		ab = rng.standard_normal((N_ANCH, n, 32))
	else:
		ab = basis[None] + 1e-4 * rng.standard_normal((N_ANCH, n, 32))
	arrs = {"Z": Z}
	for r in V.ROUTES:
		for H in (32, 64, 128):
			arrs[f"{r}_hsv_H{H}"] = spectrum(knee_k)
		arrs.update({f"{r}_basis": basis, f"{r}_anchor_bases": ab.astype(np.float32),
					 f"{r}_diag_c": diag_c, f"{r}_diag_o": diag_o})
	np.savez(path, **arrs)


def control_file(path, knee_k):
	np.savez(path, **{f"{r}_hsv_H64": spectrum(knee_k) for r in V.ROUTES})


def build(tmp, spec):
	"""spec[task] = dict(seeds=[k or None]*3, rfull=[k or None]*3, rdyn=[...], unstable, pca_like)."""
	work = Path(tmp) / "work"
	work.mkdir(parents=True)
	rng = np.random.default_rng(0)
	diag = Path(tmp) / "diagnostics"
	diag.mkdir(parents=True)
	for task, sp in spec.items():
		excluded = sp.get("excluded")               # a seed whose converted checkpoint failed D2
		for s, k in zip(CFG["models"]["seeds"], sp["seeds"]):
			if s == excluded:
				(diag / f"{task}_s{s}_conversion.json").write_text(json.dumps({"accepted": False}))
				continue
			trained_file(work / f"{task}_s{s}.npz", rng, k, sp.get("unstable", False), sp.get("pca_like", False))
		for tag in ("rfull", "rdyn"):
			for s, ts, k in zip(CFG["models"]["seeds"], CFG["gap"]["random_init_controls"]["torch_seeds"],
								sp.get(tag, [None] * 3)):
				if s != excluded:
					control_file(work / f"{task}_{tag}{ts}.npz", k)


def run(spec):
	with tempfile.TemporaryDirectory() as tmp:
		build(tmp, spec)
		models, controls = V.load(tmp, CFG)
		decisions = {r: V.decide(models[r], controls[r], CFG, r) for r in V.ROUTES}
		summary = V.write_outputs(tmp, CFG, decisions, models, controls)
		res = Path(tmp)
		files = {p.relative_to(res).as_posix() for p in res.rglob("*") if p.is_file()}
		on_disk = json.loads((res / "verdict.json").read_text())
	return summary, files, on_disk


TASKS = CFG["models"]["tasks"]


def base_spec():
	return {t: {"seeds": [6, 6, 7] if i < 4 else [None, None, None]} for i, t in enumerate(TASKS)}


def test_all_gates_pass():
	s, files, disk = run(base_spec())
	assert s["gates"] == {"A": True, "B": True, "C": True} and s["overall_survives"]
	assert s["counts_of_6_tasks"] == {"A": 4, "B": 6, "C": 6}
	assert s["routes_agree"] and disk["overall_survives"] is True
	assert s["tasks"][TASKS[0]]["B"]["k"] == 6 and s["tasks"][TASKS[5]]["B"]["k"] == 16
	for t in TASKS:
		for sd in (1, 2, 3):
			assert f"pca/{t}_s{sd}.json" in files and f"stability/{t}_s{sd}.json" in files
			for H in (32, 64, 128):
				assert f"hsv/{t}_s{sd}_H{H}.npz" in files
		for ts in (1000, 1001, 1002):
			assert f"hsv/{t}_rfull{ts}_H64.npz" in files and f"hsv/{t}_rdyn{ts}_H64.npz" in files


def test_controls_kill_gate_a():
	spec = base_spec()
	for t in TASKS[:2]:
		spec[t]["rdyn"] = [6, 6, None]            # two of three R-dyn controls reproduce
	s, _, _ = run(spec)
	assert s["counts_of_6_tasks"]["A"] == 2 and not s["gates"]["A"] and not s["overall_survives"]
	assert "random-init control" in s["tasks"][TASKS[0]]["A"]["reason"]


def test_pca_collapse_kills_gate_b():
	spec = base_spec()
	for t in TASKS[:3]:
		spec[t]["pca_like"] = True
	s, _, _ = run(spec)
	assert s["counts_of_6_tasks"]["B"] == 3 and not s["gates"]["B"] and not s["overall_survives"]
	assert s["gates"]["A"] and s["gates"]["C"]


def test_instability_kills_gate_c():
	spec = base_spec()
	for t in TASKS[:3]:
		spec[t]["unstable"] = True
	s, _, _ = run(spec)
	assert s["counts_of_6_tasks"]["C"] == 3 and not s["gates"]["C"] and not s["overall_survives"]


def test_excluded_seed_makes_rules_two_of_two():
	"""D2: with seed 3 excluded, a task passes only if BOTH remaining seeds pass."""
	spec = base_spec()
	spec[TASKS[4]] = {"seeds": [6, 6, None], "excluded": 3}          # 2 of 2 have the gap: pass
	s, files, _ = run(spec)
	assert s["tasks"][TASKS[4]]["A"]["pass"] and s["counts_of_6_tasks"]["A"] == 5
	assert f"pca/{TASKS[4]}_s3.json" not in files and f"hsv/{TASKS[4]}_rfull1002_H64.npz" not in files
	spec[TASKS[4]] = {"seeds": [6, None, 6], "excluded": 3}          # only 1 of 2: fail
	s, _, _ = run(spec)
	assert not s["tasks"][TASKS[4]]["A"]["pass"]
	spec[TASKS[4]] = {"seeds": [6, 6, None], "excluded": 3, "rdyn": [6, 6, None]}   # 2 of 2 controls reproduce
	s, _, _ = run(spec)
	assert not s["tasks"][TASKS[4]]["A"]["pass"]


def test_missing_file_without_exclusion_is_an_error():
	with tempfile.TemporaryDirectory() as tmp:
		build(tmp, base_spec())
		(Path(tmp) / "work" / f"{TASKS[0]}_s2.npz").unlink()
		try:
			V.load(tmp, CFG)
		except FileNotFoundError:
			return
	raise AssertionError("a silently missing model must not be dropped")


def test_figures_render():
	from scripts import make_figures as MF
	spec = base_spec()
	spec[TASKS[1]]["rfull"] = [6, None, None]
	spec[TASKS[4]] = {"seeds": [None, None, None], "excluded": 3}
	with tempfile.TemporaryDirectory() as tmp:
		build(tmp, spec)
		models, controls = V.load(tmp, CFG)
		decisions = {r: V.decide(models[r], controls[r], CFG, r) for r in V.ROUTES}
		V.write_outputs(tmp, CFG, decisions, models, controls)
		MF.main(tmp, Path(tmp) / "figures")
		names = sorted(p.name for p in (Path(tmp) / "figures").iterdir())
	assert names == ["fig1_hsv_spectra.pdf", "fig2_log_drops.pdf", "fig3_spearman.pdf",
					 "fig4_principal_angles.pdf", "fig5_stability.pdf"]


if __name__ == "__main__":
	tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
	failed = 0
	for name, fn in tests:
		try:
			fn()
			print(f"PASS  {name}")
		except Exception as e:
			failed += 1
			import traceback; traceback.print_exc()
			print(f"FAIL  {name}: {type(e).__name__}: {e}")
	print(f"\n{len(tests) - failed}/{len(tests)} passed")
	sys.exit(1 if failed else 0)
