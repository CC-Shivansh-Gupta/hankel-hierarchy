"""The mechanical verdict (§10). Reads only config/prereg.yaml and results/. Writes the §12 result
files and results/verdict.json.

Input: one file per analysed model in results/work/, as written by collect_anchors.py:
  {task}_s{seed}.npz      trained model
  {task}_rfull{seed}.npz  R-full control (torch seed 1000-1002)
  {task}_rdyn{seed}.npz   R-dyn control
Each file holds, for route r in {gramian, factor}:
  r_hsv_H{H}                 pooled HSVs
  r_basis                    pooled balanced basis at H0 (448 x 32, tangent coordinates)
  r_anchor_bases             per-anchor balanced bases at H0 (N x 448 x 32), trained only
  r_diag_c, r_diag_o         diag(U W̄ Uᵀ) in original coordinates (512), trained only
and, for trained models, Z (anchor states, N x 512, original coordinates).

Both routes are decided independently. Under §14 the gramian route is the verdict of record
(DEVIATIONS.md D1), and any disagreement between the routes is reported.

    python -m scripts.verdict [results_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

from scripts import gate_a_spectrum as GA
from scripts import gate_b_pca as GB
from scripts import gate_c_stability as GC
from scripts.gramians import tangent_basis
from scripts.prereg import ROOT, load_cfg, read_npz

ROUTES = ("gramian", "factor")
RECORD = "gramian"


def horizons(cfg):
	return [cfg["gramians"]["primary_horizon"]] + cfg["gramians"]["robustness_horizons"]


def decide(models, controls, cfg, route):
	"""models[task][seed] -> dict of arrays for this route (keys without the route prefix).
	controls[task][family] -> list of primary-horizon HSV arrays.
	Returns per-task and per-gate verdicts plus every per-model statistic."""
	H0 = cfg["gramians"]["primary_horizon"]
	floor_rel = cfg["gramians"]["numerical_floor_rel"]
	U = tangent_basis(cfg["models"]["architecture_expected"]["latent_dim"],
					  cfg["models"]["architecture_expected"]["simnorm_group"])
	need_tasks = cfg["decision"]["min_tasks_per_gate"]
	out = {"tasks": {}, "per_model": {}}

	for task in cfg["models"]["tasks"]:
		seeds = cfg["models"]["seeds"]
		# Gate A
		kstars, per_model = [], {}
		for s in seeds:
			m = models[task][s]
			hsv_by_H = {H: m[f"hsv_H{H}"] for H in horizons(cfg)}
			ks = GA.model_gap(hsv_by_H, cfg)
			kstars.append(ks)
			per_model[s] = {"k_star": ks,
							"valid_gaps_H": {H: {int(k): float(d) for k, d in
												 GA.valid_gaps(hsv_by_H[H], H, cfg["gap"], floor_rel).items()}
											 for H in horizons(cfg)}}
		a_pass, a_k, a_detail = GA.task_gate_a(kstars, controls[task], cfg)

		# Gates B and C share k
		k = GB.task_k(a_pass, a_k, cfg)
		b_res, c_res = [], []
		for s in seeds:
			m = models[task][s]
			Z = m["Z"]
			h = GB.hankel_importance(m["diag_c"], m["diag_o"])
			v = Z.var(axis=0)
			S_bal = m["basis"][:, :k]
			S_pca = GB.pca_basis(Z @ U, k)
			b = GB.model_gate_b(h, v, S_bal, S_pca, cfg)
			c = GC.model_gate_c(list(m["anchor_bases"].astype(np.float64)), m["basis"], k, cfg)
			b_res.append(b)
			c_res.append(c)
			per_model[s].update(k=k, gate_b=b, gate_c={kk: vv for kk, vv in c.items()
														 if kk != "anchor_angles_deg"},
								anchor_angles_deg=c["anchor_angles_deg"])
		out["per_model"][task] = per_model
		out["tasks"][task] = {
			"A": {"pass": a_pass, **_jsonable(a_detail)},
			"B": {"divergence": GB.task_divergence(b_res, cfg), "k": k},
			"C": {"stable": GC.task_stable(c_res, cfg), "k": k},
		}

	t = out["tasks"]
	gates = {
		"A": sum(v["A"]["pass"] for v in t.values()) >= need_tasks,
		"B": sum(v["B"]["divergence"] for v in t.values()) >= need_tasks,
		"C": sum(v["C"]["stable"] for v in t.values()) >= need_tasks,
	}
	out["gates"] = gates
	out["counts"] = {g: sum(v[g][key] for v in t.values())
					 for g, key in (("A", "pass"), ("B", "divergence"), ("C", "stable"))}
	out["overall_survives"] = all(gates.values())
	return out


def _booleans(d):
	"""The decision-bearing part of one route's result: every task flag, every gate, the overall."""
	return ({t: (v["A"]["pass"], v["B"]["divergence"], v["C"]["stable"]) for t, v in d["tasks"].items()},
			d["gates"], d["overall_survives"])


def _jsonable(x):
	if isinstance(x, dict):
		return {str(k): _jsonable(v) for k, v in x.items()}
	if isinstance(x, (list, tuple)):
		return [_jsonable(v) for v in x]
	if isinstance(x, (np.floating, np.integer)):
		return x.item()
	return x


def load(results_dir, cfg):
	"""Read results/work/*.npz into decide()'s inputs, per route."""
	work = Path(results_dir) / "work"
	H0 = cfg["gramians"]["primary_horizon"]
	models = {r: {} for r in ROUTES}
	controls = {r: {} for r in ROUTES}
	for task in cfg["models"]["tasks"]:
		for r in ROUTES:
			models[r][task], controls[r][task] = {}, {"r_full": [], "r_dyn": []}
		for s in cfg["models"]["seeds"]:
			f = read_npz(work / f"{task}_s{s}.npz")
			for r in ROUTES:
				models[r][task][s] = {
					**{f"hsv_H{H}": f[f"{r}_hsv_H{H}"] for H in horizons(cfg)},
					"basis": f[f"{r}_basis"], "anchor_bases": f[f"{r}_anchor_bases"],
					"diag_c": f[f"{r}_diag_c"], "diag_o": f[f"{r}_diag_o"], "Z": f["Z"]}
		for fam, tag in (("r_full", "rfull"), ("r_dyn", "rdyn")):
			for ts in cfg["gap"]["random_init_controls"]["torch_seeds"]:
				f = read_npz(work / f"{task}_{tag}{ts}.npz")
				for r in ROUTES:
					controls[r][task][fam].append(f[f"{r}_hsv_H{H0}"])
	return models, controls


def write_outputs(results_dir, cfg, decisions, models, controls):
	res = Path(results_dir)
	floor_rel = cfg["gramians"]["numerical_floor_rel"]
	for sub in ("hsv", "pca", "stability"):
		(res / sub).mkdir(parents=True, exist_ok=True)
	for task in cfg["models"]["tasks"]:
		for s in cfg["models"]["seeds"]:
			for H in horizons(cfg):
				arrs = {}
				for r in ROUTES:
					hsv = models[r][task][s][f"hsv_H{H}"]
					d = GA.log_drops(hsv, floor_rel)
					arrs[f"{r}_hsv"] = hsv
					arrs[f"{r}_d"] = np.array([d[k] for k in sorted(d)])
					ks = decisions[r]["per_model"][task][s]["k_star"]
					arrs[f"{r}_k_star"] = np.array(-1 if ks is None else ks)
				np.savez(res / "hsv" / f"{task}_s{s}_H{H}.npz", **arrs)
			pm = {r: decisions[r]["per_model"][task][s] for r in ROUTES}
			with open(res / "pca" / f"{task}_s{s}.json", "w") as fh:
				json.dump({r: {"k": pm[r]["k"], **pm[r]["gate_b"]} for r in ROUTES}, fh, indent=1)
			with open(res / "stability" / f"{task}_s{s}.json", "w") as fh:
				json.dump({r: {"k": pm[r]["k"], **pm[r]["gate_c"],
							   "anchor_angles_deg": pm[r]["anchor_angles_deg"]} for r in ROUTES}, fh, indent=1)
		H0 = cfg["gramians"]["primary_horizon"]
		for fam, tag in (("r_full", "rfull"), ("r_dyn", "rdyn")):
			for i, ts in enumerate(cfg["gap"]["random_init_controls"]["torch_seeds"]):
				np.savez(res / "hsv" / f"{task}_{tag}{ts}_H{H0}.npz",
						 **{f"{r}_hsv": controls[r][task][fam][i] for r in ROUTES})

	summary = {
		"protocol_version": cfg["protocol_version"],
		"verdict_of_record_route": RECORD,
		"overall_survives": decisions[RECORD]["overall_survives"],
		"gates": decisions[RECORD]["gates"],
		"counts_of_6_tasks": decisions[RECORD]["counts"],
		"tasks": decisions[RECORD]["tasks"],
		"routes": {r: {"overall_survives": decisions[r]["overall_survives"],
					   "gates": decisions[r]["gates"], "tasks": decisions[r]["tasks"]} for r in ROUTES},
		"routes_agree": all(_booleans(decisions[r]) == _booleans(decisions[RECORD]) for r in ROUTES),
	}
	with open(res / "verdict.json", "w") as fh:
		json.dump(_jsonable(summary), fh, indent=1)
	return summary


def main(results_dir=ROOT / "results"):
	cfg = load_cfg()
	models, controls = load(results_dir, cfg)
	decisions = {r: decide(models[r], controls[r], cfg, r) for r in ROUTES}
	summary = write_outputs(results_dir, cfg, decisions, models, controls)
	print(json.dumps({k: summary[k] for k in ("overall_survives", "gates", "counts_of_6_tasks",
											  "routes_agree")}, indent=1))


if __name__ == "__main__":
	main(*sys.argv[1:])
