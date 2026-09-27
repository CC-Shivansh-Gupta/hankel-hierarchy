"""Protocol 2 verdict (prereg-2 §6). Reads only prereg-2/config/prereg.yaml and results2/.

    python -m scripts.h2_verdict [results2_dir]

Held-out (results2/work) produces results2/verdict.json, the only verdict-bearing output. The factor route
decides. Any hypothesis or task on which the gramian route disagrees is flagged, but the decision is not
changed. Discovery (results2/discovery/work), if present, produces results2/discovery/summary.json, which is
labelled EXPLORATORY and has no verdict role.
"""
import json
import sys
from pathlib import Path

from scripts.h2 import decide
from scripts.prereg import ROOT, load_cfg

P1 = load_cfg()
P2 = load_cfg(ROOT / "prereg-2" / "config" / "prereg.yaml")


def paired_ts(seed):
	return P2["controls"]["torch_seeds"][[1, 2, 3].index(seed)]


def load_set(base, seeds_by_task):
	"""{task: [(trained, control)]}, dropping only seeds with a recorded load failure."""
	per_task, excluded = {}, []
	for task, seeds in seeds_by_task.items():
		pairs = []
		for s in seeds:
			if (base / "diagnostics" / f"{task}_s{s}_load_failure.json").exists():
				excluded.append(f"{task}-{s}")
				continue
			trained = base / "work" / f"{task}_s{s}.json"
			if not trained.exists():
				raise FileNotFoundError(f"{trained} missing with no recorded load failure")
			ctl = json.loads((base / "work" / f"{task}_rfull{paired_ts(s)}.json").read_text())
			pairs.append((json.loads(trained.read_text()), ctl))
		per_task[task] = pairs
	return per_task, excluded


def summarise(per_task, excluded, label):
	d = {r: decide(per_task, P2, r) for r in ("factor", "gramian")}
	flags = [f"{h}" for h in ("H2.1", "H2.2", "H2.3")
			 if d["factor"]["hypotheses"][h]["verdict"] != d["gramian"]["hypotheses"][h]["verdict"]]
	flags += [f"{t}:{h}" for t in per_task for h in ("H2.1", "H2.2", "H2.3")
			  if d["factor"]["tasks"][t][h] != d["gramian"]["tasks"][t][h]]
	return {"label": label, "protocol_version": P2["protocol_version"], "deciding_route": "factor",
			"verdicts": {h: v["verdict"] for h, v in d["factor"]["hypotheses"].items()},
			"tasks_supporting": {h: v["tasks_supporting"] for h, v in d["factor"]["hypotheses"].items()},
			"tasks": d["factor"]["tasks"], "excluded_load_failures": excluded,
			"route_disagreements": flags, "gramian_route": d["gramian"]}


def main(results_dir=ROOT / "results2"):
	res = Path(results_dir)
	per_task, excl = load_set(res, P2["models"]["seeds_by_task"])
	out = summarise(per_task, excl, "CONFIRMATORY (held-out, verdict-bearing)")
	(res / "verdict.json").write_text(json.dumps(out, indent=1))
	print(json.dumps({k: out[k] for k in ("verdicts", "tasks_supporting", "excluded_load_failures",
										  "route_disagreements")}, indent=1))
	disc = res / "discovery"
	if (disc / "work").exists():
		seeds = {t: P1["models"]["seeds"] for t in P1["models"]["tasks"]}
		dt, dx = load_set(disc, seeds)
		ds = summarise(dt, dx, "EXPLORATORY (discovery set; no verdict role)")
		(disc / "summary.json").write_text(json.dumps(ds, indent=1))
		print("discovery (exploratory):", ds["verdicts"])


if __name__ == "__main__":
	main(*sys.argv[1:])
