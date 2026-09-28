"""Protocol 2 result tables and figure, read from results2/ and prereg-2/config/prereg.yaml only.

    python -m scripts.h2_report [results2_dir]

Writes RESULTS2-tables.md (the tables quoted in RESULTS2.md) and figures/fig6_h2_primary.pdf. It computes no
new verdict-bearing quantity. The confirmatory section restates the §5 statistics and §6 conditions that
scripts/h2_verdict.py already decides on. The exploratory section (§7.1) is labelled as such, and it holds the
Gramian route, the §5.2 ablations and the protocol-1 §7 gap rule on the growth-normalised HSVs.
"""
import json
import sys
from pathlib import Path

import numpy as np

from scripts import gate_a_spectrum as GA
from scripts.h2 import model_criteria
from scripts.prereg import ROOT, load_cfg

P1 = load_cfg()
P2 = load_cfg(ROOT / "prereg-2" / "config" / "prereg.yaml")
C1, C2, C3 = P2["h2_1_growth_dominance"], P2["h2_2_growth_normalised"], P2["h2_3_trajectory_conditioned"]
PAIR = dict(zip([1, 2, 3], P2["controls"]["torch_seeds"]))


def load(res):
	rows = []
	for task, seeds in P2["models"]["seeds_by_task"].items():
		for s in seeds:
			diag = json.loads((res / "diagnostics" / f"{task}_s{s}.json").read_text())
			m = json.loads((res / "work" / f"{task}_s{s}.json").read_text())
			r = json.loads((res / "work" / f"{task}_rfull{PAIR[s]}.json").read_text())
			rows.append((task, s, diag, m, r))
	return rows


def conditions(m, r, route="factor"):
	"""Every §5 condition separately, so the table shows which one failed."""
	a, b = m["routes"][route], r["routes"][route]
	return {
		"H2.1": {"kc>=.5": m["kappa_c"] >= C1["min_energy_share"], "ko>=.5": m["kappa_o"] >= C1["min_energy_share"],
				 "rc>=.7": m["rho_c"] >= C1["min_spearman_growth"], "ro>=.7": m["rho_o"] >= C1["min_spearman_growth"],
				 "bulk-top>=15": a["theta_bulk"] - a["theta_top"] >= C1["min_bulk_minus_top_deg"],
				 "k>kR": m["kappa"] > r["kappa"]},
		"H2.2": {"th<=30": a["theta_hat_stab"] <= C2["max_theta_stab_deg"],
				 "th<=R-15": a["theta_hat_stab"] <= b["theta_hat_stab"] - C2["min_margin_vs_rfull_deg"]},
		"H2.3": {"near<=30": a["theta_near"] <= C3["max_theta_near_deg"], "D>=15": a["contrast"] >= C3["min_contrast_deg"],
				 "D>=DR+10": a["contrast"] >= b["contrast"] + C3["min_contrast_margin_vs_rfull_deg"]},
	}


def mark(ok):
	return "✓" if ok else "✗"


def f(x, n=2):
	return f"{x:.{n}f}"


def receipt(rows, res):
	cfg_sha = P2["models"]["checkpoint_sha256"]
	L = ["## Execution receipt", "",
		 "| model | checkpoint SHA-256 = frozen config | mean return ± sd (10 episodes) | collection time (min) |",
		 "|---|---|---|---|"]
	for task, s, d, _, _ in rows:
		ret = np.array(d["episode_returns"])
		L.append(f"| {task}-{s} | {mark(d['checkpoint_sha256'] == cfg_sha[f'{task}-{s}.pt'])} "
				 f"| {ret.mean():.0f} ± {ret.std():.0f} | {d['seconds'] / 60:.1f} |")
	L += ["", f"Total trained-model collection time: {sum(r[2]['seconds'] for r in rows) / 3600:.2f} h.",
		  "", "Environment (`results2/ENVIRONMENT.txt`):", "", "```", (res / "ENVIRONMENT.txt").read_text().strip(), "```", ""]
	return L


def confirmatory(rows, verdict):
	L = ["## Confirmatory statistics (held-out, factor route; verdict-bearing)", "",
		 "Every §5 statistic for every analysed model, with its paired R-full control. ✓/✗ marks each §5 condition.", "",
		 "### H2.1: growth dominance (H = 64)", "",
		 "| model | κ_c | κ_o | ρ_c | ρ_o | θ_bulk − θ_top (°) | κ vs R-full κ | conditions | model |",
		 "|---|---|---|---|---|---|---|---|---|"]
	for task, s, _, m, r in rows:
		a = m["routes"]["factor"]
		c = conditions(m, r)["H2.1"]
		L.append(f"| {task}-{s} | {f(m['kappa_c'])} | {f(m['kappa_o'])} | {f(m['rho_c'])} | {f(m['rho_o'])} "
				 f"| {f(a['theta_bulk'] - a['theta_top'], 1)} | {f(m['kappa'])} vs {f(r['kappa'])} "
				 f"| {''.join(mark(v) for v in c.values())} | {mark(all(c.values()))} |")
	L += ["", "Condition order: κ_c ≥ 0.5, κ_o ≥ 0.5, ρ_c ≥ 0.7, ρ_o ≥ 0.7, θ_bulk − θ_top ≥ 15°, κ > R-full κ.", "",
		  "### H2.2: growth-normalised stability (H = 64, k = 16)", "",
		  "| model | γ_m | θ̂_stab (°) | R-full θ̂_stab (°) | margin (°) | conditions | model |", "|---|---|---|---|---|---|---|"]
	for task, s, _, m, r in rows:
		a, b = m["routes"]["factor"], r["routes"]["factor"]
		c = conditions(m, r)["H2.2"]
		L.append(f"| {task}-{s} | {f(m['gamma'], 3)} | {f(a['theta_hat_stab'], 1)} | {f(b['theta_hat_stab'], 1)} "
				 f"| {f(b['theta_hat_stab'] - a['theta_hat_stab'], 1)} | {''.join(mark(v) for v in c.values())} "
				 f"| {mark(all(c.values()))} |")
	L += ["", "Condition order: θ̂_stab ≤ 30°, θ̂_stab ≤ R-full θ̂_stab − 15°.", "",
		  "### H2.3: trajectory-conditioned structure (H = 16, k = 4)", "",
		  "| model | θ_near (°) | θ_far (°) | Δ (°) | R-full θ_near / θ_far / Δ (°) | pairs near / far | conditions | model |",
		  "|---|---|---|---|---|---|---|---|"]
	for task, s, _, m, r in rows:
		a, b = m["routes"]["factor"], r["routes"]["factor"]
		c = conditions(m, r)["H2.3"]
		L.append(f"| {task}-{s} | {f(a['theta_near'], 1)} | {f(a['theta_far'], 1)} | {f(a['contrast'], 1)} "
				 f"| {f(b['theta_near'], 1)} / {f(b['theta_far'], 1)} / {f(b['contrast'], 1)} "
				 f"| {m['n_near_pairs']} / {m['n_far_pairs']} | {''.join(mark(v) for v in c.values())} "
				 f"| {mark(all(c.values()))} |")
	L += ["", "Condition order: θ_near ≤ 30°, Δ ≥ 15°, Δ ≥ R-full Δ + 10°.", "", "### Condition failure counts (17 models)", "",
		  "| hypothesis | condition | models meeting it |", "|---|---|---|"]
	for h in ("H2.1", "H2.2", "H2.3"):
		keys = list(conditions(rows[0][3], rows[0][4])[h])
		for k in keys:
			L.append(f"| {h} | {k} | {sum(conditions(m, r)[h][k] for *_, m, r in rows)} / {len(rows)} |")
		L.append(f"| {h} | **all** | **{sum(all(conditions(m, r)[h].values()) for *_, m, r in rows)} / {len(rows)}** |")
	# The per-model marks must agree with the verdict script's own per-seed decisions.
	for task, s, _, m, r in rows:
		ref = verdict["tasks"][task]["per_seed"][P2["models"]["seeds_by_task"][task].index(s)]
		assert model_criteria(m, r, P2) == ref, (task, s)
		assert {h: all(v.values()) for h, v in conditions(m, r).items()} == ref, (task, s)
	L += ["", "### Verdict mapping (§6)", "", "| hypothesis | tasks supporting (need ≥ 4 of 6) | verdict |", "|---|---|---|"]
	for h in ("H2.1", "H2.2", "H2.3"):
		L.append(f"| {h} | {verdict['tasks_supporting'][h]} of 6 | **{verdict['verdicts'][h]}** |")
	L += ["", "| task | seeds | H2.1 | H2.2 | H2.3 |", "|---|---|---|---|---|"]
	for task, v in verdict["tasks"].items():
		L.append(f"| {task} | {v['n_seeds']} ({'2-of-2' if v['n_seeds'] == 2 else '≥ 2 of 3'}) "
				 f"| {sum(x['H2.1'] for x in v['per_seed'])} | {sum(x['H2.2'] for x in v['per_seed'])} "
				 f"| {sum(x['H2.3'] for x in v['per_seed'])} |")
	L += ["", "Cells are the number of seeds meeting the model criterion.", ""]
	return L


def exploratory(rows, verdict):
	L = ["## EXPLORATORY (§7.1: reported and labelled; no verdict role)", "",
		 "### E-a. Gramian numerical route", "",
		 f"Route disagreements flagged by `h2_verdict.py`: {verdict['route_disagreements'] or 'none'}. "
		 f"Gramian-route verdicts: " + ", ".join(f"{h} {v['verdict']} ({v['tasks_supporting']}/6)"
												 for h, v in verdict["gramian_route"]["hypotheses"].items()) + ".", ""]
	keys = ["theta_top", "theta_bulk", "theta_hat_stab", "theta_near", "theta_far", "contrast"]
	dev = max(abs(x["routes"]["factor"][k] - x["routes"]["gramian"][k]) for *_, m, r in rows for x in (m, r) for k in keys)
	L += [f"Largest |factor − gramian| difference over those angle statistics, trained and R-full: {dev:.2g}°.", "",
		  "### E-b. §5.2 single-change ablations and the protocol-1 pooling (H = 64, k = 16, angles in °)", "",
		  "| model | protocol-1 pooling | discount only | equal-energy only | both (θ̂_stab) | protocol-1 §7 gap on growth-normalised HSVs |",
		  "|---|---|---|---|---|---|"]
	gcfg, floor = P1["gap"], P2["numerics"]["numerical_floor_rel"]
	for task, s, _, m, r in rows:
		a = m["routes"]["factor"]
		hsv = np.array(a["hsv_growth_normalised"])
		ks = GA.k_star(hsv, C2["horizon"], gcfg, floor)
		d = GA.log_drops(hsv, floor)
		kmax = int(C2["horizon"] * gcfg["k_max_fraction_of_H"])
		big = max(range(gcfg["k_min"], kmax + 1), key=lambda k: d[k])
		L.append(f"| {task}-{s} | {f(a['theta_stab_protocol1_pooling'], 1)} | {f(a['ablation_discount_only'], 1)} "
				 f"| {f(a['ablation_equal_energy_only'], 1)} | {f(a['theta_hat_stab'], 1)} "
				 f"| {'k* = ' + str(ks) if ks else 'none'} (largest drop {d[big]:.2f} dec at k = {big}) |")
	L += ["", "The gap column applies protocol-1 §7 conditions 1–4 at H = 64 only. This construction has no H = 32/128 "
		  "robustness spectra, so the horizon-reappearance condition cannot be applied.", "",
		  "### E-c. Growth rates", "",
		  "| model | trained γ_m | R-full γ_m | trained median ‖Φ(τ+64, τ)‖₂ | R-full median ‖Φ(τ+64, τ)‖₂ |", "|---|---|---|---|---|"]
	for task, s, _, m, r in rows:
		L.append(f"| {task}-{s} | {f(m['gamma'], 3)} | {f(r['gamma'], 3)} | {np.median(m['per_anchor']['g_f']):.2g} "
				 f"| {np.median(r['per_anchor']['g_f']):.2g} |")
	L += ["", "Discovery set: **not run.** `results2/discovery/` does not exist (DEVIATIONS P2).", ""]
	return L


def figure(rows, out):
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	names = [f"{t}-{s}" for t, s, *_ in rows]
	x = np.arange(len(rows))
	stats = [("H2.1: κ = min(κ_c, κ_o)", lambda m: m["kappa"], [(C1["min_energy_share"], "≥ 0.5")]),
			 ("H2.2: θ̂_stab (°)", lambda m: m["routes"]["factor"]["theta_hat_stab"], [(C2["max_theta_stab_deg"], "≤ 30°")]),
			 ("H2.3: Δ = θ_far − θ_near (°)", lambda m: m["routes"]["factor"]["contrast"], [(C3["min_contrast_deg"], "≥ 15°")])]
	fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
	for ax, (title, g, lines) in zip(axes, stats):
		ax.bar(x - 0.2, [g(m) for *_, m, _ in rows], 0.4, label="trained", color="#2a6fb0")
		ax.bar(x + 0.2, [g(r) for *_, _, r in rows], 0.4, label="paired R-full", color="#b0b0b0")
		for y, lab in lines:
			ax.axhline(y, color="#c0392b", lw=1, ls="--", label=f"threshold {lab}")
		lo, hi = ax.get_ylim()
		ax.set_ylim(lo, hi + 0.45 * (hi - lo))       # headroom so the legend covers no bar
		ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.9)
		ax.set_title(title, fontsize=10, loc="left")
		ax.axhline(0, color="black", lw=0.5)
	axes[-1].set_xticks(x, names, rotation=60, ha="right", fontsize=8)
	fig.suptitle("Protocol 2, held-out primary statistics (factor route). No model meets any hypothesis's full criterion.",
				 fontsize=10)
	fig.tight_layout()
	out.parent.mkdir(exist_ok=True)
	fig.savefig(out)


def main(results_dir=ROOT / "results2"):
	res = Path(results_dir)
	rows = load(res)
	verdict = json.loads((res / "verdict.json").read_text(encoding="utf-8"))
	L = ["# Protocol 2 result tables (generated by `python -m scripts.h2_report`; do not edit)", ""]
	L += receipt(rows, res) + confirmatory(rows, verdict) + exploratory(rows, verdict)
	# Written outside results2/, so the run's outputs stay byte-identical to the hashes sealed in DEVIATIONS P1.
	(ROOT / "RESULTS2-tables.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
	figure(rows, ROOT / "figures" / "fig6_h2_primary.pdf")
	print("wrote RESULTS2-tables.md and figures/fig6_h2_primary.pdf")


if __name__ == "__main__":
	main(*sys.argv[1:])
