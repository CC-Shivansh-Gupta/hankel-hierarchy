"""The five §12 figures, from results/ only. They use the verdict-of-record route (gramian, D1).

    python -m scripts.make_figures [results_dir]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import gate_a_spectrum as GA
from scripts import gate_b_pca as GB
from scripts.prereg import ROOT, load_cfg, read_npz
from scripts.verdict import RECORD

SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]     # categorical slots 1-3, fixed order
CONTROL_GREY = "#8a8a85"
INK, MUTED = "#1f1f1e", "#6b6b66"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
					 "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
					 "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e1",
					 "grid.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False})


def grid(cfg, title):
	fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), constrained_layout=True)
	fig.suptitle(title, color=INK)
	return fig, dict(zip(cfg["models"]["tasks"], axes.ravel()))


def fig1_spectra(res, cfg, out):
	H0 = cfg["gramians"]["primary_horizon"]
	kmax = int(H0 * cfg["gap"]["k_max_fraction_of_H"])
	fig, ax = grid(cfg, f"Pooled Hankel singular values, H = {H0} (σ/σ₁)")
	for task, a in ax.items():
		for i, s in enumerate(cfg["models"]["seeds"]):
			f = read_npz(res / "hsv" / f"{task}_s{s}_H{H0}.npz")
			h = f[f"{RECORD}_hsv"][:kmax + 8]
			a.semilogy(np.arange(1, len(h) + 1), h / h[0], color=SEED_COLORS[i], label=f"seed {s}")
			ks = int(f[f"{RECORD}_k_star"])
			if ks > 0:
				a.axvline(ks + 0.5, color=SEED_COLORS[i], lw=0.8, ls=(0, (2, 2)))
		for tag, ls, lab in (("rfull", "--", "R-full controls"), ("rdyn", ":", "R-dyn controls")):
			for j, ts in enumerate(cfg["gap"]["random_init_controls"]["torch_seeds"]):
				h = read_npz(res / "hsv" / f"{task}_{tag}{ts}_H{H0}.npz")[f"{RECORD}_hsv"][:kmax + 8]
				if h[0] > 0:
					a.semilogy(np.arange(1, len(h) + 1), h / h[0], color=CONTROL_GREY, ls=ls, lw=1.0,
							   label=lab if j == 0 else None)
		a.axhline(cfg["gramians"]["numerical_floor_rel"], color=MUTED, lw=0.8)
		a.axvspan(kmax + 0.5, kmax + 8.5, color="#f2f2ee", zorder=0)
		a.set_title(task, color=INK)
		a.set_xlabel("k")
	ax[cfg["models"]["tasks"][0]].legend(fontsize=7)
	fig.savefig(out / "fig1_hsv_spectra.pdf")
	plt.close(fig)


def fig2_log_drops(res, cfg, out):
	H0 = cfg["gramians"]["primary_horizon"]
	g = cfg["gap"]
	kmax = int(H0 * g["k_max_fraction_of_H"])
	fig, ax = grid(cfg, f"Log-drops d_k at H = {H0}, with the §7 thresholds")
	for task, a in ax.items():
		for i, s in enumerate(cfg["models"]["seeds"]):
			hsv = read_npz(res / "hsv" / f"{task}_s{s}_H{H0}.npz")[f"{RECORD}_hsv"]
			d = GA.log_drops(hsv, cfg["gramians"]["numerical_floor_rel"])
			ks = np.arange(1, kmax + 1)
			dk = np.array([min(d[k], 12.0) for k in ks])
			thr = []
			for k in ks:
				nb = [d[j] for j in range(max(1, k - g["local_window"]), min(kmax, k + g["local_window"]) + 1)
					  if j != k]
				thr.append(max(g["min_log10_drop"], g["min_ratio_to_local_median_drop"] * np.median(nb)))
			a.plot(ks, dk, color=SEED_COLORS[i], marker="o", ms=3, label=f"seed {s} d_k")
			a.plot(ks, np.minimum(thr, 12.0), color=SEED_COLORS[i], lw=0.9, ls="--",
				   label=f"seed {s} threshold" if task == cfg["models"]["tasks"][0] else None)
		a.axhline(g["min_log10_drop"], color=MUTED, lw=0.8)
		a.set_title(task, color=INK)
		a.set_xlabel("k")
		a.set_ylabel("decades")
	ax[cfg["models"]["tasks"][0]].legend(fontsize=6, ncol=2)
	fig.savefig(out / "fig2_log_drops.pdf")
	plt.close(fig)


def fig3_spearman(res, cfg, out):
	fig, ax = grid(cfg, "Hankel importance h_j against latent variance v_j (512 coordinates)")
	for task, a in ax.items():
		rhos = []
		for i, s in enumerate(cfg["models"]["seeds"]):
			f = read_npz(res / "work" / f"{task}_s{s}.npz")
			h = GB.hankel_importance(f[f"{RECORD}_diag_c"], f[f"{RECORD}_diag_o"])
			v = f["Z"].var(axis=0)
			a.loglog(v + 1e-300, h + 1e-300, "o", ms=2.5, alpha=0.5, color=SEED_COLORS[i], label=f"seed {s}")
			rhos.append(GB.spearman(h, v))
		a.set_title(f"{task}   ρ_S = " + ", ".join(f"{r:.2f}" for r in rhos), color=INK)
		a.set_xlabel("v_j")
		a.set_ylabel("h_j")
	ax[cfg["models"]["tasks"][0]].legend(fontsize=7)
	fig.savefig(out / "fig3_spearman.pdf")
	plt.close(fig)


def fig4_angles(res, cfg, out, verdict):
	tasks = cfg["models"]["tasks"]
	fig, a = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
	w = 0.25
	for i, s in enumerate(cfg["models"]["seeds"]):
		th = [json.loads((res / "pca" / f"{t}_s{s}.json").read_text())[RECORD]["theta_pca_deg"] for t in tasks]
		a.bar(np.arange(len(tasks)) + (i - 1) * w, th, w * 0.9, color=SEED_COLORS[i], label=f"seed {s}")
	for j, t in enumerate(tasks):
		k = verdict["tasks"][t]["B"]["k"]
		ref = GB.random_subspace_angle_reference(448, k)
		a.plot([j - 1.5 * w, j + 1.5 * w], [ref, ref], color=MUTED, lw=1.2,
			   label="random k-subspaces" if j == 0 else None)
	a.axhline(cfg["pca_control"]["principal_angle_collapse_deg"], color=INK, lw=1.0, ls="--",
			  label=f"collapse line ({cfg['pca_control']['principal_angle_collapse_deg']:.0f}°)")
	a.set_xticks(range(len(tasks)), [f"{t}\nk={verdict['tasks'][t]['B']['k']}" for t in tasks])
	a.set_ylabel("mean principal angle θ_pca (deg)")
	a.set_ylim(0, 92)
	a.set_title("Balanced vs PCA subspace", color=INK)
	a.legend(fontsize=7, ncol=3, loc="lower right")
	fig.savefig(out / "fig4_principal_angles.pdf")
	plt.close(fig)


def fig5_stability(res, cfg, out):
	tasks = cfg["models"]["tasks"]
	fig, a = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
	w = 0.25
	for i, s in enumerate(cfg["models"]["seeds"]):
		data = [json.loads((res / "stability" / f"{t}_s{s}.json").read_text())[RECORD]["anchor_angles_deg"]
				for t in tasks]
		bp = a.boxplot(data, positions=np.arange(len(tasks)) + (i - 1) * w, widths=w * 0.8,
					   patch_artist=True, showfliers=False, medianprops={"color": INK})
		for b in bp["boxes"]:
			b.set(facecolor=SEED_COLORS[i], alpha=0.75, edgecolor=SEED_COLORS[i])
		a.plot([], [], "s", color=SEED_COLORS[i], label=f"seed {s}")
	a.axhline(cfg["stability"]["max_mean_angle_deg"], color=INK, lw=1.0, ls="--",
			  label=f"stability line ({cfg['stability']['max_mean_angle_deg']:.0f}°, on the mean)")
	a.set_xticks(range(len(tasks)), tasks)
	a.set_ylabel("per-anchor angle to pooled S_bal (deg)")
	a.set_ylim(0, 92)
	a.set_title("Per-anchor stability of the balanced subspace, H = 64", color=INK)
	a.legend(fontsize=7, ncol=4)
	fig.savefig(out / "fig5_stability.pdf")
	plt.close(fig)


def main(results_dir=ROOT / "results", figures_dir=ROOT / "figures"):
	cfg = load_cfg()
	res, out = Path(results_dir), Path(figures_dir)
	out.mkdir(parents=True, exist_ok=True)
	verdict = json.loads((res / "verdict.json").read_text())
	fig1_spectra(res, cfg, out)
	fig2_log_drops(res, cfg, out)
	fig3_spearman(res, cfg, out)
	fig4_angles(res, cfg, out, verdict)
	fig5_stability(res, cfg, out)
	print(f"wrote 5 figures to {out}")


if __name__ == "__main__":
	main(*sys.argv[1:])
