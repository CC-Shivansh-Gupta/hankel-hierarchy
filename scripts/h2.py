"""Protocol 2 statistics (prereg-2/PREREGISTRATION.md §5): growth dominance (H2.1), growth-normalised
pooled stability (H2.2), and trajectory-conditioned local structure (H2.3).

Pure numpy, and validated only on synthetic LTV systems (tests/test_h2_synthetic.py, cases S1–S5)
before any model is touched. Jacobians arrive through jac(t) -> (Ã_t, B̃_t, C̃_t) in tangent
coordinates, as in scripts.analyse. Both numerical routes are computed: "factor" decides and
"gramian" is the check.

Two passes over the same trajectories. γ_m (§5.2) is a median over every anchor of the model,
so it is known only after pass 1:
  pass 1: per-anchor factors at H = 64, the growth norms g_f and g_b, energies, per-anchor bases
          S(τ; 64, 16) and S(τ; 16, 4), and the energy-averaged and equal-energy pooled factors;
  pass 2: the same factors, discounted by γ_m. Φ̂ = Φ / γ^age, so every controllability block
          and every observability row is simply rescaled; no new Jacobian quantity is needed.
"""
import numpy as np
from scipy.stats import spearmanr

from scripts.analyse import JacobianWindow, ctrl_factor_window, obs_factor_window, transition_norm
from scripts.gramians import PooledFactor, balance, balance_factors, mean_principal_angle_deg

ROUTES = ("factor", "gramian")          # factor decides (prereg-2 §4)


def _bases(R, O, k):
	"""Per-anchor balanced bases (first k columns) by both routes."""
	return {"factor": balance_factors(R, O.T)[1][:, :k],
			"gramian": balance(R @ R.T, O.T @ O)[1][:, :k]}


class _Pool:
	"""Pooled Gramian of one construction, kept both as a QR factor and as a dense sum."""

	def __init__(self, n):
		self.f = {"c": PooledFactor(n), "o": PooledFactor(n)}
		self.W = {"c": np.zeros((n, n)), "o": np.zeros((n, n))}
		self.N = 0

	def add(self, R, O):
		self.f["c"].add(R)
		self.f["o"].add(O.T)
		self.W["c"] += R @ R.T
		self.W["o"] += O.T @ O
		self.N += 1

	def basis_and_hsv(self, k):
		hf, Tf = balance_factors(self.f["c"].factor(), self.f["o"].factor())
		hg, Tg = balance(self.W["c"] / self.N, self.W["o"] / self.N)
		return {"factor": (Tf[:, :k], hf), "gramian": (Tg[:, :k], hg)}


class H2Accumulator:
	"""All protocol-2 statistics for one model (trained or R-full control)."""

	def __init__(self, n, cfg):
		self.n = n
		self.H = cfg["h2_1_growth_dominance"]["horizon"]              # 64
		self.k = cfg["h2_1_growth_dominance"]["k"]                    # 16
		self.H3 = cfg["h2_3_trajectory_conditioned"]["horizon"]       # 16
		self.k3 = cfg["h2_3_trajectory_conditioned"]["k"]             # 4
		assert cfg["h2_2_growth_normalised"]["horizon"] == self.H and cfg["h2_2_growth_normalised"]["k"] == self.k
		self.cfg = cfg
		self.anchor_ids = []                                          # (episode, τ) in pass-1 order
		self.gf, self.gb, self.ec, self.eo = [], [], [], []
		self.S64 = {r: [] for r in ROUTES}
		self.S16 = {r: [] for r in ROUTES}
		self.pool_energy = _Pool(n)          # protocol-1 pooling: the mean of W(τ)
		self.pool_equal = _Pool(n)           # ablation: equal energy, no discount
		self.pool_h3 = _Pool(n)              # reported: pooled H = 16 stability (synthetic case S3)
		self.gamma = None
		self.pool_disc_equal = _Pool(n)      # H2.2 construction
		self.pool_disc_energy = _Pool(n)     # ablation: discount only
		self.Shat = {r: [] for r in ROUTES}
		self._pass2_ids = []

	# -- pass 1 ---------------------------------------------------------------------------
	def pass1_episode(self, jac, anchors, episode):
		win = JacobianWindow(jac)
		H = self.H
		for tau in anchors:
			win.evict_before(tau - H)
			R = ctrl_factor_window(win, tau, H)
			O = obs_factor_window(win, tau, H)
			m = R.shape[1] // H
			self.anchor_ids.append((episode, tau))
			self.gf.append(transition_norm(win, tau, H))
			self.gb.append(transition_norm(win, tau - H, H))
			self.ec.append(float(np.sum(R * R)))
			self.eo.append(float(np.sum(O * O)))
			for r, b in _bases(R, O, self.k).items():
				self.S64[r].append(b)
			R3, O3 = R[:, R.shape[1] - self.H3 * m:], O[:self.H3]
			for r, b in _bases(R3, O3, self.k3).items():
				self.S16[r].append(b)
			self.pool_h3.add(R3, O3)
			self.pool_energy.add(R, O)
			self.pool_equal.add(R / np.linalg.norm(R), O / np.linalg.norm(O))

	def set_gamma(self):
		"""§5.2: γ_m = exp(median_τ log ‖Φ(τ+H, τ)‖₂ / H)."""
		self.gamma = float(np.exp(np.median(np.log(self.gf)) / self.H))
		return self.gamma

	# -- pass 2 ---------------------------------------------------------------------------
	def pass2_episode(self, jac, anchors, episode):
		assert self.gamma is not None, "call set_gamma() after pass 1"
		win = JacobianWindow(jac)
		H, g = self.H, self.gamma
		for tau in anchors:
			win.evict_before(tau - H)
			R = ctrl_factor_window(win, tau, H)
			O = obs_factor_window(win, tau, H)
			m = R.shape[1] // H
			age_c = np.repeat(np.arange(H - 1, -1, -1), m)             # blocks s = τ-H … τ-1
			Rh = R * g ** (-age_c.astype(float))[None, :]
			Oh = O * g ** (-np.arange(H, dtype=float))[:, None]        # rows s = τ … τ+H-1
			self._pass2_ids.append((episode, tau))
			for r, b in _bases(Rh, Oh, self.k).items():
				self.Shat[r].append(b)
			self.pool_disc_equal.add(Rh / np.linalg.norm(Rh), Oh / np.linalg.norm(Oh))
			self.pool_disc_energy.add(Rh, Oh)

	# -- statistics -----------------------------------------------------------------------
	def finalize(self):
		assert self._pass2_ids == self.anchor_ids, "pass 2 must visit the same anchors in the same order"
		c1, c3 = self.cfg["h2_1_growth_dominance"], self.cfg["h2_3_trajectory_conditioned"]
		ec, eo, gf, gb = map(np.asarray, (self.ec, self.eo, self.gf, self.gb))
		N = len(ec)
		top = int(np.ceil(c1["top_fraction"] * N))

		def share(e):
			return float(np.sort(e)[::-1][:top].sum() / e.sum())

		kappa_c, kappa_o = share(ec), share(eo)
		rank = np.argsort(-(ec * eo))
		top_idx, bulk_idx = rank[:top], rank[N // 2:]
		ep = np.array([e for e, _ in self.anchor_ids])
		ts = np.array([t for _, t in self.anchor_ids])
		near_pairs, far_pairs = [], []
		for i in range(N):
			for j in range(i + 1, N):
				if ep[i] != ep[j]:
					continue
				d = abs(ts[j] - ts[i])
				if d == c3["near_offset"]:
					near_pairs.append((i, j))
				elif d >= c3["far_min_offset"]:
					far_pairs.append((i, j))

		P = self.pool_energy.basis_and_hsv(self.k)
		Eq = self.pool_equal.basis_and_hsv(self.k)
		Dq = self.pool_disc_equal.basis_and_hsv(self.k)
		De = self.pool_disc_energy.basis_and_hsv(self.k)
		P3 = self.pool_h3.basis_and_hsv(self.k3)
		out = {"N_anchors": N, "gamma": self.gamma,
			   "kappa_c": kappa_c, "kappa_o": kappa_o, "kappa": min(kappa_c, kappa_o),
			   "rho_c": float(spearmanr(ec, gb).statistic), "rho_o": float(spearmanr(eo, gf).statistic),
			   "per_anchor": {"episode": ep.tolist(), "tau": ts.tolist(), "e_c": ec.tolist(),
							  "e_o": eo.tolist(), "g_f": gf.tolist(), "g_b": gb.tolist()},
			   "n_near_pairs": len(near_pairs), "n_far_pairs": len(far_pairs), "routes": {}}
		for r in ROUTES:
			S64 = self.S64[r]
			th = np.array([mean_principal_angle_deg(S, P[r][0]) for S in S64])
			th_hat = np.array([mean_principal_angle_deg(S, Dq[r][0]) for S in self.Shat[r]])
			S16 = self.S16[r]
			near = [mean_principal_angle_deg(S16[i], S16[j]) for i, j in near_pairs]
			far = [mean_principal_angle_deg(S16[i], S16[j]) for i, j in far_pairs]
			out["routes"][r] = {
				"theta_top": float(th[top_idx].mean()), "theta_bulk": float(th[bulk_idx].mean()),
				"theta_stab_protocol1_pooling": float(th.mean()),
				"theta_hat_stab": float(th_hat.mean()),
				"ablation_equal_energy_only": float(np.mean([mean_principal_angle_deg(S, Eq[r][0]) for S in S64])),
				"ablation_discount_only": float(np.mean([mean_principal_angle_deg(S, De[r][0]) for S in self.Shat[r]])),
				"hsv_growth_normalised": Dq[r][1][:self.n].tolist(),
				"theta_near": float(np.mean(near)), "theta_far": float(np.mean(far)),
				"contrast": float(np.mean(far) - np.mean(near)),
				"theta_stab_h16_pooled": float(np.mean([mean_principal_angle_deg(S, P3[r][0]) for S in S16])),
				"per_anchor_theta": th.tolist(), "per_anchor_theta_hat": th_hat.tolist(),
			}
		return out


# ---------------------------------------------------------------------------
# §5–§6 criteria
# ---------------------------------------------------------------------------

def model_criteria(s, ctl, cfg, route="factor"):
	"""Per-model pass/fail for H2.1, H2.2 and H2.3, given trained stats `s` and paired R-full stats `ctl`."""
	c1, c2, c3 = cfg["h2_1_growth_dominance"], cfg["h2_2_growth_normalised"], cfg["h2_3_trajectory_conditioned"]
	a, b = s["routes"][route], ctl["routes"][route]
	h21 = (s["kappa_c"] >= c1["min_energy_share"] and s["kappa_o"] >= c1["min_energy_share"]
		   and s["rho_c"] >= c1["min_spearman_growth"] and s["rho_o"] >= c1["min_spearman_growth"]
		   and a["theta_bulk"] - a["theta_top"] >= c1["min_bulk_minus_top_deg"]
		   and s["kappa"] > ctl["kappa"])
	h22 = (a["theta_hat_stab"] <= c2["max_theta_stab_deg"]
		   and a["theta_hat_stab"] <= b["theta_hat_stab"] - c2["min_margin_vs_rfull_deg"])
	h23 = (a["theta_near"] <= c3["max_theta_near_deg"] and a["contrast"] >= c3["min_contrast_deg"]
		   and a["contrast"] >= b["contrast"] + c3["min_contrast_margin_vs_rfull_deg"])
	return {"H2.1": bool(h21), "H2.2": bool(h22), "H2.3": bool(h23)}


def decide(per_task, cfg, route="factor"):
	"""per_task[task] = list of (trained_stats, control_stats), one per analysed seed.
	A task supports a hypothesis if ≥ min_seeds_per_task seeds meet it. With 2 seeds that is 2-of-2,
	and a task left with 1 seed cannot support anything. A hypothesis is supported if ≥ min_tasks
	tasks support it."""
	need_seeds, need_tasks = cfg["decision"]["min_seeds_per_task"], cfg["decision"]["min_tasks"]
	out = {"tasks": {}, "hypotheses": {}}
	for task, pairs in per_task.items():
		crit = [model_criteria(s, c, cfg, route) for s, c in pairs]
		out["tasks"][task] = {h: sum(x[h] for x in crit) >= need_seeds for h in ("H2.1", "H2.2", "H2.3")}
		out["tasks"][task]["n_seeds"] = len(pairs)
		out["tasks"][task]["per_seed"] = crit
	for h in ("H2.1", "H2.2", "H2.3"):
		n = sum(v[h] for v in out["tasks"].values())
		out["hypotheses"][h] = {"tasks_supporting": n, "verdict": "SUPPORTED" if n >= need_tasks else "REJECTED"}
	return out
