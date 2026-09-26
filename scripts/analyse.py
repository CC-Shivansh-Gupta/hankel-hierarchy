"""Per-model accumulator: tangent Jacobians in, pooled HSVs and balanced bases out, by both routes.

Pure numpy, so it is validated on synthetic LTV systems (tests/test_analyse_synthetic.py) before
any checkpoint exists. Jacobians arrive through a callable, jac(t) -> (A_t, B_t, C_t), already in
tangent coordinates and float64. Only the window [τ - H_max, τ + H_max) around the current anchor
is held in memory.

Routes (DEVIATIONS.md D1):
  "gramian": §6 as written. Form W̄ = mean of per-anchor Gramians, then balance() by
             symmetric eigendecomposition. This is the verdict of record.
  "factor":  the per-anchor square-root factors pooled by QR, then balance_factors().

Every horizon is sliced from the H_max factors. The observability rows for H are the first H
rows, and the controllability columns for H are the last H·m columns.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, svds

from scripts.gramians import PooledFactor, balance, balance_factors

ROUTES = ("gramian", "factor")


class JacobianWindow:
	"""Caches jac(t) and evicts steps that no later anchor can need (anchors come in ascending order)."""

	def __init__(self, jac):
		self.jac, self.cache = jac, {}

	def get(self, t):
		if t not in self.cache:
			self.cache[t] = self.jac(t)
		return self.cache[t]

	def evict_before(self, t0):
		for t in [t for t in self.cache if t < t0]:
			del self.cache[t]


def ctrl_factor_window(win, tau, H):
	"""Columns Φ(τ,s+1)B_s for s = τ-H … τ-1, by propagating the B blocks forward:
	cost m·H²/2 matrix-vector products, not H full matrix products."""
	X = win.get(tau - H)[1]
	for j in range(tau - H + 1, tau):
		A_j, B_j, _ = win.get(j)
		X = np.hstack([A_j @ X, B_j])
	return X


def obs_factor_window(win, tau, H):
	"""Rows C_sΦ(s,τ) for s = τ … τ+H-1, built from the far end inwards: Y ← [C_{j+1}; Y] A_j."""
	Y = win.get(tau + H - 1)[2]
	for j in range(tau + H - 2, tau - 1, -1):
		A_j, _, C_j = win.get(j)
		Y = np.vstack([C_j, Y @ A_j])
	return Y


def transition_norm(win, tau, H):
	"""‖Φ(τ+H,τ)‖₂, from matrix-free products. Reported only (§6)."""
	n = win.get(tau)[0].shape[0]

	def mv(v):
		v = np.asarray(v).ravel()
		for j in range(tau, tau + H):
			v = win.get(j)[0] @ v
		return v

	def rmv(v):
		v = np.asarray(v).ravel()
		for j in range(tau + H - 1, tau - 1, -1):
			v = win.get(j)[0].T @ v
		return v

	op = LinearOperator((n, n), matvec=mv, rmatvec=rmv, dtype=np.float64)
	return float(svds(op, k=1, return_singular_vectors=False, random_state=0)[0])


class ModelAccumulator:
	"""Accumulates one model over all its episodes."""

	def __init__(self, n, horizons, primary_H, k_keep=32, anchor_bases=True, diagnostics=True):
		self.n, self.Hs, self.H0, self.Hmax = n, sorted(horizons), primary_H, max(horizons)
		self.k_keep, self.want_bases, self.want_diag = k_keep, anchor_bases, diagnostics
		self.W = {(r, H, g): np.zeros((n, n)) for r in ("gramian",) for H in self.Hs for g in "co"}
		self.F = {(H, g): PooledFactor(n) for H in self.Hs for g in "co"}
		self.N = 0
		self.bases = {r: [] for r in ROUTES}
		self.diag = {"phi_norm": {H: [] for H in self.Hs}}

	def add_episode(self, jac, anchors):
		win = JacobianWindow(jac)
		for tau in anchors:
			win.evict_before(tau - self.Hmax)
			R_max = ctrl_factor_window(win, tau, self.Hmax)
			O_max = obs_factor_window(win, tau, self.Hmax)
			m = R_max.shape[1] // self.Hmax
			p = O_max.shape[0] // self.Hmax
			for H in self.Hs:
				R, O = R_max[:, R_max.shape[1] - H * m:], O_max[:H * p]
				Wc, Wo = R @ R.T, O.T @ O
				self.W["gramian", H, "c"] += Wc
				self.W["gramian", H, "o"] += Wo
				self.F[H, "c"].add(R)
				self.F[H, "o"].add(O.T)
				if H == self.H0 and self.want_bases:
					self.bases["gramian"].append(balance(Wc, Wo)[1][:, :self.k_keep].astype(np.float32))
					self.bases["factor"].append(balance_factors(R, O.T)[1][:, :self.k_keep].astype(np.float32))
				if self.want_diag:
					self.diag["phi_norm"][H].append(transition_norm(win, tau, H))
			self.N += 1

	def finalize(self):
		"""Returns {route: {"hsv": {H: σ}, "basis": pooled balanced basis at H0 (n x k_keep),
		"Wc_diag_factor"/"Wc": what Gate B needs}}."""
		out = {}
		g = {}
		for H in self.Hs:
			Wc, Wo = self.W["gramian", H, "c"] / self.N, self.W["gramian", H, "o"] / self.N
			hsv, T = balance(Wc, Wo)
			g[H] = hsv
			if H == self.H0:
				out["gramian"] = {"basis": T[:, :self.k_keep], "Wc": Wc, "Wo": Wo}
		out["gramian"]["hsv"] = g
		f = {}
		for H in self.Hs:
			Lc, Lo = self.F[H, "c"].factor(), self.F[H, "o"].factor()
			hsv, T = balance_factors(Lc, Lo)
			f[H] = hsv[:self.n]
			if H == self.H0:
				out["factor"] = {"basis": T[:, :self.k_keep], "Lc": Lc, "Lo": Lo}
		out["factor"]["hsv"] = f
		for r in ROUTES:
			out[r]["anchor_bases"] = np.stack(self.bases[r]) if self.bases[r] else None
		out["diagnostics"] = self.diag
		return out
