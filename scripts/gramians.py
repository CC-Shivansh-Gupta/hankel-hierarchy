"""Tangent projection, finite-horizon LTV Gramians, square-root HSVs and balancing.

Implements PREREGISTRATION.md §4 (tangent coordinates) and §6 (Gramians, HSVs).
Everything is float64 numpy. Nothing here touches a checkpoint.

Indexing convention: A[t], B[t], C[t] are the Jacobians at agent step t of one trajectory,
A[t] = df/dz and B[t] = df/da at (z_t, a_t), and C[t] = dQbar/dz at (z_t, a_t). Shapes are
(T, n, n), (T, n, m) and (T, p, n), with p = 1 for the primary output map.
"""
import numpy as np


# ---------------------------------------------------------------------------
# §4 tangent coordinates
# ---------------------------------------------------------------------------

def tangent_basis(latent_dim=512, group=8):
	"""Orthonormal U (latent_dim x (latent_dim - latent_dim/group)) spanning the complement of
	the per-group sum directions. SimNorm keeps each group on a simplex, so those directions
	are structurally degenerate.

	Built block-diagonally: within one group the complement of 1/sqrt(g) is spanned by the
	last g-1 left singular vectors of the centring matrix I - 11^T/g.
	"""
	assert latent_dim % group == 0
	n_groups = latent_dim // group
	centre = np.eye(group) - np.full((group, group), 1.0 / group)
	u, s, _ = np.linalg.svd(centre)
	block = u[:, :group - 1]             # s = [1,...,1,0]; drop the null direction
	assert np.allclose(s[:group - 1], 1.0) and abs(s[-1]) < 1e-12
	U = np.zeros((latent_dim, n_groups * (group - 1)))
	for g in range(n_groups):
		U[g*group:(g+1)*group, g*(group-1):(g+1)*(group-1)] = block
	return U


def project(A, B, C, U):
	"""Ã = UᵀAU, B̃ = UᵀB, C̃ = CU. Works on single matrices or stacks along axis 0."""
	Ut = U.T
	return Ut @ A @ U, Ut @ B, C @ U


# ---------------------------------------------------------------------------
# §6 finite-horizon Gramians
# ---------------------------------------------------------------------------

def controllability_gramian(A, B, tau, H):
	"""W_c(τ) = Σ_{s=τ-H}^{τ-1} Φ(τ,s+1) B_s B_sᵀ Φ(τ,s+1)ᵀ, with Φ(t,s) = A_{t-1}⋯A_s.

	Forward recursion P_{s+1} = A_s P_s A_sᵀ + B_s B_sᵀ from P_{τ-H} = 0 gives P_τ = W_c(τ).
	The update for s = τ-H multiplies A_{τ-H} into P = 0, so A_{τ-H} is never used.
	"""
	assert tau - H >= 0, "anchor too close to the episode start"
	n = A.shape[1]
	P = np.zeros((n, n))
	for s in range(tau - H, tau):
		P = A[s] @ P @ A[s].T + B[s] @ B[s].T
	return P


def observability_gramian(A, C, tau, H):
	"""W_o(τ) = Σ_{s=τ}^{τ+H-1} Φ(s,τ)ᵀ C_sᵀ C_s Φ(s,τ).

	Backward recursion Q_s = C_sᵀ C_s + A_sᵀ Q_{s+1} A_s from Q_{τ+H} = 0 gives Q_τ = W_o(τ).
	"""
	assert tau + H - 1 < A.shape[0], "anchor too close to the episode end"
	n = A.shape[1]
	Q = np.zeros((n, n))
	for s in range(tau + H - 1, tau - 1, -1):
		Q = C[s].T @ C[s] + A[s].T @ Q @ A[s]
	return Q


def anchor_gramians(A, B, C, anchors, H):
	"""Per-anchor (W_c, W_o) lists for one trajectory."""
	return ([controllability_gramian(A, B, t, H) for t in anchors],
			[observability_gramian(A, C, t, H) for t in anchors])


def pool(gramians):
	"""§6 pooling: the mean of per-anchor Gramians (an energy average)."""
	return np.mean(np.stack(gramians), axis=0)


def anchor_set(T, margin=128, stride=10):
	"""τ ∈ {margin, margin+stride, …} with τ ≤ T - margin."""
	return list(range(margin, T - margin + 1, stride))


# ---------------------------------------------------------------------------
# Square-root factors without forming the Gramians (DEVIATIONS.md D1)
#
# With a scalar output W_o can have condition number ~1e18. Forming it and then
# eigendecomposing it, as §6 specifies, loses relative accuracy in HSVs near the 1e-6 floor.
# These routines build L with W = L Lᵀ directly from the Jacobians. The W and the HSVs are
# the same mathematically, but the condition number is never squared.
# ---------------------------------------------------------------------------

def controllability_factor(A, B, tau, H):
	"""R (n x H·m) with W_c(τ) = R Rᵀ. Its columns are Φ(τ,s+1) B_s for s = τ-H … τ-1."""
	assert tau - H >= 0
	n = A.shape[1]
	M = np.eye(n)                        # Φ(τ, s+1), starting at s = τ-1
	blocks = []
	for s in range(tau - 1, tau - H - 1, -1):
		blocks.append(M @ B[s])
		if s > tau - H:
			M = M @ A[s]
	return np.hstack(blocks[::-1])


def observability_factor(A, C, tau, H):
	"""O (H·p x n) with W_o(τ) = Oᵀ O. Its rows are C_s Φ(s,τ) for s = τ … τ+H-1."""
	assert tau + H - 1 < A.shape[0]
	n = A.shape[1]
	M = np.eye(n)                        # Φ(s, τ), starting at s = τ
	rows = []
	for s in range(tau, tau + H):
		rows.append(C[s] @ M)
		M = A[s] @ M
	return np.vstack(rows)


class PooledFactor:
	"""Running square-root factor of the mean of per-anchor Gramians.

	The mean of W_τ = F_τ F_τᵀ over N anchors is Zᵀ Z / N, where Z stacks the F_τᵀ. Z is
	compressed by QR after every anchor, so memory stays n x n. factor() returns L = Rᵀ/√N,
	which satisfies mean_τ W_τ = L Lᵀ.
	"""

	def __init__(self, n):
		self.R = np.zeros((0, n))
		self.N = 0

	def add(self, F):
		self.R = np.linalg.qr(np.vstack([self.R, F.T]), mode="r")
		self.N += 1

	def factor(self):
		return self.R.T / np.sqrt(self.N)


def compress(L):
	"""n x r factor -> n x n factor with the same L Lᵀ, when r > n."""
	if L.shape[1] <= L.shape[0]:
		return L
	return np.linalg.qr(L.T, mode="r").T


def balance_factors(Lc, Lo):
	"""Square-root balancing from factors (W_c = Lc Lcᵀ, W_o = Lo Loᵀ). Same outputs as balance().

	A factor wider than it is tall is first compressed to n x n by QR of its transpose.
	L Lᵀ is unchanged, and QR does not square the condition number.
	"""
	Lc, Lo = compress(Lc), compress(Lo)
	Uh, hsv, Vht = np.linalg.svd(Lo.T @ Lc, full_matrices=False)
	with np.errstate(divide="ignore", invalid="ignore"):
		T = (Lc @ Vht.T) / np.sqrt(hsv)
	return hsv, T


# ---------------------------------------------------------------------------
# §6 square-root balancing (the method as preregistered)
# ---------------------------------------------------------------------------

def psd_factor(W):
	"""L with W = L Lᵀ, by symmetric eigendecomposition, round-off negatives clipped to zero."""
	W = 0.5 * (W + W.T)
	lam, V = np.linalg.eigh(W)
	return V * np.sqrt(np.clip(lam, 0.0, None))


def balance(Wc, Wo):
	"""Square-root method. Returns (hsv, T) where hsv is descending and the columns of
	T = L_c V_h Σ^{-1/2} are the balanced directions, so S_bal(k) = range(T[:, :k]).

	Columns whose HSV is exactly zero are returned as NaN, because they have no balanced
	direction. They always lie past the numerical floor, so no rule ever uses them.
	"""
	Lc, Lo = psd_factor(Wc), psd_factor(Wo)
	Uh, hsv, Vht = np.linalg.svd(Lo.T @ Lc)
	with np.errstate(divide="ignore", invalid="ignore"):
		T = (Lc @ Vht.T) / np.sqrt(hsv)
	return hsv, T


def floor_hsv(hsv, rel=1e-6):
	"""§6 numerical floor: HSVs below rel * σ₁ are set to exactly zero."""
	hsv = np.asarray(hsv, dtype=np.float64).copy()
	hsv[hsv < rel * hsv[0]] = 0.0
	return hsv


def balanced_subspace(Wc, Wo, k):
	"""Orthonormal basis of S_bal(k) = range(V_k)."""
	_, T = balance(Wc, Wo)
	Q, _ = np.linalg.qr(T[:, :k])
	return Q


# ---------------------------------------------------------------------------
# §8/§9 subspace comparison
# ---------------------------------------------------------------------------

def mean_principal_angle_deg(X, Y):
	"""Mean principal angle in degrees between range(X) and range(Y), which have equal dimension.

	The angles come from the singular values of Q_Xᵀ Q_Y. They are clipped to [0, 1] before
	arccos, which is accurate enough for thresholds at 15° and 30°.
	"""
	Qx, _ = np.linalg.qr(X)
	Qy, _ = np.linalg.qr(Y)
	s = np.linalg.svd(Qx.T @ Qy, compute_uv=False)
	return float(np.degrees(np.arccos(np.clip(s, 0.0, 1.0))).mean())
