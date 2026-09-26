"""§8 PCA and subspace controls (Gate B).

All subspaces are in the 448-dimensional tangent coordinates of §4. Anchor states are given in
the original 512 coordinates and projected here.
"""
import numpy as np
from scipy.stats import spearmanr

from scripts.gramians import mean_principal_angle_deg


def gramian_diag_original(U, W=None, L=None):
	"""diag(U W Uᵀ), the Gramian's diagonal in the original 512 coordinates. Takes the dense
	W (§6 route) or a factor L with W = L Lᵀ (D1 route)."""
	if L is not None:
		return np.sum((U @ L) ** 2, axis=1)
	return np.einsum("ij,jk,ik->i", U, W, U)


def hankel_importance(diag_c, diag_o):
	"""h_j = √((U W̄_c Uᵀ)_jj · (U W̄_o Uᵀ)_jj)."""
	return np.sqrt(np.clip(diag_c, 0, None) * np.clip(diag_o, 0, None))


def spearman(h, v):
	return float(spearmanr(h, v).statistic)


def pca_basis(Z_tangent, k):
	"""Top-k eigenvectors of the anchor-state covariance. Z_tangent has shape (N, n)."""
	cov = np.cov(Z_tangent, rowvar=False)
	lam, V = np.linalg.eigh(cov)
	return V[:, np.argsort(lam)[::-1][:k]]


def model_gate_b(h, v, S_bal, S_pca, cfg):
	"""A model collapses to PCA if ρ_S > 0.9 OR θ_pca < 15°."""
	pc = cfg["pca_control"]
	rho = spearman(h, v)
	theta = mean_principal_angle_deg(S_bal, S_pca)
	collapse = rho > pc["spearman_collapse_threshold"] or theta < pc["principal_angle_collapse_deg"]
	return {"rho_s": rho, "theta_pca_deg": theta, "collapse": bool(collapse)}


def task_divergence(seed_results, cfg):
	"""A task shows divergence if at least 2 of its 3 seeds do NOT collapse."""
	need = cfg["decision"]["min_seeds_per_task"]
	return sum(not r["collapse"] for r in seed_results) >= need


def random_subspace_angle_reference(n, k, samples=200, seed=0):
	"""Mean principal angle between two random k-dim subspaces of ℝⁿ. Reported only."""
	rng = np.random.default_rng(seed)
	return float(np.mean([mean_principal_angle_deg(rng.standard_normal((n, k)),
												   rng.standard_normal((n, k)))
						  for _ in range(samples)]))


def task_k(gate_a_passed, gate_a_k, cfg):
	"""§8 k per task: the Gate A median k* if the task passed Gate A, else 16. A half-integer
	median is rounded half-up (DEVIATIONS.md C6)."""
	if gate_a_passed:
		return int(np.floor(gate_a_k + 0.5))
	return int(cfg["pca_control"]["k_if_task_failed_gate_a"])
