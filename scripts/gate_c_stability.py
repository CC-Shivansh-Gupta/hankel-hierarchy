"""§9 trajectory and subspace stability test (Gate C)."""
import numpy as np

from scripts.gramians import mean_principal_angle_deg


def theta_stab(anchor_bases, pooled_basis, k):
	"""Mean over anchors of the mean principal angle between S_bal^(τ)(k) and the pooled
	S_bal(k). Bases are balanced-direction matrices whose leading k columns span the subspace."""
	angles = [mean_principal_angle_deg(Tb[:, :k], pooled_basis[:, :k]) for Tb in anchor_bases]
	return float(np.mean(angles)), angles


def model_gate_c(anchor_bases, pooled_basis, k, cfg):
	th, angles = theta_stab(anchor_bases, pooled_basis, k)
	return {"theta_stab_deg": th, "anchor_angles_deg": angles,
			"stable": bool(th <= cfg["stability"]["max_mean_angle_deg"])}


def task_stable(seed_results, cfg):
	"""A task is stable if at least 2 of its 3 seeds are stable."""
	return sum(r["stable"] for r in seed_results) >= cfg["decision"]["min_seeds_per_task"]
