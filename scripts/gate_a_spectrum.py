"""§7 spectral-gap rule and the Gate A task verdict.

Every threshold is read from config/prereg.yaml (key `gap`). k is 1-indexed throughout, as in
PREREGISTRATION.md: σ_1 ≥ σ_2 ≥ …, and d_k = log10 σ_k − log10 σ_{k+1}.

Readings of ambiguous text are fixed in DEVIATIONS.md (C1–C5), and every one was chosen before any
checkpoint was loaded. In short: C1, a drop into the floor has d = +inf. C2, controls are judged at
the primary horizon only. C3, seed agreement is about the median of the gap seeds. C4, reappearance
means any valid gap within ±25%. C5, tolerances are inclusive.
"""
import numpy as np

from scripts.gramians import floor_hsv


def within(k, ref, tol):
	return abs(k - ref) <= tol * ref


def log_drops(hsv, floor_rel):
	"""d[k] for k = 1 … len-1, returned as a dict {k: d_k}. Floored HSVs count as zero (C1)."""
	s = floor_hsv(hsv, floor_rel)
	with np.errstate(divide="ignore"):
		logs = np.log10(s)
	d = {}
	for k in range(1, len(s)):
		a, b = logs[k - 1], logs[k]
		if np.isneginf(b):
			d[k] = np.inf        # drop into the floor, or floored to floored (C1)
		else:
			d[k] = float(a - b)
	return d


def valid_gaps(hsv, H, gap_cfg, floor_rel):
	"""All k satisfying §7 conditions 1–4 at horizon H, as {k: d_k}."""
	s = floor_hsv(hsv, floor_rel)
	d = log_drops(hsv, floor_rel)
	k_hi = int(H * gap_cfg["k_max_fraction_of_H"])
	out = {}
	for k in range(gap_cfg["k_min"], k_hi + 1):
		if k + 1 > len(s):
			break
		if s[k] == 0.0:                                     # cond 2: σ_{k+1} above the floor
			continue
		if d[k] < gap_cfg["min_log10_drop"]:                # cond 3
			continue
		w = gap_cfg["local_window"]
		nbrs = [d[j] for j in range(max(1, k - w), min(k_hi, k + w) + 1) if j != k and j in d]
		if not nbrs:
			continue
		if d[k] < gap_cfg["min_ratio_to_local_median_drop"] * float(np.median(nbrs)):   # cond 4
			continue
		out[k] = d[k]
	return out


def k_star(hsv, H, gap_cfg, floor_rel):
	"""The qualifying k with the largest d_k, or None."""
	g = valid_gaps(hsv, H, gap_cfg, floor_rel)
	return max(g, key=g.get) if g else None


def model_gap(hsv_by_H, cfg):
	"""§7 'a model has a gap'. hsv_by_H maps horizon -> pooled HSVs. Returns k* or None."""
	gcfg, floor_rel = cfg["gap"], cfg["gramians"]["numerical_floor_rel"]
	H0 = cfg["gramians"]["primary_horizon"]
	tol = gcfg["horizon_reappearance_tolerance"]
	ks = k_star(hsv_by_H[H0], H0, gcfg, floor_rel)
	if ks is None:
		return None
	for H in cfg["gramians"]["robustness_horizons"]:
		if H < H0 and ks > int(H * gcfg["k_max_fraction_of_H"]):
			continue                # H = 32 is required only when k* ≤ 16
		if not any(within(k, ks, tol) for k in valid_gaps(hsv_by_H[H], H, gcfg, floor_rel)):
			return None
	return ks


def control_shows_gap(hsv_primary, ref_k, cfg):
	"""A random-init control reproduces the trained gap if it has a valid gap within ±tol of
	the trained median, at the primary horizon (C2)."""
	gcfg, floor_rel = cfg["gap"], cfg["gramians"]["numerical_floor_rel"]
	H0 = cfg["gramians"]["primary_horizon"]
	tol = gcfg["seed_agreement"]["k_tolerance"]
	return any(within(k, ref_k, tol) for k in valid_gaps(hsv_primary, H0, gcfg, floor_rel))


def task_gate_a(seed_kstars, control_hsv_by_family, cfg):
	"""§7 task verdict.

	seed_kstars: list of k* (or None) for the 3 trained seeds, from model_gap.
	control_hsv_by_family: {"r_full": [hsv, hsv, hsv], "r_dyn": [...]}, primary-horizon HSVs.
	Returns (passed, task_median_k, detail).

	Seed agreement: take the seeds that have a gap and their median m. The agreeing seeds are
	those with k* within ±25% of m. At least 2 must agree. The task's k is the median over the
	agreeing seeds.
	"""
	gcfg = cfg["gap"]
	sa = gcfg["seed_agreement"]
	have = [k for k in seed_kstars if k is not None]
	detail = {"seed_kstars": seed_kstars}
	if len(have) < sa["min_seeds"]:
		return False, None, {**detail, "reason": "too few seeds with a gap"}
	m = float(np.median(have))
	agree = [k for k in have if within(k, m, sa["k_tolerance"])]
	if len(agree) < sa["min_seeds"]:
		return False, None, {**detail, "reason": "seeds disagree on k*"}
	k_task = float(np.median(agree))
	limit = gcfg["random_init_controls"]["max_reproducing_per_family"]
	counts = {fam: sum(control_shows_gap(h, k_task, cfg) for h in hsvs)
			  for fam, hsvs in control_hsv_by_family.items()}
	detail.update(task_median_k=k_task, control_reproductions=counts)
	if any(c > limit for c in counts.values()):
		return False, k_task, {**detail, "reason": "random-init control reproduces the gap"}
	return True, k_task, detail
