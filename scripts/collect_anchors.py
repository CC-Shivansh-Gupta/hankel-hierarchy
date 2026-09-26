"""Rollouts, anchor states, per-step Jacobians and per-model accumulation (§4, §5, §7 controls).

The only script that touches TD-MPC2. It needs a CUDA GPU (the TD-MPC2 agent hard-codes cuda:0)
and the TD-MPC2 code at the pinned commit. For every task and seed it:

  1. checks the checkpoint's SHA-256 against config/prereg.yaml, and refuses a mismatch;
  2. runs 10 episodes, env seed e = 0…9 with torch seed e, and actions from
     agent.act(eval_mode=True) (DEVIATIONS.md C7);
  3. re-encodes every observation with a float64 copy of the model and takes
     A = ∂f/∂z, B = ∂f/∂a and C = ∂Q̄/∂z by autograd (C8), then projects to tangent coordinates;
  4. feeds scripts.analyse.ModelAccumulator, and writes results/work/{task}_s{seed}.npz;
  5. evaluates the R-full and R-dyn controls at the trained anchor states and actions.

    python -m scripts.collect_anchors --tdmpc2 PATH/tdmpc2/tdmpc2 --ckpt-dir PATH [--tasks ...]
    python -m scripts.collect_anchors --tdmpc2 ... --smoke   # engineering checks, no checkpoint

--smoke loads no checkpoint. It builds a random-init network at torch seed 999, which is not a
control seed, and checks the code paths: Jacobians against finite differences, float32 against
float64, rollout determinism, the C ≡ 0 trap, and array shapes. It computes no Gramian or
spectrum (DEVIATIONS.md C9).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from scripts.analyse import ModelAccumulator, ROUTES
from scripts.gate_b_pca import gramian_diag_original
from scripts.gramians import anchor_set, tangent_basis
from scripts.prereg import ROOT, load_cfg

PRE = load_cfg()
H0 = PRE["gramians"]["primary_horizon"]
HORIZONS = [H0] + PRE["gramians"]["robustness_horizons"]
K_KEEP = int(H0 * PRE["gap"]["k_max_fraction_of_H"])       # every k ≤ H0/2 is nested in these
U_NP = tangent_basis(PRE["models"]["architecture_expected"]["latent_dim"],
					 PRE["models"]["architecture_expected"]["simnorm_group"])


# ---------------------------------------------------------------------------
# TD-MPC2 plumbing
# ---------------------------------------------------------------------------

def import_tdmpc2(path):
	sys.path.insert(0, str(Path(path).resolve()))
	import hydra.utils
	hydra.utils.get_original_cwd = os.getcwd          # parse_cfg expects a hydra run
	from omegaconf import OmegaConf
	from common import math as tmath
	from common.parser import parse_cfg
	from common.seed import set_seed
	from common.world_model import WorldModel
	import envs.dmcontrol  # noqa: F401  (envs/__init__ swallows this import's errors; surface them here)
	from envs import make_env
	from tdmpc2 import TDMPC2
	return dict(OmegaConf=OmegaConf, parse_cfg=parse_cfg, set_seed=set_seed, make_env=make_env,
				TDMPC2=TDMPC2, WorldModel=WorldModel, tmath=tmath, root=Path(path).resolve())


def make_cfg(T, task, checkpoint, seed=1):
	cfg = T["OmegaConf"].load(T["root"] / "config.yaml")
	cfg.task, cfg.obs, cfg.model_size = task, "state", PRE["models"]["model_size"]
	cfg.checkpoint, cfg.seed, cfg.compile = str(checkpoint), seed, False
	cfg.enable_wandb, cfg.save_video = False, False
	return T["parse_cfg"](cfg)


def check_architecture(cfg, model):
	exp = PRE["models"]["architecture_expected"]
	assert cfg.latent_dim == exp["latent_dim"] and cfg.simnorm_dim == exp["simnorm_group"]
	assert cfg.num_q == exp["num_q"] and cfg.num_bins == exp["num_bins"]
	assert [cfg.vmin, cfg.vmax] == exp["value_bins_symlog_range"]
	assert cfg.episode_length == exp["episode_length_agent_steps"]
	assert repr(model._dynamics[-1].act) == f"SimNorm(dim={exp['simnorm_group']})"
	assert repr(model._encoder["state"][-1].act) == f"SimNorm(dim={exp['simnorm_group']})"


# DEVIATIONS.md D2: humanoid-walk-3.pt (uploaded 24 Oct 2023) predates the first public TD-MPC2 commit.
# It stores flat Sequentials (Linear, LayerNorm, Mish, ...) where the pinned code has NormedLinear blocks
# (Linear with .ln). The weights are only renamed, never altered. The old encoder's index 0 has no
# parameters and is taken to be the identity. A converted checkpoint is analysed only if its mean
# return reaches CONVERTED_MIN_RETURN_FRAC of its published final return (results/tdmpc2/*.csv at the
# pinned commit). Both rules were fixed before the conversion was ever run.
CONVERTED_MIN_RETURN_FRAC = 0.8
FLAT = {"enc": {1: "0.", 2: "0.ln.", 4: "1.", 5: "1.ln."},
		"mlp": {0: "0.", 1: "0.ln.", 3: "1.", 4: "1.ln.", 6: "2."}}


def convert_flat_layout(sd):
	"""Returns (state_dict, converted?). Only renames keys; every tensor is passed through unchanged."""
	if "_dynamics.0.0.weight" not in sd:
		return sd, False
	out = {}
	for key, val in sd.items():
		head, _, rest = key.partition(".")
		if key.startswith("_encoder.state."):
			idx, leaf = key[len("_encoder.state."):].split(".", 1)
			out["_encoder.state." + FLAT["enc"][int(idx)] + leaf] = val
		elif key.startswith("_dynamics.0."):
			idx, leaf = key[len("_dynamics.0."):].split(".", 1)
			out["_dynamics." + FLAT["mlp"][int(idx)] + leaf] = val
		elif key.startswith("_dynamics.1."):
			out["_dynamics.2.ln." + key[len("_dynamics.1."):]] = val
		elif head in ("_reward", "_pi"):
			idx, leaf = rest.split(".", 1)
			out[f"{head}." + FLAT["mlp"][int(idx)] + leaf] = val
		else:
			out[key] = val
	return out, True


def published_final_return(T, task, seed):
	import csv
	rows = [r for r in csv.DictReader(open(T["root"].parent / "results" / "tdmpc2" / f"{task}.csv"))
			if int(r["seed"]) == seed]
	return float(max(rows, key=lambda r: float(r["step"]))["reward"])


def sha256(path):
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


# ---------------------------------------------------------------------------
# rollouts (§4)
# ---------------------------------------------------------------------------

def rollout(T, cfg, agent, env_seed):
	"""One episode in the released evaluation mode. Returns obs (501 x d), actions (500 x m), rewards."""
	cfg.seed = env_seed
	env = T["make_env"](cfg)
	T["set_seed"](env_seed)                           # torch seed = env seed
	obs, done, t = env.reset(), False, 0
	O, A, R = [obs.numpy().copy()], [], []
	while not done:
		a = agent.act(obs, t0=t == 0, eval_mode=True)
		obs, r, done, _ = env.step(a)
		O.append(obs.numpy().copy())
		A.append(a.numpy().copy())
		R.append(float(r))
		t += 1
	return np.stack(O).astype(np.float64), np.stack(A).astype(np.float64), np.array(R)


# ---------------------------------------------------------------------------
# Jacobians (§5), float64
# ---------------------------------------------------------------------------

def to_float64(model):
	m = deepcopy(model).double().eval()
	for name, p in m.named_parameters():
		assert p.dtype == torch.float64, name
	return m


def make_jacobian_fn(T, cfg, dyn_model, q_model, device):
	"""Returns jac(z, a) -> (A, B, C) as float64 torch tensors, original coordinates.
	dyn_model supplies f. q_model supplies Q̄ (the online ensemble `_Qs`, in eval mode)."""
	two_hot_inv = T["tmath"].two_hot_inv

	def f(z, a):
		return dyn_model.next(z[None], a[None], None)[0]

	def qbar(z, a):
		out = q_model._Qs(torch.cat([z, a])[None])          # (num_q, 1, num_bins)
		return two_hot_inv(out, cfg).mean(0)[0, 0]           # mean_i symexp(E[Q_i])

	jf = torch.func.jacrev(f, argnums=(0, 1))

	def jac(z, a):
		Az, Ba = jf(z, a)
		zz = z.detach().clone().requires_grad_(True)
		(Cz,) = torch.autograd.grad(qbar(zz, a.detach()), zz)    # scalar output: one backward pass
		return Az.detach(), Ba.detach(), Cz[None]
	return jac


def episode_jacobians(jac, Z, Acts, U):
	"""Tangent Jacobians for every step of one episode, as numpy float64 arrays."""
	out_A, out_B, out_C = [], [], []
	for t in range(Acts.shape[0]):
		A, B, C = jac(Z[t], Acts[t])
		out_A.append((U.T @ A @ U).cpu().numpy())
		out_B.append((U.T @ B).cpu().numpy())
		out_C.append((C @ U).cpu().numpy())
	return np.stack(out_A), np.stack(out_B), np.stack(out_C)


def encode64(model64, obs, device):
	with torch.no_grad():
		return model64.encode(torch.as_tensor(obs, device=device), None)


# ---------------------------------------------------------------------------
# random-initialisation controls (§7)
# ---------------------------------------------------------------------------

def control_model(T, cfg, torch_seed, device):
	"""The WorldModel constructor at the pinned commit, under torch seed `torch_seed`. The
	zero-initialised final Q layer is then re-drawn like every other linear layer (R-full)."""
	T["set_seed"](torch_seed)
	m = T["WorldModel"](cfg).to(device).eval()
	with torch.no_grad():
		w = m._Qs.params["2", "weight"]
		assert torch.count_nonzero(w) == 0, "expected the zero-initialised final Q layer"
		torch.nn.init.trunc_normal_(w, std=0.02)
		m._Qs.params["2", "bias"].zero_()
	return m


# ---------------------------------------------------------------------------
# per-model driver
# ---------------------------------------------------------------------------

def accumulate(jac, trajs, U, anchors, anchor_bases, diagnostics):
	acc = ModelAccumulator(U.shape[1], HORIZONS, H0, k_keep=K_KEEP,
						   anchor_bases=anchor_bases, diagnostics=diagnostics)
	rho = []
	for Z, Acts in trajs:
		Aj, Bj, Cj = episode_jacobians(jac, Z, Acts, U)
		if diagnostics:
			rho.extend(np.max(np.abs(np.linalg.eigvals(Aj)), axis=1).tolist())
		acc.add_episode(lambda t: (Aj[t], Bj[t], Cj[t]), anchors)
		del Aj, Bj, Cj
	return acc.finalize(), rho


def save_model(path, res, extra):
	arrs = dict(extra)
	for r in ROUTES:
		for H in HORIZONS:
			arrs[f"{r}_hsv_H{H}"] = res[r]["hsv"][H]
		arrs[f"{r}_basis"] = res[r]["basis"]
		if res[r]["anchor_bases"] is not None:
			arrs[f"{r}_anchor_bases"] = res[r]["anchor_bases"]
	if "Wc" in res["gramian"]:
		arrs["gramian_diag_c"] = gramian_diag_original(U_NP, W=res["gramian"]["Wc"])
		arrs["gramian_diag_o"] = gramian_diag_original(U_NP, W=res["gramian"]["Wo"])
		arrs["factor_diag_c"] = gramian_diag_original(U_NP, L=res["factor"]["Lc"])
		arrs["factor_diag_o"] = gramian_diag_original(U_NP, L=res["factor"]["Lo"])
	np.savez(path, **arrs)


def run_task(T, task, ckpt_dir, out, device):
	work, diag_dir, traj_dir = out / "work", out / "diagnostics", out / "work" / "traj"
	for d in (work, diag_dir, traj_dir):
		d.mkdir(parents=True, exist_ok=True)
	U = torch.as_tensor(U_NP, device=device)
	env_seeds = PRE["rollouts"]["env_seeds"]
	for seed in PRE["models"]["seeds"]:
		name = f"{task}-{seed}.pt"
		target = work / f"{task}_s{seed}.npz"
		if target.exists():
			print(f"skip {target.name} (exists)")
			continue
		ckpt = Path(ckpt_dir) / name
		digest = sha256(ckpt)
		assert digest == PRE["models"]["checkpoint_sha256"][name], f"hash mismatch for {name}; not analysed"
		t0 = time.time()
		cfg = make_cfg(T, task, ckpt)
		T["make_env"](cfg)                            # fills obs_shape, action_dim, episode_length
		agent = T["TDMPC2"](cfg)
		raw = torch.load(str(ckpt), map_location=device, weights_only=False)
		sd, converted = convert_flat_layout(raw["model"] if "model" in raw else raw)
		agent.load(sd)
		check_architecture(cfg, agent.model)

		# 1. rollouts
		trajs_raw, returns = [], []
		for e in env_seeds:
			O, A, R = rollout(T, cfg, agent, e)
			trajs_raw.append((O, A))
			returns.append(float(R.sum()))
		if converted:
			pub = published_final_return(T, task, seed)
			ok = np.mean(returns) >= CONVERTED_MIN_RETURN_FRAC * pub
			rec = {"checkpoint": name, "converted_from_flat_layout": True, "episode_returns": returns,
				   "published_final_return": pub, "threshold_frac": CONVERTED_MIN_RETURN_FRAC, "accepted": bool(ok)}
			with open(diag_dir / f"{task}_s{seed}_conversion.json", "w") as fh:
				json.dump(rec, fh, indent=1)
			print(f"{task} s{seed}: converted checkpoint, mean return {np.mean(returns):.1f} vs published {pub:.1f}"
				  f" -> {'ACCEPTED' if ok else 'EXCLUDED'}", flush=True)
			if not ok:
				continue
		np.savez(traj_dir / f"{task}_s{seed}.npz", obs=np.stack([o for o, _ in trajs_raw]),
				 actions=np.stack([a for _, a in trajs_raw]), returns=np.array(returns))

		# 2. float64 linearisation at encoded real states
		m64 = to_float64(agent.model)
		trajs, Zanch, consist = [], [], []
		anchors = anchor_set(cfg.episode_length, PRE["anchors"]["margin"], PRE["anchors"]["stride"])
		for O, A in trajs_raw:
			Z = encode64(m64, O, device)
			Acts = torch.as_tensor(A, device=device)
			with torch.no_grad():
				consist.extend(torch.linalg.norm(m64.next(Z[:-1], Acts, None) - Z[1:], dim=1).cpu().tolist())
			trajs.append((Z[:-1], Acts))
			Zanch.append(Z[anchors].cpu().numpy())
		Zanch = np.concatenate(Zanch)

		jac = make_jacobian_fn(T, cfg, m64, m64, device)
		res, rho = accumulate(jac, trajs, U, anchors, anchor_bases=True, diagnostics=True)

		# 3. the paired controls (C10): torch seed 1000/1001/1002 with trained seed 1/2/3, at that
		#    seed's anchor states and actions
		ts = PRE["gap"]["random_init_controls"]["torch_seeds"][PRE["models"]["seeds"].index(seed)]
		ctl = to_float64(control_model(T, cfg, ts, device))
		for tag, qm in (("rfull", ctl), ("rdyn", m64)):
			res_c, _ = accumulate(make_jacobian_fn(T, cfg, ctl, qm, device), trajs, U, anchors,
								  anchor_bases=False, diagnostics=False)
			np.savez(work / f"{task}_{tag}{ts}.npz",
					 **{f"{r}_hsv_H{H}": res_c[r]["hsv"][H] for r in ROUTES for H in HORIZONS})

		# 4. the trained model's file last, so its existence means the seed is complete
		with open(diag_dir / f"{task}_s{seed}.json", "w") as fh:
			json.dump({"checkpoint_sha256": digest, "episode_returns": returns,
					   "spectral_radius": rho,
					   "phi_norm": {str(H): v for H, v in res["diagnostics"]["phi_norm"].items()},
					   "one_step_consistency_error": consist,
					   "seconds": time.time() - t0}, fh)
		save_model(target, res, {"Z": Zanch})
		print(f"{task} s{seed}: mean return {np.mean(returns):.1f}, {time.time() - t0:.0f}s", flush=True)
		del agent, m64, trajs
		torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# smoke test: engineering only, no checkpoint, no spectra
# ---------------------------------------------------------------------------

def smoke(T, device, task="walker-walk"):
	cfg = make_cfg(T, task, "none")
	T["make_env"](cfg)
	agent = T["TDMPC2"](cfg)
	T["set_seed"](999)
	agent.model = T["WorldModel"](cfg).to(device).eval()      # seed 999: not a control seed
	check_architecture(cfg, agent.model)
	with torch.no_grad():
		agent.model._Qs.params["2", "weight"].normal_(0, 0.02)  # so C is not identically zero here
	m64 = to_float64(agent.model)

	O1, A1, _ = rollout(T, cfg, agent, 0)
	O2, A2, _ = rollout(T, cfg, agent, 0)
	assert O1.shape == (501, cfg.obs_shape["state"][0]) and A1.shape == (500, cfg.action_dim)
	print("rollout deterministic under a fixed seed:", np.array_equal(A1, A2))

	Z = encode64(m64, O1, device)
	jac = make_jacobian_fn(T, cfg, m64, m64, device)
	rng = np.random.default_rng(0)
	for t in (0, 137, 499):
		z, a = Z[t], torch.as_tensor(A1[t], device=device)
		A, B, C = jac(z, a)
		assert A.shape == (512, 512) and B.shape == (512, cfg.action_dim) and C.shape == (1, 512)
		eps = 1e-6
		v = torch.as_tensor(rng.standard_normal(512), device=device)
		w = torch.as_tensor(rng.standard_normal(cfg.action_dim), device=device)
		fd_A = (m64.next((z + eps * v)[None], a[None], None) - m64.next((z - eps * v)[None], a[None], None))[0] / (2 * eps)
		fd_B = (m64.next(z[None], (a + eps * w)[None], None) - m64.next(z[None], (a - eps * w)[None], None))[0] / (2 * eps)
		q = lambda zz: T["tmath"].two_hot_inv(m64._Qs(torch.cat([zz, a])[None]), cfg).mean(0)[0, 0]
		fd_C = (q(z + eps * v) - q(z - eps * v)) / (2 * eps)
		rel = lambda x, y: float(torch.linalg.norm(x - y) / (torch.linalg.norm(y) + 1e-300))
		print(f"t={t}: FD rel err A {rel(A @ v, fd_A):.1e}  B {rel(B @ w, fd_B):.1e}  "
			  f"C {abs(float(C[0] @ v - fd_C)) / (abs(float(fd_C)) + 1e-300):.1e}")
		with torch.no_grad():
			z32, a32 = z.float()[None], a.float()[None]
			d = float(torch.linalg.norm(agent.model.next(z32, a32, None).double() - m64.next(z[None], a[None], None))
					  / torch.linalg.norm(m64.next(z[None], a[None], None)))
			q32 = T["tmath"].two_hot_inv(agent.model._Qs(torch.cat([z32, a32], 1)), cfg).mean(0)[0, 0]
			dq = abs(float(q32) - float(q(z))) / (abs(float(q(z))) + 1e-12)
		print(f"t={t}: float32 vs float64 rel diff: dynamics {d:.1e}, Q {dq:.1e}")

	fresh = T["WorldModel"](cfg).to(device).eval()
	Cz = make_jacobian_fn(T, cfg, to_float64(fresh), to_float64(fresh), device)(Z[10], torch.as_tensor(A1[10], device=device))[2]
	print("constructor Q head gives C == 0 exactly (the R-full trap):", bool(torch.count_nonzero(Cz) == 0))
	ctl = control_model(T, cfg, 999, device)
	print("re-drawn control has non-zero final Q layer:", bool(torch.count_nonzero(ctl._Qs.params["2", "weight"]) > 0))
	U = torch.as_tensor(U_NP, device=device)
	Aj, Bj, Cj = episode_jacobians(jac, Z[:3], torch.as_tensor(A1[:3], device=device), U)
	print("tangent shapes:", Aj.shape, Bj.shape, Cj.shape)
	print("SMOKE DONE (no Gramian or spectrum was computed)")


def write_environment(out):
	import platform
	import subprocess
	lines = [f"python {platform.python_version()}", f"platform {platform.platform()}",
			 f"torch {torch.__version__}  cuda {torch.version.cuda}",
			 f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
			 f"numpy {np.__version__}"]
	for pkg in ("scipy", "mujoco", "dm_control", "tensordict", "gymnasium", "omegaconf", "hydra"):
		try:
			mod = __import__(pkg)
			lines.append(f"{pkg} {getattr(mod, '__version__', '?')}")
		except Exception as e:
			lines.append(f"{pkg} unavailable ({e})")
	try:
		lines.append("cpu " + subprocess.run(["lscpu"], capture_output=True, text=True).stdout.split("Model name:")[1].split("\n")[0].strip())
	except Exception:
		lines.append(f"cpu {platform.processor()}")
	(out / "ENVIRONMENT.txt").write_text("\n".join(lines) + "\n")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tdmpc2", required=True, help="path to the tdmpc2/tdmpc2 package directory")
	ap.add_argument("--ckpt-dir")
	ap.add_argument("--out", default=str(ROOT / "results"))
	ap.add_argument("--tasks", nargs="*", default=PRE["models"]["tasks"])
	ap.add_argument("--smoke", action="store_true")
	args = ap.parse_args()
	assert torch.cuda.is_available(), "TD-MPC2 hard-codes cuda:0"
	device = torch.device("cuda:0")
	T = import_tdmpc2(args.tdmpc2)
	if args.smoke:
		return smoke(T, device)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	write_environment(out)
	for task in args.tasks:
		run_task(T, task, args.ckpt_dir, out, device)


if __name__ == "__main__":
	main()
