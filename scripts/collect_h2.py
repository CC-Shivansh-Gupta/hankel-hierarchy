"""Protocol 2 driver (prereg-2/PREREGISTRATION.md). It needs a CUDA GPU and TD-MPC2 at the pinned commit.

    python -m scripts.collect_h2 --tdmpc2 PATH/tdmpc2/tdmpc2 --ckpt-dir DIR --set heldout   [--tasks ...]
    python -m scripts.collect_h2 --tdmpc2 PATH/tdmpc2/tdmpc2 --ckpt-dir DIR --set discovery [--tasks ...]
    python -m scripts.collect_h2 --tdmpc2 PATH/tdmpc2/tdmpc2 --smoke

For each model (held-out: prereg-2 seeds_by_task; discovery: the protocol-1 set) it:
  1. checks the SHA-256 against the right config. A checkpoint in the pre-release layout is recorded as a
     load failure and excluded; it is never converted (prereg-2 §3);
  2. runs the protocol-1 rollouts (env seeds 0–9, eval_mode=True) and float64 linearisation (inherited, §4);
  3. makes the two H2Accumulator passes over the tangent Jacobians, recomputing them in pass 2 to bound memory;
  4. does the same for the paired R-full control at the same states and actions;
  5. writes results2/work/{task}_s{seed}.json and {task}_rfull{ts}.json, the trained file last.
--smoke builds a random-init network (torch seed 999, not a control seed), runs one episode through both
passes, and checks that every statistic is present and finite. It prints no statistic, and no trained model
is involved (prereg-2 §9).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts import collect_anchors as CA
from scripts.gramians import anchor_set
from scripts.h2 import H2Accumulator
from scripts.prereg import ROOT, load_cfg

P1 = load_cfg()
P2 = load_cfg(ROOT / "prereg-2" / "config" / "prereg.yaml")


def model_list(which):
	"""[(task, seed, filename, sha256)] for the held-out or the discovery set."""
	if which == "heldout":
		hashes = P2["models"]["checkpoint_sha256"]
		return [(t, s, f"{t}-{s}.pt", hashes[f"{t}-{s}.pt"])
				for t, seeds in P2["models"]["seeds_by_task"].items() for s in seeds]
	hashes = P1["models"]["checkpoint_sha256"]
	return [(t, s, f"{t}-{s}.pt", hashes[f"{t}-{s}.pt"])
			for t in P1["models"]["tasks"] for s in P1["models"]["seeds"]]


def two_passes(jac_fn, trajs, U, anchors, n):
	"""Both H2Accumulator passes. Jacobians are recomputed per episode in each pass."""
	acc = H2Accumulator(n, P2)
	for p in (1, 2):
		if p == 2:
			acc.set_gamma()
		for e, (Z, Acts) in enumerate(trajs):
			Aj, Bj, Cj = CA.episode_jacobians(jac_fn, Z, Acts, U)
			f = (lambda t: (Aj[t], Bj[t], Cj[t]))
			(acc.pass1_episode if p == 1 else acc.pass2_episode)(f, anchors, e)
			del Aj, Bj, Cj
	return acc.finalize()


def run_model(T, task, seed, name, want, ckpt_dir, work, diag, device):
	target = work / f"{task}_s{seed}.json"
	if target.exists():
		print(f"skip {target.name} (exists)", flush=True)
		return
	ckpt = Path(ckpt_dir) / name
	digest = CA.sha256(ckpt)
	assert digest == want, f"hash mismatch for {name}; not analysed"
	raw = torch.load(str(ckpt), map_location=device, weights_only=False)
	sd = raw["model"] if "model" in raw else raw
	if "_dynamics.0.0.weight" in sd:                 # pre-release layout: exclusion, never conversion
		(diag / f"{task}_s{seed}_load_failure.json").write_text(json.dumps(
			{"checkpoint": name, "reason": "pre-release state-dict layout", "excluded": True}, indent=1))
		print(f"{task} s{seed}: pre-release layout -> EXCLUDED", flush=True)
		return
	t0 = time.time()
	cfg = CA.make_cfg(T, task, ckpt)
	T["make_env"](cfg)
	agent = T["TDMPC2"](cfg)
	agent.load(sd)
	CA.check_architecture(cfg, agent.model)

	trajs_raw, returns = [], []
	for e in P1["rollouts"]["env_seeds"]:
		O, A, R = CA.rollout(T, cfg, agent, e)
		trajs_raw.append((O, A))
		returns.append(float(R.sum()))
	m64 = CA.to_float64(agent.model)
	trajs = [(CA.encode64(m64, O, device)[:-1], torch.as_tensor(A, device=device)) for O, A in trajs_raw]
	anchors = anchor_set(cfg.episode_length, P1["anchors"]["margin"], P1["anchors"]["stride"])
	U = torch.as_tensor(CA.U_NP, device=device)
	n = CA.U_NP.shape[1]

	s = two_passes(CA.make_jacobian_fn(T, cfg, m64, m64, device), trajs, U, anchors, n)
	ts = P1["gap"]["random_init_controls"]["torch_seeds"][P1["models"]["seeds"].index(seed)]
	ctl = CA.to_float64(CA.control_model(T, cfg, ts, device))
	c = two_passes(CA.make_jacobian_fn(T, cfg, ctl, ctl, device), trajs, U, anchors, n)
	(work / f"{task}_rfull{ts}.json").write_text(json.dumps(c))
	(diag / f"{task}_s{seed}.json").write_text(json.dumps(
		{"checkpoint_sha256": digest, "episode_returns": returns, "seconds": time.time() - t0}))
	target.write_text(json.dumps(s))                 # last: its existence means the seed is complete
	print(f"{task} s{seed}: mean return {np.mean(returns):.1f}, {time.time() - t0:.0f}s", flush=True)
	del agent, m64, ctl, trajs
	torch.cuda.empty_cache()


def smoke(T, device, task="walker-walk"):
	cfg = CA.make_cfg(T, task, "none")
	T["make_env"](cfg)
	agent = T["TDMPC2"](cfg)
	T["set_seed"](999)
	agent.model = T["WorldModel"](cfg).to(device).eval()
	with torch.no_grad():
		agent.model._Qs.params["2", "weight"].normal_(0, 0.02)
	O, A, _ = CA.rollout(T, cfg, agent, 0)
	m64 = CA.to_float64(agent.model)
	trajs = [(CA.encode64(m64, O, device)[:-1], torch.as_tensor(A, device=device))]
	anchors = anchor_set(cfg.episode_length, P1["anchors"]["margin"], P1["anchors"]["stride"])
	s = two_passes(CA.make_jacobian_fn(T, cfg, m64, m64, device), trajs,
				   torch.as_tensor(CA.U_NP, device=device), anchors, CA.U_NP.shape[1])
	scalars = [s["kappa"], s["rho_c"], s["rho_o"], s["gamma"]] + \
			  [v for r in s["routes"].values() for k, v in r.items() if isinstance(v, float)]
	assert all(np.isfinite(scalars)), "non-finite statistic"
	assert s["N_anchors"] == len(anchors) and s["n_near_pairs"] > 0 and s["n_far_pairs"] > 0
	print(f"H2 SMOKE DONE: {len(scalars)} statistics present and finite over {len(anchors)} anchors "
		  f"({s['n_near_pairs']} near / {s['n_far_pairs']} far pairs). Values not printed.")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--tdmpc2", required=True)
	ap.add_argument("--ckpt-dir")
	ap.add_argument("--set", choices=["heldout", "discovery"], default="heldout")
	ap.add_argument("--out", default=str(ROOT / "results2"))
	ap.add_argument("--tasks", nargs="*")
	ap.add_argument("--smoke", action="store_true")
	args = ap.parse_args()
	assert torch.cuda.is_available(), "TD-MPC2 hard-codes cuda:0"
	device = torch.device("cuda:0")
	T = CA.import_tdmpc2(args.tdmpc2)
	if args.smoke:
		return smoke(T, device)
	base = Path(args.out) / ("discovery" if args.set == "discovery" else "")
	work, diag = base / "work", base / "diagnostics"
	for d in (work, diag):
		d.mkdir(parents=True, exist_ok=True)
	CA.write_environment(Path(args.out))
	for task, seed, name, want in model_list(args.set):
		if args.tasks and task not in args.tasks:
			continue
		run_model(T, task, seed, name, want, args.ckpt_dir, work, diag, device)


if __name__ == "__main__":
	main()
