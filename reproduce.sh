#!/usr/bin/env bash
# One entry point (PREREGISTRATION.md §12). Needs a CUDA GPU (TD-MPC2 hard-codes cuda:0) and internet.
#
#   bash reproduce.sh setup      # install the pinned stack, fetch TD-MPC2 at the pinned commit
#   bash reproduce.sh smoke      # engineering checks on a random-init network; loads NO checkpoint
#   bash reproduce.sh test       # the synthetic validation suites (§13); no GPU needed
#   bash reproduce.sh run [tasks...]   # download + hash-check checkpoints, collect, then verdict + figures
#   bash reproduce.sh verdict    # verdict + figures from existing results/work
#   Protocol 2: h2-smoke | h2-heldout [tasks] | h2-discovery [tasks] | h2-verdict
set -euo pipefail
cd "$(dirname "$0")"
EXT=ext
CKPT=ckpt
cfg() { python -c "from scripts.prereg import load_cfg; c=load_cfg(); print($1)"; }

setup() {
	pip install -q -r requirements-tdmpc2.txt
	local commit; commit=$(cfg "c['models']['code_commit']")
	if [ ! -d "$EXT/tdmpc2/.git" ]; then git clone -q https://github.com/nicklashansen/tdmpc2 "$EXT/tdmpc2"; fi
	git -C "$EXT/tdmpc2" checkout -q "$commit"
	test "$(git -C "$EXT/tdmpc2" rev-parse HEAD)" = "$commit"
	echo "tdmpc2 at $commit"
}

download() {
	local rev; rev=$(cfg "c['models']['checkpoint_revision']")
	mkdir -p "$CKPT"
	python - "$rev" "$CKPT" "$@" <<'PY'
import sys, urllib.request, hashlib
from pathlib import Path
from scripts.prereg import load_cfg
rev, out, tasks = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
c = load_cfg()
for name, want in c["models"]["checkpoint_sha256"].items():
	if tasks and name.rsplit("-", 1)[0] not in tasks:
		continue
	p = out / name
	if not p.exists():
		urllib.request.urlretrieve(f"https://huggingface.co/nicklashansen/tdmpc2/resolve/{rev}/dmcontrol/{name}", p)
	got = hashlib.sha256(p.read_bytes()).hexdigest()
	print(("OK   " if got == want else "BAD  ") + name)
	if got != want:
		sys.exit(f"hash mismatch for {name}: this checkpoint is not analysed")
PY
}

download2() {   # protocol-2 held-out checkpoints, hash-checked against prereg-2/config/prereg.yaml
	mkdir -p "$CKPT"
	python - "$CKPT" <<'PY'
import sys, urllib.request, hashlib
from pathlib import Path
from scripts.prereg import ROOT, load_cfg
out = Path(sys.argv[1]); c = load_cfg(ROOT / "prereg-2" / "config" / "prereg.yaml")
rev = c["models"]["checkpoint_revision"]
for t, seeds in c["models"]["seeds_by_task"].items():
	for s in seeds:
		name = f"{t}-{s}.pt"; want = c["models"]["checkpoint_sha256"][name]; p = out / name
		if not p.exists():
			urllib.request.urlretrieve(f"https://huggingface.co/nicklashansen/tdmpc2/resolve/{rev}/dmcontrol/{name}", p)
		got = hashlib.sha256(p.read_bytes()).hexdigest()
		print(("OK   " if got == want else "BAD  ") + name)
		if got != want:
			sys.exit(f"hash mismatch for {name}")
PY
}

case "${1:-}" in
	h2-smoke)   python -m scripts.collect_h2 --tdmpc2 "$EXT/tdmpc2/tdmpc2" --smoke ;;
	h2-heldout) shift; download2
	            python -m scripts.collect_h2 --tdmpc2 "$EXT/tdmpc2/tdmpc2" --ckpt-dir "$CKPT" --set heldout ${@:+--tasks "$@"} ;;
	h2-discovery) shift; download
	            python -m scripts.collect_h2 --tdmpc2 "$EXT/tdmpc2/tdmpc2" --ckpt-dir "$CKPT" --set discovery ${@:+--tasks "$@"} ;;
	h2-verdict) python -m scripts.h2_verdict ;;
	setup) setup ;;
	smoke) python -m scripts.collect_anchors --tdmpc2 "$EXT/tdmpc2/tdmpc2" --smoke ;;
	test)  for t in tests/test_*.py; do python "$t"; done ;;
	run)   shift; download "$@"
	       python -m scripts.collect_anchors --tdmpc2 "$EXT/tdmpc2/tdmpc2" --ckpt-dir "$CKPT" ${@:+--tasks "$@"} ;;
	verdict) python -m scripts.verdict && python -m scripts.make_figures ;;
	*) sed -n 2,9p "$0"; exit 1 ;;
esac
