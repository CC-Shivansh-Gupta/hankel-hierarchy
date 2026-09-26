"""Loads config/prereg.yaml, the only source of thresholds (PREREGISTRATION.md §10.3)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path=ROOT / "config" / "prereg.yaml"):
	with open(path, encoding="utf-8") as f:
		return yaml.safe_load(f)


def read_npz(path):
	"""All arrays of an .npz, with the file closed afterwards (Windows keeps open files locked)."""
	import numpy as np
	with np.load(path) as f:
		return {k: f[k] for k in f.files}
