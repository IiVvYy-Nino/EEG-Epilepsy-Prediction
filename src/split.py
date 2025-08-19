"""Create patient-level train/val/test splits and save to JSON.

Outputs a dict of record_id lists: {"train": [...], "val": [...], "test": [...]}.
"""
import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np

from .utils import ensure_dir, pair_edf_tse, parse_patient_id, save_json


def split_by_patient_ids(
	pairs: List[Tuple[str, Optional[str], str]], val_ratio: float, test_ratio: float, seed: int
) -> Dict[str, List[str]]:
	patients: Dict[str, List[str]] = {}
	for _edf, _tse, rec in pairs:
		pid = parse_patient_id(rec)
		patients.setdefault(pid, []).append(rec)
	pids = list(patients.keys())
	rng = np.random.RandomState(seed)
	rng.shuffle(pids)
	n = len(pids)
	n_test = int(round(n * test_ratio))
	n_val = int(round(n * val_ratio))
	val_ids = set(pids[:n_val])
	test_ids = set(pids[n_val:n_val + n_test])
	train_ids = set(pids[n_val + n_test:])
	def collect(idset):
		res: List[str] = []
		for pid in idset:
			res.extend(patients[pid])
		return res
	return {"train": collect(train_ids), "val": collect(val_ids), "test": collect(test_ids)}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", type=str, default="data/Dataset_train_dev")
	ap.add_argument("--out", type=str, default="outputs/splits.json")
	ap.add_argument("--val_ratio", type=float, default=0.2)
	ap.add_argument("--test_ratio", type=float, default=0.0)
	ap.add_argument("--seed", type=int, default=42)
	args = ap.parse_args()

	pairs = pair_edf_tse(args.data_dir)
	splits = split_by_patient_ids(pairs, args.val_ratio, args.test_ratio, args.seed)
	ensure_dir(os.path.dirname(args.out))
	save_json(splits, args.out)
	print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
	raise SystemExit(main())
