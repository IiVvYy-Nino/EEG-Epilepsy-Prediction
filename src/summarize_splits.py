"""Visualize split label distributions from splits.json.

Outputs:
- CSV summary with per-split class durations (seconds), record and patient counts
- PNG bar chart of top-K classes across splits
"""
import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from .utils import pair_edf_tse, parse_patient_id, load_json, ensure_dir, save_json


def _read_tse_durations(tse_path: str, alias_map: Dict[str, str], background: str) -> Dict[str, float]:
	acc: Dict[str, float] = {}
	try:
		with open(tse_path, "r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) < 3:
					continue
				lab = parts[2]
				if lab.lower().startswith("tse_v"):
					continue
				try:
					start = float(parts[0])
					end = float(parts[1])
				duration = max(0.0, end - start)
				canon = alias_map.get(lab.lower(), lab)
				acc[canon] = acc.get(canon, 0.0) + duration
		except Exception:
			pass
	# ensure background is present
	acc.setdefault(background, 0.0)
	return acc


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--splits_json", type=str, default="outputs/splits.json")
	ap.add_argument("--data_dir", type=str, default="data/Dataset_train_dev")
	ap.add_argument("--labels_json", type=str, default=None)
	ap.add_argument("--out_csv", type=str, default="outputs/splits_summary.csv")
	ap.add_argument("--out_png", type=str, default="outputs/splits_summary.png")
	ap.add_argument("--top_k", type=int, default=12)
	ap.add_argument("--include_background", action="store_true")
	args = ap.parse_args()

	# Load splits
	splits = load_json(args.splits_json)
	# Map record_id -> tse path
	pairs = pair_edf_tse(args.data_dir)
	rec2tse: Dict[str, Optional[str]] = {rec: tse for (_edf, tse, rec) in pairs}

	# Labels and aliases
	alias_map: Dict[str, str] = {}
	background = "bckg"
	if args.labels_json and os.path.exists(args.labels_json):
		lj = load_json(args.labels_json)
		alias_map = {k.lower(): v for k, v in (lj.get("aliases", {}) or {}).items()}
		background = lj.get("background", background)

	# Aggregate durations per split
	per_split: Dict[str, Dict[str, float]] = {}
	split_rec_counts: Dict[str, int] = {}
	split_pat_counts: Dict[str, int] = {}
	for split_name, rec_list in splits.items():
		per_split[split_name] = {}
		split_rec_counts[split_name] = len(rec_list or [])
		pids = {parse_patient_id(r) for r in (rec_list or [])}
		split_pat_counts[split_name] = len(pids)
		for rec in (rec_list or []):
			tse = rec2tse.get(rec)
			if not tse or not os.path.exists(tse):
				continue
			durs = _read_tse_durations(tse, alias_map, background)
			for lab, sec in durs.items():
				per_split[split_name][lab] = per_split[split_name].get(lab, 0.0) + sec

	# Collect all classes
	all_classes = set()
	for d in per_split.values():
		all_classes.update(d.keys())
	# Optionally drop background
	if not args.include_background and background in all_classes:
		all_classes.remove(background)
	classes = sorted(all_classes)

	# Write CSV
	ensure_dir(os.path.dirname(args.out_csv) or ".")
	import csv
	with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		header = ["split", "num_patients", "num_records"] + classes
		writer.writerow(header)
		for split_name in ["train", "val", "test"]:
			row = [split_name, split_pat_counts.get(split_name, 0), split_rec_counts.get(split_name, 0)]
			vals = [per_split.get(split_name, {}).get(c, 0.0) for c in classes]
			row.extend([f"{v:.3f}" for v in vals])
			writer.writerow(row)

	# Plot top-K classes by total duration
	totals = np.array([sum(per_split.get(s, {}).get(c, 0.0) for s in per_split.keys()) for c in classes], dtype=float)
	if totals.size == 0:
		print("No durations found; skip plotting")
		return
	idx = np.argsort(-totals)[: max(1, min(args.top_k, len(classes)))]
	cls_top = [classes[i] for i in idx]

	splits_order = [s for s in ["train", "val", "test"] if s in per_split]
	M = len(splits_order)
	K = len(cls_top)
	data = np.zeros((M, K), dtype=float)
	for i, s in enumerate(splits_order):
		for j, c in enumerate(cls_top):
			data[i, j] = per_split.get(s, {}).get(c, 0.0)

	# Normalize to hours for readability
	data_hours = data / 3600.0

	plt.figure(figsize=(max(8, K * 0.6), 4 + 0.4 * M))
	width = 0.8 / max(M, 1)
	x = np.arange(K)
	for i, s in enumerate(splits_order):
		plt.bar(x + i * width, data_hours[i], width=width, label=f"{s}")
	plt.xticks(x + width * (M - 1) / 2, cls_top, rotation=45, ha="right")
	plt.ylabel("Duration (hours)")
	plt.title("Split label durations (top-{} classes)".format(K))
	plt.legend()
	plt.tight_layout()
	ensure_dir(os.path.dirname(args.out_png) or ".")
	plt.savefig(args.out_png, dpi=160)
	print({"csv": args.out_csv, "png": args.out_png})


if __name__ == "__main__":
	raise SystemExit(main())


