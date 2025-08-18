# Source Generated with Decompyle++
# File: scan_thresholds.cpython-311.pyc (Python 3.11)

"""Threshold grid search for event decoding.

Sweeps combinations of post-process parameters (prob, smooth, confirm, cooldown)
and selects the best under a false-alarms-per-hour constraint, writing results to
JSON and optionally writing the best thresholds back into configs/config.yaml.

For consistent label mapping, provide --labels_json exported during training.
"""

import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils import ensure_dir, load_json, scan_label_set
from dataset import SequenceDataset
from model import BiLSTMClassifier
from postprocess import decode_pred_events
from metrics import match_events_by_iou, precision_recall_f1, false_alarms_per_hour


def decode_gt_events(labels: np.ndarray, centers: np.ndarray, bg_idx: int) -> List[Dict]:
	lab = labels.copy()
	lab[lab < 0] = bg_idx
	start = None
	cur = None
	events: List[Dict] = []
	for i, y in enumerate(lab.tolist()):
		if y != bg_idx and start is None:
			start, cur = i, int(y)
		elif (y == bg_idx or y != cur) and start is not None:
			end = i - 1
			events.append({"start": float(centers[start]), "end": float(centers[end]), "label": cur})
			start, cur = (None, None)
	if start is not None:
		end = len(centers) - 1
		events.append({"start": float(centers[start]), "end": float(centers[end]), "label": int(cur)})
	return events


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", type=str, default="data/Dataset_train_dev")
	parser.add_argument("--cache_dir", type=str, default="data_cache")
	parser.add_argument("--window_sec", type=float, default=2.0)
	parser.add_argument("--hop_sec", type=float, default=0.25)
	parser.add_argument("--resample_hz", type=float, default=256.0)
	parser.add_argument("--bandpass", type=float, nargs=2, default=[0.5, 45.0])
	parser.add_argument("--notch_hz", type=float, default=50.0)
	parser.add_argument("--bg_label", type=str, default="bckg")
	parser.add_argument("--probs", type=float, nargs="*", default=[0.6, 0.7, 0.8, 0.9])
	parser.add_argument("--smooth", type=float, nargs="*", default=[0.0, 0.25])
	parser.add_argument("--confirm", type=int, nargs="*", default=[1, 2, 3])
	parser.add_argument("--cooldown", type=float, nargs="*", default=[0.0, 0.5, 1.0])
	parser.add_argument("--min_duration", type=float, default=0.0)
	parser.add_argument("--iou", type=float, default=0.5)
	parser.add_argument("--max_fa_per_hour", type=float, default=2.0)
	parser.add_argument("--out", type=str, default="outputs/threshold_grid.json")
	parser.add_argument("--write_config", type=str, default="configs/config.yaml")
	parser.add_argument("--labels_json", type=str, default=None, help="Path to labels.json from training (for class set and aliases)")
	args = parser.parse_args()

	ensure_dir(os.path.dirname(args.out))

	# Build label set from labels.json if provided
	label_to_index: Dict[str, int]
	alias_map: Dict[str, str] = {}
	if args.labels_json and os.path.exists(args.labels_json):
		lj = load_json(args.labels_json)
		label_names = lj.get("label_names", []) or []
		alias_map = {k.lower(): v for k, v in (lj.get("aliases", {}) or {}).items()}
		if args.bg_label not in label_names:
			label_names = [args.bg_label] + [n for n in label_names if n != args.bg_label]
		else:
			label_names = [args.bg_label] + [n for n in label_names if n != args.bg_label]
		label_to_index = {name: i for i, name in enumerate(label_names)}
	else:
		label_to_index = {args.bg_label: 0}
	ds = SequenceDataset(
		data_dir=args.data_dir,
		cache_dir=args.cache_dir,
		label_to_index=label_to_index,
		window_sec=args.window_sec,
		hop_sec=args.hop_sec,
		resample_hz=args.resample_hz,
		bandpass=(args.bandpass[0], args.bandpass[1]),
		notch_hz=args.notch_hz,
		require_labels=True,
		standardize=False,
		labels_json=args.labels_json if (args.labels_json and os.path.exists(args.labels_json)) else None,
	)

	# Validate TSE labels are covered by label set (considering aliases)
	tse_paths = [t for (_e, t, _r) in ds.items if t]
	if tse_paths and len(label_to_index) >= 1:
		present = scan_label_set(tse_paths, args.bg_label)
		unknown = []
		known = set(label_to_index.keys())
		for lab in present:
			if lab.lower() == args.bg_label.lower():
				continue
			canon = alias_map.get(lab.lower(), lab)
			if canon not in known:
				unknown.append(lab)
		if unknown:
			raise RuntimeError(f"Unknown labels in TSE not covered by training label set: {sorted(set(unknown))}. Provide --labels_json from training or update your label config.")

	# 简化：使用随机权重模型（或在此处加载你的 checkpoint）
	x0 = ds[0]["x"]
	input_dim = x0.shape[-1]
	model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=1, num_classes=len(label_to_index))
	model.eval()

	bg_idx = label_to_index[args.bg_label]
	results = []
	best = {"f1": -1.0}
	for p in args.probs:
		for s in args.smooth:
			for cfm in args.confirm:
				for cd in args.cooldown:
					TP = FP = FN = 0
					total_seconds = 0.0
					for item in ds.items:
						data = ds[ds.items.index(item)]
						X = data["x"]; centers = data["centers"]; Y = data["y"]
						total_seconds += float(centers[-1] - centers[0]) if len(centers) > 1 else 0.0
						with torch.no_grad():
							logits = model(torch.from_numpy(X[None, ...]))
							probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
						pred_events = decode_pred_events(
							probs=probs,
							centers=centers,
							bg_idx=bg_idx,
							prob_threshold=p,
							min_duration=args.min_duration,
							confirm_windows=cfm,
							cooldown_sec=cd,
							smooth_sec=s,
							hop_sec=args.hop_sec,
						)
						gt_events = decode_gt_events(Y, centers, bg_idx)
						tp, fp, fn, pairs = match_events_by_iou(pred_events, gt_events, args.iou)
						TP += tp; FP += fp; FN += fn
					fa_per_h = false_alarms_per_hour(FP, total_seconds / 3600.0)
					prec, rec, f1 = precision_recall_f1(TP, FP, FN)
					rec_row = {"prob": p, "smooth": s, "confirm": cfm, "cooldown": cd, "min_duration": args.min_duration, "precision": prec, "recall": rec, "f1": f1, "fa_per_h": fa_per_h}
					results.append(rec_row)
					if fa_per_h <= args.max_fa_per_hour and f1 > best.get("f1", -1):
						best = rec_row

	ensure_dir(os.path.dirname(args.out))
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump({"best": best, "results": results}, f, ensure_ascii=False, indent=2)
	print(json.dumps({"best": best}, indent=2))

	# auto write-back best thresholds into config yaml
	try:
		import yaml  # lazy import
		if best.get("f1", -1) >= 0 and args.write_config and os.path.exists(args.write_config):
			with open(args.write_config, "r", encoding="utf-8") as rf:
				cfg_all = yaml.safe_load(rf) or {}
			cfg_all.setdefault("postprocess", {})
			cfg_all["postprocess"]= {
				"prob": best.get("prob"),
				"smooth": best.get("smooth"),
				"confirm": best.get("confirm"),
				"cooldown": best.get("cooldown"),
				"min_duration": best.get("min_duration"),
			}
			with open(args.write_config, "w", encoding="utf-8") as wf:
				yaml.safe_dump(cfg_all, wf, allow_unicode=True, sort_keys=False)
			print("Best thresholds written to", args.write_config)
	except Exception:
		pass


if __name__ == '__main__':
	raise SystemExit(main())
