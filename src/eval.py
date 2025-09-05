"""Event-level evaluation.

Loads a trained checkpoint, runs frame-level inference, decodes events via
post-processing, then computes IoU-based TP/FP/FN, PR/F1, FA/h, and onset/offset
latencies at one or multiple IoU thresholds. Results are written to JSON/CSV.

Postprocess thresholds are read from configs/config.yaml unless overridden.
Label set is loaded from labels.json when provided, with alias normalization
and consistency checks against TSE files.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from .utils import ensure_dir, load_json, scan_label_set
from .dataset import SequenceDataset
from .model import BiLSTMClassifier
from .postprocess import decode_pred_events
from .metrics import match_events_by_iou, precision_recall_f1, false_alarms_per_hour, interval_iou, mean_onset_offset_latency


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


def load_config(path: str) -> Dict:
	if not path or not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	parser.add_argument("--data_dir", type=str, default="data/Dataset_train_dev")
	parser.add_argument("--cache_dir", type=str, default="data_cache")
	parser.add_argument("--window_sec", type=float, default=2.0)
	parser.add_argument("--hop_sec", type=float, default=0.25)
	parser.add_argument("--resample_hz", type=float, default=256.0)
	parser.add_argument("--bandpass", type=float, nargs=2, default=[0.5, 45.0])
	parser.add_argument("--notch_hz", type=float, default=50.0)
	parser.add_argument("--bg_label", type=str, default="bckg")
	parser.add_argument("--checkpoint", type=str, default="outputs/best.pt")
	parser.add_argument("--out_json", type=str, default="outputs/eval_summary.json")
	parser.add_argument("--out_csv", type=str, default="outputs/eval_records.csv")
	parser.add_argument("--ious", type=float, nargs="*", default=[0.3, 0.5])
	# optional split filtering
	parser.add_argument("--splits_json", type=str, default=None, help="If provided, restrict evaluation to records listed in the given split JSON")
	parser.add_argument("--use_split", type=str, choices=["train", "val", "test"], default="test")
	# postprocess override (fallback to config.postprocess if present)
	parser.add_argument("--prob", type=float, default=None)
	parser.add_argument("--smooth", type=float, default=None)
	parser.add_argument("--confirm", type=int, default=None)
	parser.add_argument("--cooldown", type=float, default=None)
	parser.add_argument("--min_duration", type=float, default=None)
	# labels
	parser.add_argument("--labels_json", type=str, default=None, help="Path to labels.json generated during training (for class set and aliases)")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		try:
			_idx = torch.cuda.current_device()
			_name = torch.cuda.get_device_name(_idx)
			device_desc = f"cuda:{_idx} ({_name})"
		except Exception:
			device_desc = "cuda"
	else:
		device_desc = "cpu"
	print({"device": device_desc})

	cfg = load_config(args.config)
	pp = cfg.get("postprocess", {}) if cfg else {}
	prob = args.prob if args.prob is not None else pp.get("prob", 0.8)
	smooth = args.smooth if args.smooth is not None else pp.get("smooth", 0.25)
	confirm = args.confirm if args.confirm is not None else pp.get("confirm", 2)
	cooldown = args.cooldown if args.cooldown is not None else pp.get("cooldown", 0.5)
	min_duration = args.min_duration if args.min_duration is not None else pp.get("min_duration", 0.0)

	# Load label set consistent with training
	labels_cfg = cfg.get("labels", {}) if cfg else {}
	labels_json_path = args.labels_json or labels_cfg.get("json_out")
	label_to_index: Dict[str, int]
	alias_map: Dict[str, str] = {}
	if labels_json_path and os.path.exists(labels_json_path):
		lj = load_json(labels_json_path)
		label_names = lj.get("label_names", []) or []
		alias_map = {k.lower(): v for k, v in (lj.get("aliases", {}) or {}).items()}
		# ensure background present and at index 0
		if args.bg_label not in label_names:
			label_names = [args.bg_label] + [n for n in label_names if n != args.bg_label]
		else:
			label_names = [args.bg_label] + [n for n in label_names if n != args.bg_label]
		label_to_index = {name: i for i, name in enumerate(label_names)}
	else:
		label_to_index = {args.bg_label: 0}

	ensure_dir(os.path.dirname(args.out_json))
	ensure_dir(os.path.dirname(args.out_csv))

	# Build dataset (require labels for evaluation)
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
		labels_json=labels_json_path,
	)
	# If a split file is provided, restrict to that subset (e.g., test fold)
	if args.splits_json and os.path.exists(args.splits_json):
		try:
			splits = load_json(args.splits_json)
			rec_list = splits.get(args.use_split, []) or []
			allowed = set(rec_list)
			# filter ds.items by record_id
			ds.items = [it for it in ds.items if it[2] in allowed]
		except Exception:
			pass

	# Validate that TSE labels are covered by the configured label set (considering aliases)
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

	# Build model and load checkpoint
	x0 = ds[0]["x"]
	input_dim = x0.shape[-1]
	# 🧠 使用注意力机制模型（与训练时保持一致）
	model = BiLSTMClassifier(
		input_dim=input_dim, 
		hidden_dim=256, 
		num_layers=3, 
		num_classes=len(label_to_index),
		dropout=0.15,
		use_attention=True,
		attention_heads=8
	)
	if args.checkpoint and os.path.exists(args.checkpoint):
		ckpt = torch.load(args.checkpoint, map_location="cpu")
		state = ckpt.get("model", ckpt)
		# Validate classifier head size vs label count
		if isinstance(state, dict) and any(k.endswith("proj.weight") for k in state.keys()):
			for k, v in state.items():
				if k.endswith("proj.weight"):
					trained_classes = int(v.shape[0])
					if trained_classes != len(label_to_index):
						raise RuntimeError(f"Checkpoint classifier classes ({trained_classes}) do not match current label set size ({len(label_to_index)}). Use matching --labels_json or checkpoint.")
		model.load_state_dict(state, strict=False)
	model = model.to(device)
	model.eval()

	bg_idx = label_to_index[args.bg_label]
	global_metrics: Dict[str, Dict[str, float]] = {}
	per_record_rows: List[Dict] = []
	micro_totals: Dict[str, Dict[str, float]] = {}
	for iou_thr in args.ious:
		TP = FP = FN = 0
		on_lat_sum = 0.0
		off_lat_sum = 0.0
		lat_count = 0
		total_seconds = 0.0
		for item in ds.items:
			data = ds[ds.items.index(item)]
			X = data["x"]; centers = data["centers"]; Y = data["y"]
			total_seconds += float(centers[-1] - centers[0]) if len(centers) > 1 else 0.0
			with torch.no_grad():
				inp = torch.from_numpy(X[None, ...]).to(device)
				# 🧠 注意力机制：模型返回logits和attention权重
				logits, _ = model(inp)
				probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
			pred_events = decode_pred_events(
				probs=probs,
				centers=centers,
				bg_idx=bg_idx,
				prob_threshold=prob,
				min_duration=min_duration,
				confirm_windows=confirm,
				cooldown_sec=cooldown,
				smooth_sec=smooth,
				hop_sec=args.hop_sec,
			)
			gt_events = decode_gt_events(Y, centers, bg_idx)
			tp, fp, fn, pairs = match_events_by_iou(pred_events, gt_events, iou_thr)
			TP += tp; FP += fp; FN += fn
			mo, mf = mean_onset_offset_latency(pred_events, gt_events, pairs)
			on_lat_sum += mo; off_lat_sum += mf; lat_count += (1 if pairs else 0)
			# per-record duration for potential aggregation
			rec_dur = float(centers[-1] - centers[0]) if len(centers) > 1 else 0.0
			per_record_rows.append({
				"record_id": data["record_id"],
				"pred_events": len(pred_events),
				"gt_events": len(gt_events),
				"tp": tp, "fp": fp, "fn": fn,
				"onset_latency": mo, "offset_latency": mf,
				"iou": iou_thr,
				"duration_sec": rec_dur,
			})
		fa_per_h = false_alarms_per_hour(FP, total_seconds / 3600.0)
		prec, rec, f1 = precision_recall_f1(TP, FP, FN)
		mean_on = (on_lat_sum / max(lat_count, 1)) if lat_count > 0 else 0.0
		mean_off = (off_lat_sum / max(lat_count, 1)) if lat_count > 0 else 0.0
		key = str(iou_thr)
		global_metrics[key] = {
			"precision": prec,
			"recall": rec,
			"f1": f1,
			"fa_per_h": fa_per_h,
			"mean_onset_latency": mean_on,
			"mean_offset_latency": mean_off,
		}
		# store micro totals for downstream aggregation
		micro_totals[key] = {
			"TP": float(TP),
			"FP": float(FP),
			"FN": float(FN),
			"total_seconds": float(total_seconds),
		}

	with open(args.out_json, "w", encoding="utf-8") as jf:
		json.dump({
			"config": {
				"prob": prob, "smooth": smooth, "confirm": confirm, "cooldown": cooldown, "min_duration": min_duration
			},
			"metrics": global_metrics,
			"micro_totals": micro_totals,
		}, jf, ensure_ascii=False, indent=2)

	# CSV per-record
	with open(args.out_csv, "w", newline="", encoding="utf-8") as cf:
		writer = csv.DictWriter(cf, fieldnames=list(per_record_rows[0].keys()) if per_record_rows else ["record_id"]) 
		writer.writeheader()
		for r in per_record_rows:
			writer.writerow(r)

	print(json.dumps(global_metrics, indent=2))


if __name__ == "__main__":
	raise SystemExit(main())


