"""Single-EDF prediction utility.

Runs end-to-end inference on one EDF file without labels, returning whether
any seizure events are detected, how many, their type names, and start/end
times. Outputs a concise JSON summary for downstream consumption.

It reuses the same preprocessing/feature settings as in configs/config.yaml
unless overridden, and decodes events with postprocess thresholds. For proper
class names (beyond background), pass --labels_json from training.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import yaml

from .utils import ensure_dir, load_json
from .edf_reader import load_edf
from .features import extract_features_multichannel
from .postprocess import decode_pred_events
from .model import BiLSTMClassifier


def load_config(path: str) -> Dict:
	if not path or not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--edf", type=str, required=True)
	parser.add_argument("--checkpoint", type=str, default="outputs/best.pt")
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	parser.add_argument("--labels_json", type=str, default=None, help="labels.json from training (for class names and aliases)")
	# feature/extract overrides
	parser.add_argument("--window_sec", type=float, default=None)
	parser.add_argument("--hop_sec", type=float, default=None)
	parser.add_argument("--resample_hz", type=float, default=None)
	parser.add_argument("--bandpass", type=float, nargs=2, default=None)
	parser.add_argument("--notch_hz", type=float, default=None)
	# postprocess overrides
	parser.add_argument("--prob", type=float, default=None)
	parser.add_argument("--smooth", type=float, default=None)
	parser.add_argument("--confirm", type=int, default=None)
	parser.add_argument("--cooldown", type=float, default=None)
	parser.add_argument("--min_duration", type=float, default=None)
	parser.add_argument("--out", type=str, default="outputs/pred_events.json")
	args = parser.parse_args()

	cfg = load_config(args.config)
	train_cfg = cfg.get("train", {}) if cfg else {}
	pp = cfg.get("postprocess", {}) if cfg else {}

	window_sec = args.window_sec if args.window_sec is not None else train_cfg.get("window_sec", 2.0)
	hop_sec = args.hop_sec if args.hop_sec is not None else train_cfg.get("hop_sec", 0.25)
	resample_hz = args.resample_hz if args.resample_hz is not None else train_cfg.get("resample_hz", 256.0)
	bandpass = tuple(args.bandpass) if args.bandpass is not None else tuple(train_cfg.get("bandpass", [0.5, 45.0]))
	notch_hz = args.notch_hz if args.notch_hz is not None else train_cfg.get("notch_hz", 50.0)
	bg_label = train_cfg.get("bg_label", "bckg")

	prob = args.prob if args.prob is not None else pp.get("prob", 0.8)
	smooth = args.smooth if args.smooth is not None else pp.get("smooth", 0.25)
	confirm = args.confirm if args.confirm is not None else pp.get("confirm", 2)
	cooldown = args.cooldown if args.cooldown is not None else pp.get("cooldown", 0.5)
	min_duration = args.min_duration if args.min_duration is not None else pp.get("min_duration", 0.0)

	label_names: List[str]
	if args.labels_json and os.path.exists(args.labels_json):
		lj = load_json(args.labels_json)
		ln = lj.get("label_names", []) or []
		if bg_label not in ln:
			label_names = [bg_label] + [n for n in ln if n != bg_label]
		else:
			label_names = [bg_label] + [n for n in ln if n != bg_label]
	else:
		label_names = [bg_label]

	# Read EDF and extract features
	signals, _ch_names, fs = load_edf(
		args.edf,
		resample_hz=resample_hz,
		bandpass=bandpass,
		notch_hz=notch_hz,
	)
	X, centers = extract_features_multichannel(signals, fs=fs, window_sec=window_sec, hop_sec=hop_sec)

	# Build model and run
	input_dim = X.shape[-1]
	num_classes = len(label_names)
	model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=1, num_classes=num_classes)
	if args.checkpoint and os.path.exists(args.checkpoint):
		state = torch.load(args.checkpoint, map_location="cpu")
		if isinstance(state, dict) and "model" in state:
			state = state["model"]
		# Sanity check on classifier size
		for k, v in state.items():
			if k.endswith("proj.weight") and int(v.shape[0]) != num_classes:
				raise RuntimeError(f"Checkpoint classifier classes ({int(v.shape[0])}) != current label set ({num_classes}). Provide matching --labels_json.")
		model.load_state_dict(state, strict=False)
	model.eval()
	with torch.no_grad():
		logits = model(torch.from_numpy(X[None, ...]).float())
		probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

	events_idx = decode_pred_events(
		probs=probs,
		centers=centers,
		bg_idx=label_names.index(bg_label),
		prob_threshold=prob,
		min_duration=min_duration,
		confirm_windows=confirm,
		cooldown_sec=cooldown,
		smooth_sec=smooth,
		hop_sec=hop_sec,
	)
	# Map label index to name
	events = []
	for ev in events_idx:
		lab_idx = int(ev.get("label", 0))
		name = label_names[lab_idx] if 0 <= lab_idx < len(label_names) else str(lab_idx)
		events.append({"start": ev["start"], "end": ev["end"], "label": name, "score": ev.get("score", 0.0)})

	ensure_dir(os.path.dirname(args.out) or ".")
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump({
			"edf": args.edf,
			"has_seizure": bool(len(events) > 0),
			"num_events": len(events),
			"events": events,
		}, f, ensure_ascii=False, indent=2)
	print(json.dumps({"has_seizure": bool(len(events) > 0), "num_events": len(events)}, indent=2))


if __name__ == "__main__":
	raise SystemExit(main())


