"""Utility helpers: pairing files, JSON I/O, patient splits, and label tools."""

import os
import json
import re
from typing import List, Tuple, Dict, Optional

import numpy as np


def pair_edf_tse(data_dir: str) -> List[Tuple[str, Optional[str], str]]:
	"""扫描目录，按 record_id 配对 .edf 和 .tse 文件。
	返回 (edf_path, tse_path_or_None, record_id)。
	"""
	files = os.listdir(data_dir)
	edfs = {}
	tses = {}
	for name in files:
		base, ext = os.path.splitext(name)
		path = os.path.join(data_dir, name)
		if ext.lower() == ".edf":
			edfs[base] = path
		elif ext.lower() == ".tse":
			tses[base] = path
	pairs: List[Tuple[str, Optional[str], str]] = []
	for base, edf_path in edfs.items():
		pairs.append((edf_path, tses.get(base), base))
	return pairs


def scan_label_set(tse_paths: List[str], background_label: str) -> List[str]:
	labels = set()
	for p in tse_paths:
		if not p or not os.path.exists(p):
			continue
		with open(p, "r", encoding="utf-8", errors="ignore") as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) >= 3:
					lab = parts[2]
					# skip version/header tokens
					if lab.lower().startswith("tse_v"):
						continue
					labels.add(lab)
	labels.discard(background_label)
	return [background_label] + sorted(labels)


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
	ensure_dir(os.path.dirname(path) or ".")
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str):
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def parse_patient_id(record_id: str) -> str:
	m = re.match(r"^(\d{8})_", record_id)
	return m.group(1) if m else record_id


def split_records_by_patient(
	pairs: List[Tuple[str, Optional[str], str]],
	val_ratio: float,
	test_ratio: float,
	seed: int = 42,
) -> Dict[str, List[Tuple[str, Optional[str], str]]]:
	labeled = [p for p in pairs if p[1]]
	rng = np.random.RandomState(seed)
	patients: Dict[str, List[Tuple[str, Optional[str], str]]] = {}
	for _, _, rec in labeled:
		pid = parse_patient_id(rec)
		patients.setdefault(pid, [])
	for item in labeled:
		pid = parse_patient_id(item[2])
		patients[pid].append(item)
	pids = list(patients.keys())
	rng.shuffle(pids)
	n = len(pids)
	n_test = int(round(n * test_ratio))
	n_val = int(round(n * val_ratio))
	val_ids = set(pids[:n_val])
	test_ids = set(pids[n_val:n_val + n_test])
	train_ids = set(pids[n_val + n_test:])
	def collect(idset):
		res: List[Tuple[str, Optional[str], str]] = []
		for pid in idset:
			res.extend(patients[pid])
		return res
	return {
		"train": collect(train_ids),
		"val": collect(val_ids),
		"test": collect(test_ids),
	}


def merge_non_background_segments(
	frame_centers_sec: np.ndarray,
	probs: np.ndarray,
	label_names: List[str],
	background_label: str,
	prob_threshold: float = 0.5,
	min_duration_sec: float = 0.0,
) -> List[Dict]:
	"""从每帧概率生成片段列表。"""
	bg_idx = label_names.index(background_label)
	fg_prob = 1.0 - probs[:, bg_idx]
	is_fg = fg_prob >= prob_threshold
	segments: List[Dict] = []
	start = None
	for i, flag in enumerate(is_fg):
		if flag and start is None:
			start = i
		elif (not flag) and start is not None:
			end = i - 1
			dur = frame_centers_sec[end] - frame_centers_sec[start]
			if dur >= min_duration_sec:
				# 多数类
				sum_probs = probs[start:end + 1].sum(axis=0)
				cls = int(np.argmax(sum_probs))
				segments.append({
					"start": float(frame_centers_sec[start]),
					"end": float(frame_centers_sec[end]),
					"label": label_names[cls],
					"score": float(fg_prob[start:end + 1].max()),
				})
			start = None
	if start is not None:
		end = len(frame_centers_sec) - 1
		dur = frame_centers_sec[end] - frame_centers_sec[start]
		if dur >= min_duration_sec:
			sum_probs = probs[start:end + 1].sum(axis=0)
			cls = int(np.argmax(sum_probs))
			segments.append({
				"start": float(frame_centers_sec[start]),
				"end": float(frame_centers_sec[end]),
				"label": label_names[cls],
				"score": float(fg_prob[start:end + 1].max()),
			})
	return segments


def normalize_label_with_alias(label: str, aliases: Optional[Dict[str, str]]) -> str:
	"""用别名表归一化标签名（大小写不敏感）。找不到则原样返回。"""
	if not aliases:
		return label
	return aliases.get(label.lower(), label)

