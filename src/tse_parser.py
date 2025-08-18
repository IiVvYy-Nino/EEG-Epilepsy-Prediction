# Source Generated with Decompyle++
# File: tse_parser.cpython-311.pyc (Python 3.11)

"""Parse .tse label files and map segments to frame labels.

Format example per line:
0.0000 36.8868 bckg 1.0000
"""
import os
from typing import List, Tuple


def parse_tse(tse_path: str) -> List[Tuple[float, float, str, float]]:
	"""Parse a .tse file into a list of (start_sec, end_sec, label, confidence)."""
	segments: List[Tuple[float, float, str, float]] = []
	if not tse_path or not os.path.exists(tse_path):
		return segments
	with open(tse_path, "r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			if len(parts) < 3:
				continue
			try:
				start = float(parts[0])
				end = float(parts[1])
				label = parts[2]
				# ignore header/version tokens (e.g., tse_v1.0.0)
				if label.lower().startswith("tse_v"):
					continue
				conf = float(parts[3]) if len(parts) >= 4 else 1.0
				segments.append((start, end, label, conf))
			except Exception:
				continue
	return segments


def build_frame_labels(frame_centers: List[float], segments: List[Tuple[float, float, str, float]], background_label: str) -> List[str]:
	"""Assign a label to each frame center time based on segments; default background."""
	labels: List[str] = [background_label] * len(frame_centers)
	for (s, e, lab, _c) in segments:
		for i, t in enumerate(frame_centers):
			if s <= t <= e:
				labels[i] = lab
	return labels

