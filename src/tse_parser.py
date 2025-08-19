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


def build_frame_labels(
	frame_centers: List[float],
	window_sec: float,
	segments: List[Tuple[float, float, str, float]],
	background_label: str,
	overlap_ratio: float = 0.2,
	min_seg_duration: float = 0.0,
) -> List[str]:
	"""Assign labels per window using segment-window overlap ratio.

	A window centered at t spans [t-w/2, t+w/2]. If overlap/window_len >= overlap_ratio,
	assign the segment label (pick the one with max overlap if multiple).
	Segments shorter than min_seg_duration are ignored.
	"""
	labels: List[str] = [background_label] * len(frame_centers)
	w = float(max(window_sec, 1e-6))
	half = 0.5 * w
	# filter tiny segments
	filt: List[Tuple[float, float, str]] = []
	for (s, e, lab, _c) in segments:
		if (e - s) >= float(min_seg_duration):
			filt.append((float(s), float(e), lab))
	if not filt:
		return labels
	for i, t in enumerate(frame_centers):
		ws = float(t) - half
		we = float(t) + half
		best_lab = None
		best_ov = 0.0
		for (s, e, lab) in filt:
			ov = max(0.0, min(we, e) - max(ws, s))
			if ov <= 0:
				continue
			if (ov / w) >= overlap_ratio and ov > best_ov:
				best_ov = ov
				best_lab = lab
		if best_lab is not None:
			labels[i] = best_lab
	return labels

