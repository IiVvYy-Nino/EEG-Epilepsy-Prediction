"""Event decoding and post-processing utilities.

Converts per-frame probabilities to event segments with steps:
- smoothing foreground probability;
- thresholding to foreground mask;
- confirmation window filter;
- merge adjacent same-class segments with cooldown;
- enforce minimum duration.
"""

from typing import List, Dict, Optional

import numpy as np


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
	if win <= 1:
		return x
	ker = np.ones((win,), dtype=np.float32) / float(win)
	return np.convolve(x, ker, mode="same")


def _apply_confirm(is_fg: np.ndarray, confirm_windows: int) -> np.ndarray:
	if confirm_windows <= 1:
		return is_fg
	count = 0
	res = np.zeros_like(is_fg)
	for i, v in enumerate(is_fg.tolist()):
		if v:
			count += 1
		else:
			count = 0
		res[i] = count >= confirm_windows
	return res


def _merge_with_cooldown(events: List[Dict], cooldown_sec: float) -> List[Dict]:
	if not events:
		return events
	merged: List[Dict] = []
	cur = dict(events[0])
	for ev in events[1:]:
		if ev["start"] - cur["end"] <= cooldown_sec and ev["label"] == cur["label"]:
			cur["end"] = max(cur["end"], ev["end"])
			cur["score"] = max(cur.get("score", 0.0), ev.get("score", 0.0))
		else:
			merged.append(cur)
			cur = dict(ev)
	merged.append(cur)
	return merged


def decode_pred_events(
	probs: np.ndarray,
	centers: np.ndarray,
	bg_idx: int,
	prob_threshold: float,
	min_duration: float = 0.0,
	confirm_windows: int = 1,
	cooldown_sec: float = 0.0,
	smooth_sec: float = 0.0,
	hop_sec: float = 0.5,
) -> List[Dict]:
	"""Convert per-frame probs to events with simple smoothing/confirmation/cooldown merging.

	Returns list of {start, end, label, score}.
	"""
	fg_prob = 1.0 - probs[:, bg_idx]
	if smooth_sec and smooth_sec > 1e-8:
		win = int(max(1, round(smooth_sec / max(hop_sec, 1e-8))))
		fg_prob = _moving_average(fg_prob, win)
	is_fg = (fg_prob >= prob_threshold).astype(np.int8)
	is_fg = _apply_confirm(is_fg, confirm_windows)
	start: Optional[int] = None
	events: List[Dict] = []
	for i, flag in enumerate(is_fg.tolist()):
		if flag and start is None:
			start = i
		elif (not flag) and start is not None:
			end = i - 1
			dur = float(centers[end] - centers[start])
			if dur >= min_duration:
				cls = int(np.argmax(probs[start:end + 1].sum(axis=0)))
				score = float(fg_prob[start:end + 1].max())
				events.append({"start": float(centers[start]), "end": float(centers[end]), "label": cls, "score": score})
			start = None
	if start is not None:
		end = len(centers) - 1
		dur = float(centers[end] - centers[start])
		if dur >= min_duration:
			cls = int(np.argmax(probs[start:end + 1].sum(axis=0)))
			score = float(fg_prob[start:end + 1].max())
			events.append({"start": float(centers[start]), "end": float(centers[end]), "label": cls, "score": score})
	if cooldown_sec and cooldown_sec > 1e-8:
		events = _merge_with_cooldown(events, cooldown_sec)
	return events


