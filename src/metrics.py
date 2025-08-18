"""Event-level metric utilities: IoU matching, PR/F1, FA/h, latencies."""

from typing import List, Dict, Tuple


def interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
	"""IoU between two 1D intervals [a0, a1], [b0, b1]."""
	a0, a1 = float(a[0]), float(a[1])
	b0, b1 = float(b[0]), float(b[1])
	inter = max(0.0, min(a1, b1) - max(a0, b0))
	union = max(a1 - a0, 0.0) + max(b1 - b0, 0.0) - inter
	return inter / union if union > 0 else 0.0


def match_events_by_iou(pred_events: List[Dict], gt_events: List[Dict], iou_thr: float) -> Tuple[int, int, int, List[Tuple[int, int]]]:
	matched_pred = set()
	matched_gt = set()
	pairs: List[Tuple[int, int]] = []
	for i, pe in enumerate(pred_events):
		for j, ge in enumerate(gt_events):
			if j in matched_gt:
				continue
			iou = interval_iou((pe["start"], pe["end"]), (ge["start"], ge["end"]))
			if iou >= iou_thr and pe.get("label") == ge.get("label"):
				matched_pred.add(i)
				matched_gt.add(j)
				pairs.append((i, j))
				break
	tp = len(matched_pred)
	fp = len(pred_events) - tp
	fn = len(gt_events) - len(matched_gt)
	return tp, fp, fn, pairs


def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
	prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
	return prec, rec, f1


def false_alarms_per_hour(num_false: int, total_hours: float) -> float:
	if total_hours <= 0:
		return 0.0
	return float(num_false) / float(total_hours)


def mean_onset_offset_latency(pred_events: List[Dict], gt_events: List[Dict], pairs: List[Tuple[int, int]]) -> Tuple[float, float]:
	"""Compute mean absolute onset/offset latency among matched pairs."""
	if not pairs:
		return 0.0, 0.0
	onsets = []
	offsets = []
	for i, j in pairs:
		pe = pred_events[i]
		ge = gt_events[j]
		onsets.append(abs(float(pe["start"]) - float(ge["start"])))
		offsets.append(abs(float(pe["end"]) - float(ge["end"])))
	mo = sum(onsets) / len(onsets)
	mf = sum(offsets) / len(offsets)
	return mo, mf


