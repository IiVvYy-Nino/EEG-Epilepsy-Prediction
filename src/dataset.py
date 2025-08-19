# Source Generated with Decompyle++
# File: dataset.cpython-311.pyc (Python 3.11)

"""Sequence dataset per record with disk caching and optional standardization.

Pipeline:
- pair `.edf` and `.tse` via utils.pair_edf_tse;
- read/align/filter/resample signals via edf_reader.load_edf;
- extract per-frame features via features.extract_features_multichannel;
- if label set is provided, align frame labels via tse_parser;
- cache features/labels into `.npz` to avoid recomputation.
"""
import os
import hashlib
from typing import List, Tuple, Dict, Optional
import numpy as np
from .edf_reader import load_edf
from .features import extract_features_multichannel
from .tse_parser import parse_tse, build_frame_labels
from .utils import pair_edf_tse, ensure_dir, save_json, load_json, normalize_label_with_alias


def _cache_key(
	record_id,
	window_sec,
	hop_sec,
	resample_hz=None,
	bandpass=None,
	notch_hz=None,
	notch_q=None,
	label_overlap_ratio=None,
	min_seg_duration=None,
):
	text = (
		f"{record_id}|w{window_sec}|h{hop_sec}|fs{resample_hz}|bp{bandpass}|"
		f"notch{notch_hz}|q{notch_q}|ol{label_overlap_ratio}|mindur{min_seg_duration}|fv3"
	)
	return hashlib.md5(text.encode("utf-8")).hexdigest()


class SequenceDataset:
	"""单条记录为单位的序列数据集，含磁盘缓存与可选标准化。"""

	def __init__(
		self,
		data_dir: str,
		cache_dir: str,
		label_to_index: Optional[Dict[str, int]],
		window_sec: float,
		hop_sec: float,
		resample_hz: float,
		bandpass: Optional[Tuple[Optional[float], Optional[float]]],
		notch_hz: Optional[float] = None,
		notch_q: Optional[float] = None,
		require_labels: bool = False,
		standardize: bool = False,
		stats_path: Optional[str] = None,
		labels_json: Optional[str] = None,
		label_overlap_ratio: float = 0.2,
		min_seg_duration: float = 0.0,
	):
		self.data_dir = data_dir
		self.cache_dir = cache_dir
		ensure_dir(self.cache_dir)
		self.label_to_index = label_to_index
		self.window_sec = window_sec
		self.hop_sec = hop_sec
		self.resample_hz = resample_hz
		self.bandpass = bandpass
		self.notch_hz = notch_hz
		self.notch_q = notch_q
		self.require_labels = require_labels
		self.standardize = standardize
		self.stats_path = stats_path
		self.labels_json = labels_json
		self.label_overlap_ratio = label_overlap_ratio
		self.min_seg_duration = min_seg_duration

		self._aliases: Optional[Dict[str, str]] = None
		self._background: Optional[str] = None
		if self.labels_json and os.path.exists(self.labels_json):
			lab = load_json(self.labels_json)
			self._aliases = {k.lower(): v for k, v in lab.get("aliases", {}).items()}
			self._background = lab.get("background")

		pairs = pair_edf_tse(self.data_dir)
		items = []
		for edf_path, tse_path, rec_id in pairs:
			if require_labels and not tse_path:
				continue
			items.append((edf_path, tse_path, rec_id))
		self.items = items

		self._mean = None
		self._std = None
		if self.standardize and self.stats_path and os.path.exists(self.stats_path):
			stats = load_json(self.stats_path)
			self._mean = np.asarray(stats.get("mean", []), dtype=np.float32) if stats else None
			self._std = np.asarray(stats.get("std", []), dtype=np.float32) if stats else None

	def __len__(self):
		return len(self.items)

	def _cache_path(self, record_id: str) -> str:
		key = _cache_key(
			record_id,
			self.window_sec,
			self.hop_sec,
			self.resample_hz,
			self.bandpass,
			self.notch_hz,
			self.notch_q,
			self.label_overlap_ratio,
			self.min_seg_duration,
		)
		return os.path.join(self.cache_dir, f"{record_id}_{key}.npz")

	def _load_or_compute(self, edf_path: str, tse_path: Optional[str], record_id: str):
		cpath = self._cache_path(record_id)
		if os.path.exists(cpath):
			data = np.load(cpath, allow_pickle=False)
			X = data["X"]
			centers = data["centers"]
			Y = data["Y"] if "Y" in data.files else None
			return X, Y, centers
		# compute
		signals, ch_names, fs = load_edf(
			edf_path,
			resample_hz=self.resample_hz,
			bandpass=self.bandpass,
			notch_hz=self.notch_hz,
			notch_q=self.notch_q or 30.0,
		)
		X, centers = extract_features_multichannel(signals, fs=fs, window_sec=self.window_sec, hop_sec=self.hop_sec)
		Y = None
		if tse_path and self.label_to_index is not None:
			segments = parse_tse(tse_path)
			# 归一化标签别名
			if self._aliases:
				segments = [(s, e, normalize_label_with_alias(lab, self._aliases), conf) for (s, e, lab, conf) in segments]
			bg = list(self.label_to_index.keys())[0]
			if self._background:
				bg = self._background
			# 使用重叠比例阈值与最小时长过滤做帧标注
			labels = build_frame_labels(
				centers.tolist(),
				self.window_sec,
				segments,
				background_label=bg,
				overlap_ratio=float(self.label_overlap_ratio),
				min_seg_duration=float(self.min_seg_duration),
			)
			Y = np.asarray([self.label_to_index.get(lab, -100) for lab in labels], dtype=np.int64)
		np.savez_compressed(cpath, X=X.astype(np.float32), centers=centers.astype(np.float32), Y=Y if Y is not None else np.array([], dtype=np.int64))
		return X, Y, centers

	def _apply_standardize(self, x: np.ndarray) -> np.ndarray:
		if not self.standardize:
			return x
		if self._mean is None or self._std is None or self._mean.size == 0:
			return x
		std = np.where(self._std == 0, 1.0, self._std)
		return (x - self._mean) / std

	def __getitem__(self, idx: int):
		edf_path, tse_path, record_id = self.items[idx]
		X, Y, centers = self._load_or_compute(edf_path, tse_path, record_id)
		X = self._apply_standardize(X)
		item = {"x": X, "centers": centers, "record_id": record_id}
		if Y is not None:
			item["y"] = Y
		return item

	def fit_and_save_stats(self, out_path: str, progress_path: Optional[str] = None):
		"""遍历当前数据集计算特征维度均值与标准差并保存到 JSON。"""
		done = set()
		acc_sum = None
		acc_sq = None
		n = 0
		for edf_path, tse_path, record_id in self.items:
			X, Y, centers = self._load_or_compute(edf_path, tse_path, record_id)
			if acc_sum is None:
				acc_sum = np.zeros((X.shape[1],), dtype=np.float64)
				acc_sq = np.zeros((X.shape[1],), dtype=np.float64)
			acc_sum += X.sum(axis=0)
			acc_sq += (X ** 2).sum(axis=0)
			n += X.shape[0]
		mean = (acc_sum / max(n, 1)).astype(np.float32)
		std = np.sqrt(np.maximum(acc_sq / max(n, 1) - mean.astype(np.float64) ** 2, 0.0)).astype(np.float32)
		save_json({"mean": mean.tolist(), "std": std.tolist()}, out_path)


