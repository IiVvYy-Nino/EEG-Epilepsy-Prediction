# Source Generated with Decompyle++
# File: features.cpython-311.pyc (Python 3.11)

"""Spectral bandpower and simple statistics feature extraction.

Default bands: delta/theta/alpha/beta/gamma. For each time window, compute Welch
PSD per channel, integrate band powers and aggregate {mean, std} across channels
plus broadband energy and time-domain RMS statistics to form frame features.
"""
from typing import Tuple
import numpy as np
from scipy.signal import welch

EEG_BANDS = {
	'delta': (0.5, 4.0),
	'theta': (4.0, 8.0),
	'alpha': (8.0, 13.0),
	'beta': (13.0, 30.0),
	'gamma': (30.0, 45.0),
}


def _bandpower_from_psd(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
	idx = np.logical_and(freqs >= band[0], freqs < band[1])
	if not np.any(idx):
		return np.zeros(psd.shape[:-1], dtype=psd.dtype)
	return np.trapz(psd[..., idx], freqs[idx], axis=-1)


def extract_features_multichannel(
	signals: np.ndarray,
	fs: float,
	window_sec: float,
	hop_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Extract per-window features aggregated across channels.

	Returns (features [T,F], frame_centers_sec [T])
	Base features: for each band {mean, std} across channels + broadband {mean, std}
	Extra features: simple time-domain RMS {mean, std} across channels
	"""
	if signals.ndim != 2:
		raise ValueError("signals must be 2D [C, N]")
	n_channels, n_samples = signals.shape
	win = int(round(window_sec * fs))
	hop = int(round(hop_sec * fs))
	if win <= 0 or hop <= 0 or win > n_samples:
		raise ValueError("Invalid window/hop settings")
	starts = np.arange(0, n_samples - win + 1, hop, dtype=int)
	centers = (starts + win // 2) / float(fs)

	# 🔥 OOM修复：预分配特征数组，避免动态增长
	n_frames = len(starts)
	n_features = len(EEG_BANDS) * 2 + 2 + 2  # 5频带*2 + 宽带*2 + RMS*2 = 14
	X = np.zeros((n_frames, n_features), dtype=np.float32)
	
	# 🔥 内存优化：批量处理而非逐帧
	nperseg = min(win, 256)
	for i, s in enumerate(starts.tolist()):
		seg = signals[:, s:s + win]  # [C, win]
		
		# 🔥 内存优化：使用预分配数组
		psd_array = np.zeros((n_channels, nperseg // 2 + 1), dtype=np.float32)
		
		# PSD per channel - 内存优化版本
		for c in range(n_channels):
			f, p = welch(seg[c], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling='density')
			psd_array[c] = p.astype(np.float32)  # 确保类型一致
		
		# 直接填充特征数组
		feat_idx = 0
		
		# 频带功率 mean/std across channels
		for band in EEG_BANDS.values():
			bp = _bandpower_from_psd(f, psd_array, band)  # [C]
			X[i, feat_idx] = np.mean(bp)
			X[i, feat_idx + 1] = np.std(bp)
			feat_idx += 2
		
		# 宽频功率
		broad = (0.5, 45.0)
		bp_broad = _bandpower_from_psd(f, psd_array, broad)
		X[i, feat_idx] = np.mean(bp_broad)
		X[i, feat_idx + 1] = np.std(bp_broad)
		feat_idx += 2
		
		# 时域 RMS mean/std
		rms = np.sqrt(np.mean(seg.astype(np.float32) ** 2, axis=1))  # [C]
		X[i, feat_idx] = np.mean(rms)
		X[i, feat_idx + 1] = np.std(rms)
		
		# 🔥 内存优化：及时释放临时数组
		del psd_array, seg, rms

	return X, centers

