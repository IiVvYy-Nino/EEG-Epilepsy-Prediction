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

	rows = []
	for s in starts.tolist():
		seg = signals[:, s:s + win]  # [C, win]
		# PSD per channel
		# nperseg不超过窗口长度；使用汉宁窗
		nperseg = min(win, 256)
		# welch 对二维需逐通道
		psd_list = []
		for c in range(n_channels):
			f, p = welch(seg[c], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, scaling='density')
			psd_list.append(p)
		psd = np.stack(psd_list, axis=0)  # [C, F]

		feats = []
		# 频带功率 mean/std across channels
		for band in EEG_BANDS.values():
			bp = _bandpower_from_psd(f, psd, band)  # [C]
			feats.append(float(np.mean(bp)))
			feats.append(float(np.std(bp)))
		# 宽频功率
		broad = (0.5, 45.0)
		bp_broad = _bandpower_from_psd(f, psd, broad)
		feats.append(float(np.mean(bp_broad)))
		feats.append(float(np.std(bp_broad)))
		# 时域 RMS mean/std
		rms = np.sqrt(np.mean(seg.astype(np.float32) ** 2, axis=1))  # [C]
		feats.append(float(np.mean(rms)))
		feats.append(float(np.std(rms)))
		rows.append(feats)

	X = np.asarray(rows, dtype=np.float32)
	return X, centers

