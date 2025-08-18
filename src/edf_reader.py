"""EDF I/O and preprocessing utilities.

Provides:
- optional band/high/low-pass filtering;
- optional powerline notch filtering;
- optional resampling to a unified sampling rate;
- alignment/stacking of channels to shape [C, N].
"""

from typing import Tuple, List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly, iirnotch, tf2zpk, zpk2sos, filtfilt
import pyedflib


def _butter_bandpass_sos(
	lowcut: Optional[float],
	highcut: Optional[float],
	fs: float,
	order: int = 4,
) -> Optional[np.ndarray]:
	# 若均为 None 则不构造滤波器
	if lowcut is None and highcut is None:
		return None
	if fs is None:
		raise ValueError("fs must be provided")
	nyq = fs * 0.5
	if lowcut is not None and highcut is not None:
		wn = [lowcut / nyq, highcut / nyq]
		btype = 'band'
	elif lowcut is not None:
		wn = lowcut / nyq
		btype = 'high'
	else:
		wn = highcut / nyq
		btype = 'low'
	return butter(order, wn, btype=btype, output='sos')


def _apply_notch(x: np.ndarray, fs: float, notch_hz: Optional[float], q: float = 30.0) -> np.ndarray:
	# x: [C, N]
	if notch_hz is None:
		return x
	b, a = iirnotch(w0=notch_hz, Q=q, fs=fs)
	return filtfilt(b, a, x, axis=1)


def _ba_to_sos(b: np.ndarray, a: np.ndarray) -> np.ndarray:
	z, p, k = tf2zpk(b, a)
	return zpk2sos(z, p, k)


def _align_channels_to_same_length(channels: List[np.ndarray]) -> np.ndarray:
	if not channels:
		raise ValueError("no channels provided")
	min_len = min(ch.shape[-1] for ch in channels)
	trimmed = [ch[..., :min_len] for ch in channels]
	return np.stack(trimmed, axis=0)


def load_edf(
	filepath: str,
	resample_hz: Optional[float] = None,
	bandpass: Optional[Tuple[Optional[float], Optional[float]]] = None,
	notch_hz: Optional[float] = None,
	notch_q: float = 30.0,
) -> Tuple[np.ndarray, List[str], float]:
	"""读取 EDF 并返回 (signals [C,N], channel_names, fs)。"""
	with pyedflib.EdfReader(filepath) as r:
		n_sig = r.signals_in_file
		ch_names = [r.getLabel(i) for i in range(n_sig)]
		ch_fs = [float(r.getSampleFrequency(i)) for i in range(n_sig)]
		raw = [r.readSignal(i).astype(np.float32) for i in range(n_sig)]

	# 选择目标采样率
	if resample_hz is None:
		values, counts = np.unique(np.asarray(ch_fs), return_counts=True)
		resample_fs = float(values[np.argmax(counts)]) if len(values) else float(ch_fs[0])
	else:
		resample_fs = float(resample_hz)

	lowcut, highcut = (bandpass or (None, None))
	sos = _butter_bandpass_sos(lowcut, highcut, fs=resample_fs) if (lowcut is not None or highcut is not None) else None

	proc_channels: List[np.ndarray] = []
	for x, fs in zip(raw, ch_fs):
		# 去直流
		x = x - float(np.nanmean(x))
		cur_fs = float(fs)
		if abs(cur_fs - resample_fs) > 1e-6:
			# 近似整数比重采样
			num = int(round(resample_fs))
			den = int(round(cur_fs))
			g = np.gcd(num, den) or 1
			x = resample_poly(x, num // g, den // g).astype(np.float32)
		# 形状 [N] -> [1, N] 以便逐通道滤波
		x = x[None, :]
		# 陷波
		x = _apply_notch(x, fs=resample_fs, notch_hz=notch_hz, q=notch_q)
		# 带通/高通/低通
		if sos is not None:
			x = sosfiltfilt(sos, x, axis=1)
		proc_channels.append(x[0])

	signals = _align_channels_to_same_length(proc_channels)
	return signals, ch_names, resample_fs

