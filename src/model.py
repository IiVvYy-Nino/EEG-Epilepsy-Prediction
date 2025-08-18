# Source Generated with Decompyle++
# File: model.cpython-311.pyc (Python 3.11)

"""Sequence classifier definition.

Bi-directional LSTM backbone with a per-frame linear classification head that
outputs logits of shape [B, T, C]. Optionally supports packed sequences.
"""
import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.1):
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			bidirectional=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.proj = nn.Linear(hidden_dim * 2, num_classes)

	def forward(self, x, lengths=None):
		"""
		x: [B, T, F]
		lengths: Optional[List[int]] for packing (unused if None)
		return logits [B, T, C]
		"""
		if lengths is not None:
			packed = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
			o, _ = self.lstm(packed)
			o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
		else:
			o, _ = self.lstm(x)
		logits = self.proj(o)
		return logits

