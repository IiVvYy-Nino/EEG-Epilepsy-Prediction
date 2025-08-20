"""Training entrypoint for EEG seizure detection and typing.

This script:
- builds label set (optionally from Excel) and validates TSE labels coverage;
- constructs feature-cached datasets and splits by patient for train/val/test;
- trains a BiLSTM frame classifier with optional scheduler and augmentations;
- logs to TensorBoard and saves the best checkpoint to outputs/best.pt.

CLI is usually driven by configs/config.yaml; command-line flags can override.
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml
import logging
from tqdm import tqdm

from .utils import ensure_dir, parse_patient_id, load_json, scan_label_set, pair_edf_tse, save_json
from .dataset import SequenceDataset
from .model import BiLSTMClassifier
from .features import EEG_BANDS

try:
	from .labels import build_labels_from_excels, write_labels_json
except Exception:
	build_labels_from_excels = None
	write_labels_json = None


def collate_batch(batch):
	# batch: list of dict(x[T,F], centers[T], record_id, optional y[T])
	lengths = [b["x"].shape[0] for b in batch]
	max_t = max(lengths)
	feat_dim = batch[0]["x"].shape[1]
	B = len(batch)
	x_pad = np.zeros((B, max_t, feat_dim), dtype=np.float32)
	y_pad = np.full((B, max_t), fill_value=-100, dtype=np.int64)
	for i, b in enumerate(batch):
		t = b["x"].shape[0]
		x_pad[i, :t] = b["x"]
		if "y" in b and b["y"] is not None and b["y"].size:
			y_pad[i, :t] = b["y"]
	return torch.from_numpy(x_pad), torch.from_numpy(y_pad), torch.tensor(lengths, dtype=torch.long)


def load_config(path: str) -> Dict:
	if not path or not os.path.exists(path):
		return {}
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


def setup_logger(log_path: str) -> logging.Logger:
	logger = logging.getLogger("train")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
	ch = logging.StreamHandler()
	ch.setFormatter(fmt)
	logger.addHandler(ch)
	if log_path:
		fh = logging.FileHandler(log_path, encoding="utf-8")
		fh.setFormatter(fmt)
		logger.addHandler(fh)
	return logger


def split_by_patient(items: List[Tuple[str, str, str]], val_ratio: float, test_ratio: float, seed: int = 42):
	# items: (edf_path, tse_path, record_id)
	patients: Dict[str, List[Tuple[str, str, str]]] = {}
	for it in items:
		pid = parse_patient_id(it[2])
		patients.setdefault(pid, []).append(it)
	pids = list(patients.keys())
	rng = np.random.RandomState(seed)
	rng.shuffle(pids)
	n = len(pids)
	n_test = int(round(n * test_ratio))
	n_val = int(round(n * val_ratio))
	val_ids = set(pids[:n_val])
	test_ids = set(pids[n_val:n_val + n_test])
	train_ids = set(pids[n_val + n_test:])
	def collect(idset):
		res: List[Tuple[str, str, str]] = []
		for pid in idset:
			res.extend(patients[pid])
		return res
	return collect(train_ids), collect(val_ids), collect(test_ids)


def evaluate(model: BiLSTMClassifier, dl: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
	model.eval()
	device = next(model.parameters()).device
	loss_sum = 0.0
	num = 0
	correct = 0
	count = 0
	with torch.no_grad():
		for x, y, lengths in dl:
			x = x.float().to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)
			# lengths kept on CPU for packing utilities
			logits = model(x, lengths=lengths)
			B, T, C = logits.shape
			loss = criterion(logits.reshape(B * T, C), y.reshape(B * T))
			loss_sum += float(loss.item())
			num += 1
			pred = torch.argmax(logits, dim=-1)  # [B,T]
			mask = (y != -100)
			correct += int((pred[mask] == y[mask]).sum().item())
			count += int(mask.sum().item())
	avg_loss = loss_sum / max(num, 1)
	acc = (correct / max(count, 1)) if count > 0 else 0.0
	return avg_loss, acc


def _spec_augment(x: torch.Tensor, time_mask_ratio: float, time_masks: int, feat_mask_ratio: float, feat_masks: int) -> torch.Tensor:
	# x: [B, T, F]
	B, T, F = x.shape
	if time_mask_ratio > 0 and time_masks > 0:
		L = max(1, int(round(T * time_mask_ratio)))
		for _ in range(time_masks):
			start = int(torch.randint(low=0, high=max(T - L + 1, 1), size=(1,)).item())
			x[:, start:start + L, :] = 0.0
	if feat_mask_ratio > 0 and feat_masks > 0:
		W = max(1, int(round(F * feat_mask_ratio)))
		for _ in range(feat_masks):
			start = int(torch.randint(low=0, high=max(F - W + 1, 1), size=(1,)).item())
			x[:, :, start:start + W] = 0.0
	return x


def _mixup(x: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float):
	# x: [B,T,F], y: [B,T] with -100 ignored
	if alpha <= 0 or x.size(0) < 2:
		return x, y, None  # no soft targets
	lam = float(np.random.beta(alpha, alpha))
	perm = torch.randperm(x.size(0))
	x2 = x[perm]
	y2 = y[perm]
	xm = lam * x + (1 - lam) * x2
	# build soft targets via two one-hot maps then linear combination
	B, T = y.shape
	idx = y.clone(); mask = (y != -100); idx[~mask] = 0
	idx2 = y2.clone(); mask2 = (y2 != -100); idx2[~mask2] = 0
	t1 = torch.zeros((B, T, num_classes), dtype=torch.float32, device=x.device)
	t2 = torch.zeros((B, T, num_classes), dtype=torch.float32, device=x.device)
	t1.scatter_(2, idx.unsqueeze(-1), 1.0)
	t2.scatter_(2, idx2.unsqueeze(-1), 1.0)
	target = lam * t1 + (1 - lam) * t2
	# positions where y is ignored will be masked later in loss
	return xm, y, target


def _soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor, ignore_mask: torch.Tensor) -> torch.Tensor:
	# logits: [B,T,C], soft_targets: [B,T,C], ignore_mask: [B,T] (True keep)
	logp = torch.log_softmax(logits, dim=-1)
	loss = -(soft_targets * logp).sum(dim=-1)  # [B,T]
	loss = loss[ignore_mask].mean() if ignore_mask.any() else loss.mean()
	return loss


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/config.yaml")
	parser.add_argument("--data_dir", type=str, default="data/Dataset_train_dev")
	parser.add_argument("--cache_dir", type=str, default="data_cache")
	parser.add_argument("--window_sec", type=float, default=2.0)
	parser.add_argument("--hop_sec", type=float, default=0.25)
	parser.add_argument("--resample_hz", type=float, default=256.0)
	parser.add_argument("--bandpass", type=float, nargs=2, default=[0.5, 45.0])
	parser.add_argument("--notch_hz", type=float, default=50.0)
	parser.add_argument("--bg_label", type=str, default="bckg")
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--val_ratio", type=float, default=0.2)
	parser.add_argument("--test_ratio", type=float, default=0.0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--out_dir", type=str, default="outputs")
	# dataloader
	parser.add_argument("--num_workers", type=int, default=0)
	# scheduler
	parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "onecycle"], default="none")
	parser.add_argument("--max_lr", type=float, default=1e-3)
	parser.add_argument("--clip_grad", type=float, default=0.0)
	# augment
	parser.add_argument("--mixup_alpha", type=float, default=0.0)
	parser.add_argument("--spec_time_mask_ratio", type=float, default=0.0)
	parser.add_argument("--spec_time_masks", type=int, default=0)
	parser.add_argument("--spec_feat_mask_ratio", type=float, default=0.0)
	parser.add_argument("--spec_feat_masks", type=int, default=0)
	parser.add_argument("--aug_noise_std", type=float, default=0.0)
	# optional: estimate class weights from a few batches before training (disabled by default for speed)
	parser.add_argument("--estimate_class_weights", action="store_true")
	# logging/progress
	parser.add_argument("--log_interval", type=int, default=0)
	parser.add_argument("--progress", type=str, choices=["none", "bar"], default="none")
	# evaluation control
	parser.add_argument("--eval_at_start", action="store_true")
	# label alignment thresholds
	parser.add_argument("--label_overlap_ratio", type=float, default=0.2)
	parser.add_argument("--min_seg_duration", type=float, default=0.0)
	# precompute cache to avoid long first batch stall
	parser.add_argument("--precompute_cache", type=str, choices=["none", "first_batch", "all"], default="first_batch")
	# fixed splits & stratification
	parser.add_argument("--splits_json", type=str, default=None)
	parser.add_argument("--stratify", type=str, choices=["none", "has_seizure", "multiclass"], default="multiclass")
	# optional warm-start from checkpoint (use with care to avoid leakage in LOSO)
	parser.add_argument("--init_checkpoint", type=str, default=None)
	# resume from last checkpoint (model+optimizer+scheduler+epoch)
	parser.add_argument("--resume_from", type=str, default=None)
	# early stopping
	parser.add_argument("--early_stop_patience", type=int, default=0, help="Epochs without val improvement before stop; 0 disables")
	parser.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum improvement in val_loss to reset patience")
	parser.add_argument("--early_stop_warmup", type=int, default=0, help="Minimum epochs before early stopping can trigger")
	args = parser.parse_args()

	# Prefer 'spawn' when using multiple workers (safer with CUDA and heavy C-extensions)
	try:
		import torch.multiprocessing as mp  # type: ignore
		if (args.num_workers and args.num_workers > 0):
			cur = mp.get_start_method(allow_none=True)
			if cur != "spawn":
				mp.set_start_method("spawn", force=True)
	except Exception:
		pass

	# Device selection (auto CUDA if available)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# human-readable device description
	if device.type == "cuda":
		try:
			_idx = torch.cuda.current_device()
			_name = torch.cuda.get_device_name(_idx)
			device_desc = f"cuda:{_idx} ({_name})"
		except Exception:
			device_desc = "cuda"
	else:
		device_desc = "cpu"

	# Load YAML and override defaults
	# Load YAML; let CLI override config: only fill args still at parser defaults.
	cfg_all = load_config(args.config) if args.config else {}
	cfg_train = cfg_all.get("train", {}) if cfg_all else {}
	if cfg_train:
		defaults = {k: parser.get_default(k) for k in vars(args)}
		for k, v in cfg_train.items():
			if hasattr(args, k) and getattr(args, k) == defaults.get(k):
				setattr(args, k, v)

	ensure_dir(args.cache_dir)
	ensure_dir(args.out_dir)
	logger = setup_logger(os.path.join(args.out_dir, "train.log"))
	writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))
	logger.info("Using device: %s", device_desc)
	logger.info("Starting training with args: %s", {k: getattr(args, k) for k in vars(args)})

	label_to_index: Dict[str, int] = {args.bg_label: 0}

	# Labels from Excel or existing labels.json (optional)
	labels_cfg = cfg_all.get("labels", {}) if cfg_all else {}
	labels_json_path = None
	if labels_cfg and build_labels_from_excels is not None:
		bg = labels_cfg.get("background", args.bg_label)
		types_xlsx = labels_cfg.get("excel_types")
		periods_xlsx = labels_cfg.get("excel_periods")
		json_out = labels_cfg.get("json_out", os.path.join(args.out_dir, "labels.json"))
		try:
			label_names, aliases = build_labels_from_excels(types_xlsx, periods_xlsx, background=bg)
			label_to_index = {name: i for i, name in enumerate(label_names)}
			if write_labels_json is not None:
				write_labels_json(json_out, label_names, aliases, background=bg)
			labels_json_path = json_out
		except Exception:
			labels_json_path = None
	# Fallback: load existing labels.json if provided and exists
	if (labels_json_path is None) and labels_cfg:
		lj_path = labels_cfg.get("json_out")
		if lj_path and os.path.exists(lj_path):
			lj = load_json(lj_path)
			ln = (lj.get("label_names", []) or [])
			# ensure background at index 0
			if args.bg_label not in ln:
				label_names = [args.bg_label] + [n for n in ln if n != args.bg_label]
			else:
				label_names = [args.bg_label] + [n for n in ln if n != args.bg_label]
			label_to_index = {name: i for i, name in enumerate(label_names)}
			labels_json_path = lj_path

	# Early label consistency check before dataset split
	try:
		pairs = pair_edf_tse(args.data_dir)
		tse_paths = [t for (_e, t, _r) in pairs if t]
		if tse_paths:
			alias_map: Dict[str, str] = {}
			if labels_json_path and os.path.exists(labels_json_path):
				lj = load_json(labels_json_path)
				alias_map = {k.lower(): v for k, v in (lj.get("aliases", {}) or {}).items()}
			present = scan_label_set(tse_paths, args.bg_label)
			known = set(label_to_index.keys())
			unknown: List[str] = []
			for lab in present:
				if lab.lower() == args.bg_label.lower():
					continue
				canon = alias_map.get(lab.lower(), lab)
				if canon not in known:
					unknown.append(lab)
			if unknown:
				auto_from_tse = bool(cfg_train.get("auto_labels_from_tse", False))
				if auto_from_tse and (not labels_cfg or not (labels_cfg.get("excel_types") or labels_cfg.get("json_out"))):
					# Derive label set from TSE present labels
					canon_set = []
					for lab in present:
						if lab.lower() == args.bg_label.lower():
							continue
						canon = alias_map.get(lab.lower(), lab)
						if canon not in canon_set:
							canon_set.append(canon)
					label_names = [args.bg_label] + sorted(canon_set)
					label_to_index = {name: i for i, name in enumerate(label_names)}
					logger.info("Auto-derived label set from TSE: %s", label_names)
				else:
					raise RuntimeError(
						"Found labels in TSE not covered by training label set: "
						f"{sorted(set(unknown))}. Configure labels via configs/labels or provide labels.json."
					)
	except Exception as e:
		# If directory missing or other non-fatal issues, continue and let later steps surface errors
		if isinstance(e, RuntimeError):
			raise

	ds_full = SequenceDataset(
		data_dir=args.data_dir,
		cache_dir=args.cache_dir,
		label_to_index=label_to_index,
		window_sec=args.window_sec,
		hop_sec=args.hop_sec,
		resample_hz=args.resample_hz,
		bandpass=(args.bandpass[0], args.bandpass[1]),
		notch_hz=args.notch_hz,
		require_labels=False,
		standardize=False,
		labels_json=labels_json_path,
		label_overlap_ratio=args.label_overlap_ratio,
		min_seg_duration=args.min_seg_duration,
	)
	# Use fixed splits if provided; otherwise random split by patient.
	# If splits_json path is provided but missing, auto-create and save to that path.
	if args.splits_json:
		id2item = {rec: it for (edf, tse, rec) in ds_full.items for it in [(edf, tse, rec)]}
		if os.path.exists(args.splits_json):
			sp = load_json(args.splits_json)
		else:
			logger.info("splits_json not found. Auto-creating patient-level split → %s", args.splits_json)
			# build by patient from current items with optional stratification
			all_items = ds_full.items
			# map patient -> records and tse paths
			patient_to_records: Dict[str, List[str]] = {}
			patient_to_tses: Dict[str, List[str]] = {}
			for _edf, _tse, rec in all_items:
				pid = parse_patient_id(rec)
				patient_to_records.setdefault(pid, []).append(rec)
				if _tse:
					patient_to_tses.setdefault(pid, []).append(_tse)
			pids = list(patient_to_records.keys())
			rng = np.random.RandomState(args.seed)
			rng.shuffle(pids)
			if args.stratify == "has_seizure":
				# classify patients by whether any non-background label appears in their TSEs
				alias_map: Dict[str, str] = {}
				if labels_json_path and os.path.exists(labels_json_path):
					lj = load_json(labels_json_path)
					alias_map = {k.lower(): v for k, v in (lj.get("aliases", {}) or {}).items()}
				bg_lower = args.bg_label.lower()
				seizure_pids: List[str] = []
				background_pids: List[str] = []
				for pid in pids:
					paths = patient_to_tses.get(pid, [])
					found_seizure = False
					for tp in paths:
						try:
							with open(tp, "r", encoding="utf-8", errors="ignore") as f:
								for line in f:
									parts = line.strip().split()
									if len(parts) >= 3:
										lab = parts[2]
										lab_canon = alias_map.get(lab.lower(), lab)
										if lab_canon.lower() != bg_lower:
											found_seizure = True
											break
						except Exception:
							pass
					if found_seizure:
						seizure_pids.append(pid)
					else:
						background_pids.append(pid)
				# shuffle groups deterministically
				rng.shuffle(seizure_pids)
				rng.shuffle(background_pids)
				def slice_group(group: List[str], val_r: float, test_r: float) -> Tuple[set, set, set]:
					n = len(group)
					n_test = int(round(n * test_r))
					n_val = int(round(n * val_r))
					val_ids = set(group[:n_val])
					test_ids = set(group[n_val:n_val + n_test])
					train_ids = set(group[n_val + n_test:])
					return train_ids, val_ids, test_ids
				tr_a, va_a, te_a = slice_group(seizure_pids, args.val_ratio, args.test_ratio)
				tr_b, va_b, te_b = slice_group(background_pids, args.val_ratio, args.test_ratio)
				train_ids = tr_a.union(tr_b)
				val_ids = va_a.union(va_b)
				test_ids = te_a.union(te_b)
			elif args.stratify == "multiclass":
				# Multi-class patient-level stratification by class presence vector
				# Build canonical label list (exclude background)
				label_names = list(label_to_index.keys())
				classes = [ln for ln in label_names if ln.lower() != args.bg_label.lower()]
				if not classes:
					# fallback
					n = len(pids)
					n_test = int(round(n * args.test_ratio))
					n_val = int(round(n * args.val_ratio))
					val_ids = set(pids[:n_val])
					test_ids = set(pids[n_val:n_val + n_test])
					train_ids = set(pids[n_val + n_test:])
				else:
					# Build per-patient set of present classes
					alias_map: Dict[str, str] = {}
					if labels_json_path and os.path.exists(labels_json_path):
						lj = load_json(labels_json_path)
						alias_map = {k.lower(): v for k, v in (lj.get("aliases", {}) or {}).items()}
					bg_lower = args.bg_label.lower()
					patient_classes: Dict[str, set] = {pid: set() for pid in pids}
					for pid in pids:
						for tp in patient_to_tses.get(pid, []):
							try:
								with open(tp, "r", encoding="utf-8", errors="ignore") as f:
									for line in f:
										parts = line.strip().split()
										if len(parts) >= 3:
											lab = parts[2]
											if lab.lower().startswith("tse_v"):
												continue
											lab_canon = alias_map.get(lab.lower(), lab)
											if lab_canon.lower() == bg_lower:
												continue
											# only keep labels present in training label set
											if lab_canon not in label_to_index:
												continue
											patient_classes[pid].add(lab_canon)
							except Exception:
								pass
					# total counts per class across patients
					total_per_class: Dict[str, int] = {c: 0 for c in classes}
					for pid in pids:
						for c in patient_classes[pid]:
							if c in total_per_class:
								total_per_class[c] += 1
					# per split targets
					r_train = 1.0 - (args.val_ratio + args.test_ratio)
					r_val = args.val_ratio
					r_test = args.test_ratio
					target_train = {c: int(round(total_per_class[c] * r_train)) for c in classes}
					target_val = {c: int(round(total_per_class[c] * r_val)) for c in classes}
					target_test = {c: int(round(total_per_class[c] * r_test)) for c in classes}
					# split capacities in patients count
					n = len(pids)
					n_test = int(round(n * args.test_ratio))
					n_val = int(round(n * args.val_ratio))
					cap = {"train": n - n_val - n_test, "val": n_val, "test": n_test}
					# current counts
					cur = {"train": {c: 0 for c in classes}, "val": {c: 0 for c in classes}, "test": {c: 0 for c in classes}}
					sizes = {"train": 0, "val": 0, "test": 0}
					# patient order: prioritize rare-class holders
					rarity = {c: max(total_per_class[c], 1) for c in classes}
					def rarity_score(pid: str) -> float:
						return sum(1.0 / rarity.get(c, 1) for c in patient_classes[pid])
					order = sorted(pids, key=lambda p: (-rarity_score(p), len(patient_classes[p])), reverse=False)
					assign: Dict[str, str] = {}
					for pid in order:
						labels = list(patient_classes[pid])
						# compute need score per split
						def need(split: str) -> float:
							need_sum = 0.0
							for c in labels:
								if split == "train":
									target = target_train[c]
									curc = cur["train"][c]
								elif split == "val":
									target = target_val[c]
									curc = cur["val"][c]
								else:
									target = target_test[c]
									curc = cur["test"][c]
								need_sum += max(target - curc, 0)
							return need_sum
						candidates = [s for s in ["train", "val", "test"] if sizes[s] < cap[s]] or ["train", "val", "test"]
						best_split = max(candidates, key=lambda s: (need(s), -sizes[s]))
						assign[pid] = best_split
						sizes[best_split] += 1
						for c in labels:
							cur[best_split][c] += 1
					train_ids = {p for p, s in assign.items() if s == "train"}
					val_ids = {p for p, s in assign.items() if s == "val"}
					test_ids = {p for p, s in assign.items() if s == "test"}
			else:
				# plain random by patient
				n = len(pids)
				n_test = int(round(n * args.test_ratio))
				n_val = int(round(n * args.val_ratio))
				val_ids = set(pids[:n_val])
				test_ids = set(pids[n_val:n_val + n_test])
				train_ids = set(pids[n_val + n_test:])
			def collect(idset):
				res = []
				for pid in idset:
					res.extend(patient_to_records[pid])
				return res
			sp = {"train": collect(train_ids), "val": collect(val_ids), "test": collect(test_ids)}
			ensure_dir(os.path.dirname(args.splits_json) or ".")
			save_json(sp, args.splits_json)
			logger.info("Created splits: train=%d val=%d test=%d", len(sp.get("train", [])), len(sp.get("val", [])), len(sp.get("test", [])))
		train_items = [id2item[r] for r in (sp.get("train", []) or []) if r in id2item]
		val_items = [id2item[r] for r in (sp.get("val", []) or []) if r in id2item]
		_ = [id2item[r] for r in (sp.get("test", []) or []) if r in id2item]
		if not train_items or not val_items:
			train_items, val_items, _ = split_by_patient(ds_full.items, args.val_ratio, args.test_ratio, seed=args.seed)
	else:
		train_items, val_items, _ = split_by_patient(ds_full.items, args.val_ratio, args.test_ratio, seed=args.seed)
	# Create shallow dataset views by overriding items
	ds_train = ds_full
	ds_train.items = train_items
	ds_val = SequenceDataset(
		data_dir=args.data_dir,
		cache_dir=args.cache_dir,
		label_to_index=label_to_index,
		window_sec=args.window_sec,
		hop_sec=args.hop_sec,
		resample_hz=args.resample_hz,
		bandpass=(args.bandpass[0], args.bandpass[1]),
		notch_hz=args.notch_hz,
		require_labels=False,
		standardize=False,
		labels_json=labels_json_path,
		label_overlap_ratio=args.label_overlap_ratio,
		min_seg_duration=args.min_seg_duration,
	)
	ds_val.items = val_items

	pin_mem = bool(torch.cuda.is_available())
	dl_train = DataLoader(
		ds_train,
		batch_size=args.batch_size,
		shuffle=True,
		collate_fn=collate_batch,
		num_workers=args.num_workers,
		pin_memory=pin_mem,
		prefetch_factor=1 if args.num_workers and args.num_workers > 0 else None,
		persistent_workers=False,
	)
	dl_val = DataLoader(
		ds_val,
		batch_size=args.batch_size,
		shuffle=False,
		collate_fn=collate_batch,
		num_workers=args.num_workers,
		pin_memory=pin_mem,
		prefetch_factor=1 if args.num_workers and args.num_workers > 0 else None,
		persistent_workers=False,
	)

	# Optional: precompute features to warm up cache and show progress
	if args.precompute_cache != "none":
		if args.precompute_cache == "all":
			warm_n = len(ds_train.items)
		else:
			warm_n = min(len(ds_train.items), args.batch_size)
		logger.info("Precomputing features for %d record(s) to warm up cache...", warm_n)
		# Always build cache sequentially to avoid worker boot/OMP contention during prewarm
		for i in tqdm(range(warm_n), desc="Building cache", unit="rec"):
			_ = ds_train[i]

	# Infer feature dimension without touching data (fixed by feature design)
	input_dim = int(len(EEG_BANDS) * 2 + 2 + 2)  # bands{mean,std} + broadband{mean,std} + RMS{mean,std}
	num_classes = len(label_to_index)
	model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=1, num_classes=num_classes)
	model = model.to(device)
	# Warm start
	if args.init_checkpoint and os.path.exists(args.init_checkpoint):
		try:
			ckpt = torch.load(args.init_checkpoint, map_location="cpu")
			state = ckpt.get("model", ckpt)
			# best-effort load
			model.load_state_dict(state, strict=False)
			logger.info("Loaded init checkpoint from %s", args.init_checkpoint)
		except Exception as e:
			logger.info("Failed to load init checkpoint %s: %s", args.init_checkpoint, e)
	# class weights (optional): compute inverse-frequency weights from a small sample
	weights = None
	if args.estimate_class_weights:
		try:
			logger.info("Estimating class weights from a few batches…")
			# sample a few batches to estimate class freq
			freq = np.zeros((num_classes,), dtype=np.int64)
			cnt = 0
			for _i, (_x, _y, _l) in zip(range(3), dl_train):
				mask = (_y != -100)
				vals, counts = torch.unique(_y[mask], return_counts=True)
				for v, c in zip(vals.tolist(), counts.tolist()):
					freq[int(v)] += int(c)
				cnt += 1
			if cnt > 0 and freq.sum() > 0:
				f = freq.astype(np.float64)
				inv = (1.0 / np.maximum(f, 1.0))
				w = inv / inv.sum() * num_classes
				weights = torch.tensor(w, dtype=torch.float32)
		except Exception:
			weights = None
	criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
	criterion = criterion.to(device)
	optim = torch.optim.Adam(model.parameters(), lr=args.max_lr)

	# Scheduler
	scheduler = None
	steps_per_epoch = max(1, len(dl_train))
	if args.scheduler == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
	elif args.scheduler == "onecycle":
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.max_lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)

	best_val = float("inf")
	best_path = os.path.join(args.out_dir, "best.pt")
	last_path = os.path.join(args.out_dir, "last.pt")
	start = time.time()
	global_step = 0

	# Optionally resume training state
	resume_epoch = 0
	if args.resume_from and os.path.exists(args.resume_from):
		try:
			ckpt = torch.load(args.resume_from, map_location="cpu")
			state = ckpt.get("model", ckpt)
			model.load_state_dict(state, strict=False)
			if "optimizer" in ckpt:
				optim.load_state_dict(ckpt["optimizer"])
			if scheduler is not None and ("scheduler" in ckpt):
				scheduler.load_state_dict(ckpt["scheduler"])
			resume_epoch = int(ckpt.get("epoch", 0))
			best_val = float(ckpt.get("best_val", best_val))
			global_step = int(ckpt.get("global_step", global_step))
			logger.info("Resumed from %s @ epoch %d (best_val=%.6f)", args.resume_from, resume_epoch, best_val)
		except Exception as e:
			logger.info("Failed to resume from %s: %s", args.resume_from, e)

	# Optional baseline evaluation before any training
	if args.eval_at_start:
		val_loss0, val_acc0 = evaluate(model, dl_val, criterion)
		is_best0 = val_loss0 < best_val
		logger.info(
			"Epoch %d/%d | train_loss: %s | val_loss: %.4f | val_acc: %.4f | lr: %.2e | best: %s",
			0, args.epochs, "-", val_loss0, val_acc0, optim.param_groups[0]["lr"], "✓" if is_best0 else "-",
		)
		writer.add_scalar("val/loss", float(val_loss0), 0)
		writer.add_scalar("val/acc", float(val_acc0), 0)
		if is_best0:
			best_val = val_loss0
			torch.save({"model": model.state_dict(), "epoch": 0, "val_loss": val_loss0}, best_path)
			logger.info("Best checkpoint updated → %s", best_path)
	for epoch in range(max(1, resume_epoch + 1), args.epochs + 1):
		model.train()
		epoch_loss_sum = 0.0
		epoch_loss_count = 0
		for it, (x, y, lengths) in enumerate(dl_train, start=1):
			x = x.float().to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)
			# augment: gaussian noise
			if args.aug_noise_std > 0:
				x = x + torch.randn_like(x) * args.aug_noise_std
			# SpecAugment style masks
			if args.spec_time_masks > 0 or args.spec_feat_masks > 0:
				x = _spec_augment(x, args.spec_time_mask_ratio, args.spec_time_masks, args.spec_feat_mask_ratio, args.spec_feat_masks)
			# Mixup
			xm, y_orig, soft_targets = _mixup(x, y, num_classes, args.mixup_alpha)
			logits = model(xm, lengths=lengths)
			B, T, C = logits.shape
			if soft_targets is not None:
				ignore_mask = (y_orig != -100)
				loss = _soft_ce_loss(logits, soft_targets, ignore_mask)
			else:
				loss = criterion(logits.reshape(B * T, C), y.reshape(B * T))
			epoch_loss_sum += float(loss.item())
			epoch_loss_count += 1
			optim.zero_grad()
			loss.backward()
			if args.clip_grad and args.clip_grad > 0:
				nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
			optim.step()
			if args.scheduler == "onecycle" and scheduler is not None:
				scheduler.step()
			global_step += 1

		# epoch end
		if args.scheduler == "cosine" and scheduler is not None:
			scheduler.step()

		# once per epoch logging (pretty one-line summary)
		avg_train = (epoch_loss_sum / max(epoch_loss_count, 1)) if epoch_loss_count > 0 else 0.0
		lr = optim.param_groups[0]["lr"]
		val_loss, val_acc = evaluate(model, dl_val, criterion)
		is_best = val_loss < best_val
		logger.info(
			"Epoch %d/%d | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f | lr: %.2e | best: %s",
			epoch, args.epochs, avg_train, val_loss, val_acc, lr, "✓" if is_best else "-",
		)
		writer.add_scalar("train/epoch_avg_loss", float(avg_train), epoch)
		writer.add_scalar("train/lr", float(lr), epoch)
		writer.add_scalar("val/loss", float(val_loss), epoch)
		writer.add_scalar("val/acc", float(val_acc), epoch)
		if is_best:
			best_val = val_loss
			torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "best_val": best_val}, best_path)
			logger.info("Best checkpoint updated → %s", best_path)

		# Early stopping check
		if not hasattr(main, "_no_improve"):
			setattr(main, "_no_improve", 0)
		no_improve = getattr(main, "_no_improve")
		if is_best and (best_val + args.early_stop_min_delta <= val_loss):
			# this branch logically won't hit because best_val was updated; keep for clarity
			setattr(main, "_no_improve", 0)
		elif is_best:
			setattr(main, "_no_improve", 0)
		else:
			setattr(main, "_no_improve", no_improve + 1)
		no_improve = getattr(main, "_no_improve")
		if args.early_stop_patience > 0 and epoch >= args.early_stop_warmup:
			if no_improve >= args.early_stop_patience:
				logger.info("Early stopping triggered at epoch %d (no improvement for %d epochs)", epoch, no_improve)
				# save last state before break
				to_save = {
					"model": model.state_dict(),
					"optimizer": optim.state_dict(),
					"epoch": epoch,
					"best_val": best_val,
					"global_step": global_step,
				}
				if scheduler is not None:
					to_save["scheduler"] = scheduler.state_dict()
				torch.save(to_save, last_path)
				break
		# always save last checkpoint for resume
		to_save = {
			"model": model.state_dict(),
			"optimizer": optim.state_dict(),
			"epoch": epoch,
			"best_val": best_val,
			"global_step": global_step,
		}
		if scheduler is not None:
			to_save["scheduler"] = scheduler.state_dict()
		torch.save(to_save, last_path)

	elapsed = time.time() - start
	logger.info("Training finished in %.1fs", elapsed)
	writer.close()
	print("train finished (with scheduler & augment)")


if __name__ == "__main__":
	raise SystemExit(main())
