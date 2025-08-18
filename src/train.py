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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import logging

from utils import ensure_dir, parse_patient_id, load_json, scan_label_set, pair_edf_tse
from dataset import SequenceDataset
from model import BiLSTMClassifier

try:
	from labels import build_labels_from_excels, write_labels_json
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
	loss_sum = 0.0
	num = 0
	correct = 0
	count = 0
	with torch.no_grad():
		for x, y, lengths in dl:
			x = x.float()
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
	lam = np.random.beta(alpha, alpha)
	perm = torch.randperm(x.size(0))
	x2 = x[perm]
	y2 = y[perm]
	xm = lam * x + (1 - lam) * x2
	# build soft targets
	B, T = y.shape
	target = torch.zeros((B, T, num_classes), dtype=torch.float32, device=x.device)
	mask = (y != -100)
	idx = y.clone()
	idx[~mask] = 0
	target.scatter_(2, idx.unsqueeze(-1), lam)
	mask2 = (y2 != -100)
	idx2 = y2.clone()
	idx2[~mask2] = 0
	target.scatter_add_(2, idx2.unsqueeze(-1), (1 - lam))
	# positions where both are ignored remain zeros and will be masked later
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
	args = parser.parse_args()

	# Load YAML and override defaults
	cfg = load_config(args.config).get("train", {}) if args.config else {}
	for k, v in cfg.items():
		if hasattr(args, k):
			setattr(args, k, v)

	ensure_dir(args.cache_dir)
	ensure_dir(args.out_dir)
	logger = setup_logger(os.path.join(args.out_dir, "train.log"))
	writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tb"))
	logger.info("Starting training with args: %s", {k: getattr(args, k) for k in vars(args)})

	label_to_index: Dict[str, int] = {args.bg_label: 0}

	# Labels from Excel (optional)
	labels_cfg = cfg.get("labels", {}) if cfg else {}
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
	else:
		labels_json_path = None

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
	)
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
	)
	ds_val.items = val_items

	dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
	dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

	# Infer dims
	x0, y0, lengths0 = next(iter(dl_train))
	input_dim = x0.shape[-1]
	num_classes = len(label_to_index)
	model = BiLSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=1, num_classes=num_classes)
	criterion = nn.CrossEntropyLoss(ignore_index=-100)
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
	start = time.time()
	global_step = 0
	for epoch in range(1, args.epochs + 1):
		model.train()
		for it, (x, y, lengths) in enumerate(dl_train, start=1):
			x = x.float()
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
			optim.zero_grad()
			loss.backward()
			if args.clip_grad and args.clip_grad > 0:
				nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
			optim.step()
			if args.scheduler == "onecycle" and scheduler is not None:
				scheduler.step()
			global_step += 1
			if it % 10 == 0:
				lr = optim.param_groups[0]["lr"]
				logger.info("epoch %d iter %d loss %.4f lr %.2e", epoch, it, loss.item(), lr)
				writer.add_scalar("train/loss", float(loss.item()), global_step)
				writer.add_scalar("train/lr", float(lr), global_step)

		# epoch end
		if args.scheduler == "cosine" and scheduler is not None:
			scheduler.step()

		val_loss, val_acc = evaluate(model, dl_val, criterion)
		logger.info("epoch %d val_loss %.4f val_acc %.4f", epoch, val_loss, val_acc)
		writer.add_scalar("val/loss", float(val_loss), epoch)
		writer.add_scalar("val/acc", float(val_acc), epoch)
		if val_loss < best_val:
			best_val = val_loss
			torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, best_path)
			logger.info("Saved best checkpoint at %s", best_path)

	elapsed = time.time() - start
	logger.info("Training finished in %.1fs", elapsed)
	writer.close()
	print("train finished (with scheduler & augment)")


if __name__ == "__main__":
	raise SystemExit(main())
