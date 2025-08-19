"""Leave-One-Subject-Out CV driver.

For each patient ID (parsed from record_id prefix), generate a split where
that patient's records are used as test, and the remaining patients are
patient-level split into train/val using utils.split_records_by_patient.

Optionally, run training per fold.
"""
import argparse
import json
import os
import subprocess
from typing import Dict, List, Tuple, Optional
import yaml
import torch
import optuna

from .utils import ensure_dir, pair_edf_tse, parse_patient_id, split_records_by_patient, load_json


def build_loso_splits(
	data_dir: str,
	val_ratio: float,
	seed: int,
	config_path: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
	"""Return mapping patient_id -> splits dict with record_id lists."""
	pairs = pair_edf_tse(data_dir)
	# Group by patient
	patient_to_records: Dict[str, List[Tuple[str, Optional[str], str]]] = {}
	for edf, tse, rec in pairs:
		pid = parse_patient_id(rec)
		patient_to_records.setdefault(pid, []).append((edf, tse, rec))
	# Build per-patient LOSO splits
	loso: Dict[str, Dict[str, List[str]]] = {}
	# Load optional label config for stratification (background & aliases)
	cfg = {}
	bg = "bckg"
	alias_map: Dict[str, str] = {}
	# Try best-effort to read provided config (or fallback to configs/config.yaml)
	try:
		cfg_path = config_path or os.path.join("configs", "config.yaml")
		if os.path.exists(cfg_path):
			with open(cfg_path, "r", encoding="utf-8") as f:
				cfg = yaml.safe_load(f) or {}
			bg = (cfg.get("train", {}) or {}).get("bg_label", bg)
			lj_path = (cfg.get("labels", {}) or {}).get("json_out")
			if lj_path and os.path.exists(lj_path):
				lj = load_json(lj_path)
				alias_map = {k.lower(): v for k, v in (lj.get("aliases", {}) or {}).items()}
	except Exception:
		pass

	for idx, pid in enumerate(sorted(patient_to_records.keys())):
		# test: all records of this patient
		test_recs = [rec for (_e, _t, rec) in patient_to_records[pid]]
		# remaining pairs for train/val split
		remain: List[Tuple[str, Optional[str], str]] = []
		for other, items in patient_to_records.items():
			if other == pid:
				continue
			remain.extend(items)
		# Patient-level stratified split on remaining: balance per-class presence into val
		# Build patient -> records and patient -> tse list
		patient_to_recs: Dict[str, List[str]] = {}
		patient_to_tses: Dict[str, List[str]] = {}
		for _edf, _tse, _rec in remain:
			pp = parse_patient_id(_rec)
			patient_to_recs.setdefault(pp, []).append(_rec)
			if _tse:
				patient_to_tses.setdefault(pp, []).append(_tse)
		pids = list(patient_to_recs.keys())
		# Build per-patient set of present classes (canonical, excluding background)
		bg_lower = bg.lower()
		patient_classes: Dict[str, set] = {p: set() for p in pids}
		for pp in pids:
			for tp in patient_to_tses.get(pp, []):
				try:
					with open(tp, "r", encoding="utf-8", errors="ignore") as f:
						for line in f:
							parts = line.strip().split()
							if len(parts) >= 3:
								lab = parts[2]
								if lab.lower().startswith("tse_v"):
									continue
								canon = alias_map.get(lab.lower(), lab)
								if canon.lower() == bg_lower:
									continue
								patient_classes[pp].add(canon)
				except Exception:
					pass
		# Totals per class (#patients containing the class)
		total_per_class: Dict[str, int] = {}
		for pp in pids:
			for c in patient_classes[pp]:
				total_per_class[c] = total_per_class.get(c, 0) + 1
		n = len(pids)
		n_val = int(round(n * val_ratio))
		# Targets for val split in patient counts
		target_val = {c: int(round(total_per_class[c] * val_ratio)) for c in total_per_class}
		cur_val = {c: 0 for c in total_per_class}
		# Order patients by rarity-first
		rarity = {c: max(total_per_class.get(c, 1), 1) for c in total_per_class}
		def rarity_score(p: str) -> float:
			labels = patient_classes[p]
			return sum(1.0 / rarity.get(c, 1) for c in labels) if labels else 0.0
		order = sorted(pids, key=lambda p: (-rarity_score(p), len(patient_classes[p])), reverse=False)
		val_ids: List[str] = []
		train_ids: List[str] = []
		for pp in order:
			# compute need for val
			need = 0.0
			for c in patient_classes[pp]:
				need += max(target_val.get(c, 0) - cur_val.get(c, 0), 0)
			# assign to val if we still have capacity and there is need
			if len(val_ids) < n_val and need > 0:
				val_ids.append(pp)
				for c in patient_classes[pp]:
					cur_val[c] = cur_val.get(c, 0) + 1
			else:
				train_ids.append(pp)
		# If val is underfilled due to zero-need patients, move some from train -> val
		if len(val_ids) < n_val:
			move = min(n_val - len(val_ids), len(train_ids))
			val_ids.extend(train_ids[:move])
			train_ids = train_ids[move:]
		# Collect records
		def collect(ids: List[str]) -> List[str]:
			res: List[str] = []
			for p in ids:
				res.extend(patient_to_recs.get(p, []))
			return res
		splits_rem = {"train": collect(train_ids), "val": collect(val_ids)}
		loso[pid] = {
			"train": splits_rem.get("train", []),
			"val": splits_rem.get("val", []),
			"test": test_recs,
		}
	return loso


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data_dir", type=str, default="data/Dataset_train_dev")
	ap.add_argument("--out_dir", type=str, default="outputs/losocv")
	ap.add_argument("--val_ratio", type=float, default=0.2)
	ap.add_argument("--seed", type=int, default=42)
	# optional training
	ap.add_argument("--run_train", action="store_true")
	ap.add_argument("--config", type=str, default="configs/config.yaml")
	ap.add_argument("--epochs", type=int, default=10)
	ap.add_argument("--batch_size", type=int, default=4)
	# optional per-fold Bayesian optimization for label overlap params
	ap.add_argument("--auto_optimize", action="store_true", help="Per-fold Optuna tuning of label_overlap_ratio and min_seg_duration before final training")
	ap.add_argument("--opt_trials", type=int, default=15)
	ap.add_argument("--opt_epochs", type=int, default=3)
	ap.add_argument("--resume", action="store_true", help="Resume training for folds/trials if last.pt exists")
	ap.add_argument("--early_stop_patience", type=int, default=0)
	ap.add_argument("--early_stop_min_delta", type=float, default=0.0)
	ap.add_argument("--early_stop_warmup", type=int, default=0)
	args = ap.parse_args()

	loso = build_loso_splits(args.data_dir, val_ratio=args.val_ratio, seed=args.seed, config_path=args.config)
	ensure_dir(args.out_dir)

	# write per-patient splits
	for pid, sp in loso.items():
		path = os.path.join(args.out_dir, f"{pid}.json")
		with open(path, "w", encoding="utf-8") as f:
			json.dump(sp, f, ensure_ascii=False, indent=2)

	# concise plan summary
	plan = {
		"folds": len(loso),
		"data_dir": args.data_dir,
		"out_dir": args.out_dir,
		"val_ratio": args.val_ratio,
		"seed": args.seed,
		"auto_optimize": bool(args.auto_optimize),
		"opt_trials": (args.opt_trials if args.auto_optimize else None),
		"opt_epochs": (args.opt_epochs if args.auto_optimize else None),
		"epochs": args.epochs,
		"batch_size": args.batch_size,
		"config": args.config,
	}
	print("[LOSO] Plan:", json.dumps(plan))

	if args.run_train:
		for i, pid in enumerate(sorted(loso.keys()), start=1):
			sp_path = os.path.join(args.out_dir, f"{pid}.json")
			fold_out = os.path.join(args.out_dir, f"fold_{pid}")
			ensure_dir(fold_out)
			# brief fold summary
			sp = loso[pid]
			print(f"[Fold {i}/{len(loso)}] PID={pid} | train={len(sp.get('train', []))} val={len(sp.get('val', []))} test={len(sp.get('test', []))}")

			best_params_path = os.path.join(fold_out, "optuna_best.json")
			best_overlap = None
			best_min_dur = None

			def run_trial(trial_idx: int, overlap: float, min_dur: float) -> float:
				trial_out = os.path.join(fold_out, "optuna", f"trial_{trial_idx}")
				ensure_dir(trial_out)
				cmd_trial = [
					"python", "-m", "src.train",
					"--config", args.config,
					"--splits_json", sp_path,
					"--out_dir", trial_out,
					"--epochs", str(args.opt_epochs),
					"--batch_size", str(args.batch_size),
					"--label_overlap_ratio", str(overlap),
					"--min_seg_duration", str(min_dur),
					"--early_stop_patience", str(args.early_stop_patience),
					"--early_stop_min_delta", str(args.early_stop_min_delta),
					"--early_stop_warmup", str(args.early_stop_warmup),
				]
				# resume from last.pt if requested and available
				last_ckpt = os.path.join(trial_out, "last.pt")
				if args.resume and os.path.exists(last_ckpt):
					cmd_trial += ["--resume_from", last_ckpt]
				subprocess.run(cmd_trial, check=False)
				ckpt = os.path.join(trial_out, "best.pt")
				if os.path.exists(ckpt):
					try:
						state = torch.load(ckpt, map_location="cpu")
						return float(state.get("val_loss", 1e9))
					except Exception:
						return 1e9
				return 1e9

			if args.auto_optimize:
				print(f"[Fold {i}/{len(loso)}] PID={pid} -> running Optuna ({args.opt_trials} trials, {args.opt_epochs} epochs/trial)")
				ensure_dir(os.path.join(fold_out, "optuna"))
				trial_counter = {"n": 0}
				def objective(trial: optuna.Trial) -> float:
					trial_counter["n"] += 1
					overlap = trial.suggest_float("label_overlap_ratio", 0.05, 0.6)
					min_dur = trial.suggest_float("min_seg_duration", 0.0, 3.0)
					return run_trial(trial_counter["n"], overlap, min_dur)
				study = optuna.create_study(direction="minimize")
				study.optimize(objective, n_trials=int(args.opt_trials))
				best = {"label_overlap_ratio": study.best_params.get("label_overlap_ratio"), "min_seg_duration": study.best_params.get("min_seg_duration"), "value": study.best_value}
				with open(best_params_path, "w", encoding="utf-8") as bf:
					json.dump(best, bf, ensure_ascii=False, indent=2)
				best_overlap = float(best["label_overlap_ratio"]) if best.get("label_overlap_ratio") is not None else None
				best_min_dur = float(best["min_seg_duration"]) if best.get("min_seg_duration") is not None else None
				print(f"[Fold {i}/{len(loso)}] PID={pid} -> Optuna best: overlap={best_overlap}, min_dur={best_min_dur}, val_loss={best.get('value')}")

			# Final training for the fold (use best params if available)
			cmd = [
				"python", "-m", "src.train",
				"--config", args.config,
				"--splits_json", sp_path,
				"--out_dir", fold_out,
				"--epochs", str(args.epochs),
				"--batch_size", str(args.batch_size),
				"--early_stop_patience", str(args.early_stop_patience),
				"--early_stop_min_delta", str(args.early_stop_min_delta),
				"--early_stop_warmup", str(args.early_stop_warmup),
			]
			# resume from last.pt if requested and available
			last_ckpt_final = os.path.join(fold_out, "last.pt")
			if args.resume and os.path.exists(last_ckpt_final):
				cmd += ["--resume_from", last_ckpt_final]
			if best_overlap is not None:
				cmd += ["--label_overlap_ratio", str(best_overlap)]
			if best_min_dur is not None:
				cmd += ["--min_seg_duration", str(best_min_dur)]
			print(f"[Fold {i}/{len(loso)}] PID={pid} -> final train → {fold_out}")
			subprocess.run(cmd, check=False)

			# Evaluate on this fold's test set and save under the fold directory
			eval_json = os.path.join(fold_out, "eval_summary.json")
			eval_csv = os.path.join(fold_out, "eval_records.csv")
			ecmd = [
				"python", "-m", "src.eval",
				"--config", args.config,
				"--data_dir", args.data_dir,
				"--cache_dir", os.path.join("data_cache"),
				"--checkpoint", os.path.join(fold_out, "best.pt"),
				"--out_json", eval_json,
				"--out_csv", eval_csv,
				"--splits_json", sp_path,
				"--use_split", "test",
			]
			print(f"[Fold {i}/{len(loso)}] PID={pid} -> evaluating test…")
			subprocess.run(ecmd, check=False)
			# brief metric line @ IoU 0.5 if available
			try:
				with open(eval_json, "r", encoding="utf-8") as rf:
					obj = json.load(rf)
					metrics = obj.get("metrics", {}) or {}
					iou_key = "0.5" if "0.5" in metrics else (list(metrics.keys())[0] if metrics else None)
					if iou_key:
						m = metrics[iou_key]
						print(f"[Fold {i}/{len(loso)}] PID={pid} | IoU={iou_key} | F1={m.get('f1', 0):.3f} P={m.get('precision', 0):.3f} R={m.get('recall', 0):.3f} FA/h={m.get('fa_per_h', 0):.3f}")
			except Exception:
				pass

		# After all folds, aggregate metrics across folds (macro and micro averages)
		agg_path = os.path.join(args.out_dir, "loso_eval_aggregate.json")
		try:
			macro = {}
			micro = {}
			micro_accum = {}
			for pid in sorted(loso.keys()):
				fold_out = os.path.join(args.out_dir, f"fold_{pid}")
				fj = os.path.join(fold_out, "eval_summary.json")
				if os.path.exists(fj):
					with open(fj, "r", encoding="utf-8") as f:
						obj = json.load(f)
						metrics = obj.get("metrics", {})
						mtot = obj.get("micro_totals", {})
						for iou_key, vals in metrics.items():
							for k, v in vals.items():
								macro.setdefault(iou_key, {}).setdefault(k, []).append(v)
						# micro totals accumulation
						for iou_key, totals in mtot.items():
							acc = micro_accum.setdefault(iou_key, {"TP":0.0,"FP":0.0,"FN":0.0,"total_seconds":0.0})
							acc["TP"] += float(totals.get("TP", 0.0))
							acc["FP"] += float(totals.get("FP", 0.0))
							acc["FN"] += float(totals.get("FN", 0.0))
							acc["total_seconds"] += float(totals.get("total_seconds", 0.0))
			# macro average (simple mean per fold)
			macro_avg = {}
			for iou_key, kv in macro.items():
				macro_avg[iou_key] = {k: (sum(vs) / max(len(vs), 1)) for k, vs in kv.items()}
			# micro average from accumulated TP/FP/FN and total_seconds
			micro_avg = {}
			for iou_key, acc in micro_accum.items():
				TP = acc.get("TP", 0.0); FP = acc.get("FP", 0.0); FN = acc.get("FN", 0.0)
				seconds = max(acc.get("total_seconds", 0.0), 1e-9)
				prec = TP / max(TP + FP, 1e-9)
				rec = TP / max(TP + FN, 1e-9)
				f1 = (2 * prec * rec / max(prec + rec, 1e-9)) if (prec + rec) > 0 else 0.0
				fa_per_h = FP / (seconds / 3600.0)
				micro_avg[iou_key] = {"precision": prec, "recall": rec, "f1": f1, "fa_per_h": fa_per_h}
			with open(agg_path, "w", encoding="utf-8") as af:
				json.dump({"folds": len(loso), "macro_avg": macro_avg, "micro_avg": micro_avg}, af, ensure_ascii=False, indent=2)
			# brief aggregate line @ IoU 0.5
			iou_key = "0.5" if "0.5" in macro_avg else (next(iter(macro_avg.keys())) if macro_avg else None)
			if iou_key:
				ma = macro_avg[iou_key]; mi = micro_avg.get(iou_key, {})
				print(f"[LOSO] aggregate @ IoU={iou_key} | macro F1={ma.get('f1', 0):.3f} P={ma.get('precision', 0):.3f} R={ma.get('recall', 0):.3f} | micro F1={mi.get('f1', 0):.3f} FA/h={mi.get('fa_per_h', 0):.3f}")
			print({"aggregate_metrics_path": agg_path})
		except Exception:
			pass


if __name__ == "__main__":
	raise SystemExit(main())



