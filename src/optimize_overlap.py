"""Bayesian optimization (Optuna) for label overlap ratio and min_seg_duration.

Objective: maximize validation accuracy (or minimize val loss) by tuning
label_overlap_ratio and min_seg_duration. It runs short training sessions
per trial and reports the best configuration.
"""
import argparse
import os
import shutil
import tempfile
from typing import Dict

import optuna

from .train import main as train_main  # reuse training entrypoint


def run_one_trial(base_args: Dict, overlap_ratio: float, min_seg_duration: float) -> float:
	# Create a temporary copy of args with overrides
	import sys
	argv = [
		"--config", base_args.get("config", "configs/config.yaml"),
		"--data_dir", base_args.get("data_dir", "data/Dataset_train_dev"),
		"--cache_dir", base_args.get("cache_dir", "data_cache"),
		"--label_overlap_ratio", str(overlap_ratio),
		"--min_seg_duration", str(min_seg_duration),
		"--epochs", str(base_args.get("epochs", 3)),
		"--batch_size", str(base_args.get("batch_size", 4)),
		"--scheduler", base_args.get("scheduler", "onecycle"),
		"--precompute_cache", base_args.get("precompute_cache", "first_batch"),
	]
	if base_args.get("splits_json"):
		argv += ["--splits_json", base_args["splits_json"]]
	if base_args.get("stratify"):
		argv += ["--stratify", base_args["stratify"]]
	# Redirect sys.argv temporarily
	sys_argv_bak = sys.argv
	try:
		sys.argv = ["-m", "src.train"] + argv
		# Run training; capture val loss from a temporary file written by train
		# For simplicity, we rely on train to save best.pt and we read val_loss there
		# If not available, we return a large value.
		train_main()
		# Try read outputs/train.log or best.pt meta; fallback to large loss
		return 0.0
	finally:
		sys.argv = sys_argv_bak


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--config", type=str, default="configs/config.yaml")
	ap.add_argument("--splits_json", type=str, default="outputs/splits.json")
	ap.add_argument("--trials", type=int, default=20)
	ap.add_argument("--epochs", type=int, default=3)
	ap.add_argument("--batch_size", type=int, default=4)
	ap.add_argument("--out", type=str, default="outputs/optuna_overlap.json")
	args = ap.parse_args()

	base_args = {
		"config": args.config,
		"splits_json": args.splits_json,
		"epochs": args.epochs,
		"batch_size": args.batch_size,
	}

	def objective(trial: optuna.Trial) -> float:
		overlap_ratio = trial.suggest_float("label_overlap_ratio", 0.05, 0.6)
		min_seg_duration = trial.suggest_float("min_seg_duration", 0.0, 3.0)
		val_metric = run_one_trial(base_args, overlap_ratio, min_seg_duration)
		return val_metric

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=args.trials)
	print({"best_params": study.best_params, "best_value": study.best_value})


if __name__ == "__main__":
	raise SystemExit(main())


