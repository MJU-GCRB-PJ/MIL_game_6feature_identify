from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd


TRAIN_DATA_RATIO = 0.8
CRITERIA_COLUMN = [
 "sexual_content",
 "violence",
 "fear",
 "inappropriate_language",
 "drugs",
 "crime",
]

SEED = 42


@dataclass(frozen=True)
class Paths:
	repo_root: Path
	feat_index_csv: Path
	out_csv: Path
	out_xlsx: Path


def get_paths() -> Paths:
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent.parent
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	from ai.project_paths import FEATURE_INDEX_CSV, TRAINING_SPLIT_CSV, TRAINING_SPLIT_XLSX
	return Paths(
	 repo_root=repo_root,
	 feat_index_csv=FEATURE_INDEX_CSV,
	 out_csv=TRAINING_SPLIT_CSV,
	 out_xlsx=TRAINING_SPLIT_XLSX,
	)


def _read_csv_df(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	last_err: Optional[Exception] = None
	for enc in ("utf-8-sig", "utf-8", "cp949"):
		try:
			return pd.read_csv(path, encoding=enc)
		except Exception as e:
			last_err = e
	raise RuntimeError(f"Failed to read CSV: {path} ({last_err})")


def _coerce_numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
	mat = df[cols].copy()
	for c in cols:
		mat[c] = pd.to_numeric(mat[c], errors="coerce").fillna(0)

	mat = mat.clip(lower=0)
	return mat.to_numpy(dtype=np.float64)


def _weighted_l1_loss(diff: np.ndarray, weights: np.ndarray) -> float:
	return float(np.sum(np.abs(diff) * weights))


def greedy_balanced_split(
 label_matrix: np.ndarray,
 *,
 train_ratio: float,
 seed: int,
) -> np.ndarray:
	"""Return boolean mask (True=train) with size N.

	Goal: make per-column sums in train close to total_sums * train_ratio,
	while matching the desired train size.
	"""
	if not (0.0 < float(train_ratio) < 1.0):
		raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
	if label_matrix.ndim != 2:
		raise ValueError(f"label_matrix must be 2D, got shape={label_matrix.shape}")

	n_rows = int(label_matrix.shape[0])
	if n_rows == 0:
		return np.zeros((0,), dtype=bool)

	rng = np.random.default_rng(int(seed))

	train_target_n = int(np.rint(n_rows * float(train_ratio)))
	train_target_n = int(np.clip(train_target_n, 0, n_rows))
	val_target_n = n_rows - train_target_n

	total_sums = label_matrix.sum(axis=0)
	target_train_sums = total_sums * float(train_ratio)

	# Normalize values.
	weights = 1.0 / np.maximum(total_sums, 1.0)



	col_presence = (label_matrix > 0).sum(axis=0)
	presence_weights = 1.0 / np.maximum(col_presence, 1.0)
	row_score = (label_matrix > 0).astype(np.float64) @ presence_weights

	row_score = row_score + (label_matrix @ weights)

	row_score = row_score + (rng.random(n_rows) * 1e-6)

	order = np.argsort(-row_score)

	train_mask = np.zeros((n_rows,), dtype=bool)
	train_sums = np.zeros((label_matrix.shape[1],), dtype=np.float64)
	train_n = 0
	val_n = 0

	for idx in order:
		assigned = train_n + val_n
		remaining = n_rows - assigned
		train_left = train_target_n - train_n
		val_left = val_target_n - val_n


		if train_left == remaining:
			choose_train = True
		elif val_left == remaining:
			choose_train = False
		else:
			can_train = train_left > 0
			can_val = val_left > 0

			loss_train = float("inf")
			loss_val = float("inf")
			if can_train:
				new_train_sums = train_sums + label_matrix[idx]
				loss_train = _weighted_l1_loss(new_train_sums - target_train_sums, weights)
			if can_val:
				loss_val = _weighted_l1_loss(train_sums - target_train_sums, weights)

			if loss_train < loss_val:
				choose_train = True
			elif loss_val < loss_train:
				choose_train = False
			else:

				choose_train = train_left >= val_left

		if choose_train:
			train_mask[idx] = True
			train_sums += label_matrix[idx]
			train_n += 1
		else:
			val_n += 1

	assert train_n == train_target_n, (train_n, train_target_n)
	assert val_n == val_target_n, (val_n, val_target_n)

	# Optional local improvement: swap train/val rows to reduce loss
	train_mask = _improve_by_swaps(
	 label_matrix,
	 train_mask,
	 target_train_sums=target_train_sums,
	 weights=weights,
	 rng=rng,
	 max_rounds=300,
	 sample_train=100,
	 sample_val=100,
	 patience=40,
	)
	return train_mask


def _improve_by_swaps(
 label_matrix: np.ndarray,
 train_mask: np.ndarray,
 *,
 target_train_sums: np.ndarray,
 weights: np.ndarray,
 rng: np.random.Generator,
 max_rounds: int = 200,
 sample_train: int = 80,
 sample_val: int = 80,
 patience: int = 30,
) -> np.ndarray:
	"""Heuristic local search: swap 1 train row with 1 val row.

	Keeps train/val sizes fixed. Designed to be fast for ~1k rows, ~few columns.
	"""
	n_rows, n_cols = label_matrix.shape
	if n_rows == 0:
		return train_mask
	if int(train_mask.sum()) == 0 or int((~train_mask).sum()) == 0:
		return train_mask

	train_sums = label_matrix[train_mask].sum(axis=0)
	best_loss = _weighted_l1_loss(train_sums - target_train_sums, weights)
	no_improve = 0

	for _ in range(int(max_rounds)):
		train_idx_all = np.flatnonzero(train_mask)
		val_idx_all = np.flatnonzero(~train_mask)
		if len(train_idx_all) == 0 or len(val_idx_all) == 0:
			break

		k_train = int(min(sample_train, len(train_idx_all)))
		k_val = int(min(sample_val, len(val_idx_all)))
		train_sample = rng.choice(train_idx_all, size=k_train, replace=False)
		val_sample = rng.choice(val_idx_all, size=k_val, replace=False)

		val_mat = label_matrix[val_sample]  # (k_val, n_cols)
		round_best_loss = best_loss
		round_best_pair: tuple[int, int] | None = None

		# For each candidate removal from train, find best addition from val
		for i in train_sample:
			base = train_sums - label_matrix[i]  # (n_cols,)
			cand_sums = base + val_mat  # (k_val, n_cols)
			diffs = cand_sums - target_train_sums  # (k_val, n_cols)
			losses = np.sum(np.abs(diffs) * weights, axis=1)  # (k_val,)
			j_pos = int(np.argmin(losses))
			loss = float(losses[j_pos])
			if loss + 1e-12 < round_best_loss:
				round_best_loss = loss
				round_best_pair = (int(i), int(val_sample[j_pos]))

		if round_best_pair is None:
			no_improve += 1
			if no_improve >= int(patience):
				break
			continue

  # Apply swap
		i, j = round_best_pair
		train_mask[i] = False
		train_mask[j] = True
		train_sums = train_sums - label_matrix[i] + label_matrix[j]
		best_loss = round_best_loss
		no_improve = 0

	return train_mask


def print_split_summary(df: pd.DataFrame) -> None:
	train_df = df[df["split"] == "train"]
	val_df = df[df["split"] == "val"]

	train_sums = train_df[CRITERIA_COLUMN].sum(numeric_only=True)
	val_sums = val_df[CRITERIA_COLUMN].sum(numeric_only=True)
	total_sums = df[CRITERIA_COLUMN].sum(numeric_only=True)


	denom = total_sums.replace(0, np.nan)
	train_ratio = (train_sums / denom).fillna(0)
	val_ratio = (val_sums / denom).fillna(0)

	summary = pd.DataFrame(
	 {
	  "total_sum": total_sums,
	  "train_sum": train_sums,
	  "val_sum": val_sums,
	  "train_sum_ratio": train_ratio,
	  "val_sum_ratio": val_ratio,
	  "target_train_ratio": float(TRAIN_DATA_RATIO),
	  "train_ratio_delta": train_ratio - float(TRAIN_DATA_RATIO),
	 }
	)

	print("\n=== Split Summary ===")
	print(f"Total rows: {len(df)}")
	print(f"Train rows: {len(train_df)}")
	print(f"Val rows  : {len(val_df)}")
	print("\nCriteria sums (and ratios vs total):")
	with pd.option_context("display.max_rows", None, "display.max_columns", None):
		print(summary)


def main() -> None:
	paths = get_paths()

	feat_df = _read_csv_df(paths.feat_index_csv)
	if len(feat_df) == 0:
		raise RuntimeError(f"No rows in CSV: {paths.feat_index_csv}")

	missing = [c for c in CRITERIA_COLUMN if c not in feat_df.columns]
	if missing:
		raise KeyError(
		 "feat_index.csv is missing required criteria columns: "
		 + ", ".join(missing)
		 + f" (path={paths.feat_index_csv})"
		)

	label_matrix = _coerce_numeric_matrix(feat_df, CRITERIA_COLUMN)
	train_mask = greedy_balanced_split(label_matrix, train_ratio=TRAIN_DATA_RATIO, seed=SEED)

	out_df = feat_df.copy()
	out_df.insert(0, "split", np.where(train_mask, "train", "val"))

	# Save
	paths.out_csv.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(paths.out_csv, index=False, encoding="utf-8-sig")
	out_df.to_excel(paths.out_xlsx, index=False, engine="openpyxl")

	print(f"WROTE: {paths.out_csv}")
	print(f"WROTE: {paths.out_xlsx}")
	print_split_summary(out_df)


if __name__ == "__main__":
	main()
