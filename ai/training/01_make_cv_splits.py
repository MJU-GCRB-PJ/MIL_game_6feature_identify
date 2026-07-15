from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
	if str(import_path) not in sys.path:
		sys.path.insert(0, str(import_path))

from ai.data_manifest import read_data_manifest  # noqa: E402
from ai.project_paths import DATA_LIST_XLSX, FEATURE_INDEX_CSV  # noqa: E402
from cv_config import (  # noqa: E402
	BASE_SEED,
	CLASS_NAMES,
	CV_OUTPUT_DIR,
	EXPECTED_SAMPLE_COUNT,
	N_FOLDS,
	fold_data_csv,
	fold_dir,
)


SEED = BASE_SEED
CLASS_NAME = list(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create the deterministic paper-aligned cross-validation splits."
	)
	parser.add_argument("--manifest", type=Path, default=DATA_LIST_XLSX)
	parser.add_argument("--feature-index", type=Path, default=FEATURE_INDEX_CSV)
	parser.add_argument("--output-root", type=Path, default=CV_OUTPUT_DIR)
	parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLE_COUNT)
	parser.add_argument(
		"--assignments-only",
		action="store_true",
		help="Write fold assignments and summary without materializing feature-index data files.",
	)
	return parser.parse_args()


def read_csv_df(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	last_err: Optional[Exception] = None
	for enc in ("utf-8-sig", "utf-8", "cp949"):
		try:
			return pd.read_csv(path, encoding=enc)
		except Exception as e:
			last_err = e
	raise RuntimeError(f"Failed to read CSV: {path} ({last_err})")


def coerce_label_matrix(df: pd.DataFrame) -> np.ndarray:
	missing = [c for c in CLASS_NAME if c not in df.columns]
	if missing:
		raise KeyError(f"Missing label columns: {missing}")

	mat = df[CLASS_NAME].copy()
	for c in CLASS_NAME:
		numeric = pd.to_numeric(mat[c], errors="coerce")
		invalid = numeric.isna() | ~numeric.isin([0, 1])
		if invalid.any():
			raise ValueError(f"Label column '{c}' must contain only 0 or 1")
		mat[c] = numeric
	return mat.to_numpy(dtype=np.float64)


def target_fold_sizes(n_rows: int, n_folds: int) -> list[int]:
	base = int(n_rows) // int(n_folds)
	rem = int(n_rows) % int(n_folds)
	return [base + (1 if i < rem else 0) for i in range(int(n_folds))]


def make_balanced_fold_ids(
	label_matrix: np.ndarray,
	*,
	n_folds: int = N_FOLDS,
	seed: int = SEED,
) -> np.ndarray:
	"""Return 0-based validation fold id for every row.

	The heuristic keeps validation fold sizes fixed and greedily minimizes the
	per-class label-sum distance from the ideal 1/K validation distribution.
	"""
	if label_matrix.ndim != 2:
		raise ValueError(f"label_matrix must be 2D, got shape={label_matrix.shape}")
	n_rows = int(label_matrix.shape[0])
	if n_rows == 0:
		return np.zeros((0,), dtype=np.int64)
	if n_folds < 2:
		raise ValueError(f"n_folds must be >= 2, got {n_folds}")
	if n_rows < n_folds:
		raise ValueError(f"n_rows({n_rows}) must be >= n_folds({n_folds})")

	rng = np.random.default_rng(int(seed))
	fold_sizes = target_fold_sizes(n_rows, n_folds)

	total_sums = label_matrix.sum(axis=0)
	target_sums = total_sums / float(n_folds)
	label_weights = 1.0 / np.maximum(total_sums, 1.0)

	presence = (label_matrix > 0).astype(np.float64)
	presence_count = presence.sum(axis=0)
	presence_weights = 1.0 / np.maximum(presence_count, 1.0)
	row_score = (presence @ presence_weights) + (label_matrix @ label_weights)
	row_score = row_score + (rng.random(n_rows) * 1e-6)
	order = np.argsort(-row_score)

	fold_ids = np.full((n_rows,), -1, dtype=np.int64)
	fold_counts = np.zeros((n_folds,), dtype=np.int64)
	fold_sums = np.zeros((n_folds, label_matrix.shape[1]), dtype=np.float64)

	def objective(sums: np.ndarray) -> float:
		return float(np.sum(np.abs(sums - target_sums[None, :]) * label_weights[None, :]))

	for row_idx in order:
		best_fold = -1
		best_loss = float("inf")
		row_labels = label_matrix[int(row_idx)]

		for fold_idx in range(n_folds):
			if fold_counts[fold_idx] >= fold_sizes[fold_idx]:
				continue

			next_all_sums = fold_sums.copy()
			next_all_sums[fold_idx] += row_labels
			label_loss = objective(next_all_sums)
			size_ratio = (float(fold_counts[fold_idx] + 1) / float(fold_sizes[fold_idx]))
			size_loss = abs(size_ratio - 1.0) * 0.01
			tie_break = float(rng.random() * 1e-9)
			loss = label_loss + size_loss + tie_break

			if loss < best_loss:
				best_loss = loss
				best_fold = fold_idx

		if best_fold < 0:
			raise RuntimeError("No fold capacity left while assigning rows")

		fold_ids[int(row_idx)] = int(best_fold)
		fold_counts[best_fold] += 1
		fold_sums[best_fold] += row_labels

	if np.any(fold_ids < 0):
		raise RuntimeError("Some rows were not assigned to a fold")
	if fold_counts.tolist() != fold_sizes:
		raise RuntimeError(f"Fold sizes mismatch: got={fold_counts.tolist()}, target={fold_sizes}")
	fold_ids = improve_by_swaps(
		label_matrix,
		fold_ids,
		target_sums=target_sums,
		label_weights=label_weights,
		seed=seed,
		max_rounds=50,
	)
	return fold_ids


def improve_by_swaps(
	label_matrix: np.ndarray,
	fold_ids: np.ndarray,
	*,
	target_sums: np.ndarray,
	label_weights: np.ndarray,
	seed: int,
	max_rounds: int = 50,
) -> np.ndarray:
	"""Swap rows between folds while preserving fold sizes and reducing label imbalance."""
	rng = np.random.default_rng(int(seed) + 10_000)
	n_folds = int(fold_ids.max()) + 1
	fold_sums = np.stack(
		[label_matrix[fold_ids == fold_idx].sum(axis=0) for fold_idx in range(n_folds)],
		axis=0,
	)

	def fold_loss(sums: np.ndarray) -> float:
		return float(np.sum(np.abs(sums - target_sums) * label_weights))

	def total_loss() -> float:
		return float(sum(fold_loss(fold_sums[fold_idx]) for fold_idx in range(n_folds)))

	initial_loss = total_loss()
	for _ in range(int(max_rounds)):
		improved = False
		fold_pairs = [(a, b) for a in range(n_folds) for b in range(a + 1, n_folds)]
		rng.shuffle(fold_pairs)

		for fold_a, fold_b in fold_pairs:
			idx_a = np.flatnonzero(fold_ids == fold_a)
			idx_b = np.flatnonzero(fold_ids == fold_b)
			if len(idx_a) == 0 or len(idx_b) == 0:
				continue

			rng.shuffle(idx_a)
			rng.shuffle(idx_b)
			current_pair_loss = fold_loss(fold_sums[fold_a]) + fold_loss(fold_sums[fold_b])
			best_pair: tuple[int, int] | None = None
			best_pair_loss = current_pair_loss

			for i in idx_a:
				labels_i = label_matrix[int(i)]
				base_a = fold_sums[fold_a] - labels_i
				base_b = fold_sums[fold_b] + labels_i
				for j in idx_b:
					labels_j = label_matrix[int(j)]
					next_a = base_a + labels_j
					next_b = base_b - labels_j
					next_pair_loss = fold_loss(next_a) + fold_loss(next_b)
					if next_pair_loss + 1e-12 < best_pair_loss:
						best_pair_loss = next_pair_loss
						best_pair = (int(i), int(j))

			if best_pair is None:
				continue

			i, j = best_pair
			labels_i = label_matrix[i]
			labels_j = label_matrix[j]
			fold_ids[i] = fold_b
			fold_ids[j] = fold_a
			fold_sums[fold_a] = fold_sums[fold_a] - labels_i + labels_j
			fold_sums[fold_b] = fold_sums[fold_b] - labels_j + labels_i
			improved = True

		if not improved:
			break

	print(f"Fold label-balance loss: initial={initial_loss:.6f}, final={total_loss():.6f}")
	return fold_ids


def build_summary(df: pd.DataFrame, fold_ids: np.ndarray) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	total_rows = int(len(df))
	total_sums = df[CLASS_NAME].apply(pd.to_numeric, errors="coerce").fillna(0).sum()

	for fold_no in range(1, N_FOLDS + 1):
		val_mask = fold_ids == (fold_no - 1)
		train_mask = ~val_mask
		val_df = df.loc[val_mask]
		train_df = df.loc[train_mask]

		row: dict[str, object] = {
			"fold": fold_no,
			"total_rows": total_rows,
			"train_rows": int(len(train_df)),
			"val_rows": int(len(val_df)),
			"train_ratio": float(len(train_df) / total_rows),
			"val_ratio": float(len(val_df) / total_rows),
		}

		train_sums = train_df[CLASS_NAME].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
		val_sums = val_df[CLASS_NAME].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
		for c in CLASS_NAME:
			denom = float(total_sums[c]) if float(total_sums[c]) > 0 else 1.0
			row[f"train_{c}_sum"] = float(train_sums[c])
			row[f"val_{c}_sum"] = float(val_sums[c])
			row[f"val_{c}_ratio"] = float(val_sums[c] / denom)
		rows.append(row)

	return pd.DataFrame(rows)


def build_assignments(df: pd.DataFrame, fold_ids: np.ndarray) -> pd.DataFrame:
	columns = ["file_name", *CLASS_NAME]
	missing = [column for column in columns if column not in df.columns]
	if missing:
		raise KeyError(f"Cannot build fold assignments; missing columns: {missing}")
	assignments = df[columns].copy()
	assignments.insert(1, "validation_fold", fold_ids.astype(np.int64) + 1)
	return assignments


def attach_assignments(feature_df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
	if "file_name" not in feature_df.columns:
		raise KeyError("Feature index is missing 'file_name'")
	if feature_df["file_name"].duplicated().any():
		duplicates = feature_df.loc[
			feature_df["file_name"].duplicated(keep=False), "file_name"
		].tolist()
		raise ValueError(f"Feature index has duplicate file_name values: {duplicates}")

	manifest_files = set(assignments["file_name"].astype(str))
	feature_files = set(feature_df["file_name"].astype(str))
	missing = sorted(manifest_files - feature_files)
	extra = sorted(feature_files - manifest_files)
	if missing or extra:
		raise ValueError(
			"Feature index and manifest file sets differ: "
			f"missing={missing[:10]} ({len(missing)}), extra={extra[:10]} ({len(extra)})"
		)

	# Labels and fold ids always come from the current canonical manifest.
	manifest_columns = ["file_name", "validation_fold", *CLASS_NAME]
	columns_to_replace = [
		column for column in ["validation_fold", *CLASS_NAME] if column in feature_df.columns
	]
	feature_without_manifest_values = feature_df.drop(columns=columns_to_replace)
	return feature_without_manifest_values.merge(
		assignments[manifest_columns],
		on="file_name",
		how="left",
		sort=False,
		validate="one_to_one",
	)


def write_fold_files(df: pd.DataFrame, output_root: Path) -> None:
	output_root.mkdir(parents=True, exist_ok=True)
	for fold_no in range(1, N_FOLDS + 1):
		current_fold_dir = fold_dir(fold_no, output_root)
		current_fold_dir.mkdir(parents=True, exist_ok=True)

		out_df = df.copy()
		split_values = np.where(out_df["validation_fold"] == fold_no, "val", "train")
		if "split" in out_df.columns:
			out_df["split"] = split_values
		else:
			out_df.insert(0, "split", split_values)

		csv_path = fold_data_csv(fold_no, output_root)
		xlsx_path = current_fold_dir / "data.xlsx"
		out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
		out_df.to_excel(xlsx_path, index=False, engine="openpyxl")
		print(f"WROTE: {csv_path}")
		print(f"WROTE: {xlsx_path}")


def validate_fold_ids(df: pd.DataFrame, fold_ids: np.ndarray) -> None:
	if len(fold_ids) != len(df):
		raise RuntimeError(f"fold_ids length mismatch: {len(fold_ids)} != {len(df)}")
	unique, counts = np.unique(fold_ids, return_counts=True)
	if unique.tolist() != list(range(N_FOLDS)):
		raise RuntimeError(f"Unexpected fold ids: {unique.tolist()}")
	expected = target_fold_sizes(len(df), N_FOLDS)
	if counts.tolist() != expected:
		raise RuntimeError(f"Unexpected fold sizes: {counts.tolist()} != {expected}")
	if "file_name" in df.columns and df["file_name"].nunique(dropna=False) != len(df):
		raise RuntimeError("file_name is not unique; validation fold coverage cannot be verified safely")


def main() -> None:
	args = parse_args()
	output_root = args.output_root.expanduser().resolve()
	manifest = read_data_manifest(
		args.manifest,
		expected_rows=int(args.expected_samples) if args.expected_samples > 0 else None,
	)
	label_matrix = coerce_label_matrix(manifest)
	fold_ids = make_balanced_fold_ids(label_matrix, n_folds=N_FOLDS, seed=SEED)
	validate_fold_ids(manifest, fold_ids)

	output_root.mkdir(parents=True, exist_ok=True)
	assignments = build_assignments(manifest, fold_ids)
	assignments_csv = output_root / "fold_assignments.csv"
	assignments_xlsx = output_root / "fold_assignments.xlsx"
	assignments.to_csv(assignments_csv, index=False, encoding="utf-8-sig")
	assignments.to_excel(assignments_xlsx, index=False, engine="openpyxl")
	print(f"WROTE: {assignments_csv}")
	print(f"WROTE: {assignments_xlsx}")

	summary = build_summary(manifest, fold_ids)
	summary_csv = output_root / "fold_split_summary.csv"
	summary_xlsx = output_root / "fold_split_summary.xlsx"
	summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
	summary.to_excel(summary_xlsx, index=False, engine="openpyxl")
	print(f"WROTE: {summary_csv}")
	print(f"WROTE: {summary_xlsx}")

	with pd.option_context("display.max_columns", None, "display.width", 240):
		print(summary)

	if args.assignments_only:
		return

	feature_df = read_csv_df(args.feature_index)
	if len(feature_df) == 0:
		raise RuntimeError(f"No rows in feature index: {args.feature_index}")
	joined = attach_assignments(feature_df, assignments)
	write_fold_files(joined, output_root)


if __name__ == "__main__":
	main()
