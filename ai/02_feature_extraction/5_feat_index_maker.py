from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Paths:
	repo_root: Path
	data_list_xlsx: Path
	index_csv: Path
	feature_root: Path


def get_paths() -> Paths:
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent.parent
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	from ai.project_paths import (
		DATA_LIST_XLSX,
		FEATURE_ROOT,
		PREPROCESS_INDEX_CSV,
	)
	return Paths(
		repo_root=repo_root,
		data_list_xlsx=DATA_LIST_XLSX,
		index_csv=PREPROCESS_INDEX_CSV,
		feature_root=FEATURE_ROOT,
	)


def _safe_str(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, float) and value != value:
		return ""
	return str(value)


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


def _resolve_feature_path(manifest_path: Path, raw_feature_path: str) -> Path:
	"""Resolve a feature_path field from manifest.

	- In this repo's extractors it's usually absolute.
	- If relative, resolve relative to manifest's parent.
	"""
	raw_feature_path = raw_feature_path.strip()
	if not raw_feature_path:
		return Path("")
	p = Path(raw_feature_path)
	if p.is_absolute():
		return p
	return (manifest_path.parent / p).resolve()


def _iter_manifest_rows(manifest_path: Path) -> Iterable[dict[str, str]]:
	with manifest_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			yield {k: _safe_str(v) for k, v in row.items()}


def _check_manifest_feature_paths_exist(
	manifest_path: Path,
	*,
	require_audio_types: Optional[set[str]] = None,
	feature_path_key: str = "feature_path",
	max_rows: int = 1_000_000,
) -> bool:
	if not manifest_path.exists():
		return False

	found_types: set[str] = set()
	seen_any = False
	try:
		for i, row in enumerate(_iter_manifest_rows(manifest_path)):
			if i >= max_rows:
				break
			seen_any = True
			if require_audio_types is not None:
				found_types.add(row.get("audio_type", ""))

			feat_raw = row.get(feature_path_key, "")
			feat_path = _resolve_feature_path(manifest_path, feat_raw)
			if not feat_path or not feat_path.exists():
				return False
	except Exception:
		return False

	if not seen_any:
		return False
	if require_audio_types is not None:
		return require_audio_types.issubset(found_types)
	return True


def _all_exist(paths: Iterable[Path]) -> bool:
	for p in paths:
		if not p.exists():
			return False
	return True


def build_feature_columns(file_name: str, feature_root: Path) -> dict[str, object]:
	file_name = Path(file_name).name
	if not file_name:
		return {
			"feature_root": str(feature_root),
			"feature_base_dir": "",
			"vivit_dir": "",
			"vivit_manifest": "",
			"vivit_complete": False,
			"ast_dir": "",
			"ast_manifest": "",
			"ast_original_dir": "",
			"ast_vocal_dir": "",
			"ast_nonvocal_dir": "",
			"ast_complete": False,
			"stt_dir": "",
			"stt_emb": "",
			"stt_mask": "",
			"stt_meta": "",
			"stt_units": "",
			"stt_map": "",
			"stt_text": "",
			"stt_complete": False,
			"ocr_dir": "",
			"ocr_emb": "",
			"ocr_mask": "",
			"ocr_meta": "",
			"ocr_units": "",
			"ocr_map": "",
			"ocr_text": "",
			"ocr_complete": False,
		}

	base_dir = feature_root / file_name

	# ViViT
	vivit_dir = base_dir / "vivit_feat"
	vivit_manifest = vivit_dir / "manifest.csv"
	vivit_complete = _check_manifest_feature_paths_exist(vivit_manifest)

	# AST
	ast_dir = base_dir / "ast_feat"
	ast_manifest = ast_dir / "manifest.csv"
	ast_original_dir = ast_dir / "original_ast_feat"
	ast_vocal_dir = ast_dir / "vocal_ast_feat"
	ast_nonvocal_dir = ast_dir / "non-vocal_ast_feat"
	ast_complete = _check_manifest_feature_paths_exist(
		ast_manifest,
		require_audio_types={"original", "vocal", "non-vocal"},
	)

	# STT
	stt_dir = base_dir / "stt_feat"
	stt_emb = stt_dir / "stt_8s_emb.npy"
	stt_mask = stt_dir / "stt_8s_mask.npy"
	stt_meta = stt_dir / "stt_8s_meta.json"
	stt_units = stt_dir / "stt_units.jsonl"
	stt_map = stt_dir / "stt_8s_map.json"
	stt_text = stt_dir / "stt_8s_text.json"
	stt_complete = _all_exist([stt_emb, stt_mask, stt_meta, stt_units, stt_map, stt_text])

	# OCR
	ocr_dir = base_dir / "ocr_feat"
	ocr_emb = ocr_dir / "ocr_8s_emb.npy"
	ocr_mask = ocr_dir / "ocr_8s_mask.npy"
	ocr_meta = ocr_dir / "ocr_8s_meta.json"
	ocr_units = ocr_dir / "ocr_units.jsonl"
	ocr_map = ocr_dir / "ocr_8s_map.json"
	ocr_text = ocr_dir / "ocr_8s_text.json"
	ocr_complete = _all_exist([ocr_emb, ocr_mask, ocr_meta, ocr_units, ocr_map, ocr_text])

	return {
		"feature_root": str(feature_root),
		"feature_base_dir": str(base_dir),
		"vivit_dir": str(vivit_dir),
		"vivit_manifest": str(vivit_manifest),
		"vivit_complete": bool(vivit_complete),
		"ast_dir": str(ast_dir),
		"ast_manifest": str(ast_manifest),
		"ast_original_dir": str(ast_original_dir),
		"ast_vocal_dir": str(ast_vocal_dir),
		"ast_nonvocal_dir": str(ast_nonvocal_dir),
		"ast_complete": bool(ast_complete),
		"stt_dir": str(stt_dir),
		"stt_emb": str(stt_emb),
		"stt_mask": str(stt_mask),
		"stt_meta": str(stt_meta),
		"stt_units": str(stt_units),
		"stt_map": str(stt_map),
		"stt_text": str(stt_text),
		"stt_complete": bool(stt_complete),
		"ocr_dir": str(ocr_dir),
		"ocr_emb": str(ocr_emb),
		"ocr_mask": str(ocr_mask),
		"ocr_meta": str(ocr_meta),
		"ocr_units": str(ocr_units),
		"ocr_map": str(ocr_map),
		"ocr_text": str(ocr_text),
		"ocr_complete": bool(ocr_complete),
	}


def _maybe_quick_validate_npy_pair(emb_path: Path, mask_path: Path) -> bool:
	"""Optional lightweight integrity check.

	Returns False if loading fails or shapes mismatch. Does not enforce dtype.
	"""
	try:
		emb = np.load(str(emb_path), mmap_mode="r")
		mask = np.load(str(mask_path), mmap_mode="r")
		if getattr(emb, "ndim", 0) != 2:
			return False
		if getattr(mask, "ndim", 0) != 1:
			return False
		if int(emb.shape[0]) != int(mask.shape[0]):
			return False
		return True
	except Exception:
		return False


def _maybe_quick_validate_json(path: Path) -> bool:
	try:
		with path.open("r", encoding="utf-8") as f:
			json.load(f)
		return True
	except Exception:
		return False


def _maybe_quick_validate_jsonl(path: Path, max_lines: int = 3) -> bool:
	try:
		with path.open("r", encoding="utf-8") as f:
			for i, line in enumerate(f):
				if i >= max_lines:
					break
				line = line.strip()
				if not line:
					continue
				json.loads(line)
		return True
	except Exception:
		return False


def quick_validate_completed_files(row: pd.Series) -> tuple[bool, bool]:
	"""Return (stt_ok, ocr_ok) with light integrity checks.

	Only runs if stt_complete/ocr_complete were True by existence.
	"""
	stt_ok = bool(row.get("stt_complete", False))
	ocr_ok = bool(row.get("ocr_complete", False))

	if stt_ok:
		emb = Path(_safe_str(row.get("stt_emb")))
		mask = Path(_safe_str(row.get("stt_mask")))
		meta = Path(_safe_str(row.get("stt_meta")))
		units = Path(_safe_str(row.get("stt_units")))
		mapp = Path(_safe_str(row.get("stt_map")))
		text = Path(_safe_str(row.get("stt_text")))
		stt_ok = (
			_maybe_quick_validate_npy_pair(emb, mask)
			and _maybe_quick_validate_json(meta)
			and _maybe_quick_validate_jsonl(units)
			and _maybe_quick_validate_json(mapp)
			and _maybe_quick_validate_json(text)
		)

	if ocr_ok:
		emb = Path(_safe_str(row.get("ocr_emb")))
		mask = Path(_safe_str(row.get("ocr_mask")))
		meta = Path(_safe_str(row.get("ocr_meta")))
		units = Path(_safe_str(row.get("ocr_units")))
		mapp = Path(_safe_str(row.get("ocr_map")))
		text = Path(_safe_str(row.get("ocr_text")))
		ocr_ok = (
			_maybe_quick_validate_npy_pair(emb, mask)
			and _maybe_quick_validate_json(meta)
			and _maybe_quick_validate_jsonl(units)
			and _maybe_quick_validate_json(mapp)
			and _maybe_quick_validate_json(text)
		)

	return stt_ok, ocr_ok


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Build feature indexes from data_list.xlsx, the preprocessing index, and extracted features.",
	)
	p.add_argument("--data-list", type=str, default="", help="Path to data/data_list.xlsx")
	p.add_argument("--index", type=str, default="", help="Path to the preprocessing index CSV")
	p.add_argument(
		"--feature-root",
		type=str,
		default="",
		help="Feature root directory (default: /data/feature_extraction)",
	)
	p.add_argument("--only-file", type=str, default="", help="Process only this file_name (exact match)")
	p.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 = all)")
	p.add_argument(
		"--no-integrity-check",
		action="store_true",
		help="Skip lightweight integrity checks for STT/OCR completed files (existence checks only)",
	)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	paths = get_paths()

	data_list_xlsx = Path(args.data_list) if args.data_list else paths.data_list_xlsx
	index_csv = Path(args.index) if args.index else paths.index_csv
	feature_root = (
		Path(args.feature_root).expanduser().resolve()
		if args.feature_root
		else paths.feature_root
	)

	if str(paths.repo_root) not in sys.path:
		sys.path.insert(0, str(paths.repo_root))
	from ai.data_manifest import read_data_manifest
	data_df = read_data_manifest(data_list_xlsx)
	index_df = _read_csv_df(index_csv)

	if "file_name" not in data_df.columns:
		raise KeyError(f"data_list is missing 'file_name' column: {data_list_xlsx}")
	if "file_name" not in index_df.columns:
		raise KeyError(f"index is missing 'file_name' column: {index_csv}")

	# Normalize key
	data_df["file_name"] = data_df["file_name"].astype(str)
	index_df["file_name"] = index_df["file_name"].astype(str)

	merged = data_df.merge(index_df, on="file_name", how="left", suffixes=("", "_index"))

	if args.only_file:
		merged = merged[merged["file_name"] == args.only_file]
	if args.limit and args.limit > 0:
		merged = merged.head(int(args.limit))

	# Build modality feature columns
	feature_rows: list[dict[str, object]] = []
	for _, row in tqdm(merged.iterrows(), total=len(merged), desc="scan features"):
		file_name = _safe_str(row.get("file_name"))
		feature_rows.append(build_feature_columns(file_name, feature_root))

	feat_df = pd.DataFrame(feature_rows)
	out_df = pd.concat([merged.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

	# Optional integrity checks: downgrade stt_complete/ocr_complete if parsing fails
	if not args.no_integrity_check and len(out_df) > 0:
		stt_ok_list: list[bool] = []
		ocr_ok_list: list[bool] = []
		for _, row in tqdm(out_df.iterrows(), total=len(out_df), desc="integrity"):
			stt_ok, ocr_ok = quick_validate_completed_files(row)
			stt_ok_list.append(stt_ok)
			ocr_ok_list.append(ocr_ok)
		out_df["stt_complete"] = stt_ok_list
		out_df["ocr_complete"] = ocr_ok_list

	# Save
	out_csv = feature_root / "feat_index.csv"
	out_xlsx = feature_root / "feat_index.xlsx"
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
	out_df.to_excel(out_xlsx, index=False, engine="openpyxl")

	print(f"WROTE: {out_csv}")
	print(f"WROTE: {out_xlsx}")


if __name__ == "__main__":
	main()
