#!/usr/bin/env python3
"""Materialize the paper's fixed No.2 ensemble from the current results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

CLASS_NAMES = [
	"sexual_content",
	"violence",
	"fear",
	"inappropriate_language",
	"drugs",
	"crime",
]

SEED = 42
THRESHOLD = 0.5
NO2_METHOD = "Weighted_Soft_Voting"
NO2_COMBINATION = "Vision+Original Audio+Vocal Audio+Non-Vocal Audio+OCR+STT"
NO2_WEIGHTS: dict[str, float] = {
	"vision": 0.1767,
	"original_audio": 0.1195,
	"vocal_audio": 0.0222,
	"non_vocal_audio": 0.0718,
	"ocr": 0.4262,
	"stt": 0.1836,
}

SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = SCRIPT_DIR / "outputs" / "ensemble"
RESULTS_XLSX = ENSEMBLE_DIR / "ensemble_results.xlsx"
SUMMARY_XLSX = ENSEMBLE_DIR / "no2_summary.xlsx"
PTH_PATH = ENSEMBLE_DIR / "best_pth" / "no2_weighted_soft_voting.pth"


def _numeric_or_str(value: Any) -> Any:
	if isinstance(value, (int, float, np.integer, np.floating)):
		return float(value)
	return str(value)


def load_no2_result() -> dict[str, Any]:
	if not RESULTS_XLSX.exists():
		raise FileNotFoundError(
			f"Ensemble results not found: {RESULTS_XLSX}. Run 08_ensemble.py first."
		)

	df = pd.read_excel(RESULTS_XLSX, sheet_name="All_Results")
	required = {"Ensemble_Method", "Model_Combination"}
	missing = required - set(df.columns)
	if missing:
		raise KeyError(f"Missing result columns: {sorted(missing)}")

	matches = df[
		(df["Ensemble_Method"].astype(str) == NO2_METHOD)
		& (df["Model_Combination"].astype(str) == NO2_COMBINATION)
	].copy()
	if matches.empty:
		raise RuntimeError(
			"The fixed No.2 ensemble is absent from the current ensemble results."
		)

	if "Val_Macro_AUC" in matches.columns:
		matches = matches.sort_values("Val_Macro_AUC", ascending=False, na_position="last")
	row = matches.iloc[0].to_dict()
	row["Paper_Rank"] = 2
	row["Model_Weights"] = str(NO2_WEIGHTS)
	return row


def save_no2_outputs(row: dict[str, Any]) -> None:
	ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
	PTH_PATH.parent.mkdir(parents=True, exist_ok=True)

	pd.DataFrame([row]).to_excel(SUMMARY_XLSX, sheet_name="No2_Summary", index=False)
	try:
		best_score = float(row.get("Val_Macro_AUC", float("nan")))
	except (TypeError, ValueError):
		best_score = float("nan")

	payload = {
		"criterion": "Paper_No2_Weighted_Soft_Voting",
		"description": "Paper No.2 Weighted Soft Voting ensemble",
		"best_score": best_score,
		"ensemble_method": NO2_METHOD,
		"model_combination": NO2_COMBINATION,
		"model_weights": NO2_WEIGHTS,
		"metrics_row": {key: _numeric_or_str(value) for key, value in row.items()},
		"class_names": CLASS_NAMES,
		"threshold": THRESHOLD,
		"seed": SEED,
	}
	torch.save(payload, PTH_PATH)

	print(f"Saved summary: {SUMMARY_XLSX}")
	print(f"Saved artifact: {PTH_PATH}")


def main() -> None:
	row = load_no2_result()
	save_no2_outputs(row)


if __name__ == "__main__":
	main()
