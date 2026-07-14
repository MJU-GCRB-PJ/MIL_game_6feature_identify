from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


N_FOLDS = 5
CLASS_NAME = [
	"sexual_content",
	"violence",
	"fear",
	"inappropriate_language",
	"drugs",
	"crime",
]


@dataclass(frozen=True)
class ModelSpec:
	key: str
	output_dir_name: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
	ModelSpec("vision", "vision_mil"),
	ModelSpec("original_audio", "original_audio_mil"),
	ModelSpec("vocal_audio", "vocal_audio_mil"),
	ModelSpec("non_vocal_audio", "non_vocal_audio_mil"),
	ModelSpec("ocr", "ocr_mil"),
	ModelSpec("stt", "stt_mil"),
)

MODEL_METRIC_COLUMNS = ["macro_auc", "macro_f1", "micro_f1"]
ENSEMBLE_METRIC_COLUMNS = [
	"Val_Macro_AUC",
	"Val_Macro_Recall",
	"Val_Macro_F1",
	"Total_Macro_AUC",
	"Train_Macro_AUC",
]
TARGET_MODEL_COMBINATION = "Vision+Original Audio+Vocal Audio+Non-Vocal Audio+OCR+STT"


def get_paths() -> tuple[Path, Path]:
	script_dir = Path(__file__).resolve().parent
	output_root = script_dir / "outputs" / "kfold"
	final_dir = output_root / "final_validation"
	return output_root, final_dir


def read_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def to_float(v: Any) -> float:
	try:
		if v is None:
			return float("nan")
		out = float(v)
		return out
	except Exception:
		return float("nan")


def numeric_summary(values: list[float]) -> dict[str, float | int]:
	arr = np.array([v for v in values if not math.isnan(float(v))], dtype=np.float64)
	if arr.size == 0:
		return {
			"count": 0,
			"mean": float("nan"),
			"std": float("nan"),
			"min": float("nan"),
			"max": float("nan"),
		}
	return {
		"count": int(arr.size),
		"mean": float(np.mean(arr)),
		"std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
		"min": float(np.min(arr)),
		"max": float(np.max(arr)),
	}


def best_metric_row(history: Any) -> dict[str, Any] | None:
	if not isinstance(history, list) or not history:
		return None
	valid_rows = [r for r in history if isinstance(r, dict)]
	if not valid_rows:
		return None

	def score(row: dict[str, Any]) -> float:
		return to_float(row.get("macro_auc"))

	scored = [(score(r), r) for r in valid_rows]
	scored_valid = [(s, r) for s, r in scored if not math.isnan(s)]
	if scored_valid:
		return max(scored_valid, key=lambda x: x[0])[1]
	return valid_rows[-1]


def collect_model_rows(output_root: Path) -> tuple[pd.DataFrame, list[str]]:
	rows: list[dict[str, Any]] = []
	missing: list[str] = []

	for fold_no in range(1, N_FOLDS + 1):
		for spec in MODEL_SPECS:
			metrics_path = output_root / f"{fold_no}_fold" / spec.output_dir_name / "metrics.json"
			if not metrics_path.exists():
				missing.append(str(metrics_path))
				continue

			best = best_metric_row(read_json(metrics_path))
			if best is None:
				missing.append(f"{metrics_path} (empty or invalid)")
				continue

			row: dict[str, Any] = {
				"fold": fold_no,
				"model": spec.key,
				"epoch": best.get("epoch"),
				"n": best.get("n"),
			}
			for col in MODEL_METRIC_COLUMNS:
				row[col] = to_float(best.get(col))
			per_class_auc = best.get("per_class_auc", {})
			if isinstance(per_class_auc, dict):
				for c in CLASS_NAME:
					row[f"auc_{c}"] = to_float(per_class_auc.get(c))
			rows.append(row)

	return pd.DataFrame(rows), missing


def summarize_model_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
	if raw_df.empty:
		return pd.DataFrame()

	metric_cols = MODEL_METRIC_COLUMNS + [f"auc_{c}" for c in CLASS_NAME if f"auc_{c}" in raw_df.columns]
	rows: list[dict[str, Any]] = []
	for model, g in raw_df.groupby("model", sort=False):
		row: dict[str, Any] = {"model": model}
		for metric in metric_cols:
			vals = [to_float(v) for v in g[metric].tolist()]
			s = numeric_summary(vals)
			row[f"{metric}_count"] = s["count"]
			row[f"{metric}_mean"] = s["mean"]
			row[f"{metric}_std"] = s["std"]
			row[f"{metric}_min"] = s["min"]
			row[f"{metric}_max"] = s["max"]
			for _, fold_row in g.sort_values("fold").iterrows():
				row[f"fold_{int(fold_row['fold'])}_{metric}"] = to_float(fold_row.get(metric))
		epoch_vals = [to_float(v) for v in g["epoch"].tolist()]
		row["best_epoch_mean"] = numeric_summary(epoch_vals)["mean"]
		rows.append(row)
	return pd.DataFrame(rows)


def collect_ensemble_rows(output_root: Path) -> tuple[pd.DataFrame, list[str]]:
	rows: list[dict[str, Any]] = []
	missing: list[str] = []

	for fold_no in range(1, N_FOLDS + 1):
		excel_path = output_root / f"{fold_no}_fold" / "ensemble" / "ensemble_results.xlsx"
		if not excel_path.exists():
			missing.append(str(excel_path))
			continue

		df = pd.read_excel(excel_path, sheet_name="All_Results")
		required_cols = {"Model_Combination", "Ensemble_Method", "Val_Macro_AUC"}
		missing_cols = sorted(required_cols - set(df.columns))
		if df.empty or missing_cols:
			missing.append(f"{excel_path} (empty or missing columns: {missing_cols})")
			continue

		combo_series = df["Model_Combination"].astype(str).str.strip()
		target_df = df.loc[combo_series == TARGET_MODEL_COMBINATION].copy()
		if target_df.empty:
			missing.append(f"{excel_path} (target combination not found: {TARGET_MODEL_COMBINATION})")
			continue

		for _, result_row in target_df.iterrows():
			result = result_row.to_dict()
			row: dict[str, Any] = {
				"fold": fold_no,
				"Ensemble_Method": result.get("Ensemble_Method"),
				"Model_Combination": str(result.get("Model_Combination", "")).strip(),
				"Model_Weights": result.get("Model_Weights"),
			}
			for col in ENSEMBLE_METRIC_COLUMNS:
				row[col] = to_float(result.get(col))
			for c in CLASS_NAME:
				col = f"Val_AUC_{c}"
				if col in result:
					row[col] = to_float(result.get(col))
			rows.append(row)

	return pd.DataFrame(rows), missing


def summarize_ensemble_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
	if raw_df.empty:
		return pd.DataFrame()

	metric_cols = [c for c in raw_df.columns if c.startswith("Val_") or c in ("Total_Macro_AUC", "Train_Macro_AUC")]
	rows: list[dict[str, Any]] = []
	for method, g in raw_df.groupby("Ensemble_Method", sort=False):
		row: dict[str, Any] = {
			"Model_Combination": TARGET_MODEL_COMBINATION,
			"Ensemble_Method": method,
		}
		for metric in metric_cols:
			vals = [to_float(v) for v in g[metric].tolist()]
			s = numeric_summary(vals)
			row[f"{metric}_count"] = s["count"]
			row[f"{metric}_mean"] = s["mean"]
			row[f"{metric}_std"] = s["std"]
			row[f"{metric}_min"] = s["min"]
			row[f"{metric}_max"] = s["max"]
			for _, fold_row in g.sort_values("fold").iterrows():
				row[f"fold_{int(fold_row['fold'])}_{metric}"] = to_float(fold_row.get(metric))
		rows.append(row)

	out_df = pd.DataFrame(rows)
	sort_col = "Val_Macro_AUC_mean"
	if sort_col in out_df.columns:
		out_df = out_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
	return out_df

def write_json_summary(
	path: Path,
	*,
	model_summary: pd.DataFrame,
	ensemble_summary: pd.DataFrame,
	missing_model: list[str],
	missing_ensemble: list[str],
) -> None:
	payload = {
		"n_folds": N_FOLDS,
		"target_model_combination": TARGET_MODEL_COMBINATION,
		"model_summary": model_summary.replace({np.nan: None}).to_dict(orient="records"),
		"ensemble_summary": ensemble_summary.replace({np.nan: None}).to_dict(orient="records"),
		"missing_model_results": missing_model,
		"missing_ensemble_results": missing_ensemble,
	}
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
	output_root, final_dir = get_paths()
	final_dir.mkdir(parents=True, exist_ok=True)

	model_raw, missing_model = collect_model_rows(output_root)
	model_summary = summarize_model_rows(model_raw)

	ensemble_raw, missing_ensemble = collect_ensemble_rows(output_root)
	ensemble_summary = summarize_ensemble_rows(ensemble_raw)

	model_xlsx = final_dir / "kfold_model_summary.xlsx"
	ensemble_xlsx = final_dir / "kfold_ensemble_summary.xlsx"
	json_path = final_dir / "final_validation_summary.json"

	with pd.ExcelWriter(model_xlsx, engine="openpyxl") as writer:
		model_summary.to_excel(writer, sheet_name="Summary", index=False)
		model_raw.to_excel(writer, sheet_name="Fold_Raw", index=False)
		if missing_model:
			pd.DataFrame({"missing": missing_model}).to_excel(writer, sheet_name="Missing", index=False)

	with pd.ExcelWriter(ensemble_xlsx, engine="openpyxl") as writer:
		ensemble_summary.to_excel(writer, sheet_name="Six_Modal_Summary", index=False)
		ensemble_raw.to_excel(writer, sheet_name="Six_Modal_Fold_Raw", index=False)
		fold_sheet_columns = [
			"fold",
			"Ensemble_Method",
			"Model_Combination",
			"Model_Weights",
			*ENSEMBLE_METRIC_COLUMNS,
			*[f"Val_AUC_{c}" for c in CLASS_NAME],
		]
		for fold_no in range(1, N_FOLDS + 1):
			if ensemble_raw.empty or "fold" not in ensemble_raw.columns:
				fold_df = pd.DataFrame(columns=fold_sheet_columns)
			else:
				fold_df = ensemble_raw[ensemble_raw["fold"] == fold_no].copy()
				existing_cols = [c for c in fold_sheet_columns if c in fold_df.columns]
				fold_df = fold_df[existing_cols]
				if "Val_Macro_AUC" in fold_df.columns:
					fold_df = fold_df.sort_values("Val_Macro_AUC", ascending=False).reset_index(drop=True)
			fold_df.to_excel(writer, sheet_name=f"fold_{fold_no}", index=False)
		if missing_ensemble:
			pd.DataFrame({"missing": missing_ensemble}).to_excel(writer, sheet_name="Missing", index=False)

	write_json_summary(
		json_path,
		model_summary=model_summary,
		ensemble_summary=ensemble_summary,
		missing_model=missing_model,
		missing_ensemble=missing_ensemble,
	)

	print(f"WROTE: {model_xlsx}")
	print(f"WROTE: {ensemble_xlsx}")
	print(f"WROTE: {json_path}")
	if missing_model:
		print(f"Missing model results: {len(missing_model)}")
	if missing_ensemble:
		print(f"Missing ensemble results: {len(missing_ensemble)}")


if __name__ == "__main__":
	main()
