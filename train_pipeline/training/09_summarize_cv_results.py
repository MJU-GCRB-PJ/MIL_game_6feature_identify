"""Summarize model and ensemble metrics across the five paper folds."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cv_config import (
    CLASS_NAMES,
    CV_OUTPUT_DIR,
    MODEL_SPECS,
    N_FOLDS,
    TARGET_MODEL_COMBINATION,
    fold_dir,
)


MODEL_METRIC_COLUMNS = ["macro_auc", "macro_f1", "micro_f1"]
ENSEMBLE_METRIC_COLUMNS = [
    "Val_Macro_AUC",
    "Val_Macro_Recall",
    "Val_Macro_F1",
    "Total_Macro_AUC",
    "Train_Macro_AUC",
]
MODEL_WEIGHT_COLUMNS = [
    "Weight_Vision",
    "Weight_Original_Audio",
    "Weight_Vocal_Audio",
    "Weight_Non_Vocal_Audio",
    "Weight_OCR",
    "Weight_STT",
]
MODEL_LABEL_TO_WEIGHT_COLUMN = {
    spec.label: weight_column
    for spec, weight_column in zip(MODEL_SPECS, MODEL_WEIGHT_COLUMNS, strict=True)
}
MODEL_KEY_TO_WEIGHT_COLUMN = {
    spec.key: weight_column
    for spec, weight_column in zip(MODEL_SPECS, MODEL_WEIGHT_COLUMNS, strict=True)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=CV_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    array = np.array(
        [value for value in values if not math.isnan(float(value))],
        dtype=np.float64,
    )
    if array.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def selected_weight_columns(model_combination: Any) -> list[str]:
    columns: list[str] = []
    for label in str(model_combination).split("+"):
        column = MODEL_LABEL_TO_WEIGHT_COLUMN.get(label.strip())
        if column is not None:
            columns.append(column)
    return columns


def parse_model_weights(weights: Any) -> Any:
    if not isinstance(weights, str):
        return weights
    text = weights.strip()
    if not text or text in {"N/A", "equal"}:
        return text
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def normalize_weight_values(values: dict[str, float]) -> dict[str, float]:
    cleaned = {
        key: max(float(value), 0.0)
        for key, value in values.items()
        if math.isfinite(float(value))
    }
    total = sum(cleaned.values())
    if total <= 0:
        return cleaned
    return {key: round(value / total, 6) for key, value in cleaned.items()}


def derive_model_weight_columns(
    model_combination: Any,
    method: Any,
    weights: Any,
) -> dict[str, float]:
    selected_columns = selected_weight_columns(model_combination)
    output = {column: float("nan") for column in MODEL_WEIGHT_COLUMNS}
    if not selected_columns:
        return output

    method_text = str(method)
    parsed = parse_model_weights(weights)
    weight_by_column: dict[str, float] = {}
    if method_text == "Individual":
        weight_by_column = {selected_columns[0]: 1.0}
    elif method_text in {"Hard_Voting", "Soft_Voting"} or parsed == "equal":
        equal_weight = 1.0 / len(selected_columns)
        weight_by_column = {column: equal_weight for column in selected_columns}
    elif method_text == "Weighted_Soft_Voting" and isinstance(parsed, dict):
        for key, column in MODEL_KEY_TO_WEIGHT_COLUMN.items():
            if column in selected_columns and key in parsed:
                weight_by_column[column] = float(parsed[key])
        weight_by_column = normalize_weight_values(weight_by_column)
    elif method_text.startswith("Stacking_") and isinstance(parsed, dict):
        accumulated = {column: 0.0 for column in selected_columns}
        for raw_values in parsed.values():
            if not isinstance(raw_values, (list, tuple)):
                continue
            values = [
                abs(float(value))
                for value in raw_values
                if isinstance(value, (int, float, np.integer, np.floating))
            ]
            if len(values) >= len(selected_columns) * len(CLASS_NAMES):
                for index, column in enumerate(selected_columns):
                    start = index * len(CLASS_NAMES)
                    accumulated[column] += float(
                        np.mean(values[start : start + len(CLASS_NAMES)])
                    )
            elif len(values) >= len(selected_columns):
                for index, column in enumerate(selected_columns):
                    accumulated[column] += values[index]
        weight_by_column = normalize_weight_values(accumulated)

    for column, value in weight_by_column.items():
        output[column] = round(float(value), 6)
    return output


def best_metric_row(history: Any) -> dict[str, Any] | None:
    if not isinstance(history, list):
        return None
    rows = [row for row in history if isinstance(row, dict)]
    if not rows:
        return None
    scored = [(to_float(row.get("macro_auc")), row) for row in rows]
    valid = [(score, row) for score, row in scored if not math.isnan(score)]
    return max(valid, key=lambda item: item[0])[1] if valid else rows[-1]


def collect_model_rows(output_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in range(1, N_FOLDS + 1):
        for spec in MODEL_SPECS:
            metrics_path = fold_dir(fold, output_root) / spec.output_dir_name / "metrics.json"
            if not metrics_path.exists():
                missing.append(str(metrics_path))
                continue
            best = best_metric_row(read_json(metrics_path))
            if best is None:
                missing.append(f"{metrics_path} (empty or invalid)")
                continue

            row: dict[str, Any] = {
                "fold": fold,
                "model": spec.key,
                "epoch": best.get("epoch"),
                "n": best.get("n"),
            }
            for column in MODEL_METRIC_COLUMNS:
                row[column] = to_float(best.get(column))
            per_class_auc = best.get("per_class_auc", {})
            if isinstance(per_class_auc, dict):
                for class_name in CLASS_NAMES:
                    row[f"auc_{class_name}"] = to_float(per_class_auc.get(class_name))
            rows.append(row)
    return pd.DataFrame(rows), missing


def summarize_model_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    metric_columns = MODEL_METRIC_COLUMNS + [
        f"auc_{class_name}"
        for class_name in CLASS_NAMES
        if f"auc_{class_name}" in raw.columns
    ]
    rows: list[dict[str, Any]] = []
    for model, group in raw.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model}
        for metric in metric_columns:
            summary = numeric_summary([to_float(value) for value in group[metric]])
            for statistic, value in summary.items():
                row[f"{metric}_{statistic}"] = value
            for _, fold_row in group.sort_values("fold").iterrows():
                row[f"fold_{int(fold_row['fold'])}_{metric}"] = to_float(
                    fold_row.get(metric)
                )
        row["best_epoch_mean"] = numeric_summary(
            [to_float(value) for value in group["epoch"]]
        )["mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_ensemble_rows(output_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for fold in range(1, N_FOLDS + 1):
        excel_path = fold_dir(fold, output_root) / "ensemble" / "ensemble_results.xlsx"
        if not excel_path.exists():
            missing.append(str(excel_path))
            continue
        result_frame = pd.read_excel(excel_path, sheet_name="All_Results")
        required = {"Model_Combination", "Ensemble_Method", "Val_Macro_AUC"}
        missing_columns = sorted(required - set(result_frame.columns))
        if result_frame.empty or missing_columns:
            missing.append(f"{excel_path} (empty or missing columns: {missing_columns})")
            continue

        combinations = result_frame["Model_Combination"].astype(str).str.strip()
        target_rows = result_frame.loc[combinations == TARGET_MODEL_COMBINATION]
        if target_rows.empty:
            missing.append(
                f"{excel_path} (target combination not found: {TARGET_MODEL_COMBINATION})"
            )
            continue

        for _, result_row in target_rows.iterrows():
            result = result_row.to_dict()
            row: dict[str, Any] = {
                "fold": fold,
                "Ensemble_Method": result.get("Ensemble_Method"),
                "Model_Combination": str(result.get("Model_Combination", "")).strip(),
                "Model_Weights": result.get("Model_Weights"),
            }
            row.update(
                derive_model_weight_columns(
                    row["Model_Combination"],
                    row["Ensemble_Method"],
                    row["Model_Weights"],
                )
            )
            for column in MODEL_WEIGHT_COLUMNS:
                if column in result:
                    row[column] = to_float(result.get(column))
            for column in ENSEMBLE_METRIC_COLUMNS:
                row[column] = to_float(result.get(column))
            for class_name in CLASS_NAMES:
                column = f"Val_AUC_{class_name}"
                if column in result:
                    row[column] = to_float(result.get(column))
            rows.append(row)
    return pd.DataFrame(rows), missing


def summarize_ensemble_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in raw.columns
        if column.startswith("Val_")
        or column in {"Total_Macro_AUC", "Train_Macro_AUC"}
    ]
    weight_columns = [column for column in MODEL_WEIGHT_COLUMNS if column in raw.columns]
    rows: list[dict[str, Any]] = []
    for method, group in raw.groupby("Ensemble_Method", sort=False):
        row: dict[str, Any] = {
            "Model_Combination": TARGET_MODEL_COMBINATION,
            "Ensemble_Method": method,
        }
        for metric in metric_columns:
            summary = numeric_summary([to_float(value) for value in group[metric]])
            for statistic, value in summary.items():
                row[f"{metric}_{statistic}"] = value
            for _, fold_row in group.sort_values("fold").iterrows():
                row[f"fold_{int(fold_row['fold'])}_{metric}"] = to_float(
                    fold_row.get(metric)
                )
        for weight_column in weight_columns:
            summary = numeric_summary([to_float(value) for value in group[weight_column]])
            row[f"{weight_column}_mean"] = summary["mean"]
            row[f"{weight_column}_std"] = summary["std"]
            for _, fold_row in group.sort_values("fold").iterrows():
                row[f"fold_{int(fold_row['fold'])}_{weight_column}"] = to_float(
                    fold_row.get(weight_column)
                )
        rows.append(row)

    output = pd.DataFrame(rows)
    if "Val_Macro_AUC_mean" in output.columns:
        output = output.sort_values("Val_Macro_AUC_mean", ascending=False).reset_index(
            drop=True
        )
    return output


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


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
        "model_summary": dataframe_records(model_summary),
        "ensemble_summary": dataframe_records(ensemble_summary),
        "missing_model_results": missing_model,
        "missing_ensemble_results": missing_ensemble,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    model_raw, missing_model = collect_model_rows(output_root)
    model_summary = summarize_model_rows(model_raw)
    ensemble_raw, missing_ensemble = collect_ensemble_rows(output_root)
    ensemble_summary = summarize_ensemble_rows(ensemble_raw)

    model_path = summary_dir / "model_summary.xlsx"
    ensemble_path = summary_dir / "ensemble_summary.xlsx"
    json_path = summary_dir / "summary.json"

    with pd.ExcelWriter(model_path, engine="openpyxl") as writer:
        model_summary.to_excel(writer, sheet_name="Summary", index=False)
        model_raw.to_excel(writer, sheet_name="Fold_Raw", index=False)
        if missing_model:
            pd.DataFrame({"missing": missing_model}).to_excel(
                writer, sheet_name="Missing", index=False
            )

    fold_columns = [
        "fold",
        "Ensemble_Method",
        "Model_Combination",
        "Model_Weights",
        *MODEL_WEIGHT_COLUMNS,
        *ENSEMBLE_METRIC_COLUMNS,
        *[f"Val_AUC_{class_name}" for class_name in CLASS_NAMES],
    ]
    with pd.ExcelWriter(ensemble_path, engine="openpyxl") as writer:
        ensemble_summary.to_excel(writer, sheet_name="Six_Modal_Summary", index=False)
        ensemble_raw.to_excel(writer, sheet_name="Six_Modal_Fold_Raw", index=False)
        for fold in range(1, N_FOLDS + 1):
            if ensemble_raw.empty or "fold" not in ensemble_raw:
                fold_frame = pd.DataFrame(columns=fold_columns)
            else:
                fold_frame = ensemble_raw.loc[ensemble_raw["fold"] == fold].copy()
                fold_frame = fold_frame[
                    [column for column in fold_columns if column in fold_frame.columns]
                ]
                if "Val_Macro_AUC" in fold_frame:
                    fold_frame = fold_frame.sort_values(
                        "Val_Macro_AUC", ascending=False
                    ).reset_index(drop=True)
            fold_frame.to_excel(writer, sheet_name=f"fold_{fold}", index=False)
        if missing_ensemble:
            pd.DataFrame({"missing": missing_ensemble}).to_excel(
                writer, sheet_name="Missing", index=False
            )

    write_json_summary(
        json_path,
        model_summary=model_summary,
        ensemble_summary=ensemble_summary,
        missing_model=missing_model,
        missing_ensemble=missing_ensemble,
    )

    print(f"WROTE: {model_path}")
    print(f"WROTE: {ensemble_path}")
    print(f"WROTE: {json_path}")
    if missing_model:
        print(f"Missing model results: {len(missing_model)}")
    if missing_ensemble:
        print(f"Missing ensemble results: {len(missing_ensemble)}")


if __name__ == "__main__":
    main()
