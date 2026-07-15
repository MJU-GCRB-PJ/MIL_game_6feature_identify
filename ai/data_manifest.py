"""Read and validate the canonical dataset manifest."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ai.project_paths import DATA_LIST_XLSX


SOURCE_LABEL_COLUMNS = {
    "sexuality": "sexual_content",
    "violence": "violence",
    "fear/horror/threatening": "fear",
    "language": "inappropriate_language",
    "alcohol/tobacco/drug": "drugs",
    "crime/anti-societal or anti-governmental messages": "crime",
    "gambling(betting)": "gambling",
}
MODEL_LABEL_COLUMNS = [
    "sexual_content",
    "violence",
    "fear",
    "inappropriate_language",
    "drugs",
    "crime",
]
REQUIRED_SOURCE_COLUMNS = ["file_name", *SOURCE_LABEL_COLUMNS]


def read_data_manifest(
    path: Path = DATA_LIST_XLSX,
    *,
    required_columns: Iterable[str] = (),
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Load data_list.xlsx and expose stable model-facing label columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    dataframe = pd.read_excel(path, engine="openpyxl")
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Dataset manifest is missing required columns: {missing}")

    if expected_rows is not None and len(dataframe) != int(expected_rows):
        raise ValueError(
            f"Dataset manifest row count mismatch: expected {int(expected_rows)}, "
            f"got {len(dataframe)} ({path})"
        )

    dataframe = dataframe.copy()
    dataframe["file_name"] = dataframe["file_name"].fillna("").astype(str).str.strip()
    if (dataframe["file_name"] == "").any():
        rows = (dataframe.index[dataframe["file_name"] == ""] + 2).tolist()
        raise ValueError(f"Dataset manifest has empty file_name values at Excel rows: {rows}")
    duplicates = dataframe.loc[dataframe["file_name"].duplicated(), "file_name"].tolist()
    if duplicates:
        raise ValueError(f"Dataset manifest has duplicate file_name values: {duplicates}")

    for source, target in SOURCE_LABEL_COLUMNS.items():
        numeric = pd.to_numeric(dataframe[source], errors="coerce")
        invalid = numeric.isna() | ~numeric.isin([0, 1])
        if invalid.any():
            rows = (dataframe.index[invalid] + 2).tolist()
            values = dataframe.loc[invalid, source].tolist()
            raise ValueError(
                f"Manifest label '{source}' must contain only 0 or 1; "
                f"invalid Excel rows={rows}, values={values}"
            )
        dataframe[target] = numeric.astype(int)

    requested = list(dict.fromkeys(required_columns))
    missing_requested = [column for column in requested if column not in dataframe.columns]
    if missing_requested:
        raise KeyError(
            f"Dataset manifest is missing requested columns: {missing_requested}"
        )
    return dataframe
