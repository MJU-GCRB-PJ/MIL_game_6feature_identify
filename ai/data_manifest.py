"""Read and validate the canonical dataset manifest."""

from __future__ import annotations

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


def read_data_manifest(path: Path = DATA_LIST_XLSX) -> pd.DataFrame:
    """Load data_list.xlsx and expose stable model-facing label columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")

    dataframe = pd.read_excel(path, engine="openpyxl")
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in dataframe.columns]
    if missing:
        raise KeyError(f"Dataset manifest is missing required columns: {missing}")

    dataframe = dataframe.copy()
    dataframe["file_name"] = dataframe["file_name"].fillna("").astype(str).str.strip()
    if (dataframe["file_name"] == "").any():
        rows = (dataframe.index[dataframe["file_name"] == ""] + 2).tolist()
        raise ValueError(f"Dataset manifest has empty file_name values at Excel rows: {rows}")
    duplicates = dataframe.loc[dataframe["file_name"].duplicated(), "file_name"].tolist()
    if duplicates:
        raise ValueError(f"Dataset manifest has duplicate file_name values: {duplicates}")

    for source, target in SOURCE_LABEL_COLUMNS.items():
        dataframe[target] = pd.to_numeric(dataframe[source], errors="coerce").fillna(0).astype(int)
    return dataframe
