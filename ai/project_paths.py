"""Canonical filesystem locations for the research pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_LIST_XLSX = DATA_DIR / "data_list.xlsx"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


RAW_PREPROCESSED_DIR = _env_path(
    "MIL_RAW_PREPROCESSED_DIR",
    DATA_DIR / "raw_pre-processed",
)
RAW_VIDEO_DIR = RAW_PREPROCESSED_DIR / "game_play_raw_video"
FRAMES_DIR = RAW_PREPROCESSED_DIR / "frames"
AUDIO_DIR = RAW_PREPROCESSED_DIR / "audio"
OCR_RESULTS_DIR = RAW_PREPROCESSED_DIR / "ocr_results"
STT_RESULTS_DIR = RAW_PREPROCESSED_DIR / "stt_results"
PREPROCESS_LOG_DIR = RAW_PREPROCESSED_DIR / "logs"
PREPROCESS_INDEX_CSV = RAW_PREPROCESSED_DIR / "index.csv"
PREPROCESS_INDEX_XLSX = RAW_PREPROCESSED_DIR / "index.xlsx"

FEATURE_ROOT = _env_path("MIL_FEATURE_ROOT", Path("/data/feature_extraction"))
FEATURE_INDEX_CSV = FEATURE_ROOT / "feat_index.csv"
FEATURE_INDEX_XLSX = FEATURE_ROOT / "feat_index.xlsx"

TRAINING_DIR = PROJECT_ROOT / "ai" / "training"
TRAINING_OUTPUT_DIR = TRAINING_DIR / "outputs"
CV_OUTPUT_DIR = TRAINING_OUTPUT_DIR / "cv"


def ensure_runtime_directories() -> None:
    """Create lightweight runtime directories; large data stays outside Git."""
    for path in (
        RAW_PREPROCESSED_DIR,
        RAW_VIDEO_DIR,
        FRAMES_DIR,
        AUDIO_DIR,
        OCR_RESULTS_DIR,
        STT_RESULTS_DIR,
        PREPROCESS_LOG_DIR,
        FEATURE_ROOT,
        TRAINING_OUTPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
