#!/usr/bin/env python3
"""Build the preprocessing index from the canonical dataset manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_pipeline.data_manifest import read_data_manifest  # noqa: E402
from train_pipeline.project_paths import (  # noqa: E402
    AUDIO_DIR,
    DATA_LIST_XLSX,
    FRAMES_DIR,
    OCR_RESULTS_DIR,
    PREPROCESS_INDEX_CSV,
    PREPROCESS_INDEX_XLSX,
    RAW_VIDEO_DIR,
    STT_RESULTS_DIR,
)


INDEX_COLUMNS = [
    "file_name",
    "video_id",
    "video_path",
    "frames_dir",
    "frames_ext",
    "n_frames",
    "audio_dir",
    "original_wav",
    "vocal_wav",
    "non_vocal_wav",
    "stt_json",
    "ocr_jsonl",
    "exists_video",
    "exists_frames_dir",
    "exists_audio_dir",
    "exists_original_wav",
    "exists_vocal_wav",
    "exists_non_vocal_wav",
    "exists_stt_json",
    "exists_ocr_jsonl",
    "ok_audio_all",
]


def _count_frames(directory: Path) -> tuple[int, str]:
    jpg_count = sum(1 for _ in directory.glob("*.jpg")) if directory.is_dir() else 0
    png_count = sum(1 for _ in directory.glob("*.png")) if directory.is_dir() else 0
    extension = "jpg" if jpg_count else ("png" if png_count else "")
    return jpg_count + png_count, extension


def build_index_row(file_name: str) -> dict[str, Any]:
    file_name = Path(file_name).name
    video_id = Path(file_name).stem
    video_path = RAW_VIDEO_DIR / file_name
    frames_dir = FRAMES_DIR / video_id
    audio_dir = AUDIO_DIR / video_id
    original_wav = audio_dir / "original.wav"
    vocal_wav = audio_dir / "vocal.wav"
    non_vocal_wav = audio_dir / "non-vocal.wav"
    stt_json = STT_RESULTS_DIR / f"{video_id}.json"
    ocr_jsonl = OCR_RESULTS_DIR / f"{video_id}.jsonl"
    n_frames, frames_ext = _count_frames(frames_dir)

    exists_original = original_wav.is_file()
    exists_vocal = vocal_wav.is_file()
    exists_non_vocal = non_vocal_wav.is_file()
    return {
        "file_name": file_name,
        "video_id": video_id,
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "frames_ext": frames_ext,
        "n_frames": n_frames,
        "audio_dir": str(audio_dir),
        "original_wav": str(original_wav),
        "vocal_wav": str(vocal_wav),
        "non_vocal_wav": str(non_vocal_wav),
        "stt_json": str(stt_json),
        "ocr_jsonl": str(ocr_jsonl),
        "exists_video": video_path.is_file(),
        "exists_frames_dir": frames_dir.is_dir(),
        "exists_audio_dir": audio_dir.is_dir(),
        "exists_original_wav": exists_original,
        "exists_vocal_wav": exists_vocal,
        "exists_non_vocal_wav": exists_non_vocal,
        "exists_stt_json": stt_json.is_file(),
        "exists_ocr_jsonl": ocr_jsonl.is_file(),
        "ok_audio_all": exists_original and exists_vocal and exists_non_vocal,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_LIST_XLSX)
    parser.add_argument("--output-csv", type=Path, default=PREPROCESS_INDEX_CSV)
    parser.add_argument("--output-xlsx", type=Path, default=PREPROCESS_INDEX_XLSX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_data_manifest(args.manifest)
    rows = [build_index_row(file_name) for file_name in manifest["file_name"]]
    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    index.to_excel(args.output_xlsx, index=False, engine="openpyxl")

    print(f"Preprocessing index written: {args.output_csv}")
    print(f"Preprocessing index written: {args.output_xlsx}")
    print(
        "Samples: {total} | videos: {videos} | frames: {frames} | audio: {audio} | "
        "STT: {stt} | OCR: {ocr}".format(
            total=len(index),
            videos=int(index["exists_video"].sum()),
            frames=int(index["exists_frames_dir"].sum()),
            audio=int(index["ok_audio_all"].sum()),
            stt=int(index["exists_stt_json"].sum()),
            ocr=int(index["exists_ocr_jsonl"].sum()),
        )
    )


if __name__ == "__main__":
    main()
