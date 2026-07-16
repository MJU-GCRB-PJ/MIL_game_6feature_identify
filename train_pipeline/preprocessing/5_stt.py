#!/usr/bin/env python3
"""Generate local Whisper STT JSON files for the canonical dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_pipeline.data_manifest import read_data_manifest  # noqa: E402
from train_pipeline.project_paths import AUDIO_DIR, DATA_LIST_XLSX, STT_RESULTS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_LIST_XLSX)
    parser.add_argument("--model", default="turbo", help="faster-whisper model name or local path")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--language", default=None, help="Optional language code such as en or ko")
    parser.add_argument("--only-file", default="", help="Process one exact manifest file_name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe(audio_path: Path, model: WhisperModel, language: str | None) -> dict[str, Any]:
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )
    output_segments = [
        {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        }
        for segment in segments
        if segment.text.strip()
    ]
    return {
        "source_audio": str(audio_path),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "segments": output_segments,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest = read_data_manifest(args.manifest)
    if args.only_file:
        manifest = manifest.loc[manifest["file_name"] == args.only_file]
        if manifest.empty:
            raise ValueError(f"file_name is not present in the manifest: {args.only_file}")

    device = resolve_device(args.device)
    compute_type = args.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    completed = skipped = missing = failed = 0
    for file_name in tqdm(manifest["file_name"], desc="STT transcription"):
        video_id = Path(file_name).stem
        audio_path = AUDIO_DIR / video_id / "vocal.wav"
        output_path = STT_RESULTS_DIR / f"{video_id}.json"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if not audio_path.exists():
            tqdm.write(f"Missing vocal audio: {audio_path}")
            missing += 1
            continue
        try:
            write_json(output_path, transcribe(audio_path, model, args.language))
            completed += 1
        except Exception as error:
            tqdm.write(f"STT failed for {file_name}: {error}")
            failed += 1

    print(
        f"STT complete: processed={completed} skipped={skipped} "
        f"missing_audio={missing} failed={failed}"
    )


if __name__ == "__main__":
    main()
