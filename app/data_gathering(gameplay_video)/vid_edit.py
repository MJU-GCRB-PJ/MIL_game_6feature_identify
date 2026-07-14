#!/usr/bin/env python3
"""Cut configured gameplay intervals using data/data_list.xlsx as the manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.project_paths import DATA_LIST_XLSX, RAW_PREPROCESSED_DIR, RAW_VIDEO_DIR  # noqa: E402


DEFAULT_SOURCE_DIR = RAW_PREPROCESSED_DIR / "downloaded_full_videos"


def normalize_time(value: str) -> str:
    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        hours, minutes = divmod(minutes, 60)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time value: {value}")
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_ranges(value: object) -> list[tuple[str, str]]:
    if value is None or pd.isna(value):
        return []
    ranges = []
    for item in str(value).split(","):
        if "-" not in item:
            continue
        start, end = item.split("-", maxsplit=1)
        ranges.append((normalize_time(start), normalize_time(end)))
    return ranges


def run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed")


def cut_video(source: Path, destination: Path, ranges: list[tuple[str, str]], overwrite: bool) -> str:
    if destination.exists() and not overwrite:
        return "skipped"
    if not source.exists():
        return "missing"
    if not ranges:
        return "no_ranges"

    destination.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    list_path = destination.with_suffix(destination.suffix + ".parts.txt")
    try:
        for index, (start, end) in enumerate(ranges):
            part = destination.with_suffix(f".part{index}.mp4")
            parts.append(part)
            run([
                "ffmpeg", "-y", "-ss", start, "-to", end, "-i", str(source),
                "-c", "copy", "-avoid_negative_ts", "1", str(part),
            ])
        if len(parts) == 1:
            parts[0].replace(destination)
        else:
            list_path.write_text(
                "".join(f"file '{part.as_posix()}'\n" for part in parts),
                encoding="utf-8",
            )
            run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
                "-c", "copy", str(destination),
            ])
        return "complete"
    finally:
        for part in parts:
            part.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_LIST_XLSX)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_excel(args.manifest, engine="openpyxl")
    required = {"file_name", "archive_area"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise KeyError(f"Manifest is missing required columns: {missing_columns}")

    tasks = []
    for row in manifest.itertuples(index=False):
        file_name = str(getattr(row, "file_name")).strip()
        ranges = parse_ranges(getattr(row, "archive_area"))
        tasks.append((args.source_dir / file_name, args.output_dir / file_name, ranges, args.overwrite))

    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(cut_video, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Editing videos"):
            status = future.result()
            counts[status] = counts.get(status, 0) + 1
    print("Video editing summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
