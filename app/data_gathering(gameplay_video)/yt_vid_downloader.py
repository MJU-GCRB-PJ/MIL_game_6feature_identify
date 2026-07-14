#!/usr/bin/env python3
"""Download manifest-listed gameplay videos with yt-dlp."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.project_paths import DATA_LIST_XLSX, RAW_PREPROCESSED_DIR  # noqa: E402


DEFAULT_OUTPUT_DIR = RAW_PREPROCESSED_DIR / "downloaded_full_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_LIST_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cookies", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def download_video(url: str, destination: Path, cookies: Path | None, overwrite: bool) -> tuple[str, str]:
    if destination.exists() and not overwrite:
        return destination.name, "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "yt-dlp",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--format", "bestvideo*+bestaudio/best",
        "--output", str(destination),
    ]
    if overwrite:
        command.append("--force-overwrites")
    if cookies:
        command.extend(["--cookies", str(cookies)])
    command.append(url)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "yt-dlp failed"
        return destination.name, f"failed: {message}"
    return destination.name, "complete"


def main() -> None:
    args = parse_args()
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is not installed or is not on PATH.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or is not on PATH.")

    cookies = args.cookies
    if cookies is None and os.environ.get("YT_DLP_COOKIES_FILE"):
        cookies = Path(os.environ["YT_DLP_COOKIES_FILE"]).expanduser()
    if cookies is not None and not cookies.exists():
        raise FileNotFoundError(f"Cookies file not found: {cookies}")

    manifest = pd.read_excel(args.manifest, engine="openpyxl")
    required = {"file_name", "video_link"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise KeyError(f"Manifest is missing required columns: {missing_columns}")

    jobs = []
    for row in manifest.itertuples(index=False):
        file_name = str(getattr(row, "file_name")).strip()
        url = str(getattr(row, "video_link") or "").strip()
        if file_name and url.startswith(("https://youtu.be/", "https://www.youtube.com/")):
            jobs.append((url, args.output_dir / file_name, cookies, args.overwrite))
    if args.limit > 0:
        jobs = jobs[: args.limit]

    counts: dict[str, int] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(download_video, *job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading videos"):
            file_name, status = future.result()
            key = status.split(":", maxsplit=1)[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "failed":
                failures.append((file_name, status))

    print("Download summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    for file_name, status in failures:
        print(f"{file_name}: {status}")


if __name__ == "__main__":
    main()
