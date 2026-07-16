#!/usr/bin/env python3
"""Download manifest-listed gameplay videos with yt-dlp."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_pipeline.project_paths import DATA_LIST_XLSX, RAW_PREPROCESSED_DIR  # noqa: E402


DEFAULT_OUTPUT_DIR = RAW_PREPROCESSED_DIR / "downloaded_full_videos"

YTDLP_FORMAT = (
    "bv*[height<=720][ext=mp4]+ba[ext=m4a]/"
    "b[height<=720][ext=mp4]/"
    "bv*[height<=720]+ba/"
    "best[height<=720]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DATA_LIST_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--concurrent-fragments", type=int, default=1)
    parser.add_argument("--cookies", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C.UTF-8"
    env["LANG"] = "C.UTF-8"
    return subprocess.run(command, capture_output=True, text=True, env=env)


def ytdlp_command_prefix() -> list[str]:
    module_check = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--version"],
        capture_output=True,
        text=True,
    )
    if module_check.returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]
    return ["yt-dlp"]


def download_with_ytdlp(
    url: str,
    output_template: Path,
    cookies: Path | None,
    overwrite: bool,
    concurrent_fragments: int,
) -> tuple[bool, str]:
    command = [
        *ytdlp_command_prefix(),
        "--no-playlist",
        "--retries", "2",
        "--fragment-retries", "2",
        "--concurrent-fragments", str(max(1, concurrent_fragments)),
        "--merge-output-format", "mp4",
        "--format", YTDLP_FORMAT,
        "--output", str(output_template),
        "--print", "after_move:filepath",
    ]
    if overwrite:
        command.append("--force-overwrites")
    if cookies:
        command.extend(["--cookies", str(cookies)])
    command.append(url)
    result = run(command)
    if result.returncode != 0:
        return False, ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    output = (result.stdout or "").strip()
    if not output:
        return False, (result.stderr or "yt-dlp output was empty.").strip()
    return True, output.splitlines()[-1].strip()


def transcode_to_h264(input_path: Path, output_path: Path) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_output.unlink(missing_ok=True)

    base_command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:a", "aac",
        "-b:a", "128k",
        "-f", "mp4",
    ]
    encoder_commands = [
        base_command[:-4] + ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"] + base_command[-4:] + [str(temp_output)],
        base_command[:-4] + ["-c:v", "libx264", "-preset", "fast", "-crf", "23"] + base_command[-4:] + [str(temp_output)],
    ]

    errors = []
    for command in encoder_commands:
        result = run(command)
        if result.returncode == 0 and temp_output.exists():
            temp_output.replace(output_path)
            return True, str(output_path)
        errors.append(((result.stdout or "") + "\n" + (result.stderr or "")).strip())
        temp_output.unlink(missing_ok=True)

    return False, errors[-1] if errors else "ffmpeg failed"


def download_video(
    url: str,
    destination: Path,
    cookies: Path | None,
    overwrite: bool,
    concurrent_fragments: int,
) -> tuple[str, str]:
    if destination.exists() and not overwrite:
        return destination.name, "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{destination.stem}_", dir=str(destination.parent)) as temp_dir:
        temp_dir_path = Path(temp_dir)
        output_template = temp_dir_path / "%(id)s.%(ext)s"
        ok, message = download_with_ytdlp(url, output_template, cookies, overwrite, concurrent_fragments)
        if not ok:
            detail = message.splitlines()[-1] if message else "yt-dlp failed"
            return destination.name, f"failed: {detail}"

        downloaded_path = Path(message)
        if not downloaded_path.exists():
            candidates = sorted(temp_dir_path.glob("*"))
            downloaded_path = candidates[0] if candidates else downloaded_path
        if not downloaded_path.exists():
            return destination.name, f"failed: downloaded file not found: {message}"

        ok, message = transcode_to_h264(downloaded_path, destination)
        if not ok:
            detail = message.splitlines()[-1] if message else "ffmpeg failed"
            return destination.name, f"failed: {detail}"

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
            jobs.append((url, args.output_dir / file_name, cookies, args.overwrite, args.concurrent_fragments))
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
