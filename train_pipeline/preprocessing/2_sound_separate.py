#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocessing script for the game-content identification pipeline."""

import os
import sys
import gc
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import torch
from tqdm.auto import tqdm


# ==================== Configuration variables ====================
# Path setup.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_pipeline.project_paths import (  # noqa: E402
    AUDIO_DIR,
    DATA_LIST_XLSX,
    PREPROCESS_LOG_DIR,
    RAW_PREPROCESSED_DIR,
)

OUTPUT_ROOT_PATH = str(RAW_PREPROCESSED_DIR)
OUTPUT_AUDIO_PATH = str(AUDIO_DIR)
DEMUCS_OUT = RAW_PREPROCESSED_DIR / ".demucs_work"
INDEX_RAW_VID_LIST = str(DATA_LIST_XLSX)
INDEX_RAW_PROCESS_INFO = str(PREPROCESS_LOG_DIR / "raw_vid_process_info.csv")
INDEX_SOUND_SEPARATE_INFO = str(PREPROCESS_LOG_DIR / "sound_separate_info.csv")

TEST_MODE = False  # Test mode.
TEST_MODE_SAMPLE_COUNT = 1  # Test mode.

# Audio source separation.
DEMUCS_MODEL = "htdemucs_ft"  # Model setup.
DEVICE = "cuda"
MEMORY_FLUSH_INTERVAL = 10  # Memory management.


# ==================== Logging colors ====================
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


# ==================== Utility functions ====================
def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Helper function for run command."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def scan_audio_directories() -> List[str]:
    """Helper function for scan audio directories."""
    audio_root = Path(OUTPUT_AUDIO_PATH)
    
    if not audio_root.exists():
        print(f"{Colors.RED}Audio directory not found: {audio_root}{Colors.RESET}")
        return []
    
    video_names = []
    
    # Scan all subfolders.
    for video_dir in audio_root.iterdir():
        if not video_dir.is_dir():
            continue
        
        original_wav = video_dir / "original.wav"
        vocal_wav = video_dir / "vocal.wav"
        non_vocal_wav = video_dir / "non-vocal.wav"
        
        # Process item.
        if original_wav.exists() and not (vocal_wav.exists() and non_vocal_wav.exists()):
            video_names.append(video_dir.name)
    
    return sorted(video_names)


def flush_memory():
    """Helper function for flush memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Audio source separation ====================
def separate_audio(video_name: str) -> Dict:
    """Helper function for separate audio."""
    audio_dir = Path(OUTPUT_AUDIO_PATH) / video_name
    original_wav = audio_dir / "original.wav"
    vocal_wav = audio_dir / "vocal.wav"
    non_vocal_wav = audio_dir / "non-vocal.wav"
    
    # Skip if already processed.
    if vocal_wav.exists() and non_vocal_wav.exists():
        return {
            "video_name": video_name,
            "status": "skipped",
            "message": "Already processed"
        }
    
    # Validation check.
    if not original_wav.exists():
        return {
            "video_name": video_name,
            "status": "failed",
            "error": f"original.wav not found: {original_wav}"
        }
    
    try:
        # Temporary output directory.
        if DEMUCS_OUT.exists():
            for item in DEMUCS_OUT.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
        else:
            DEMUCS_OUT.mkdir(parents=True, exist_ok=True)
        


        cmd_demucs = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n", DEMUCS_MODEL,
            "--two-stems=vocals",
            "-d", DEVICE,
            "-o", str(DEMUCS_OUT),
            str(original_wav)
        ]
        
        code, out, err = run_command(cmd_demucs)
        
        if code != 0:
            return {
                "video_name": video_name,
                "status": "failed",
                "error": f"Demucs execution failed. Code: {code}, Error: {err}"
            }
        
        # Move files.
        model_output_dirs = list(DEMUCS_OUT.iterdir())
        if not model_output_dirs:
            return {
                "video_name": video_name,
                "status": "failed",
                "error": f"Demucs output folder not found: {DEMUCS_OUT}"
            }
        
        sep_dir_parent = model_output_dirs[0]
        sep_dir = sep_dir_parent / original_wav.stem
        
        if not sep_dir.exists():
            return {
                "video_name": video_name,
                "status": "failed",
                "error": f"Separation output folder not found: {sep_dir}"
            }
        
        src_vocal = sep_dir / "vocals.wav"
        src_non_vocal = sep_dir / "no_vocals.wav"
        
        if not src_vocal.exists() or not src_non_vocal.exists():
            return {
                "video_name": video_name,
                "status": "failed",
                "error": f"Separated files not found. Vocals: {src_vocal.exists()}, Non-vocals: {src_non_vocal.exists()}"
            }
        
        # Move files.
        shutil.move(str(src_vocal), str(vocal_wav))
        shutil.move(str(src_non_vocal), str(non_vocal_wav))
        
        # Clean temporary folders.
        shutil.rmtree(sep_dir_parent, ignore_errors=True)
        
        return {
            "video_name": video_name,
            "status": "success",
            "vocal_path": str(vocal_wav),
            "non_vocal_path": str(non_vocal_wav)
        }
    
    except Exception as e:
        return {
            "video_name": video_name,
            "status": "failed",
            "error": str(e)
        }


# ==================== Progress saving ====================
def save_progress(results: List[Dict]) -> None:
    """Helper function for save progress."""
    df = pd.DataFrame(results)
    Path(INDEX_SOUND_SEPARATE_INFO).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_SOUND_SEPARATE_INFO, index=False, encoding='utf-8')
    print(f"\n{Colors.GREEN}Process status saved: {INDEX_SOUND_SEPARATE_INFO}{Colors.RESET}")


# ==================== Main execution ====================
def main():
    """Helper function for main."""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}  Audio source-separation pipeline started{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")
    
    # Configuration.
    print(f"{Colors.CYAN}[Configuration]{Colors.RESET}")
    print(f"  Model: {DEMUCS_MODEL}")
    print(f"  Device: {DEVICE}")
    print("  Output format: WAV")
    print(f"  Memory flush interval: every {MEMORY_FLUSH_INTERVAL} items")
    
    # Process item.
    print(f"\n{Colors.CYAN}[1/3] Scanning inputs...{Colors.RESET}")
    video_list = scan_audio_directories()
    
    if len(video_list) == 0:
        print(f"{Colors.YELLOW}No original.wav files require processing.{Colors.RESET}")
        return
    
    print(f"  Found {len(video_list)} audio files")
    
    # Test mode.
    if TEST_MODE:
        video_list = video_list[:TEST_MODE_SAMPLE_COUNT]
        print(f"{Colors.YELLOW}[TEST MODE] Processing the first {TEST_MODE_SAMPLE_COUNT} items only.{Colors.RESET}")
    
    # Temporary output directory.
    print(f"\n{Colors.CYAN}[2/3] Preparing the Demucs work directory...{Colors.RESET}")
    DEMUCS_OUT.mkdir(parents=True, exist_ok=True)
    print(f"{Colors.GREEN}  Work directory ready{Colors.RESET}")
    
    # Process item.
    print(f"\n{Colors.CYAN}[3/3] Separating audio sources...{Colors.RESET}")
    results = []
    
    with tqdm(video_list, desc="Audio separation", colour="green") as pbar:
        for idx, video_name in enumerate(pbar, start=1):
            # Audio source separation.
            result = separate_audio(video_name)
            results.append(result)
            

            if result["status"] == "success":
                pbar.set_postfix_str(f"✓ {video_name}")
            elif result["status"] == "skipped":
                pbar.set_postfix_str(f"⊘ {video_name} (skipped)")
            else:
                pbar.set_postfix_str(f"✗ {video_name} (failed)")

                tqdm.write(f"{Colors.RED}  [Error] {video_name}: {result.get('error')}{Colors.RESET}")
            
            # Memory management.
            if idx % MEMORY_FLUSH_INTERVAL == 0:
                flush_memory()
                pbar.set_postfix_str(f"Memory cleared after {idx} items")
    
    # Memory management.
    flush_memory()
    
    # Save results and summary.
    save_progress(results)
    
    # Compute statistics.
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.GREEN}  Processing complete.{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")
    
    # Print failed items.
    if failed_count > 0:
        print(f"\n{Colors.RED}Failed items:{Colors.RESET}")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['video_name']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
