"""Preprocessing script for the game-content identification pipeline."""



import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import pandas as pd
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
    FRAMES_DIR,
    PREPROCESS_LOG_DIR,
    RAW_PREPROCESSED_DIR,
    RAW_VIDEO_DIR,
)

RAW_VID_PATH = str(RAW_VIDEO_DIR)
OUTPUT_ROOT_PATH = str(RAW_PREPROCESSED_DIR)
OUTPUT_FRAME_PATH = str(FRAMES_DIR)
OUTPUT_AUDIO_PATH = str(AUDIO_DIR)
INDEX_RAW_VID_LIST = str(DATA_LIST_XLSX)
INDEX_RAW_PROCESS_INFO = str(PREPROCESS_LOG_DIR / "raw_vid_process_info.csv")

FRAME_INTERVAL_SEC = 0.25  # Frame Extraction Interval Seconds (4 FPS)
# FRAME_SIZE = (360, 203)  # width, height
SAMPLE_RATE = 44_100  # Audio Sample Rate

NUM_WORKERS = 16  # Total workers for parallel processing

TEST_MODE = False  # If True, process only first 1 video for testing

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


def check_gpu_available() -> bool:
    """Helper function for check gpu available."""
    code, out, _ = run_command(["nvidia-smi", "-L"])
    return code == 0


def load_video_list(manifest_path: str, test_mode: bool = False) -> List[str]:
    """Helper function for load video list."""
    df = pd.read_excel(manifest_path, engine="openpyxl")
    
    # Handle filename extensions.
    video_names = df['file_name'].apply(lambda x: Path(x).stem).tolist()
    
    if test_mode:
        print(f"{Colors.YELLOW}[TEST MODE] Processing the first video only.{Colors.RESET}")
        return video_names[:1]
    
    return video_names


def ensure_dir(path: Path) -> None:
    """Helper function for ensure dir."""
    path.mkdir(parents=True, exist_ok=True)


# ==================== Frame extraction ====================
def extract_frames(video_path: Path, video_name: str) -> Dict:
    """Helper function for extract frames."""
    output_dir = Path(OUTPUT_FRAME_PATH) / video_name
    
    # Skip if already processed.
    if output_dir.exists() and len(list(output_dir.glob("*.jpg"))) > 0:
        return {
            "video_name": video_name,
            "status": "skipped",
            "frames_dir": str(output_dir),
            "message": "Already processed"
        }
    
    ensure_dir(output_dir)
    
    try:

        fps = 1.0 / FRAME_INTERVAL_SEC
        
        # Fallback handling.

        cmd_gpu = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", str(video_path),

            "-vf", f"hwdownload,format=nv12,fps={fps}",
            "-start_number", "1",
            "-q:v", "2",
            str(output_dir / "%d.jpg"),
            "-hide_banner",
            "-loglevel", "error"
        ]
        

        code, _, err = run_command(cmd_gpu)
        
        # Fallback handling.
        if code != 0:
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"fps={fps}",  # Filter records.
                "-start_number", "1",
                "-q:v", "2",
                str(output_dir / "%d.jpg"),
                "-hide_banner",
                "-loglevel", "error"
            ]
            code, _, err = run_command(cmd)
        
        if code != 0:
            return {
                "video_name": video_name,
                "status": "failed",
                "error": err
            }
        
        # Validation check.
        frame_count = len(list(output_dir.glob("*.jpg")))
        
        return {
            "video_name": video_name,
            "status": "success",
            "frames_dir": str(output_dir),
            "frame_count": frame_count
        }
    
    except Exception as e:
        return {
            "video_name": video_name,
            "status": "failed",
            "error": str(e)
        }


# ==================== Audio extraction ====================
def extract_audio(video_path: Path, video_name: str) -> Dict:
    """Helper function for extract audio."""
    audio_output_dir = Path(OUTPUT_AUDIO_PATH) / video_name
    original_wav = audio_output_dir / "original.wav"
    
    # Skip if already processed.
    if original_wav.exists():
        return {
            "video_name": video_name,
            "status": "skipped",
            "audio_dir": str(audio_output_dir),
            "message": "Already processed"
        }
    
    ensure_dir(audio_output_dir)
    
    try:
        # Audio extraction.
        cmd_audio = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(original_wav),
            "-hide_banner",
            "-loglevel", "error"
        ]
        
        code, _, err = run_command(cmd_audio)
        if code != 0:
            return {
                "video_name": video_name,
                "status": "failed",
                "error": f"Audio extraction failed: {err}"
            }
        
        return {
            "video_name": video_name,
            "status": "success",
            "audio_dir": str(audio_output_dir),
            "audio_file": str(original_wav)
        }
    
    except Exception as e:
        return {
            "video_name": video_name,
            "status": "failed",
            "error": str(e)
        }



def process_frame_task(video_name: str) -> Dict:
    """Helper function for process frame task."""
    video_path = Path(RAW_VID_PATH) / f"{video_name}.mp4"
    
    if not video_path.exists():
        return {
            "video_name": video_name,
            "task": "frame",
            "status": "failed",
            "error": f"Video file not found: {video_path}"
        }
    
    result = extract_frames(video_path, video_name)
    result["task"] = "frame"
    return result




def process_audio_task(video_name: str) -> Dict:
    """Helper function for process audio task."""
    video_path = Path(RAW_VID_PATH) / f"{video_name}.mp4"
    
    if not video_path.exists():
        return {
            "video_name": video_name,
            "task": "audio",
            "status": "failed",
            "error": f"Video file not found: {video_path}"
        }
    
    result = extract_audio(video_path, video_name)
    result["task"] = "audio"
    return result


# ==================== Progress saving ====================
def save_progress(results: List[Dict]) -> None:
    """Helper function for save progress."""
    df = pd.DataFrame(results)
    Path(INDEX_RAW_PROCESS_INFO).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_RAW_PROCESS_INFO, index=False, encoding='utf-8')
    print(f"\n{Colors.GREEN}Process status saved: {INDEX_RAW_PROCESS_INFO}{Colors.RESET}")


# ==================== Main execution ====================
def main():
    """Helper function for main."""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}  Video preprocessing pipeline started{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")
    
    # Check GPU availability.
    gpu_available = check_gpu_available()
    if gpu_available:
        print(f"{Colors.GREEN}CUDA GPU detected{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}CUDA GPU not found; running in CPU mode.{Colors.RESET}")
    
    # Create output directories.
    for path in [OUTPUT_FRAME_PATH, OUTPUT_AUDIO_PATH]:
        ensure_dir(Path(path))
    
    # Load the video list.
    print(f"\n{Colors.CYAN}[1/3] Loading the video list...{Colors.RESET}")
    video_list = load_video_list(INDEX_RAW_VID_LIST, test_mode=TEST_MODE)
    print(f"  Found {len(video_list)} videos")
    
    if len(video_list) == 0:
        print(f"{Colors.RED}No videos are available for processing.{Colors.RESET}")
        return
    
    # Save results.
    all_results = []
    
    # ==================== Process item ====================
    print(f"\n{Colors.CYAN}[2/3] Extracting audio{Colors.RESET}")
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as audio_executor:
        audio_futures = {
            audio_executor.submit(process_audio_task, vid_name): vid_name 
            for vid_name in video_list
        }
        
        with tqdm(total=len(video_list), desc="Audio processing", colour="green") as pbar:
            for future in as_completed(audio_futures):
                result = future.result()
                all_results.append(result)
                
                if result["status"] == "success":
                    pbar.set_postfix_str(f"✓ {result['video_name']}")
                elif result["status"] == "skipped":
                    pbar.set_postfix_str(f"⊘ {result['video_name']} (skipped)")
                else:
                    pbar.set_postfix_str(f"✗ {result['video_name']} (failed)")
                
                pbar.update(1)
    
    # ==================== Frame extraction ====================
    print(f"\n{Colors.CYAN}[3/3] Extracting frames with multiprocessing{Colors.RESET}")
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as frame_executor:
        frame_futures = {
            frame_executor.submit(process_frame_task, vid_name): vid_name 
            for vid_name in video_list
        }
        
        with tqdm(total=len(video_list), desc="Frame extraction", colour="blue") as pbar:
            for future in as_completed(frame_futures):
                result = future.result()
                all_results.append(result)
                
                if result["status"] == "success":
                    pbar.set_postfix_str(f"✓ {result['video_name']}")
                elif result["status"] == "skipped":
                    pbar.set_postfix_str(f"⊘ {result['video_name']} (skipped)")
                else:
                    pbar.set_postfix_str(f"✗ {result['video_name']} (failed)")
                
                pbar.update(1)
    
    # ==================== Save results and summary ====================
    save_progress(all_results)
    
    # Compute statistics.
    success_count = sum(1 for r in all_results if r["status"] == "success")
    skipped_count = sum(1 for r in all_results if r["status"] == "skipped")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    
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
        for r in all_results:
            if r["status"] == "failed":
                print(f"  - {r['video_name']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
