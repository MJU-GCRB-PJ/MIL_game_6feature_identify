"""Preprocessing script for the game-content identification pipeline."""

import json
import gc
import time
import os
import re
import sys
import warnings
from io import StringIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import queue

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import logging as transformers_logging
from natsort import natsorted

# ==================== Path setup ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.project_paths import FRAMES_DIR, OCR_RESULTS_DIR, PREPROCESS_LOG_DIR  # noqa: E402

MODEL_DIR = SCRIPT_DIR / "model" / "DeepSeek-OCR"
FRAME_PATH = FRAMES_DIR
OUTPUT_PATH = OCR_RESULTS_DIR
LOG_CSV = PREPROCESS_LOG_DIR / "ocr_deepseek_process_info.csv"

# ==================== Configuration ====================
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
BASE_SIZE = 1024
IMAGE_SIZE = 640
CROP_MODE = True
USE_EVAL_MODE = False  # Save output.

# ==================== Parallel processing ====================
NUM_WORKERS = 2  # Parallel processing.
BATCH_SIZE = 16   # Memory management.

# ==================== Configuration ====================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TEST_MODE = False  # Test mode.
TEST_IMG_NUM = 10  # Test mode.

SKIP_EXISTING = True  # Process item.

# =========================
# Metric handling.
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Configuration.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ==================== Utility functions ====================


def scan_game_folders() -> List[Path]:
    """Helper function for scan game folders."""
    if not FRAME_PATH.exists():
        raise FileNotFoundError(f"Frame path not found: {FRAME_PATH}")
    
    folders = [p for p in FRAME_PATH.iterdir() if p.is_dir()]
    folders = natsorted(folders)  # Sort values.
    
    if TEST_MODE:
        print(f"[TEST_MODE] Processing one folder: {folders[0].name if folders else 'None'}")
        folders = folders[:1]
    
    return folders


def list_images(img_dir: Path) -> List[Path]:
    """Helper function for list images."""
    files = []
    for p in img_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            frame_num = extract_frame_number(p)
            if frame_num % 4 == 0:
                files.append(p)
    
    files = natsorted(files)
    
    # Test mode.
    if TEST_MODE and len(files) > TEST_IMG_NUM:
        print(f"[TEST_MODE] Limiting images from {len(files)} to {TEST_IMG_NUM}")
        files = files[:TEST_IMG_NUM]
    
    return files


def extract_frame_number(img_path: Path) -> int:
    """Helper function for extract frame number."""
    try:
        return int(img_path.stem)
    except ValueError:
        return 0


def normalize_bbox_to_1000(bbox: List[int], img_width: int, img_height: int) -> List[int]:
    """Helper function for normalize bbox to 1000."""
    x0, y0, x1, y1 = bbox
    norm_x0 = int((x0 / img_width) * 1000)
    norm_y0 = int((y0 / img_height) * 1000)
    norm_x1 = int((x1 / img_width) * 1000)
    norm_y1 = int((y1 / img_height) * 1000)
    

    return [
        max(0, min(1000, norm_x0)),
        max(0, min(1000, norm_y0)),
        max(0, min(1000, norm_x1)),
        max(0, min(1000, norm_y1))
    ]


# ==================== Utility functions ====================


def load_deepseek_model(device: str = "cuda") -> Tuple[AutoModel, AutoTokenizer]:
    """Helper function for load deepseek model."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Please download the model first using the notebook."
        )
    
    print(f"🔧 Loading DeepSeek-OCR from {MODEL_DIR}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # Configuration.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Configuration.
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token_id = tokenizer.pad_token_id
    
    # Filter records.
    warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
    transformers_logging.set_verbosity_error()
    
    try:
        model = AutoModel.from_pretrained(
            str(MODEL_DIR),
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).eval().to(device)
        print("✓ Model loaded with flash_attention_2")
    except Exception as e:
        print(f"⚠ Flash attention failed, using default: {repr(e)}")
        model = AutoModel.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).eval().to(device)
        print("✓ Model loaded with default attention")
    
    # Configuration.
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def process_image_batch(
    image_paths: List[Path],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda"
) -> List[Dict]:
    """Helper function for process image batch."""
    results = []
    
    # Batch processing.

    for img_path in image_paths:
        try:
            result = process_single_image(img_path, model, tokenizer, device)
            results.append(result)
        except Exception as e:
            frame_num = extract_frame_number(img_path)
            results.append({
                "frame_num": frame_num,
                "tokens": [],
                "image_size": [0, 0],
                "status": f"error: {type(e).__name__}"
            })
    
    return results


def process_single_image(
    image_path: Path,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda"
) -> Dict:
    """Helper function for process single image."""
    frame_num = extract_frame_number(image_path)
    
    try:
        # Save output.
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        


        pid = os.getpid()
        temp_output = OUTPUT_PATH / "temp" / f"pid_{pid}_frame_{frame_num}"
        temp_output.mkdir(parents=True, exist_ok=True)
        

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_stdout = StringIO()
        sys.stderr = captured_stderr = StringIO()
        

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                # Save output.
                model.infer(
                    tokenizer,
                    prompt=PROMPT,
                    image_file=str(image_path),
                    output_path=str(temp_output),
                    base_size=BASE_SIZE,
                    image_size=IMAGE_SIZE,
                    crop_mode=CROP_MODE,
                    save_results=True,  # Create required output.
                    test_compress=False,
                )
            finally:

                sys.stdout = old_stdout
                sys.stderr = old_stderr
                stdout_text = captured_stdout.getvalue()
                stderr_text = captured_stderr.getvalue()

                combined_output = stdout_text + "\n" + stderr_text
        

        tokens = parse_deepseek_output(temp_output, img_width, img_height, combined_output, frame_num)
        

        cleanup_temp_files(temp_output)
        
        return {
            "frame_num": frame_num,
            "tokens": tokens,
            "image_size": [img_width, img_height],
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error processing frame {frame_num}: {repr(e)}")
        return {
            "frame_num": frame_num,
            "tokens": [],
            "image_size": [0, 0],
            "status": f"error: {repr(e)}"
        }


def parse_deepseek_output(
    output_dir: Path,
    img_width: int,
    img_height: int,
    captured_text: str,
    frame_num: int = 0
) -> List[Dict]:
    """Helper function for parse deepseek output."""
    tokens = []
    
    # Split or separate data.
    lines = captured_text.split('\n')
    

    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>'
    
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.search(pattern, line)
        
        if match:
            ref_type = match.group(1).strip()
            x0 = int(match.group(2))
            y0 = int(match.group(3))
            x1 = int(match.group(4))
            y1 = int(match.group(5))
            
            # Extract required data.
            text_content = ""
            if i + 1 < len(lines):
                text_content = lines[i + 1].strip()
            

            if ref_type.lower() == "text" and text_content:
                # Normalize values.
                norm_bbox = normalize_bbox_to_1000([x0, y0, x1, y1], img_width, img_height)
                
                tokens.append({
                    "text": text_content,
                    "bbox": norm_bbox,
                    "confidence": 1.0
                })
        
        i += 1
    
    return tokens


def cleanup_temp_files(temp_dir: Path):
    """Helper function for cleanup temp files."""
    try:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠ Failed to cleanup temp files: {repr(e)}")


def load_existing_results(jsonl_path: Path) -> Dict[int, Dict]:
    """Helper function for load existing results."""
    existing_results = {}
    if not jsonl_path.exists():
        return existing_results
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    frame_num = result.get('frame_num', 0)
                    existing_results[frame_num] = result
    except Exception as e:
        print(f"⚠ Failed to load existing results: {repr(e)}")
    
    return existing_results


def filter_frames_to_process(image_paths: List[Path], existing_results: Dict[int, Dict]) -> List[Path]:
    """Helper function for filter frames to process."""
    frames_to_process = []
    
    for img_path in image_paths:
        frame_num = extract_frame_number(img_path)
        
        # Process item.
        if frame_num not in existing_results or existing_results[frame_num].get('status') != 'success':
            frames_to_process.append(img_path)
    
    return frames_to_process


def background_writer(write_queue: queue.Queue, output_path: Path, stop_event: threading.Event):
    """Helper function for background writer."""
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            while not stop_event.is_set() or not write_queue.empty():
                try:
                    # Queue handling.
                    batch_results = write_queue.get(timeout=0.1)
                    
                    # Batch processing.
                    for result in batch_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()  # Write output.
                    
                    write_queue.task_done()
                    
                except queue.Empty:
                    continue
    except Exception as e:
        print(f"\n⚠ Background writer error: {repr(e)}")


# ==================== Process item ====================


def process_folder(folder_path: Path) -> Dict[str, any]:
    """Helper function for process folder."""
    folder_name = folder_path.name
    output_jsonl = OUTPUT_PATH / f"{folder_name}.jsonl"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:

        all_image_paths = list_images(folder_path)
        if not all_image_paths:
            print(f"⚠ No images found: {folder_name}")
            return {
                "folder_name": folder_name,
                "frame_count": 0,
                "processing_time": 0.0,
                "status": "no_images"
            }
        
        # Load input.
        existing_results = load_existing_results(output_jsonl)
        
        # Process item.
        image_paths = filter_frames_to_process(all_image_paths, existing_results)
        
        # Process item.
        if not image_paths:
            print(f"⏭ Skipped (all frames already processed): {folder_name}")
            return {
                "folder_name": folder_name,
                "frame_count": len(existing_results),
                "processing_time": 0.0,
                "status": "skipped"
            }
        
        print(f"\n{'='*60}")
        print(f"📁 Processing: {folder_name}")
        print(f"📊 Total frames: {len(all_image_paths)}")
        print(f"✅ Already processed: {len(existing_results)}")
        print(f"🔄 To process: {len(image_paths)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load the model.
        model, tokenizer = load_deepseek_model(device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Create output directories.
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        
        # Batch processing.
        write_queue = queue.Queue(maxsize=10)  # Batch processing.
        stop_event = threading.Event()
        temp_jsonl = OUTPUT_PATH / f"{folder_name}_temp.jsonl"
        
        # Save output.
        writer_thread = threading.Thread(
            target=background_writer,
            args=(write_queue, temp_jsonl, stop_event),
            daemon=True
        )
        writer_thread.start()
        
        # Batch processing.
        new_results = []
        
        # Batch processing.
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"OCR {folder_name}"):
            batch_paths = image_paths[i:i + BATCH_SIZE]
            try:
                # Batch processing.
                batch_results = process_image_batch(batch_paths, model, tokenizer, device)
                new_results.extend(batch_results)
                
                # Queue handling.
                write_queue.put(batch_results)
                
            except Exception as e:
                print(f"\n❌ Error processing batch {i//BATCH_SIZE + 1}: {type(e).__name__}: {str(e)}")
                # Batch processing.
                individual_results = []
                for img_path in batch_paths:
                    try:
                        result = process_single_image(img_path, model, tokenizer, device)
                        new_results.append(result)
                        individual_results.append(result)
                            
                    except Exception as e2:
                        frame_num = extract_frame_number(img_path)
                        error_result = {
                            "frame_num": frame_num,
                            "tokens": [],
                            "image_size": [0, 0],
                            "status": f"error: {type(e2).__name__}"
                        }
                        new_results.append(error_result)
                        individual_results.append(error_result)
                
                # Queue handling.
                if individual_results:
                    write_queue.put(individual_results)
        
        # Batch processing.
        stop_event.set()
        write_queue.join()  # Queue handling.
        writer_thread.join(timeout=30)
        
        # Merge data.
        all_results = list(existing_results.values())
        

        for new_result in new_results:
            frame_num = new_result['frame_num']
            existing_results[frame_num] = new_result
        
        # Sort values.
        all_results = sorted(existing_results.values(), key=lambda x: x["frame_num"])
        
        # Write output.
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # Delete temporary data.
        if temp_jsonl.exists():
            temp_jsonl.unlink()
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in all_results if r["status"] == "success")
        
        print(f"✅ Completed: {folder_name}")
        print(f"   - Frames: {success_count}/{len(all_results)}")
        print(f"   - New processed: {len(new_results)}")
        print(f"   - Time: {elapsed:.1f}s ({elapsed/len(image_paths):.3f}s/frame)")
        print(f"   - Output: {output_jsonl}")
        
        # Memory management.
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return {
            "folder_name": folder_name,
            "frame_count": success_count,
            "processing_time": elapsed,
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error processing {folder_name}: {repr(e)}")
        import traceback
        traceback.print_exc()
        return {
            "folder_name": folder_name,
            "frame_count": 0,
            "processing_time": 0.0,
            "status": f"error: {repr(e)}"
        }


# ==================== Main execution ====================


def main():
    """Helper function for main."""
    # Configuration.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Configuration.
    
    print("\n" + "="*70)
    print("🎮 Game Frame OCR Pipeline with DeepSeek-OCR")
    print("="*70)
    print(f"📂 Input: {FRAME_PATH}")
    print(f"💾 Output: {OUTPUT_PATH}")
    print(f"🔧 Model: {MODEL_DIR}")
    print(f"🧪 Test mode: {TEST_MODE}")
    print(f"📦 Output format: LayoutXLM-compatible (bbox: 0-1000)")
    print(f"🔧 Batch size: {BATCH_SIZE}")
    print(f"👷 Workers: {NUM_WORKERS}")
    print("="*70 + "\n")
    

    game_folders = scan_game_folders()
    if not game_folders:
        print("❌ No game folders found!")
        return
    
    print(f"📊 Found {len(game_folders)} folder(s) to process\n")
    
    # Parallel processing.
    all_results = []
    
    if NUM_WORKERS > 1:
        # Process item.
        print(f"🚀 Starting multiprocessing with {NUM_WORKERS} workers...\n")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Worker thread.
            future_to_folder = {executor.submit(process_folder, folder): folder for folder in game_folders}
            

            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Worker exception for {folder.name}: {repr(e)}")
                    all_results.append({
                        "folder_name": folder.name,
                        "frame_count": 0,
                        "processing_time": 0.0,
                        "status": f"worker_error: {repr(e)}"
                    })
    else:
        # Process item.
        print("📝 Starting sequential processing...\n")
        for folder in game_folders:
            result = process_folder(folder)
            all_results.append(result)
    

    print("\n" + "="*70)
    print("📊 Processing Summary")
    print("="*70)
    
    df = pd.DataFrame(all_results)
    

    success_count = len(df[df["status"] == "success"])
    total_frames = df[df["status"] == "success"]["frame_count"].sum()
    total_time = df[df["status"] == "success"]["processing_time"].sum()
    
    print(f"✅ Successful: {success_count}/{len(df)}")
    print(f"📊 Total frames: {total_frames}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    if total_frames > 0:
        print(f"⚡ Avg speed: {total_time/total_frames:.3f}s/frame")
    
    # Save output.
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_CSV, index=False)
    print(f"\n💾 Log saved: {LOG_CSV}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
