"""Preprocessing script for the game-content identification pipeline."""

import json
import gc
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from natsort import natsorted

# ==================== Path setup ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_pipeline.project_paths import FRAMES_DIR, OCR_RESULTS_DIR, PREPROCESS_LOG_DIR  # noqa: E402

FRAME_PATH = FRAMES_DIR
OUTPUT_PATH = OCR_RESULTS_DIR
LOG_CSV = PREPROCESS_LOG_DIR / "ocr_olm_process_info.csv"

# ==================== Configuration ====================
MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
PARALLEL_PROCESS_NUM = 120  # Process item.

# ==================== Parallel processing ====================
NUM_FOLDER_WORKERS = 1  # Process item.

# ==================== Configuration ====================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TEST_MODE = False  # Test mode.
SKIP_EXISTING = True  # Process item.

# =========================
# Metric handling.
# =========================
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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
    return natsorted(files)


def extract_frame_number(img_path: Path) -> int:
    """Helper function for extract frame number."""
    try:
        return int(img_path.stem)
    except ValueError:
        # Sort values.
        return 0


def chunk_list(items: List[Path], chunk_size: int) -> List[List[Path]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def load_images(paths: List[Path]) -> List[Image.Image]:
    imgs = []
    for p in paths:

        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def build_inputs(
    processor: AutoProcessor,
    images: List[Image.Image],
    device: str,
) -> dict:

    prompt_text = (
        "Attached is one page of a document that you must process. "
        "Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.\n"
        "If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)\n"
        "Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."
    )
    

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    texts = [text] * len(images)
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


def decode_per_sample(
    processor: AutoProcessor,
    output_ids: torch.Tensor,
    input_ids: torch.Tensor,
) -> List[str]:
    """Helper function for decode per sample."""
    results = []
    B = output_ids.shape[0]
    for i in range(B):
        prompt_len = input_ids[i].shape[0]
        new_tokens = output_ids[i, prompt_len:]
        txt = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        

        if txt.startswith("---"):
            # Extract required data.
            parts = txt.split("---", 2)
            if len(parts) >= 3:
                txt = parts[2].strip()
            elif len(parts) == 2:
                txt = parts[1].strip()
        
        results.append(txt)
    return results


def try_load_model(device: str = "cuda") -> Qwen2_5_VLForConditionalGeneration:
    """Helper function for try load model."""
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype="auto",
            attn_implementation="flash_attention_2",
        ).eval().to(device)
        print("flash_attention_2 loaded successfully")
        return model
    except Exception as e:
        print(f"Failed to load flash_attention_2; using fallback: {repr(e)}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype="auto",
    ).eval().to(device)
    print("Fallback model loaded")
    return model


def generate_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    inputs: dict,
) -> torch.Tensor:
    """Helper function for generate batch."""
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=2048,  # Create required output.
            use_cache=True,
        )
    return out


# ==================== Process item ====================


def process_folder(folder_path: Path, model=None, processor=None) -> Dict[str, any]:
    """Helper function for process folder."""
    folder_name = folder_path.name
    output_jsonl = OUTPUT_PATH / f"{folder_name}.jsonl"
    
    # Process item.
    if SKIP_EXISTING and output_jsonl.exists():
        print(f"⏭ Skipped (already exists): {folder_name}")
        return {
            "folder_name": folder_name,
            "frame_count": 0,
            "processing_time": 0.0,
            "status": "skipped"
        }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:

        image_paths = list_images(folder_path)
        if not image_paths:
            print(f"⚠ No images found: {folder_name}")
            return {
                "folder_name": folder_name,
                "frame_count": 0,
                "processing_time": 0.0,
                "status": "no_images"
            }
        
        print(f"\n{'='*60}")
        print(f"📁 Processing: {folder_name}")
        print(f"📊 Total frames: {len(image_paths)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load the model.
        if model is None or processor is None:
            print("🔧 Loading model...")
            processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
            model = try_load_model(device=device)
            if device == "cuda":
                torch.cuda.synchronize()
        
        # Batch processing.
        batches = chunk_list(image_paths, PARALLEL_PROCESS_NUM)
        print(f"🔄 Batch processing: {len(batches)} batches (size={PARALLEL_PROCESS_NUM})")
        
        # Create output directories.
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        # Delete temporary data.
        if output_jsonl.exists():
            output_jsonl.unlink()
        
        processed_count = 0
        

        f_out = open(output_jsonl, "a", encoding="utf-8", buffering=65536)
        
        try:
            # Batch processing.
            next_inputs = None
            next_batch_paths = None
            
            for bi, batch_paths in enumerate(tqdm(batches, desc=f"OCR {folder_name}")):
                # Batch processing.
                if next_inputs is not None:
                    inputs = next_inputs
                    current_paths = next_batch_paths
                else:
                    images = load_images(batch_paths)
                    inputs = build_inputs(processor, images, device=device)
                    current_paths = batch_paths
                
                # Batch processing.
                if bi + 1 < len(batches):
                    next_batch_paths = batches[bi + 1]
                    next_images = load_images(next_batch_paths)
                    next_inputs = build_inputs(processor, next_images, device=device)
                else:
                    next_inputs = None
                    next_batch_paths = None
                
                # Batch processing.
                out_ids = generate_batch(model, inputs)
                
                # Parallel processing.
                texts = decode_per_sample(processor, out_ids, inputs["input_ids"])
                
                # Write output.
                for img_path, ocr_text in zip(current_paths, texts):
                    frame_num = extract_frame_number(img_path)
                    result = {
                        "frame_num": frame_num,
                        "ocr_result": ocr_text
                    }
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_count += 1
                
                # Batch processing.
                f_out.flush()
        
        finally:
            f_out.close()
        
        elapsed = time.time() - start_time
        
        print(f"✅ Completed: {folder_name}")
        print(f"   - Frames: {processed_count}")
        print(f"   - Time: {elapsed:.1f}s ({elapsed/processed_count:.3f}s/frame)")
        print(f"   - Output: {output_jsonl}")
        
        # Memory management.
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Memory management.
        gc.collect()
        
        return {
            "folder_name": folder_name,
            "frame_count": processed_count,
            "processing_time": elapsed,
            "status": "success"
        }
    
    except Exception as e:
        print(f"❌ Error processing {folder_name}: {repr(e)}")
        return {
            "folder_name": folder_name,
            "frame_count": 0,
            "processing_time": 0.0,
            "status": f"error: {repr(e)}"
        }


# ==================== Main execution ====================


def main():
    """Helper function for main."""
    print("\n" + "="*70)
    print("🎮 Game Frame OCR Pipeline with olmOCR")
    print("="*70)
    print(f"📂 Input: {FRAME_PATH}")
    print(f"💾 Output: {OUTPUT_PATH}")
    print(f"🔧 Model: {MODEL_ID}")
    print(f"⚙️  Parallel images: {PARALLEL_PROCESS_NUM}")
    print(f"🔀 Folder workers: {NUM_FOLDER_WORKERS}")
    print(f"🧪 Test mode: {TEST_MODE}")
    print("="*70 + "\n")
    

    game_folders = scan_game_folders()
    if not game_folders:
        print("❌ No game folders found!")
        return
    
    print(f"📊 Found {len(game_folders)} folder(s) to process\n")
    
    # Parallel processing.
    all_results = []
    
    if NUM_FOLDER_WORKERS == 1:
        # Model setup.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("🔧 Loading model for sequential processing...")
        processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
        model = try_load_model(device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        
        for folder in tqdm(game_folders, desc="Processing folders"):
            result = process_folder(folder, model, processor)
            all_results.append(result)
        
        # Memory management.
        del model
        del processor
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    else:
        # Parallel processing.
        with ProcessPoolExecutor(max_workers=NUM_FOLDER_WORKERS) as executor:
            futures = {executor.submit(process_folder, folder): folder for folder in game_folders}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
                folder = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Exception for {folder.name}: {repr(e)}")
                    all_results.append({
                        "folder_name": folder.name,
                        "frame_count": 0,
                        "processing_time": 0.0,
                        "status": f"exception: {repr(e)}"
                    })
    

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
