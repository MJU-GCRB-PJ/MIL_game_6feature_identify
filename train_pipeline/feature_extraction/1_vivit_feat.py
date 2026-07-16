from __future__ import annotations

import argparse
import os
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import dotenv
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import VivitImageProcessor, VivitModel


FRAME_INTERVAL_SEC = 0.25
FRAMES_PER_CHUNK = 32  # 32 frames @ 4 FPS = 8 seconds
DEFAULT_BATCH_SIZE = 30
DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_VRAM = True
DEFAULT_USE_FP16 = True
DEFAULT_PREFETCH_DEPTH = 4

VIVIT_MODEL_NAME = "google/vivit-b-16x2-kinetics400"


@dataclass(frozen=True)
class Paths:
	repo_root: Path
	index_csv: Path
	model_cache_dir: Path
	feature_root_dir: Path


def get_paths() -> Paths:
	script_dir = Path(__file__).resolve().parent  # train_pipeline/feature_extraction
	repo_root = script_dir.parent.parent
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	from train_pipeline.project_paths import FEATURE_ROOT, PREPROCESS_INDEX_CSV
	index_csv = PREPROCESS_INDEX_CSV
	model_cache_dir = script_dir / "model" / "hf_cache"
	feature_root_dir = FEATURE_ROOT
	return Paths(
	 repo_root=repo_root,
	 index_csv=index_csv,
	 model_cache_dir=model_cache_dir,
	 feature_root_dir=feature_root_dir,
	)


def _safe_str(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, float) and value != value:  # NaN
		return ""
	return str(value)


def _numeric_stem(p: Path) -> int:
	try:
		return int(p.stem)
	except Exception:
		return 1_000_000_000


def list_frame_paths(frames_dir: Path, frames_ext: str) -> list[Path]:
	frames_ext = _safe_str(frames_ext).lower().lstrip(".")
	patterns: list[str]
	if frames_ext in ("jpg", "jpeg", "png"):
		patterns = [f"*.{frames_ext}"]
	else:
		patterns = ["*.jpg", "*.png"]

	paths: list[Path] = []
	for pat in patterns:
		paths.extend(frames_dir.glob(pat))

 # Sort values.
	paths = sorted(paths, key=_numeric_stem)
	return paths


def load_frames_rgb(frame_paths: Iterable[Path]) -> list[np.ndarray]:
	import cv2

	frames: list[np.ndarray] = []
	for p in frame_paths:
		img = cv2.imread(str(p), cv2.IMREAD_COLOR)
		if img is None:
			raise FileNotFoundError(f"Failed to load image: {p}")
		frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	return frames


def _parse_chunk_index(p: Path) -> Optional[int]:
 # expects chunk_000123.npy
	stem = p.stem
	if not stem.startswith("chunk_"):
		return None
	try:
		return int(stem.split("chunk_")[-1])
	except Exception:
		return None


def _find_resume_chunk(out_dir: Path) -> int:
	if not out_dir.exists():
		return 0
	existing: set[int] = set()
	for p in out_dir.glob("chunk_*.npy"):
		idx = _parse_chunk_index(p)
		if idx is not None:
			existing.add(idx)
	resume = 0

	while resume in existing:
		resume += 1
	return resume


def _count_existing_chunks(out_dir: Path) -> int:
	if not out_dir.exists():
		return 0
	return sum(1 for _ in out_dir.glob("chunk_*.npy"))


def pad_to_length(items: list[Path], target_len: int) -> tuple[list[Path], int]:
	if len(items) >= target_len:
		return items[:target_len], 0
	if not items:
		return items, 0
	pad_count = target_len - len(items)
	return items + [items[-1]] * pad_count, pad_count


def load_vivit(cache_dir: Path, token: Optional[str], device: str) -> tuple[VivitImageProcessor, VivitModel]:
	cache_dir.mkdir(parents=True, exist_ok=True)

	# Load input.
	try:
		processor = VivitImageProcessor.from_pretrained(
		 VIVIT_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=True,
		)
		model = VivitModel.from_pretrained(
		 VIVIT_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=True,
		)
	except Exception:
	 # Download step.
		processor = VivitImageProcessor.from_pretrained(
		 VIVIT_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=False,
		)
		model = VivitModel.from_pretrained(
		 VIVIT_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=False,
		)

	model = model.to(device)
	model.eval()
	return processor, model


def extract_chunk_feature(
 *,
 processor: VivitImageProcessor,
 model: VivitModel,
 frames_rgb: list[np.ndarray],
 device: str,
) -> np.ndarray:
	inputs = processor(images=frames_rgb, return_tensors="pt")
	pixel_values = inputs.pixel_values.to(device)

	with torch.inference_mode():
		outputs = model(pixel_values=pixel_values)

		# Load input.
		# Write output.
		feat: torch.Tensor = outputs.last_hidden_state[:, 0]
	feat = feat.squeeze(0).detach().float().cpu()
	return feat.numpy().astype(np.float32)


def extract_batch_features(
 *,
 processor: VivitImageProcessor,
 model: VivitModel,
 batch_frames_rgb: list[list[np.ndarray]],
 device: str,
) -> np.ndarray:
 # returns shape: (B, hidden)
	inputs = processor(images=batch_frames_rgb, return_tensors="pt")
	pixel_values = inputs.pixel_values.to(device)
	with torch.inference_mode():
		outputs = model(pixel_values=pixel_values)
		feat = outputs.last_hidden_state[:, 0]  # (B, hidden)
	feat = feat.detach().float().cpu().numpy().astype(np.float32)
	return feat


def extract_feats_from_pixel_values(
 *,
 model: VivitModel,
 pixel_values: torch.Tensor,
 device: str,
 use_fp16: bool,
) -> torch.Tensor:
 # returns tensor shape: (B, hidden) on same device as model
	with torch.inference_mode():
		if device == "cuda" and use_fp16:
			with torch.autocast(device_type="cuda", dtype=torch.float16):
				outputs = model(pixel_values=pixel_values)
		else:
			outputs = model(pixel_values=pixel_values)
		feat = outputs.last_hidden_state[:, 0]
	return feat


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def process_one_row(
 *,
 row: dict[str, str],
 processor: VivitImageProcessor,
 model: VivitModel,
 paths: Paths,
 device: str,
 batch_size: int,
 num_workers: int,
 prefetch_vram: bool,
 use_fp16: bool,
 prefetch_depth: int,
) -> None:
	file_name = Path(_safe_str(row.get("file_name"))).name
	frames_dir = Path(_safe_str(row.get("frames_dir")))
	frames_ext = _safe_str(row.get("frames_ext"))

	if not file_name:
		return
	if not frames_dir.exists() or not frames_dir.is_dir():
		return

	frame_paths = list_frame_paths(frames_dir, frames_ext)
	if not frame_paths:
		return

	out_dir = paths.feature_root_dir / file_name / "vivit_feat"
	ensure_dir(out_dir)
	manifest_path = out_dir / "manifest.csv"
	resume_chunk = _find_resume_chunk(out_dir)


	n_frames = len(frame_paths)
	num_chunks = (n_frames + FRAMES_PER_CHUNK - 1) // FRAMES_PER_CHUNK
	existing_chunks = _count_existing_chunks(out_dir)
	if existing_chunks > 0:
		tqdm.write(
		 f"{file_name}: found {existing_chunks}/{num_chunks} existing chunks; resume=chunk_{resume_chunk:06d}"
		)
	else:
		tqdm.write(f"{file_name}: chunks 0/{num_chunks} (new)")
	if resume_chunk > 0 and resume_chunk < num_chunks:
		tqdm.write(f"RESUME {file_name}: starting at chunk_{resume_chunk:06d}")

	manifest_rows: list[dict[str, object]] = []

	for chunk_idx in range(num_chunks):
		start = chunk_idx * FRAMES_PER_CHUNK
		end = min((chunk_idx + 1) * FRAMES_PER_CHUNK, n_frames)
		# Create required output.
		n_pad = max(0, FRAMES_PER_CHUNK - (end - start))
		out_path = out_dir / f"chunk_{chunk_idx:06d}.npy"
		manifest_rows.append(
		 {
		  "chunk_idx": chunk_idx,
		  "start_frame": start + 1,
		  "end_frame": end,
		  "n_pad": n_pad,
		  "start_sec": start * FRAME_INTERVAL_SEC,
		  "end_sec": (start + FRAMES_PER_CHUNK) * FRAME_INTERVAL_SEC,
		  "feature_path": str(out_path),
		 }
		)

 # Create required output.
	chunk_indices = [i for i in range(resume_chunk, num_chunks) if not (out_dir / f"chunk_{i:06d}.npy").exists()]
	if not chunk_indices:

		pass
	else:
		batch_size = max(1, int(batch_size))
		num_workers = max(1, int(num_workers))
		prefetch_depth = max(0, int(prefetch_depth))

		def _chunk_paths_for_idx(chunk_idx: int) -> list[Path]:
			start = chunk_idx * FRAMES_PER_CHUNK
			end = min((chunk_idx + 1) * FRAMES_PER_CHUNK, n_frames)
			chunk_paths = frame_paths[start:end]
			chunk_paths, _ = pad_to_length(chunk_paths, FRAMES_PER_CHUNK)
			return chunk_paths

		def _submit_batch(ex: ThreadPoolExecutor, batch_idxs: list[int]) -> dict[int, "object"]:
		 # return idx -> Future
			futs = {}
			for idx in batch_idxs:
				futs[idx] = ex.submit(load_frames_rgb, _chunk_paths_for_idx(idx))
			return futs

  # Batch processing.
		batches: list[list[int]] = [chunk_indices[i : i + batch_size] for i in range(0, len(chunk_indices), batch_size)]
		inner = tqdm(total=len(chunk_indices), desc=f"{file_name} chunks", leave=False)

		with ThreadPoolExecutor(max_workers=num_workers) as ex:
			prefetch_stream = None
			if device == "cuda" and prefetch_vram:
				prefetch_stream = torch.cuda.Stream()

			def _prepare_pixel_values_cpu(batch_idxs: list[int]) -> torch.Tensor:
			 # load frames in threads
				futs = _submit_batch(ex, batch_idxs)
				loaded = {idx: futs[idx].result() for idx in batch_idxs}
				batch_frames = [loaded[idx] for idx in batch_idxs]
				inputs = processor(images=batch_frames, return_tensors="pt")
				pv = inputs.pixel_values
				if device == "cuda":
					try:
						pv = pv.pin_memory()
					except Exception:
						pass
				return pv

			def _to_device(pv_cpu: torch.Tensor) -> torch.Tensor:
				if device != "cuda":
					return pv_cpu.to(device)
				dtype = torch.float16 if use_fp16 else torch.float32
				if prefetch_stream is None:
					return pv_cpu.to(device, dtype=dtype, non_blocking=True)
				with torch.cuda.stream(prefetch_stream):
					return pv_cpu.to(device, dtype=dtype, non_blocking=True)

   # --- VRAM prefetch queue (prefetch_depth batches ahead) ---
   # target inflight includes current batch + future batches
			target_inflight = 1 + prefetch_depth

			# pending CPU prep futures, in order
			pending_cpu: list[tuple[list[int], "object"]] = []
			# ready GPU tensors, in order
			ready_gpu: list[tuple[list[int], torch.Tensor]] = []
			next_batch_ptr = 0

			def _fill_pending() -> None:
				nonlocal next_batch_ptr
				while (len(pending_cpu) + len(ready_gpu)) < target_inflight and next_batch_ptr < len(batches):
					b = batches[next_batch_ptr]
					fut = ex.submit(_prepare_pixel_values_cpu, b)
					pending_cpu.append((b, fut))
					next_batch_ptr += 1

			def _promote_ready() -> None:
			 # keep ordering: only promote from the front
				while pending_cpu:
					b, fut = pending_cpu[0]
					if not fut.done():
						break
					pv_cpu = fut.result()
					pv_gpu = _to_device(pv_cpu)
					ready_gpu.append((b, pv_gpu))
					pending_cpu.pop(0)

			_fill_pending()
			_promote_ready()

			while ready_gpu or pending_cpu:
			 # ensure we have at least one GPU-ready batch
				if not ready_gpu:
				 # block on the next pending batch, then promote
					_promote_ready()
					if not ready_gpu and pending_cpu:
						b, fut = pending_cpu.pop(0)
						pv_cpu = fut.result()
						pv_gpu = _to_device(pv_cpu)
						ready_gpu.append((b, pv_gpu))

				batch_idxs, pv_gpu = ready_gpu.pop(0)
				# make sure current stream sees prefetch stream copies
				if device == "cuda" and prefetch_stream is not None:
					torch.cuda.current_stream().wait_stream(prefetch_stream)

    # 1) launch forward (async)
				feat_gpu = extract_feats_from_pixel_values(
				 model=model,
				 pixel_values=pv_gpu,
				 device=device,
				 use_fp16=use_fp16,
				)

				# 2) while GPU is working, keep CPU prep and H2D for future batches going
				_fill_pending()
				_promote_ready()

				# 3) sync on result when moving to CPU for saving
				feat_np = feat_gpu.detach().float().cpu().numpy().astype(np.float32)
				for j, chunk_idx in enumerate(batch_idxs):
					out_path = out_dir / f"chunk_{chunk_idx:06d}.npy"
					np.save(out_path, feat_np[j], allow_pickle=False)
				inner.update(len(batch_idxs))

				# Batch processing.
				del feat_gpu
				del pv_gpu
				del feat_np

		inner.close()


	manifest_columns = [
	 "file_name",
	 "frames_dir",
	 "frames_ext",
	 "n_frames",
	 "frames_per_chunk",
	 "frame_interval_sec",
	 "chunk_idx",
	 "start_frame",
	 "end_frame",
	 "n_pad",
	 "start_sec",
	 "end_sec",
	 "feature_path",
	]
	with manifest_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=manifest_columns, extrasaction="ignore")
		writer.writeheader()
		for r in manifest_rows:
			writer.writerow(
			 {
			  "file_name": file_name,
			  "frames_dir": str(frames_dir),
			  "frames_ext": frames_ext,
			  "n_frames": n_frames,
			  "frames_per_chunk": FRAMES_PER_CHUNK,
			  "frame_interval_sec": FRAME_INTERVAL_SEC,
			  **r,
			 }
			)


def _read_index_rows(index_csv: Path) -> list[dict[str, str]]:
	for enc in ("utf-8-sig", "utf-8"):
		try:
			with index_csv.open("r", encoding=enc, newline="") as f:
				reader = csv.DictReader(f)
				return list(reader)
		except UnicodeDecodeError:
			continue
	with index_csv.open("r", newline="") as f:
		reader = csv.DictReader(f)
		return list(reader)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Extract ViViT features from saved frames (32 frames per chunk).")
	p.add_argument("--index", type=str, default="", help="Path to the preprocessing index CSV")
	p.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 = all)")
	p.add_argument("--only-file", type=str, default="", help="Process only this file_name (exact match, e.g. 1_TormentedSouls_19.mp4)")
	p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of chunks per GPU forward pass")
	p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Thread workers for frame loading")
	p.add_argument("--no-prefetch-vram", action="store_true", help="Disable VRAM prefetch for next batch")
	p.add_argument(
	 "--prefetch-depth",
	 type=int,
	 default=DEFAULT_PREFETCH_DEPTH,
	 help="How many FUTURE batches to prefetch to VRAM (0 disables lookahead; effective when CUDA+prefetch enabled)",
	)
	p.add_argument("--no-fp16", action="store_true", help="Disable fp16 (use fp32) on CUDA")
	return p.parse_args()


def main() -> None:
	args = parse_args()

	dotenv.load_dotenv()
	hf_token = os.getenv("HF_TOKEN")
	device = "cuda" if torch.cuda.is_available() else "cpu"

	paths = get_paths()
	if args.index:
		paths = Paths(
		 repo_root=paths.repo_root,
		 index_csv=Path(args.index),
		 model_cache_dir=paths.model_cache_dir,
		 feature_root_dir=paths.feature_root_dir,
		)

	if not paths.index_csv.exists():
		raise FileNotFoundError(f"index.csv not found: {paths.index_csv}")

	index_rows = _read_index_rows(paths.index_csv)
	if not index_rows:
		raise ValueError("index.csv contains no data rows.")
	if "file_name" not in index_rows[0] or "frames_dir" not in index_rows[0]:
		raise ValueError("index.csv must contain file_name and frames_dir columns.")

	processor, model = load_vivit(paths.model_cache_dir, hf_token, device)
	tqdm.write(f"DEVICE={device}")
	tqdm.write(f"MODEL_CACHE_DIR={paths.model_cache_dir}")
	tqdm.write(f"FEATURE_ROOT_DIR={paths.feature_root_dir}")

	if args.only_file:
		index_rows = [r for r in index_rows if _safe_str(r.get("file_name")) == args.only_file]
		if not index_rows:
			raise ValueError(f"No row matches --only-file: {args.only_file}")

	if args.limit and args.limit > 0:
		index_rows = index_rows[: args.limit]

	for row in tqdm(index_rows, total=len(index_rows), desc="ViViT feature extraction"):
		try:
			process_one_row(
			 row=row,
			 processor=processor,
			 model=model,
			 paths=paths,
			 device=device,
			 batch_size=args.batch_size,
			 num_workers=args.num_workers,
			 prefetch_vram=(DEFAULT_PREFETCH_VRAM and not args.no_prefetch_vram),
			 use_fp16=(DEFAULT_USE_FP16 and not args.no_fp16),
			 prefetch_depth=args.prefetch_depth,
			)
		except Exception as e:

			tqdm.write(f"Warning: processing failed file_name={_safe_str(row.get('file_name'))} err={e}")
			continue


if __name__ == "__main__":
	main()
