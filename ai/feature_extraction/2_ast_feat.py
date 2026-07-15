from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dotenv
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import ASTModel, AutoFeatureExtractor


TARGET_SR = 16_000
SEC_PER_CHUNK = 8
SAMPLES_PER_CHUNK = TARGET_SR * SEC_PER_CHUNK

DEFAULT_BATCH_SIZE = 60
DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_VRAM = True
DEFAULT_USE_FP16 = True
DEFAULT_PREFETCH_DEPTH = 4

AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


@dataclass(frozen=True)
class Paths:
	repo_root: Path
	index_csv: Path
	model_cache_dir: Path
	feature_root_dir: Path


def get_paths() -> Paths:
	script_dir = Path(__file__).resolve().parent  # ai/feature_extraction
	repo_root = script_dir.parent.parent
	if str(repo_root) not in sys.path:
		sys.path.insert(0, str(repo_root))
	from ai.project_paths import FEATURE_ROOT, PREPROCESS_INDEX_CSV
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


def ensure_dir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


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


def load_ast(
 cache_dir: Path,
 token: Optional[str],
 device: str,
) -> tuple[AutoFeatureExtractor, ASTModel]:
	cache_dir.mkdir(parents=True, exist_ok=True)

	# Load input.
	try:
		extractor = AutoFeatureExtractor.from_pretrained(
		 AST_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=True,
		)
		model = ASTModel.from_pretrained(
		 AST_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=True,
		)
	except Exception:
	 # Download step.
		extractor = AutoFeatureExtractor.from_pretrained(
		 AST_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=False,
		)
		model = ASTModel.from_pretrained(
		 AST_MODEL_NAME,
		 token=token,
		 cache_dir=str(cache_dir),
		 local_files_only=False,
		)

	model = model.to(device)
	model.eval()
	return extractor, model


def load_audio_mono_resampled(wav_path: Path, target_sr: int) -> tuple[np.ndarray, int, int]:
	"""Load wav as mono float32 and resample to target_sr.

	Returns:
		wav: shape (n_samples,), float32
		src_sr: original sample rate
		target_sr: returned sampling rate
	"""
	import soundfile as sf

	wav, src_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
	if wav.ndim == 2:
	 # (n_samples, channels) -> mono
		wav = wav.mean(axis=1)

	if int(src_sr) != int(target_sr):
		try:
			import soxr

			wav = soxr.resample(wav, src_sr, target_sr, quality="HQ")
		except Exception:
		 # fallback
			import librosa

			wav = librosa.resample(wav, orig_sr=src_sr, target_sr=target_sr)

	return wav.astype(np.float32), int(src_sr), int(target_sr)


def extract_feats_from_input_values(
 *,
 model: ASTModel,
 input_values: torch.Tensor,
 device: str,
 use_fp16: bool,
) -> torch.Tensor:
 # returns tensor shape: (B, hidden)
	with torch.inference_mode():
		if device == "cuda" and use_fp16:
			with torch.autocast(device_type="cuda", dtype=torch.float16):
				outputs = model(input_values=input_values)
		else:
			outputs = model(input_values=input_values)
		feat = outputs.last_hidden_state.mean(dim=1)
	return feat


def process_one_audio_type(
 *,
 file_name: str,
 audio_type: str,
 wav_path: Path,
 extractor: AutoFeatureExtractor,
 model: ASTModel,
 paths: Paths,
 device: str,
 batch_size: int,
 num_workers: int,
 prefetch_vram: bool,
 use_fp16: bool,
 prefetch_depth: int,
 manifest_rows: list[dict[str, object]],
) -> None:
	if not wav_path.exists():
		return

	out_base_dir = paths.feature_root_dir / file_name / "ast_feat"
	out_dir = out_base_dir / f"{audio_type}_ast_feat"
	ensure_dir(out_dir)


	wav, src_sr, tgt_sr = load_audio_mono_resampled(wav_path, TARGET_SR)
	if wav.size == 0:
		return

	n_samples = int(wav.shape[0])
	num_chunks = (n_samples + SAMPLES_PER_CHUNK - 1) // SAMPLES_PER_CHUNK

	existing_chunks = _count_existing_chunks(out_dir)
	resume_chunk = _find_resume_chunk(out_dir)
	if existing_chunks > 0:
		tqdm.write(f"{file_name} [{audio_type}]: found {existing_chunks}/{num_chunks} existing chunks; resume=chunk_{resume_chunk:06d}")
	else:
		tqdm.write(f"{file_name} [{audio_type}]: chunks 0/{num_chunks} (new)")
	if resume_chunk > 0 and resume_chunk < num_chunks:
		tqdm.write(f"RESUME {file_name} [{audio_type}]: starting at chunk_{resume_chunk:06d}")

 # Create required output.
	for chunk_idx in range(num_chunks):
		start_sample = chunk_idx * SAMPLES_PER_CHUNK
		end_sample = min((chunk_idx + 1) * SAMPLES_PER_CHUNK, n_samples)
		n_pad = max(0, SAMPLES_PER_CHUNK - (end_sample - start_sample))
		out_path = out_dir / f"chunk_{chunk_idx:06d}.npy"
		manifest_rows.append(
		 {
		  "file_name": file_name,
		  "audio_type": audio_type,
		  "audio_path": str(wav_path),
		  "src_sr": src_sr,
		  "target_sr": tgt_sr,
		  "sec_per_chunk": SEC_PER_CHUNK,
		  "chunk_idx": chunk_idx,
		  "start_sec": (start_sample / TARGET_SR),
		  "end_sec": ((start_sample + SAMPLES_PER_CHUNK) / TARGET_SR),
		  "start_sample": start_sample,
		  "end_sample": end_sample,
		  "n_pad_samples": n_pad,
		  "feature_path": str(out_path),
		 }
		)

 # Extract required data.
	chunk_indices = [
	 i
	 for i in range(resume_chunk, num_chunks)
	 if not (out_dir / f"chunk_{i:06d}.npy").exists()
	]
	if not chunk_indices:
		return

	batch_size = max(1, int(batch_size))
	num_workers = max(1, int(num_workers))
	prefetch_depth = max(0, int(prefetch_depth))
	batches: list[list[int]] = [
	 chunk_indices[i : i + batch_size] for i in range(0, len(chunk_indices), batch_size)
	]

	inner = tqdm(total=len(chunk_indices), desc=f"{file_name} [{audio_type}] chunks", leave=False)

	with ThreadPoolExecutor(max_workers=num_workers) as ex:
		prefetch_stream = None
		if device == "cuda" and prefetch_vram:
			prefetch_stream = torch.cuda.Stream()

		def _prepare_input_values_cpu(batch_idxs: list[int]) -> torch.Tensor:
			segments: list[np.ndarray] = []
			for idx in batch_idxs:
				start = idx * SAMPLES_PER_CHUNK
				end = min((idx + 1) * SAMPLES_PER_CHUNK, n_samples)
				seg = wav[start:end]
				if seg.shape[0] < SAMPLES_PER_CHUNK:
					seg = np.pad(seg, (0, SAMPLES_PER_CHUNK - seg.shape[0]))
				segments.append(seg.astype(np.float32, copy=False))

   # Create required output.
			inputs = extractor(segments, sampling_rate=TARGET_SR, return_tensors="pt")
			iv: torch.Tensor = inputs["input_values"]
			if device == "cuda":
				try:
					iv = iv.pin_memory()
				except Exception:
					pass
			return iv

		def _to_device(iv_cpu: torch.Tensor) -> torch.Tensor:
			if device != "cuda":
				return iv_cpu.to(device)
			dtype = torch.float16 if use_fp16 else torch.float32
			if prefetch_stream is None:
				return iv_cpu.to(device, dtype=dtype, non_blocking=True)
			with torch.cuda.stream(prefetch_stream):
				return iv_cpu.to(device, dtype=dtype, non_blocking=True)

		target_inflight = 1 + prefetch_depth
		pending_cpu: list[tuple[list[int], "object"]] = []
		ready_gpu: list[tuple[list[int], torch.Tensor]] = []
		next_batch_ptr = 0

		def _fill_pending() -> None:
			nonlocal next_batch_ptr
			while (len(pending_cpu) + len(ready_gpu)) < target_inflight and next_batch_ptr < len(batches):
				b = batches[next_batch_ptr]
				fut = ex.submit(_prepare_input_values_cpu, b)
				pending_cpu.append((b, fut))
				next_batch_ptr += 1

		def _promote_ready() -> None:
			while pending_cpu:
				b, fut = pending_cpu[0]
				if not fut.done():
					break
				iv_cpu = fut.result()
				iv_gpu = _to_device(iv_cpu)
				ready_gpu.append((b, iv_gpu))
				pending_cpu.pop(0)

		_fill_pending()
		_promote_ready()

		while ready_gpu or pending_cpu:
			if not ready_gpu:
				_promote_ready()
				if not ready_gpu and pending_cpu:
					b, fut = pending_cpu.pop(0)
					iv_cpu = fut.result()
					iv_gpu = _to_device(iv_cpu)
					ready_gpu.append((b, iv_gpu))

			batch_idxs, iv_gpu = ready_gpu.pop(0)
			if device == "cuda" and prefetch_stream is not None:
				torch.cuda.current_stream().wait_stream(prefetch_stream)

			feat_gpu = extract_feats_from_input_values(
			 model=model,
			 input_values=iv_gpu,
			 device=device,
			 use_fp16=use_fp16,
			)

			_fill_pending()
			_promote_ready()

			feat_np = feat_gpu.detach().float().cpu().numpy().astype(np.float32)
			for j, chunk_idx in enumerate(batch_idxs):
				out_path = out_dir / f"chunk_{chunk_idx:06d}.npy"
				np.save(out_path, feat_np[j], allow_pickle=False)
			inner.update(len(batch_idxs))

			del feat_gpu
			del iv_gpu
			del feat_np

	inner.close()


def process_one_row(
 *,
 row: dict[str, str],
 extractor: AutoFeatureExtractor,
 model: ASTModel,
 paths: Paths,
 device: str,
 batch_size: int,
 num_workers: int,
 prefetch_vram: bool,
 use_fp16: bool,
 prefetch_depth: int,
) -> None:
	file_name = Path(_safe_str(row.get("file_name"))).name
	if not file_name:
		return

	original_wav = Path(_safe_str(row.get("original_wav")))
	vocal_wav = Path(_safe_str(row.get("vocal_wav")))
	non_vocal_wav = Path(_safe_str(row.get("non_vocal_wav")))


	if not (original_wav.exists() and vocal_wav.exists() and non_vocal_wav.exists()):
		return

	out_base_dir = paths.feature_root_dir / file_name / "ast_feat"
	ensure_dir(out_base_dir)
	manifest_path = out_base_dir / "manifest.csv"



	if manifest_path.exists():
		try:
			with manifest_path.open("r", encoding="utf-8", newline="") as f:
				existing_rows = list(csv.DictReader(f))
			need_types = {"original", "vocal", "non-vocal"}
			seen_types: set[str] = set()
			all_exist = True
			for r in existing_rows:
				a_type = _safe_str(r.get("audio_type")).strip()
				if a_type:
					seen_types.add(a_type)
				feat_path = Path(_safe_str(r.get("feature_path")))
				if not feat_path.exists():
					all_exist = False
					break
			if all_exist and need_types.issubset(seen_types) and len(existing_rows) > 0:
				tqdm.write(f"{file_name}: AST feature extraction already complete; skipping")
				return
		except Exception:

			pass

	manifest_rows: list[dict[str, object]] = []

	process_one_audio_type(
	 file_name=file_name,
	 audio_type="original",
	 wav_path=original_wav,
	 extractor=extractor,
	 model=model,
	 paths=paths,
	 device=device,
	 batch_size=batch_size,
	 num_workers=num_workers,
	 prefetch_vram=prefetch_vram,
	 use_fp16=use_fp16,
	 prefetch_depth=prefetch_depth,
	 manifest_rows=manifest_rows,
	)
	process_one_audio_type(
	 file_name=file_name,
	 audio_type="vocal",
	 wav_path=vocal_wav,
	 extractor=extractor,
	 model=model,
	 paths=paths,
	 device=device,
	 batch_size=batch_size,
	 num_workers=num_workers,
	 prefetch_vram=prefetch_vram,
	 use_fp16=use_fp16,
	 prefetch_depth=prefetch_depth,
	 manifest_rows=manifest_rows,
	)
	process_one_audio_type(
	 file_name=file_name,
	 audio_type="non-vocal",
	 wav_path=non_vocal_wav,
	 extractor=extractor,
	 model=model,
	 paths=paths,
	 device=device,
	 batch_size=batch_size,
	 num_workers=num_workers,
	 prefetch_vram=prefetch_vram,
	 use_fp16=use_fp16,
	 prefetch_depth=prefetch_depth,
	 manifest_rows=manifest_rows,
	)


	manifest_columns = [
	 "file_name",
	 "audio_type",
	 "audio_path",
	 "src_sr",
	 "target_sr",
	 "sec_per_chunk",
	 "chunk_idx",
	 "start_sec",
	 "end_sec",
	 "start_sample",
	 "end_sample",
	 "n_pad_samples",
	 "feature_path",
	]
	with manifest_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=manifest_columns, extrasaction="ignore")
		writer.writeheader()
		for r in manifest_rows:
			writer.writerow(r)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Extract AST audio features from saved wav files (8 seconds per chunk).")
	p.add_argument("--index", type=str, default="", help="Path to the preprocessing index CSV")
	p.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 = all)")
	p.add_argument("--only-file", type=str, default="", help="Process only this file_name (exact match)")
	p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of chunks per GPU forward pass")
	p.add_argument(
	 "--num-workers",
	 type=int,
	 default=DEFAULT_NUM_WORKERS,
	 help="Thread workers for CPU feature extraction (feature extractor)",
	)
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

	# speed hints (safe)
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.set_float32_matmul_precision("high")

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
	if "file_name" not in index_rows[0] or "original_wav" not in index_rows[0]:
		raise ValueError("index.csv must contain file_name, original_wav, vocal_wav, and non_vocal_wav columns.")

	extractor, model = load_ast(paths.model_cache_dir, hf_token, device)
	# Convert data.

	tqdm.write(f"DEVICE={device}")
	tqdm.write(f"AST_MODEL_NAME={AST_MODEL_NAME}")
	tqdm.write(f"MODEL_CACHE_DIR={paths.model_cache_dir}")
	tqdm.write(f"FEATURE_ROOT_DIR={paths.feature_root_dir}")

	if args.only_file:
		index_rows = [r for r in index_rows if _safe_str(r.get("file_name")) == args.only_file]
		if not index_rows:
			raise ValueError(f"No row matches --only-file: {args.only_file}")

	if args.limit and args.limit > 0:
		index_rows = index_rows[: args.limit]

	for row in tqdm(index_rows, total=len(index_rows), desc="AST feature extraction"):
		try:
			process_one_row(
			 row=row,
			 extractor=extractor,
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
