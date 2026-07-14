from __future__ import annotations

import argparse
import csv
import json
import math
import random
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Metric handling.

EPOCHS = 300


MAX_CLIPS_PER_VIDEO = 384


BATCH_SIZE = 64


VAL_BATCH_SIZE = 1

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

GRAD_CLIP_NORM = 1.0
LR_SCHEDULER = "cosine"  # "none" | "cosine"
WARMUP_RATIO = 0.1

GRAD_ACCUM_STEPS = 1

INSTANCE_EMBED_DIM = 1536
ATTN_HIDDEN_DIM = 384
DROPOUT = 0.3

USE_POS_WEIGHT = True

TRAIN_SAMPLING = "contiguous"  # "random" | "contiguous"
VAL_SAMPLING = "uniform"  # "uniform" | "first"


VAL_NUM_CROPS = 8


USE_AMP = True
ENABLE_TF32 = True
FAST_CUDNN = True


NUM_WORKERS = 8
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 8
CACHE_MAX_VIDEOS = 512

# Epoch boundary stall mitigation
EPOCH_PREFETCH_FIRST_BATCH = True

# Preload-to-RAM
PRELOAD_TRAIN = False
PRELOAD_VAL = True


USE_EMA = True
EMA_DECAY = 0.999

# Early stopping (macro AUC)
EARLY_STOP_ENABLED = True
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-4

CLASS_NUM = 6
CLASS_NAME = [
 "sexual_content",
 "violence",
 "fear",
 "inappropriate_language",
 "drugs",
 "crime",
]


@dataclass(frozen=True)
class Paths:
	repo_root: Path
	data_csv: Path
	output_dir: Path


def get_paths() -> Paths:
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent.parent
	data_csv = script_dir / "outputs" / "splits" / "feat_data-ration_list.csv"
	output_dir = repo_root / "ai" / "03_mil_training" / "outputs" / "vocal_audio_mil"
	return Paths(repo_root=repo_root, data_csv=data_csv, output_dir=output_dir)


def seed_everything(seed: int) -> None:
	seed = int(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def set_fast_cuda_settings(*, enable_tf32: bool, fast_cudnn: bool) -> None:
	if torch.cuda.is_available():
		torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
		torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
		try:
			torch.set_float32_matmul_precision("high")
		except Exception:
			pass

	if bool(fast_cudnn):
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.deterministic = False


def _safe_str(v: Any) -> str:
	if v is None:
		return ""
	if isinstance(v, float) and v != v:
		return ""
	return str(v)


def _read_csv_df(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	last_err: Optional[Exception] = None
	for enc in ("utf-8-sig", "utf-8", "cp949"):
		try:
			return pd.read_csv(path, encoding=enc)
		except Exception as e:
			last_err = e
	raise RuntimeError(f"Failed to read CSV: {path} ({last_err})")


def _to_float01(x: Any) -> float:
	if x is None:
		return 0.0
	if isinstance(x, (int, np.integer)):
		return float(1.0 if int(x) != 0 else 0.0)
	if isinstance(x, (float, np.floating)):
		if float(x) != float(x):
			return 0.0
		return float(1.0 if float(x) >= 0.5 else 0.0)
	s = str(x).strip()
	if not s:
		return 0.0
	if s.lower() in ("true", "t", "yes", "y"):
		return 1.0
	if s.lower() in ("false", "f", "no", "n"):
		return 0.0
	try:
		v = float(s)
		return float(1.0 if v >= 0.5 else 0.0)
	except Exception:
		return 0.0


def _coerce_bool(x: Any) -> bool:
	if isinstance(x, bool):
		return bool(x)
	s = str(x).strip().lower()
	if s in ("1", "true", "t", "yes", "y"):
		return True
	if s in ("0", "false", "f", "no", "n", ""):
		return False
	try:
		return bool(int(float(s)))
	except Exception:
		return False


def _resolve_feature_path(manifest_path: Path, raw_feature_path: str) -> Path:
	raw_feature_path = raw_feature_path.strip()
	if not raw_feature_path:
		return Path("")
	p = Path(raw_feature_path)
	if p.is_absolute():
		return p
	return (manifest_path.parent / p).resolve()


def _iter_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
	rows: list[dict[str, str]] = []
	for enc in ("utf-8-sig", "utf-8"):
		try:
			with manifest_path.open("r", encoding=enc, newline="") as f:
				reader = csv.DictReader(f)
				for r in reader:
					rows.append({k: _safe_str(v) for k, v in r.items()})
			return rows
		except UnicodeDecodeError:
			continue
		except Exception:
			break
 # fallback
	with manifest_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		for r in reader:
			rows.append({k: _safe_str(v) for k, v in r.items()})
	return rows


def _parse_chunk_idx_from_path(p: Path) -> Optional[int]:
 # expects chunk_000123.npy
	stem = p.stem
	if not stem.startswith("chunk_"):
		return None
	try:
		return int(stem.split("chunk_")[-1])
	except Exception:
		return None


def read_ast_vocal_feature_paths(manifest_path: Path) -> list[Path]:
	if not manifest_path.exists():
		return []
	rows = _iter_manifest_rows(manifest_path)
	if not rows:
		return []

	filtered: list[tuple[int, Path]] = []
	unsorted: list[Path] = []
	for r in rows:
		if r.get("audio_type", "") != "vocal":
			continue
		raw = r.get("feature_path", "")
		p = _resolve_feature_path(manifest_path, raw)
		if not p:
			continue
		idx_raw = r.get("chunk_idx", "")
		idx: Optional[int] = None
		try:
			if idx_raw != "":
				idx = int(float(idx_raw))
		except Exception:
			idx = None
		if idx is None:
			idx = _parse_chunk_idx_from_path(p)
		if idx is None:
			unsorted.append(p)
		else:
			filtered.append((int(idx), p))

	filtered.sort(key=lambda t: t[0])
	unsorted = sorted(unsorted)
	out = [p for _, p in filtered] + unsorted
	return out


def _select_instance_indices(
 n: int,
 max_clips: int = 0,
 *,
 mode: str = "val",
 sampling: str = "uniform",
 rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], list[int]]:
	"""Return (indices, mask) — always uses ALL n instances (no truncation).

	max_clips, mode, sampling, rng are kept for API compatibility but
	no longer limit the number of instances returned.
	"""
	n = int(n)
	if n <= 0:
		return [], []
	indices = list(range(n))
	mask = [1] * n
	return indices, mask


class AstVocalBagDataset(Dataset):
	def __init__(
	 self,
	 df: pd.DataFrame,
	 *,
	 split: str,
	 max_clips_per_video: int,
	 seed: int,
	 limit: int = 0,
	 cache_max_videos: int = 0,
	 preload: bool = False,
	 include_feat_paths: bool = False,
	 train_sampling: str = TRAIN_SAMPLING,
	 val_sampling: str = VAL_SAMPLING,
	) -> None:
		split = str(split)
		if "split" not in df.columns:
			raise ValueError("CSV must contain split column")
		for c in CLASS_NAME:
			if c not in df.columns:
				raise ValueError(f"CSV missing label column: {c}")
		if "ast_manifest" not in df.columns:
			raise ValueError("CSV missing ast_manifest column")
		if "file_name" not in df.columns:
			raise ValueError("CSV missing file_name column")

		self.split = split
		self.max_clips_per_video = int(max_clips_per_video)
		self.include_feat_paths = bool(include_feat_paths)
		self.train_sampling = str(train_sampling)
		self.val_sampling = str(val_sampling)
		self.rng = np.random.default_rng(int(seed) + (0 if split == "train" else 10_000))

		self.cache_max_videos = int(max(0, cache_max_videos))
		self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
		self._preload = bool(preload)

		sub = df[df["split"].astype(str) == split].copy()
		if limit and int(limit) > 0:
			sub = sub.iloc[: int(limit)].copy()

		self.items: list[dict[str, Any]] = []
		self.feat_dim: int = 0

		skipped = 0
		for _, row in sub.iterrows():
			manifest_s = _safe_str(row.get("ast_manifest"))
			if not manifest_s:
				skipped += 1
				continue
			manifest_path = Path(manifest_s)
			feat_paths = read_ast_vocal_feature_paths(manifest_path)
			if not feat_paths:
				skipped += 1
				continue

			ok = True
			for p in feat_paths:
				if not p.exists():
					ok = False
					break
			if not ok:
				skipped += 1
				continue

			y = np.array([_to_float01(row.get(c)) for c in CLASS_NAME], dtype=np.float32)
			file_name = Path(_safe_str(row.get("file_name"))).name
			self.items.append(
			 {
			  "file_name": file_name,
			  "manifest": str(manifest_path),
			  "feat_paths": feat_paths,
			  "y": y,
			 }
			)

			if self.feat_dim == 0:
				try:
					arr = np.load(str(feat_paths[0]), allow_pickle=False)
					arr = np.asarray(arr)
					if arr.ndim == 2 and arr.shape[0] == 1:
						arr = arr[0]
					if arr.ndim != 1:
						raise ValueError(f"AST feature must be 1D, got shape={arr.shape}")
					self.feat_dim = int(arr.shape[0])
				except Exception:
					pass

		if not self.items:
			raise ValueError(f"No usable samples for split={split}. skipped={skipped}")
		if self.feat_dim <= 0:
			raise ValueError("Failed to infer feat_dim from AST features")

		self.y_arr = np.stack([it["y"] for it in self.items], axis=0).astype(np.float32)
		print(f"{split} videos={len(self.items)} (skipped={skipped})")
		print(f"feat_dim={self.feat_dim}, max_clips={self.max_clips_per_video}")

		if self._preload:
			self.preload_all(verbose=True)

	def __len__(self) -> int:
		return int(len(self.items))

	def _load_full_feats(self, feat_paths: list[Path]) -> np.ndarray:
		feats: list[np.ndarray] = []
		for p in feat_paths:
			arr = np.load(str(p), allow_pickle=False)
			arr = np.asarray(arr)
			if arr.ndim == 2 and arr.shape[0] == 1:
				arr = arr[0]
			if arr.ndim != 1:
				raise ValueError(f"AST feature must be 1D, got shape={arr.shape} path={p}")
			feats.append(arr.astype(np.float32, copy=False))
		out = np.stack(feats, axis=0).astype(np.float32, copy=False)
		return out

	def get_full_feats(self, idx: int) -> np.ndarray:
		idx = int(idx)
		if self.cache_max_videos <= 0:
			return self._load_full_feats(self.items[idx]["feat_paths"])
		if idx in self._cache:
			v = self._cache.pop(idx)
			self._cache[idx] = v
			return v
		v = self._load_full_feats(self.items[idx]["feat_paths"])
		self._cache[idx] = v
		while len(self._cache) > self.cache_max_videos:
			self._cache.popitem(last=False)
		return v

	def preload_all(self, *, verbose: bool = False) -> None:
		self.cache_max_videos = max(int(self.cache_max_videos), len(self.items))
		it = range(len(self.items))
		if verbose:
			it = tqdm(it, desc=f"preload_{self.split}")
		for i in it:
			i = int(i)
			if i in self._cache:
				continue
			v = self._load_full_feats(self.items[i]["feat_paths"])
			self._cache[i] = v
		while len(self._cache) > int(self.cache_max_videos):
			self._cache.popitem(last=False)

	def __getitem__(self, idx: int) -> dict[str, Any]:
		idx = int(idx)
		item = self.items[idx]
		full = self.get_full_feats(idx)
		n_total = int(full.shape[0])
		mode = "train" if self.split == "train" else "val"
		sampling = self.train_sampling if mode == "train" else self.val_sampling
		indices, mask = _select_instance_indices(
		 n_total,
		 self.max_clips_per_video,
		 mode=mode,
		 sampling=sampling,
		 rng=self.rng,
		)
		x = full[np.array(indices, dtype=np.int64)]
		y = item["y"].astype(np.float32, copy=False)

		out: dict[str, Any] = {
		 "x": torch.from_numpy(x),
		 "mask": torch.tensor(mask, dtype=torch.float32),
		 "y": torch.from_numpy(y),
		 "idx": idx,
		}
		if self.include_feat_paths:
			out["feat_paths"] = [str(p) for p in item["feat_paths"]]
			out["file_name"] = item["file_name"]
		return out


def bag_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
	"""Collate variable-length bags by dynamic padding to max length in batch."""
	max_len = max(b["x"].shape[0] for b in batch)
	feat_dim = batch[0]["x"].shape[-1]
	B = len(batch)
	x_padded = torch.zeros(B, max_len, feat_dim)
	mask_padded = torch.zeros(B, max_len)
	ys = []
	idxs = []
	for i, b in enumerate(batch):
		n = b["x"].shape[0]
		x_padded[i, :n] = b["x"]
		mask_padded[i, :n] = b["mask"][:n]
		ys.append(b["y"])
		idxs.append(b["idx"])
	result: dict[str, Any] = {
	 "x": x_padded,
	 "mask": mask_padded,
	 "y": torch.stack(ys),
	 "idx": idxs,
	}
	if "file_name" in batch[0]:
		result["file_name"] = [b["file_name"] for b in batch]
	if "feat_paths" in batch[0]:
		result["feat_paths"] = [b["feat_paths"] for b in batch]
	return result


def build_eval_crop_starts(n_total: int, crop_len: int, num_crops: int) -> list[int]:
	"""Deterministic crop start indices for evaluation."""
	n_total = int(n_total)
	crop_len = int(crop_len)
	num_crops = int(num_crops)
	if num_crops <= 1:
		return [0]
	if n_total <= crop_len:
		return [0] * num_crops
	max_start = n_total - crop_len
	if num_crops == 2:
		return [0, int(max_start)]
	starts = np.linspace(0, max_start, num=num_crops)
	return [int(round(s)) for s in starts.tolist()]


class GatedAttentionMIL(nn.Module):
	def __init__(
	 self,
	 *,
	 in_dim: int,
	 embed_dim: int,
	 attn_dim: int,
	 num_classes: int,
	 dropout: float = 0.1,
	) -> None:
		super().__init__()
		self.instance_encoder = nn.Sequential(
		 nn.Linear(int(in_dim), int(embed_dim)),
		 nn.ReLU(inplace=True),
		 nn.Dropout(float(dropout)),
		)
		self.attn_v = nn.Linear(int(embed_dim), int(attn_dim))
		self.attn_u = nn.Linear(int(embed_dim), int(attn_dim))
		self.attn_w = nn.Linear(int(attn_dim), 1)
		self.classifier = nn.Linear(int(embed_dim), int(num_classes))

	@staticmethod
	def _masked_softmax(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
		mask = mask.to(dtype=x.dtype)
		x = x.masked_fill(mask <= 0, float("-inf"))
		a = torch.softmax(x, dim=dim)
		a = a * mask
		denom = a.sum(dim=dim, keepdim=True).clamp_min(1e-12)
		return a / denom

	def forward(
	 self,
	 x: torch.Tensor,
	 mask: torch.Tensor,
	 *,
	 return_attention: bool = False,
	) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""x: (B, T, D), mask: (B, T) -> logits: (B, C)."""
		h = self.instance_encoder(x)  # (B, T, E)
		v = torch.tanh(self.attn_v(h))
		u = torch.sigmoid(self.attn_u(h))
		a_logits = self.attn_w(v * u).squeeze(-1)  # (B, T)
		a = self._masked_softmax(a_logits, mask, dim=1)  # (B, T)
		z = torch.sum(a.unsqueeze(-1) * h, dim=1)  # (B, E)
		logits = self.classifier(z)  # (B, C)
		if return_attention:
			return logits, a
		return logits, None


class ModelEMA:
	def __init__(self, model: nn.Module, *, decay: float) -> None:
		self.decay = float(decay)
		self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
		self._backup: Optional[dict[str, torch.Tensor]] = None

	@torch.no_grad()
	def update(self, model: nn.Module) -> None:
		d = self.decay
		msd = model.state_dict()
		for k, v in msd.items():
			if k not in self.shadow:
				self.shadow[k] = v.detach().clone()
				continue
			if v.dtype.is_floating_point:
				self.shadow[k].mul_(d).add_(v.detach(), alpha=(1.0 - d))
			else:
				self.shadow[k] = v.detach().clone()

	def store(self, model: nn.Module) -> None:
		self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

	def copy_to(self, model: nn.Module) -> None:
		model.load_state_dict(self.shadow, strict=True)

	def restore(self, model: nn.Module) -> None:
		if self._backup is None:
			return
		model.load_state_dict(self._backup, strict=True)
		self._backup = None


def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
	"""y: (N, C) in {0,1}. pos_weight = (neg/pos)."""
	if y.ndim != 2:
		raise ValueError(f"y must be 2D, got {y.shape}")
	n = float(y.shape[0])
	pos = y.sum(axis=0).astype(np.float32)
	neg = (n - pos).astype(np.float32)
	pos = np.maximum(pos, 1.0)
	w = neg / pos
	w = np.clip(w, 1.0, 100.0)
	return torch.from_numpy(w.astype(np.float32))


@torch.inference_mode()
def evaluate(
 model: nn.Module,
 loader: DataLoader,
 *,
 device: torch.device,
 threshold: float = 0.5,
 amp: bool = False,
 val_num_crops: int = 1,
 max_clips: int = MAX_CLIPS_PER_VIDEO,
 val_dataset: Optional[AstVocalBagDataset] = None,
) -> dict[str, Any]:
	model.eval()
	all_y: list[torch.Tensor] = []
	all_p: list[torch.Tensor] = []

	val_num_crops = int(val_num_crops)
	max_clips = int(max_clips)

	for batch in tqdm(loader, desc="val", leave=False):
		y = batch["y"].to(device, non_blocking=True)

		if val_num_crops > 1:
			if val_dataset is None:
				raise ValueError("val_dataset must be provided when val_num_crops > 1")
			idx = int(batch["idx"][0]) if isinstance(batch["idx"], (torch.Tensor, list)) else int(batch["idx"])
			full = val_dataset.get_full_feats(idx)
			n_total = int(full.shape[0])
			# Use ALL clips in a single forward pass (no cropping)
			mask_np = np.ones((n_total,), dtype=np.float32)
			if amp and device.type == "cuda":
				x = torch.from_numpy(full).unsqueeze(0).to(device, dtype=torch.float16, non_blocking=True)
			else:
				x = torch.from_numpy(full).unsqueeze(0).to(device, non_blocking=True)
			mask = torch.from_numpy(mask_np).unsqueeze(0).to(device, non_blocking=True)
			if amp and device.type == "cuda":
				with torch.autocast(device_type="cuda", dtype=torch.float16):
					logits, _ = model(x, mask, return_attention=False)
			else:
				logits, _ = model(x, mask, return_attention=False)
		else:
			if amp and device.type == "cuda":
				x = batch["x"].to(device, dtype=torch.float16, non_blocking=True)
			else:
				x = batch["x"].to(device, non_blocking=True)
			mask = batch["mask"].to(device, non_blocking=True)
			if amp and device.type == "cuda":
				with torch.autocast(device_type="cuda", dtype=torch.float16):
					logits, _ = model(x, mask, return_attention=False)
			else:
				logits, _ = model(x, mask, return_attention=False)

		p = torch.sigmoid(logits)
		all_y.append(y.detach().float().cpu())
		all_p.append(p.detach().float().cpu())

	y_true = torch.cat(all_y, dim=0).numpy()
	y_prob = torch.cat(all_p, dim=0).numpy()
	y_pred = (y_prob >= float(threshold)).astype(np.int64)

	per_class_auc: dict[str, float] = {}
	aucs: list[float] = []
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
		for i, name in enumerate(CLASS_NAME):
			try:

				if len(np.unique(y_true[:, i])) < 2:
					auc = float("nan")
				else:
					auc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
			except Exception:
				auc = float("nan")
			per_class_auc[name] = auc
			if auc == auc:
				aucs.append(auc)

	macro_auc = float(np.mean(aucs)) if aucs else float("nan")

	try:
		macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
		micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
	except Exception:
		macro_f1 = float("nan")
		micro_f1 = float("nan")

	return {
	 "macro_auc": macro_auc,
	 "macro_f1": macro_f1,
	 "micro_f1": micro_f1,
	 "per_class_auc": per_class_auc,
	 "n": int(y_true.shape[0]),
	}


def train_one_epoch(
 model: nn.Module,
 loader: DataLoader,
 *,
 device: torch.device,
 optimizer: torch.optim.Optimizer,
 criterion: nn.Module,
 grad_clip: float = 1.0,
 amp: bool = False,
 scaler: Optional[object] = None,
 scheduler: Optional[object] = None,
 accum_steps: int = 1,
 ema: Optional[ModelEMA] = None,
 data_iter: Optional[object] = None,
 first_batch: Optional[dict[str, Any]] = None,
) -> float:
	model.train()
	accum_steps = int(max(1, accum_steps))
	losses: list[float] = []
	optimizer.zero_grad(set_to_none=True)

	enabled_amp = bool(amp and device.type == "cuda")
	if data_iter is None:
		data_iter = iter(loader)

	def _run_batch(step: int, batch: dict[str, Any]) -> None:
		x = batch["x"].to(device, non_blocking=True)
		mask = batch["mask"].to(device, non_blocking=True)
		y = batch["y"].to(device, non_blocking=True)
		if enabled_amp and device.type == "cuda":
			x = x.to(dtype=torch.float16)

		with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled_amp):
			logits, _ = model(x, mask, return_attention=False)
			loss = criterion(logits, y) / float(accum_steps)

		if enabled_amp and scaler is not None:
			scaler.scale(loss).backward()  # pyright: ignore
		else:
			loss.backward()

		losses.append(float((loss.detach() * float(accum_steps)).cpu()))

		do_step = (step % accum_steps == 0) or (step == len(loader))
		if not do_step:
			return

		if grad_clip and float(grad_clip) > 0:
			if enabled_amp and scaler is not None:
				scaler.unscale_(optimizer)  # pyright: ignore
			torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

		if enabled_amp and scaler is not None:
			scaler.step(optimizer)  # pyright: ignore
			scaler.update()  # pyright: ignore
		else:
			optimizer.step()

		if scheduler is not None:
			scheduler.step()
		if ema is not None:
			ema.update(model)
		optimizer.zero_grad(set_to_none=True)

	total = int(len(loader))
	bar = tqdm(total=total, desc="train", leave=False)
	step = 0
	if first_batch is not None:
		step += 1
		bar.update(1)
		_run_batch(step, first_batch)

	for batch in data_iter:  # type: ignore[assignment]
		step += 1
		bar.update(1)
		_run_batch(step, batch)

	bar.close()
	return float(np.mean(losses)) if losses else float("nan")


def _start_prefetch_first_batch(
 loader: DataLoader,
) -> tuple[object, threading.Thread, dict[str, Any]]:
	data_iter = iter(loader)
	box: dict[str, Any] = {}

	def _run() -> None:
		try:
			box["batch"] = next(data_iter)
		except StopIteration:
			box["batch"] = None
		except Exception as e:
			box["error"] = e
			box["batch"] = None

	th = threading.Thread(target=_run, daemon=True)
	th.start()
	return data_iter, th, box


def build_warmup_cosine_scheduler(
 optimizer: torch.optim.Optimizer,
 *,
 total_steps: int,
 warmup_steps: int,
 min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
	"""Step-wise warmup + cosine decay scheduler.

	Call scheduler.step() once per optimizer step.
	"""
	total_steps = max(1, int(total_steps))
	warmup_steps = int(np.clip(int(warmup_steps), 0, total_steps))
	min_lr_ratio = float(np.clip(float(min_lr_ratio), 0.0, 1.0))

	def lr_lambda(step: int) -> float:
		step = int(step)
		if warmup_steps > 0 and step < warmup_steps:
			return float(step + 1) / float(warmup_steps)
		progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
		progress = float(np.clip(progress, 0.0, 1.0))
		cos = 0.5 * (1.0 + np.cos(np.pi * progress))
		return float(min_lr_ratio + (1.0 - min_lr_ratio) * cos)

	return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Vocal-audio(AST) multi-label Attention MIL (8s chunk features)")
	paths = get_paths()
	p.add_argument("--csv", type=str, default=str(paths.data_csv))
	p.add_argument("--output-dir", type=str, default=str(paths.output_dir))
	p.add_argument("--epochs", type=int, default=EPOCHS)
	p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	p.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE)
	p.add_argument("--lr", type=float, default=LEARNING_RATE)
	p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
	p.add_argument("--max-clips", type=int, default=MAX_CLIPS_PER_VIDEO)
	p.add_argument("--grad-clip", type=float, default=GRAD_CLIP_NORM)
	p.add_argument("--accum-steps", type=int, default=GRAD_ACCUM_STEPS)
	p.add_argument("--lr-scheduler", type=str, default=LR_SCHEDULER, choices=["none", "cosine"])
	p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
	p.add_argument("--dropout", type=float, default=DROPOUT)
	p.add_argument("--embed-dim", type=int, default=INSTANCE_EMBED_DIM)
	p.add_argument("--attn-dim", type=int, default=ATTN_HIDDEN_DIM)
	p.add_argument("--seed", type=int, default=42)
	p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
	p.add_argument("--val-num-workers", type=int, default=0)
	p.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR)
	p.add_argument(
	 "--persistent-workers",
	 action=argparse.BooleanOptionalAction,
	 default=PERSISTENT_WORKERS,
	)
	p.add_argument("--cache-max-videos", type=int, default=CACHE_MAX_VIDEOS)
	p.add_argument(
	 "--preload-train",
	 action=argparse.BooleanOptionalAction,
	 default=PRELOAD_TRAIN,
	 help="Preload all train features into RAM (reduces stalls, uses more memory)",
	)
	p.add_argument(
	 "--preload-val",
	 action=argparse.BooleanOptionalAction,
	 default=PRELOAD_VAL,
	 help="Preload all val features into RAM (recommended for val multi-crop)",
	)
	p.add_argument(
	 "--epoch-prefetch-first-batch",
	 action=argparse.BooleanOptionalAction,
	 default=EPOCH_PREFETCH_FIRST_BATCH,
	 help="Prefetch the next epoch's first train batch during validation/checkpointing",
	)
	p.add_argument("--limit-train", type=int, default=0)
	p.add_argument("--limit-val", type=int, default=0)
	p.add_argument("--train-sampling", type=str, default=TRAIN_SAMPLING, choices=["random", "contiguous"])
	p.add_argument("--val-sampling", type=str, default=VAL_SAMPLING, choices=["uniform", "first"])
	p.add_argument("--val-num-crops", type=int, default=VAL_NUM_CROPS)
	p.add_argument(
	 "--pos-weight",
	 action=argparse.BooleanOptionalAction,
	 default=USE_POS_WEIGHT,
	 help="Use BCE pos_weight balancing",
	)
	p.add_argument(
	 "--amp",
	 action=argparse.BooleanOptionalAction,
	 default=USE_AMP,
	 help="Enable CUDA AMP (mixed precision)",
	)
	p.add_argument(
	 "--tf32",
	 action=argparse.BooleanOptionalAction,
	 default=ENABLE_TF32,
	 help="Enable TF32 matmul/cudnn (Ampere+)",
	)
	p.add_argument(
	 "--fast-cudnn",
	 action=argparse.BooleanOptionalAction,
	 default=FAST_CUDNN,
	 help="Enable cudnn.benchmark and disable cudnn.deterministic for speed",
	)
	p.add_argument(
	 "--ema",
	 action=argparse.BooleanOptionalAction,
	 default=USE_EMA,
	 help="Use EMA weights for evaluation",
	)
	p.add_argument("--ema-decay", type=float, default=EMA_DECAY)
	p.add_argument(
	 "--early-stop",
	 action=argparse.BooleanOptionalAction,
	 default=EARLY_STOP_ENABLED,
	 help="Enable early stopping (macro AUC)",
	)
	p.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
	p.add_argument("--early-stop-min-delta", type=float, default=EARLY_STOP_MIN_DELTA)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	seed_everything(int(args.seed))
	set_fast_cuda_settings(enable_tf32=bool(args.tf32), fast_cudnn=bool(args.fast_cudnn))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	df = _read_csv_df(Path(args.csv))
	if "split" not in df.columns:
		raise ValueError("CSV must contain split column")

	train_ds = AstVocalBagDataset(
	 df,
	 split="train",
	 max_clips_per_video=int(args.max_clips),
	 seed=int(args.seed),
	 limit=int(args.limit_train),
	 cache_max_videos=int(args.cache_max_videos),
	 preload=bool(args.preload_train),
	 include_feat_paths=False,
	 train_sampling=str(args.train_sampling),
	 val_sampling=str(args.val_sampling),
	)
	val_ds = AstVocalBagDataset(
	 df,
	 split="val",
	 max_clips_per_video=int(args.max_clips),
	 seed=int(args.seed),
	 limit=int(args.limit_val),
	 cache_max_videos=int(args.cache_max_videos),
	 preload=bool(args.preload_val),
	 include_feat_paths=False,
	 train_sampling=str(args.train_sampling),
	 val_sampling=str(args.val_sampling),
	)

	train_num_workers = int(args.num_workers)
	val_num_workers = int(args.val_num_workers)
	pin_memory = device.type == "cuda"

	val_num_crops = int(args.val_num_crops)
	val_batch_size = int(args.val_batch_size)
	if val_num_crops > 1:
		val_batch_size = 1
		val_num_workers = 0

	train_dl_kwargs: dict[str, Any] = {
	 "batch_size": int(args.batch_size),
	 "num_workers": train_num_workers,
	 "pin_memory": pin_memory,
	}
	if train_num_workers > 0:
		train_dl_kwargs["prefetch_factor"] = int(args.prefetch_factor)
		train_dl_kwargs["persistent_workers"] = bool(args.persistent_workers)

	val_dl_kwargs: dict[str, Any] = {
	 "batch_size": val_batch_size,
	 "num_workers": val_num_workers,
	 "pin_memory": pin_memory,
	}
	if val_num_workers > 0:
		val_dl_kwargs["prefetch_factor"] = int(args.prefetch_factor)
		val_dl_kwargs["persistent_workers"] = bool(args.persistent_workers)

	train_loader = DataLoader(train_ds, shuffle=True, collate_fn=bag_collate_fn, **train_dl_kwargs)
	val_loader = DataLoader(val_ds, shuffle=False, collate_fn=bag_collate_fn, **val_dl_kwargs)

	model = GatedAttentionMIL(
	 in_dim=int(train_ds.feat_dim),
	 embed_dim=int(args.embed_dim),
	 attn_dim=int(args.attn_dim),
	 num_classes=CLASS_NUM,
	 dropout=float(args.dropout),
	).to(device)

	if not bool(args.pos_weight):
		pos_weight = None
	else:
		pos_weight = compute_pos_weight(train_ds.y_arr).to(device)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

	amp_enabled = bool(args.amp) and device.type == "cuda"
	try:
		scaler: Optional[object] = torch.amp.GradScaler("cuda") if amp_enabled else None
	except Exception:
		scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

	accum_steps = int(max(1, int(args.accum_steps)))
	steps_per_epoch = int(math.ceil(len(train_loader) / float(accum_steps)))
	total_steps = int(max(1, steps_per_epoch * int(args.epochs)))
	warmup_steps = int(total_steps * float(args.warmup_ratio))

	if str(args.lr_scheduler) == "cosine":
		scheduler = build_warmup_cosine_scheduler(
		 optimizer,
		 total_steps=total_steps,
		 warmup_steps=warmup_steps,
		 min_lr_ratio=0.01,
		)
	else:
		scheduler = None

	best_macro_auc = -1.0
	best_epoch = 0
	no_improve = 0
	best_path = out_dir / "best_vocal_audio_mil_model.pth"
	last_path = out_dir / "last_vocal_audio_mil_model.pth"
	metrics_path = out_dir / "metrics.json"

	history: list[dict[str, Any]] = []

	run_cfg = {
	 "task": "vocal_audio_mil",
	 "csv": str(args.csv),
	 "output_dir": str(out_dir),
	 "epochs": int(args.epochs),
	 "batch_size": int(args.batch_size),
	 "val_batch_size": int(val_batch_size),
	 "lr": float(args.lr),
	 "weight_decay": float(args.weight_decay),
	 "max_clips": int(args.max_clips),
	 "grad_clip": float(args.grad_clip),
	 "accum_steps": int(accum_steps),
	 "embed_dim": int(args.embed_dim),
	 "attn_dim": int(args.attn_dim),
	 "dropout": float(args.dropout),
	 "pos_weight": bool(args.pos_weight),
	 "lr_scheduler": str(args.lr_scheduler),
	 "warmup_ratio": float(args.warmup_ratio),
	 "amp": bool(args.amp),
	 "tf32": bool(args.tf32),
	 "fast_cudnn": bool(args.fast_cudnn),
	 "num_workers": train_num_workers,
	 "val_num_workers": val_num_workers,
	 "prefetch_factor": int(args.prefetch_factor),
	 "persistent_workers": bool(args.persistent_workers),
	 "cache_max_videos": int(args.cache_max_videos),
	 "preload_train": bool(args.preload_train),
	 "preload_val": bool(args.preload_val),
	 "epoch_prefetch_first_batch": bool(args.epoch_prefetch_first_batch),
	 "train_sampling": str(args.train_sampling),
	 "val_sampling": str(args.val_sampling),
	 "val_num_crops": int(val_num_crops),
	 "ema": bool(args.ema),
	 "ema_decay": float(args.ema_decay),
	 "early_stop": bool(args.early_stop),
	 "early_stop_patience": int(args.early_stop_patience),
	 "early_stop_min_delta": float(args.early_stop_min_delta),
	}

	print(f"DEVICE={device.type}")
	print(f"amp={amp_enabled}, tf32={bool(args.tf32)}, fast_cudnn={bool(args.fast_cudnn)}")
	print(
	 f"workers(train)={train_num_workers}, workers(val)={val_num_workers}, "
	 f"prefetch_factor={int(args.prefetch_factor)}, persistent={bool(args.persistent_workers)}"
	)
	print(f"cache_max_videos={int(args.cache_max_videos)}")
	print(f"preload_train={bool(args.preload_train)}, preload_val={bool(args.preload_val)}")
	print(f"epoch_prefetch_first_batch={bool(args.epoch_prefetch_first_batch)}")
	print(f"embed_dim={int(args.embed_dim)}, attn_dim={int(args.attn_dim)}, dropout={float(args.dropout)}")
	print(f"pos_weight={bool(args.pos_weight)}, lr_scheduler={str(args.lr_scheduler)}, warmup_ratio={float(args.warmup_ratio)}")
	print(f"accum_steps={int(accum_steps)}, train_sampling={str(args.train_sampling)}, val_sampling={str(args.val_sampling)}")
	print(f"val_batch_size={int(val_batch_size)}, val_num_crops={int(val_num_crops)}")
	print(f"ema={bool(args.ema)}, ema_decay={float(args.ema_decay)}")
	print(f"early_stop={bool(args.early_stop)}")
	print(
	 f"early_stop_patience={int(args.early_stop_patience)}, early_stop_min_delta={float(args.early_stop_min_delta)}"
	)

	ema_obj: Optional[ModelEMA] = None
	if bool(args.ema):
		ema_obj = ModelEMA(model, decay=float(args.ema_decay))

	prefetch_iter: Optional[object] = None
	prefetch_thread: Optional[threading.Thread] = None
	prefetch_box: Optional[dict[str, Any]] = None

	for epoch in range(1, int(args.epochs) + 1):
		first_batch = None
		data_iter = None
		if prefetch_thread is not None and prefetch_box is not None and prefetch_iter is not None:
			prefetch_thread.join()
			if prefetch_box.get("error") is not None:
				raise RuntimeError(f"Prefetch failed: {prefetch_box['error']}")
			first_batch = prefetch_box.get("batch")
			data_iter = prefetch_iter
			prefetch_iter, prefetch_thread, prefetch_box = None, None, None

		train_loss = train_one_epoch(
		 model,
		 train_loader,
		 device=device,
		 optimizer=optimizer,
		 criterion=criterion,
		 grad_clip=float(args.grad_clip),
		 amp=amp_enabled,
		 scaler=scaler,
		 scheduler=scheduler,
		 accum_steps=accum_steps,
		 ema=ema_obj,
		 data_iter=data_iter,
		 first_batch=first_batch,
		)

		if bool(args.epoch_prefetch_first_batch) and (epoch < int(args.epochs)):
			prefetch_iter, prefetch_thread, prefetch_box = _start_prefetch_first_batch(train_loader)

		if ema_obj is not None:
			ema_obj.store(model)
			ema_obj.copy_to(model)
		val_metrics = evaluate(
		 model,
		 val_loader,
		 device=device,
		 threshold=0.5,
		 amp=amp_enabled,
		 val_num_crops=val_num_crops,
		 max_clips=int(args.max_clips),
		 val_dataset=val_ds if val_num_crops > 1 else None,
		)
		if ema_obj is not None:
			ema_obj.restore(model)

		macro_auc = float(val_metrics.get("macro_auc", float("nan")))

		row = {
		 "epoch": int(epoch),
		 "train_loss": float(train_loss),
		 **val_metrics,
		}
		history.append(row)
		metrics_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

		print(
		 f"[Epoch {epoch:03d}] loss={train_loss:.5f} "
		 f"val_macro_auc={macro_auc:.4f} val_macro_f1={row.get('macro_f1', float('nan')):.4f}"
		)

		torch.save({"model": model.state_dict(), "epoch": epoch, "config": run_cfg}, last_path)

		wrote_best = False
		best_state_dict = ema_obj.shadow if ema_obj is not None else model.state_dict()
		if macro_auc == macro_auc and (macro_auc > best_macro_auc + float(args.early_stop_min_delta)):
			best_macro_auc = macro_auc
			best_epoch = int(epoch)
			no_improve = 0
			torch.save({"model": best_state_dict, "epoch": epoch, "config": run_cfg}, best_path)
			wrote_best = True
		else:
			if macro_auc == macro_auc:
				no_improve += 1
		if (not wrote_best) and (not best_path.exists()):
			torch.save({"model": best_state_dict, "epoch": epoch, "config": run_cfg}, best_path)

		if bool(args.early_stop) and int(args.early_stop_patience) > 0 and no_improve >= int(args.early_stop_patience):
			print(f"EARLY STOP at epoch={epoch} (best_epoch={best_epoch}, best_macro_auc={best_macro_auc:.4f})")
			prefetch_iter, prefetch_thread, prefetch_box = None, None, None
			break

	print(f"DONE. best_macro_auc={best_macro_auc:.4f} (best_epoch={best_epoch})")
	print(f"best_ckpt={best_path}")
	print(f"last_ckpt={last_path}")
	print(f"metrics={metrics_path}")


if __name__ == "__main__":
	main()
