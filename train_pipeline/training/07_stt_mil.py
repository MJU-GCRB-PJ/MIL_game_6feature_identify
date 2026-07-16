"""Train the STT MIL model across the paper's cross-validation folds."""

from __future__ import annotations

import argparse
import json
import math
import random
import threading
import warnings
from collections import OrderedDict
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

from cv_config import CV_OUTPUT_DIR, N_FOLDS, run_model_cross_validation


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

CACHE_MAX_VIDEOS = 128

# Epoch boundary stall mitigation
EPOCH_PREFETCH_FIRST_BATCH = True

# Preload-to-RAM
PRELOAD_TRAIN = False
PRELOAD_VAL = True

USE_EMA = True
EMA_DECAY = 0.999

EARLY_STOP_ENABLED = True
EARLY_STOP_PATIENCE = 50
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


def _safe_str(value: object) -> str:
	if value is None:
		return ""
	if isinstance(value, float) and value != value:  # NaN
		return ""
	return str(value)


def _to_float01(v: object) -> float:
	s = _safe_str(v).strip()
	if s == "":
		return 0.0
	try:
		return float(s)
	except Exception:
		return 0.0


def _to_bool(v: object) -> bool:
	if isinstance(v, bool):
		return bool(v)
	s = _safe_str(v).strip().lower()
	if s in ("1", "true", "t", "yes", "y"):
		return True
	if s in ("0", "false", "f", "no", "n", ""):
		return False
	try:
		return bool(int(float(s)))
	except Exception:
		return False


def _read_csv_df(path: Path) -> pd.DataFrame:
	for enc in ("utf-8-sig", "utf-8"):
		try:
			return pd.read_csv(path, encoding=enc)
		except UnicodeDecodeError:
			continue
	return pd.read_csv(path)


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
			torch.backends.cudnn.deterministic = False
			torch.backends.cudnn.benchmark = True


def _select_instance_indices(
 n: int,
 max_clips: int = 0,
 *,
 mode: str = "val",
 sampling: str = "uniform",
 rng: Optional[np.random.Generator] = None,
) -> tuple[list[int], list[int]]:
	"""Return (indices, pad_mask) — always uses ALL n instances (no truncation).

	max_clips, mode, sampling, rng are kept for API compatibility but
	no longer limit the number of instances returned.
	"""
	n = int(n)
	if n <= 0:
		return [], []
	indices = list(range(n))
	mask = [1] * n
	return indices, mask


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
	return result


def build_eval_crop_starts(n_total: int, crop_len: int, num_crops: int) -> list[int]:
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


def _make_final_mask(pad_mask: np.ndarray, presence_mask: np.ndarray) -> np.ndarray:
	pad_mask = np.asarray(pad_mask, dtype=np.float32)
	presence_mask = np.asarray(presence_mask, dtype=np.float32)
	final = pad_mask * presence_mask
	if float(final.sum()) <= 0.0:
		final = pad_mask
	return final.astype(np.float32, copy=False)


class SttBagDataset(Dataset):
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
	 include_paths: bool = False,
	 require_complete: bool = True,
	 train_sampling: str = TRAIN_SAMPLING,
	 val_sampling: str = VAL_SAMPLING,
	) -> None:
		split = str(split)
		if "split" not in df.columns:
			raise ValueError("CSV must contain split column")
		for c in CLASS_NAME:
			if c not in df.columns:
				raise ValueError(f"CSV missing label column: {c}")
		for c in ("file_name", "stt_emb", "stt_mask", "stt_complete"):
			if c not in df.columns:
				raise ValueError(f"CSV missing column: {c}")

		self.split = split
		self.max_clips_per_video = int(max_clips_per_video)
		self.include_paths = bool(include_paths)
		self.require_complete = bool(require_complete)
		self.train_sampling = str(train_sampling)
		self.val_sampling = str(val_sampling)
		self.rng = np.random.default_rng(int(seed) + (0 if split == "train" else 10_000))

		self.cache_max_videos = int(max(0, cache_max_videos))
		self._cache: "OrderedDict[int, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
		self._preload = bool(preload)

		sub = df[df["split"].astype(str) == split].copy()
		if limit and int(limit) > 0:
			sub = sub.iloc[: int(limit)].copy()

		self.items: list[dict[str, Any]] = []
		self.feat_dim: int = 0

		skipped = 0
		for _, row in sub.iterrows():
			if self.require_complete and not _to_bool(row.get("stt_complete")):
				skipped += 1
				continue
			emb_s = _safe_str(row.get("stt_emb"))
			mask_s = _safe_str(row.get("stt_mask"))
			if not emb_s or not mask_s:
				skipped += 1
				continue
			emb_path = Path(emb_s)
			mask_path = Path(mask_s)
			if not emb_path.exists() or not mask_path.exists():
				skipped += 1
				continue

   # empty bag filter
			try:
				mask_arr = np.load(str(mask_path), allow_pickle=False)
				mask_arr = np.asarray(mask_arr)
				if mask_arr.ndim != 1:
					skipped += 1
					continue
				if int(mask_arr.shape[0]) <= 0:
					skipped += 1
					continue
				if float(mask_arr.sum()) <= 0.0:
					skipped += 1
					continue
			except Exception:
				skipped += 1
				continue

			y = np.array([_to_float01(row.get(c)) for c in CLASS_NAME], dtype=np.float32)
			file_name = Path(_safe_str(row.get("file_name"))).name
			self.items.append(
			 {
			  "file_name": file_name,
			  "stt_emb": str(emb_path),
			  "stt_mask": str(mask_path),
			  "y": y,
			 }
			)

			if self.feat_dim == 0:
				try:
					arr = np.load(str(emb_path), allow_pickle=False, mmap_mode="r")
					arr = np.asarray(arr)
					if arr.ndim != 2:
						raise ValueError(f"STT emb must be 2D, got shape={arr.shape}")
					self.feat_dim = int(arr.shape[1])
				except Exception:
					pass

		if not self.items:
			raise ValueError(f"No usable samples for split={split}. skipped={skipped}")
		if self.feat_dim <= 0:
			raise ValueError("Failed to infer feat_dim from STT embeddings")

		self.y_arr = np.stack([it["y"] for it in self.items], axis=0).astype(np.float32)
		print(f"{split} videos={len(self.items)} (skipped={skipped})")
		print(f"feat_dim={self.feat_dim}, max_clips={self.max_clips_per_video}")

		if self._preload:
			self.preload_all(verbose=True)

	def __len__(self) -> int:
		return int(len(self.items))

	def _load_full(self, emb_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
		emb = np.load(str(emb_path), allow_pickle=False)
		emb = np.asarray(emb)
		if emb.ndim != 2:
			raise ValueError(f"STT emb must be 2D, got shape={emb.shape} path={emb_path}")
		mask = np.load(str(mask_path), allow_pickle=False)
		mask = np.asarray(mask)
		if mask.ndim != 1:
			raise ValueError(f"STT mask must be 1D, got shape={mask.shape} path={mask_path}")
		if int(emb.shape[0]) != int(mask.shape[0]):
			raise ValueError(
			 f"STT emb/mask length mismatch: emb_T={emb.shape[0]} mask_T={mask.shape[0]} path={emb_path}"
			)
		presence = (mask.astype(np.float32, copy=False) > 0).astype(np.float32, copy=False)
		return emb, presence

	def preload_all(self, *, verbose: bool = False) -> None:
		self.cache_max_videos = max(self.cache_max_videos, len(self.items))
		it = range(len(self.items))
		if verbose:
			it = tqdm(it, desc=f"preload_{self.split}")
		for i in it:
			i = int(i)
			if i in self._cache:
				continue
			item = self.items[i]
			v = self._load_full(Path(item["stt_emb"]), Path(item["stt_mask"]))
			self._cache[i] = v

	def get_full(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
		idx = int(idx)
		item = self.items[idx]
		emb_path = Path(item["stt_emb"])
		mask_path = Path(item["stt_mask"])

		if self.cache_max_videos <= 0:
			return self._load_full(emb_path, mask_path)
		if idx in self._cache:
			v = self._cache.pop(idx)
			self._cache[idx] = v
			return v
		v = self._load_full(emb_path, mask_path)
		self._cache[idx] = v
		while len(self._cache) > self.cache_max_videos:
			self._cache.popitem(last=False)
		return v

	def __getitem__(self, idx: int) -> dict[str, Any]:
		idx = int(idx)
		item = self.items[idx]
		full_emb, full_presence = self.get_full(idx)
		n_total = int(full_emb.shape[0])
		mode = "train" if self.split == "train" else "val"
		sampling = self.train_sampling if mode == "train" else self.val_sampling
		indices, pad_mask = _select_instance_indices(
		 n_total,
		 self.max_clips_per_video,
		 mode=mode,
		 sampling=sampling,
		 rng=self.rng,
		)
		idx_arr = np.array(indices, dtype=np.int64)
		x = full_emb[idx_arr]
		presence_sel = full_presence[idx_arr]
		final_mask = _make_final_mask(np.array(pad_mask, dtype=np.float32), presence_sel)
		y = item["y"].astype(np.float32, copy=False)

		out: dict[str, Any] = {
		 "x": torch.from_numpy(np.asarray(x)),
		 "mask": torch.from_numpy(final_mask),
		 "y": torch.from_numpy(y),
		 "idx": idx,
		}
		if self.include_paths:
			out["file_name"] = item["file_name"]
			out["stt_emb"] = item["stt_emb"]
			out["stt_mask"] = item["stt_mask"]
		return out


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

	def forward(self, x: torch.Tensor, mask: torch.Tensor, *, return_attn: bool = False) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
		if x.ndim != 3:
			raise ValueError(f"x must be 3D (B,N,D), got {x.shape}")
		if mask.ndim != 2:
			raise ValueError(f"mask must be 2D (B,N), got {mask.shape}")

		h = self.instance_encoder(x)
		v = torch.tanh(self.attn_v(h))
		u = torch.sigmoid(self.attn_u(h))
		a_logits = self.attn_w(v * u).squeeze(-1)
		a = self._masked_softmax(a_logits, mask, dim=1)
		m = (a.unsqueeze(-1) * h).sum(dim=1)
		logits = self.classifier(m)
		return logits, (a if return_attn else None)


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
 val_dataset: Optional[SttBagDataset] = None,
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
			full_emb, full_presence = val_dataset.get_full(idx)
			n_total = int(full_emb.shape[0])
			# Use ALL clips in a single forward pass (no cropping)
			pad_mask = np.ones((n_total,), dtype=np.float32)
			crop_mask = _make_final_mask(pad_mask, full_presence)

			x = torch.from_numpy(np.asarray(full_emb)).unsqueeze(0).to(device, non_blocking=True)
			m = torch.from_numpy(np.asarray(crop_mask, dtype=np.float32)).unsqueeze(0).to(device, non_blocking=True)

			if amp and device.type == "cuda":
				x = x.to(dtype=torch.float16)
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
				logits, _ = model(x, m)
			p = torch.sigmoid(logits)
		else:
			x = batch["x"].to(device, non_blocking=True)
			m = batch["mask"].to(device, non_blocking=True)
			if amp and device.type == "cuda":
				x = x.to(dtype=torch.float16)
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
				logits, _ = model(x, m)
			p = torch.sigmoid(logits)

		all_y.append(y.detach().float().cpu())
		all_p.append(p.detach().float().cpu())

	y_true = torch.cat(all_y, dim=0).numpy()
	y_prob = torch.cat(all_p, dim=0).numpy()

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
		y_pred = (y_prob >= float(threshold)).astype(np.int32)
		micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
		macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

	per_class_auc: dict[str, float] = {}
	auc_list: list[float] = []
	for i, name in enumerate(CLASS_NAME):
		try:
			a = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
			per_class_auc[name] = a
			if a == a:
				auc_list.append(a)
		except Exception:
			continue
	macro_auc = float(np.mean(auc_list)) if auc_list else float("nan")

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
 optimizer: torch.optim.Optimizer,
 criterion: nn.Module,
 device: torch.device,
 grad_clip: float,
 amp: bool,
 scaler: Optional[object],
 accum_steps: int,
 scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
 ema: Optional[ModelEMA],
 data_iter: Optional[object] = None,
 first_batch: Optional[dict[str, Any]] = None,
) -> float:
	model.train()
	loss_sum = 0.0
	n_steps = 0

	optimizer.zero_grad(set_to_none=True)
	if data_iter is None:
		data_iter = iter(loader)

	total = int(len(loader))
	bar = tqdm(total=total, desc="train", leave=False)

	step = 0
	if first_batch is not None:
		step += 1
		batch = first_batch
		bar.update(1)
		x = batch["x"].to(device, non_blocking=True)
		m = batch["mask"].to(device, non_blocking=True)
		y = batch["y"].to(device, non_blocking=True)
		if amp and device.type == "cuda":
			x = x.to(dtype=torch.float16)

		enabled_amp = bool(amp and device.type == "cuda")
		with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled_amp):
			logits, _ = model(x, m)
			loss = criterion(logits, y)
			loss = loss / float(max(1, int(accum_steps)))

		if enabled_amp and scaler is not None:
			scaler.scale(loss).backward()  # pyright: ignore
		else:
			loss.backward()

		if step % int(accum_steps) == 0:
			if enabled_amp and scaler is not None:
				scaler.unscale_(optimizer)  # pyright: ignore
			if float(grad_clip) > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
			if enabled_amp and scaler is not None:
				scaler.step(optimizer)  # pyright: ignore
				scaler.update()  # pyright: ignore
			else:
				optimizer.step()
			optimizer.zero_grad(set_to_none=True)
			if scheduler is not None:
				scheduler.step()
			if ema is not None:
				ema.update(model)
			n_steps += 1

		loss_sum += float(loss.detach().item())

	for batch in data_iter:  # type: ignore[assignment]
		step += 1
		bar.update(1)
		x = batch["x"].to(device, non_blocking=True)
		m = batch["mask"].to(device, non_blocking=True)
		y = batch["y"].to(device, non_blocking=True)
		if amp and device.type == "cuda":
			x = x.to(dtype=torch.float16)

		enabled_amp = bool(amp and device.type == "cuda")
		with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled_amp):
			logits, _ = model(x, m)
			loss = criterion(logits, y)
			loss = loss / float(max(1, int(accum_steps)))

		if enabled_amp and scaler is not None:
			scaler.scale(loss).backward()  # pyright: ignore
		else:
			loss.backward()

		if step % int(accum_steps) == 0:
			if enabled_amp and scaler is not None:
				scaler.unscale_(optimizer)  # pyright: ignore
			if float(grad_clip) > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
			if enabled_amp and scaler is not None:
				scaler.step(optimizer)  # pyright: ignore
				scaler.update()  # pyright: ignore
			else:
				optimizer.step()
			optimizer.zero_grad(set_to_none=True)
			if scheduler is not None:
				scheduler.step()
			if ema is not None:
				ema.update(model)
			n_steps += 1

		loss_sum += float(loss.detach().item())

	bar.close()
	return float(loss_sum / float(max(1, len(loader))))


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
	p = argparse.ArgumentParser(description="STT(8s embedding) multi-label Attention MIL")
	p.add_argument("--fold", type=int, choices=range(1, N_FOLDS + 1), default=None)
	p.add_argument("--folds", type=int, nargs="+", choices=range(1, N_FOLDS + 1), default=None)
	p.add_argument("--csv", type=str, default="")
	p.add_argument("--output-dir", type=str, default="")
	p.add_argument("--output-root", type=Path, default=CV_OUTPUT_DIR)
	p.add_argument("--skip-existing", action="store_true")
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
	p.add_argument("--seed", type=int, default=None)
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
	 help="Preload all train embeddings/masks into RAM (reduces stalls, uses more memory)",
	)
	p.add_argument(
	 "--preload-val",
	 action=argparse.BooleanOptionalAction,
	 default=PRELOAD_VAL,
	 help="Preload all val embeddings/masks into RAM (recommended for val multi-crop)",
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
	p.add_argument(
	 "--require-complete",
	 action=argparse.BooleanOptionalAction,
	 default=True,
	 help="Use only rows with stt_complete==True",
	)
	return p.parse_args()


def train_fold(args: argparse.Namespace) -> None:
	seed_everything(int(args.seed))
	set_fast_cuda_settings(enable_tf32=bool(args.tf32), fast_cudnn=bool(args.fast_cudnn))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	df = _read_csv_df(Path(args.csv))

	train_ds = SttBagDataset(
	 df,
	 split="train",
	 max_clips_per_video=int(args.max_clips),
	 seed=int(args.seed),
	 limit=int(args.limit_train),
	 cache_max_videos=int(args.cache_max_videos),
	 preload=bool(args.preload_train),
	 include_paths=False,
	 require_complete=bool(args.require_complete),
	 train_sampling=str(args.train_sampling),
	 val_sampling=str(args.val_sampling),
	)
	val_ds = SttBagDataset(
	 df,
	 split="val",
	 max_clips_per_video=int(args.max_clips),
	 seed=int(args.seed),
	 limit=int(args.limit_val),
	 cache_max_videos=int(args.cache_max_videos),
	 preload=bool(args.preload_val),
	 include_paths=False,
	 require_complete=bool(args.require_complete),
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

	pos_weight = compute_pos_weight(train_ds.y_arr).to(device) if bool(args.pos_weight) else None
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

	scheduler = (
	 build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=0.01)
	 if str(args.lr_scheduler) == "cosine"
	 else None
	)

	best_macro_auc = -1.0
	best_epoch = 0
	no_improve = 0
	best_path = out_dir / "best_stt_mil_model.pth"
	last_path = out_dir / "last_stt_mil_model.pth"
	metrics_path = out_dir / "metrics.json"

	history: list[dict[str, Any]] = []

	run_cfg = {
	 "fold": args.fold,
	 "task": "stt_mil",
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
	 "require_complete": bool(args.require_complete),
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
	print(f"val_batch_size={int(val_batch_size)}, val_num_crops={int(val_num_crops)}")
	print(f"early_stop={bool(args.early_stop)}")

	ema_obj: Optional[ModelEMA] = ModelEMA(model, decay=float(args.ema_decay)) if bool(args.ema) else None

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
		 optimizer=optimizer,
		 criterion=criterion,
		 device=device,
		 grad_clip=float(args.grad_clip),
		 amp=amp_enabled,
		 scaler=scaler,
		 accum_steps=accum_steps,
		 scheduler=scheduler,
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
		 amp=amp_enabled,
		 val_num_crops=int(val_num_crops),
		 max_clips=int(args.max_clips),
		 val_dataset=val_ds,
		)
		if ema_obj is not None:
			ema_obj.restore(model)

		row = {
		 "epoch": int(epoch),
		 "train_loss": float(train_loss),
		 **{k: v for k, v in val_metrics.items()},
		}
		history.append(row)
		with metrics_path.open("w", encoding="utf-8") as f:
			json.dump(history, f, ensure_ascii=False, indent=2)

		macro_auc = float(val_metrics.get("macro_auc", float("nan")))
		print(
		 f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | "
		 f"macro_auc={macro_auc:.4f} | macro_f1={float(val_metrics.get('macro_f1', 0.0)):.4f} | "
		 f"micro_f1={float(val_metrics.get('micro_f1', 0.0)):.4f}"
		)

		state = {
		 "epoch": int(epoch),
		 "model": model.state_dict(),
		 "optimizer": optimizer.state_dict(),
		 "run_cfg": run_cfg,
		 "best_macro_auc": float(best_macro_auc),
		 "best_epoch": int(best_epoch),
		}
		if ema_obj is not None:
			state["ema"] = ema_obj.shadow
		torch.save(state, str(last_path))

		improved = (macro_auc == macro_auc) and (macro_auc > best_macro_auc + float(args.early_stop_min_delta))
		if improved:
			best_macro_auc = float(macro_auc)
			best_epoch = int(epoch)
			no_improve = 0
			best_state = dict(state)
			if ema_obj is not None:
				best_state["model"] = ema_obj.shadow
			torch.save(best_state, str(best_path))
		else:
			no_improve += 1

		if bool(args.early_stop) and int(args.early_stop_patience) > 0 and no_improve >= int(args.early_stop_patience):
			print(f"Early stopping at epoch={epoch} (best_epoch={best_epoch}, best_macro_auc={best_macro_auc:.4f})")
			prefetch_iter, prefetch_thread, prefetch_box = None, None, None
			break


def main() -> None:
	run_model_cross_validation(
		parse_args(),
		model_key="stt",
		train_fold=train_fold,
	)


if __name__ == "__main__":
	main()
