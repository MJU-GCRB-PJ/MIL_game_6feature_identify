"""MIL training and evaluation script for the original train-validation pipeline."""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import random
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.multioutput import MultiOutputClassifier
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

try:
	from xgboost import XGBClassifier
	HAS_XGB = True
except ImportError:
	HAS_XGB = False

try:
	import cupy as cp
	HAS_CUPY = True
except ImportError:
	cp = None  # type: ignore[assignment]
	HAS_CUPY = False

# GPU device for metrics / optimization (set once)
_GPU_DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Configuration.
# ──────────────────────────────────────────────────────────────
CLASS_NUM = 6
CLASS_NAME = [
 "sexual_content",
 "violence",
 "fear",
 "inappropriate_language",
 "drugs",
 "crime",
]

SEED = 42
THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_BASE = SCRIPT_DIR / "outputs"
DATA_CSV = OUTPUT_BASE / "splits" / "feat_data-ration_list.csv"
ENSEMBLE_DIR = OUTPUT_BASE / "ensemble"

# Model setup.
MODEL_REGISTRY: list[dict[str, Any]] = [
 {
  "key": "vision",
  "label": "Vision",
  "ckpt_subdir": "vision_mil",
  "ckpt_file": "best_vision_mil_model.pth",
  "activation": "gelu",
  "dataset_type": "vivit",
  "default_embed_dim": 1024,
  "default_attn_dim": 256,
  "default_dropout": 0.2,
 },
 {
  "key": "original_audio",
  "label": "Original Audio",
  "ckpt_subdir": "original_audio_mil",
  "ckpt_file": "best_original_audio_mil_model.pth",
  "activation": "relu",
  "dataset_type": "ast_original",
  "default_embed_dim": 1536,
  "default_attn_dim": 384,
  "default_dropout": 0.3,
 },
 {
  "key": "vocal_audio",
  "label": "Vocal Audio",
  "ckpt_subdir": "vocal_audio_mil",
  "ckpt_file": "best_vocal_audio_mil_model.pth",
  "activation": "relu",
  "dataset_type": "ast_vocal",
  "default_embed_dim": 1536,
  "default_attn_dim": 384,
  "default_dropout": 0.3,
 },
 {
  "key": "non_vocal_audio",
  "label": "Non-Vocal Audio",
  "ckpt_subdir": "non_vocal_audio_mil",
  "ckpt_file": "best_non_vocal_audio_mil_model.pth",
  "activation": "relu",
  "dataset_type": "ast_non_vocal",
  "default_embed_dim": 1536,
  "default_attn_dim": 384,
  "default_dropout": 0.3,
 },
 {
  "key": "ocr",
  "label": "OCR",
  "ckpt_subdir": "ocr_mil",
  "ckpt_file": "best_ocr_mil_model.pth",
  "activation": "relu",
  "dataset_type": "ocr",
  "default_embed_dim": 1536,
  "default_attn_dim": 384,
  "default_dropout": 0.3,
 },
 {
  "key": "stt",
  "label": "STT",
  "ckpt_subdir": "stt_mil",
  "ckpt_file": "best_stt_mil_model.pth",
  "activation": "relu",
  "dataset_type": "stt",
  "default_embed_dim": 1536,
  "default_attn_dim": 384,
  "default_dropout": 0.3,
 },
]

# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────

def seed_everything(seed: int = SEED) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	if torch.cuda.is_available():
		torch.set_float32_matmul_precision("high")


def _safe_str(v: Any) -> str:
	if v is None:
		return ""
	if isinstance(v, float) and v != v:
		return ""
	return str(v)


def _to_float01(x: Any) -> float:
	if x is None:
		return 0.0
	if isinstance(x, (int, np.integer)):
		return float(1.0 if int(x) > 0 else 0.0)
	if isinstance(x, (float, np.floating)):
		if float(x) != float(x):
			return 0.0
		return float(1.0 if float(x) > 0.0 else 0.0)
	s = str(x).strip()
	if not s:
		return 0.0
	if s.lower() in ("true", "t", "yes", "y"):
		return 1.0
	if s.lower() in ("false", "f", "no", "n"):
		return 0.0
	try:
		return float(1.0 if float(s) > 0.0 else 0.0)
	except Exception:
		return 0.0


def _to_bool(v: Any) -> bool:
	if isinstance(v, bool):
		return v
	s = _safe_str(v).strip().lower()
	if s in ("1", "true", "t", "yes", "y"):
		return True
	return False


def _read_csv_df(path: Path) -> pd.DataFrame:
	for enc in ("utf-8-sig", "utf-8", "cp949"):
		try:
			return pd.read_csv(path, encoding=enc)
		except Exception:
			continue
	return pd.read_csv(path)


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
	def __init__(
	 self,
	 *,
	 in_dim: int,
	 embed_dim: int,
	 attn_dim: int,
	 num_classes: int = CLASS_NUM,
	 dropout: float = 0.1,
	 activation: str = "relu",
	) -> None:
		super().__init__()
		act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)
		self.instance_encoder = nn.Sequential(
		 nn.Linear(int(in_dim), int(embed_dim)),
		 act,
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

	def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		"""x: (B, T, D), mask: (B, T) → logits: (B, C)"""
		h = self.instance_encoder(x)
		v = torch.tanh(self.attn_v(h))
		u = torch.sigmoid(self.attn_u(h))
		a_logits = self.attn_w(v * u).squeeze(-1)
		a = self._masked_softmax(a_logits, mask, dim=1)
		z = (a.unsqueeze(-1) * h).sum(dim=1)
		logits = self.classifier(z)
		return logits


# ──────────────────────────────────────────────────────────────
# Model setup.
# ──────────────────────────────────────────────────────────────

def load_model(reg: dict[str, Any], device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
	"""Helper function for load model."""
	ckpt_path = OUTPUT_BASE / reg["ckpt_subdir"] / reg["ckpt_file"]
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	state_dict = ckpt["model"]


	cfg = ckpt.get("config") or ckpt.get("run_cfg") or {}

	feat_dim = cfg.get("feat_dim")
	if feat_dim is None:
		feat_dim = state_dict["instance_encoder.0.weight"].shape[1]
	feat_dim = int(feat_dim)

	embed_dim = int(cfg.get("embed_dim", reg["default_embed_dim"]))
	attn_dim = int(cfg.get("attn_dim", reg["default_attn_dim"]))
	dropout = float(cfg.get("dropout", reg["default_dropout"]))

	model = GatedAttentionMIL(
	 in_dim=feat_dim,
	 embed_dim=embed_dim,
	 attn_dim=attn_dim,
	 num_classes=CLASS_NUM,
	 dropout=dropout,
	 activation=reg["activation"],
	)
	model.load_state_dict(state_dict, strict=True)
	model.to(device).eval()
	return model, cfg


# ──────────────────────────────────────────────────────────────
# Read input.
# ──────────────────────────────────────────────────────────────

def _read_vivit_manifest_feature_paths(manifest_path: Path) -> list[Path]:
	if not manifest_path.exists():
		return []
	rows: list[dict[str, str]] = []
	for enc in ("utf-8-sig", "utf-8"):
		try:
			with manifest_path.open("r", encoding=enc, newline="") as f:
				reader = csv.DictReader(f)
				rows = list(reader)
			break
		except UnicodeDecodeError:
			continue
	if not rows:
		return []

	def _chunk_idx(r: dict) -> int:
		try:
			return int(float(_safe_str(r.get("chunk_idx"))))
		except Exception:
			return 10**9
	rows = sorted(rows, key=_chunk_idx)

	out: list[Path] = []
	for r in rows:
		p = Path(_safe_str(r.get("feature_path")))
		if p.exists():
			out.append(p)
	return out


def _resolve_feature_path(manifest_path: Path, raw: str) -> Path:
	raw = raw.strip()
	if not raw:
		return Path("")
	p = Path(raw)
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
	return rows


def _read_ast_feature_paths(manifest_path: Path, audio_type: str) -> list[Path]:
	"""Helper function for read ast feature paths."""
	if not manifest_path.exists():
		return []
	rows = _iter_manifest_rows(manifest_path)
	if not rows:
		return []
	filtered: list[tuple[int, Path]] = []
	unsorted: list[Path] = []
	for r in rows:
		if r.get("audio_type", "") != audio_type:
			continue
		raw = r.get("feature_path", "")
		p = _resolve_feature_path(manifest_path, raw)
		if not p or not p.exists():
			continue
		idx_raw = r.get("chunk_idx", "")
		idx: Optional[int] = None
		try:
			if idx_raw:
				idx = int(float(idx_raw))
		except Exception:
			pass
		if idx is not None:
			filtered.append((idx, p))
		else:
			unsorted.append(p)
	filtered.sort(key=lambda t: t[0])
	return [p for _, p in filtered] + sorted(unsorted)


def _load_1d_npy_stack(paths: list[Path]) -> np.ndarray:
	"""Helper function for load 1d npy stack."""
	feats = []
	for p in paths:
		arr = np.load(str(p), allow_pickle=False)
		arr = np.asarray(arr)
		if arr.ndim == 2 and arr.shape[0] == 1:
			arr = arr[0]
		feats.append(arr.astype(np.float32, copy=False))
	return np.stack(feats, axis=0)


def _load_2d_emb_and_mask(emb_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
	"""Helper function for load 2d emb and mask."""
	emb = np.load(str(emb_path), allow_pickle=False).astype(np.float32, copy=False)
	mask = np.load(str(mask_path), allow_pickle=False).astype(np.float32, copy=False)
	presence = (mask > 0).astype(np.float32)
	return emb, presence


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────


@torch.inference_mode()
def _gpu_roc_auc_scores(y_true: np.ndarray, y_score: np.ndarray) -> list[float]:
	"""Per-column ROC AUC on GPU. Returns list of C floats (nan if undefined)."""
	yt = torch.from_numpy(y_true.astype(np.float32)).to(_GPU_DEVICE)
	ys = torch.from_numpy(y_score.astype(np.float32)).to(_GPU_DEVICE)
	N = yt.shape[0]
	results: list[float] = []
	for c in range(yt.shape[1]):
		t, s = yt[:, c], ys[:, c]
		n_pos = t.sum()
		if n_pos == 0 or n_pos == N:
			results.append(float("nan"))
			continue
		n_neg = N - n_pos
		desc = torch.argsort(s, descending=True)
		t_sorted = t[desc]
		tps = torch.cumsum(t_sorted, 0)
		fps = torch.arange(1, N + 1, device=_GPU_DEVICE, dtype=torch.float32) - tps
		tpr = torch.cat([torch.zeros(1, device=_GPU_DEVICE), tps / n_pos])
		fpr = torch.cat([torch.zeros(1, device=_GPU_DEVICE), fps / n_neg])
		results.append(float(torch.trapezoid(tpr, fpr).cpu()))
	return results


@torch.inference_mode()
def _gpu_recall_f1(
 y_true: np.ndarray, y_pred: np.ndarray,
) -> tuple[list[float], list[float], float, float]:
	"""Per-column recall & F1 + macro values on GPU (zero_division=0)."""
	yt = torch.from_numpy(y_true.astype(np.float32)).to(_GPU_DEVICE)
	yp = torch.from_numpy(y_pred.astype(np.float32)).to(_GPU_DEVICE)
	tp = (yt * yp).sum(0)
	fn = (yt * (1 - yp)).sum(0)
	fp = ((1 - yt) * yp).sum(0)
	recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
	prec = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
	f1 = torch.where(
	 prec + recall > 0,
	 2 * prec * recall / (prec + recall),
	 torch.zeros_like(tp),
	)
	return (
	 recall.cpu().tolist(),
	 f1.cpu().tolist(),
	 float(recall.mean().cpu()),
	 float(f1.mean().cpu()),
	)


def _cupy_roc_auc_single(y_true_cp: Any, y_score_cp: Any) -> float:
	"""Single-column AUC using CuPy (for DE objective hot loop)."""
	n = len(y_true_cp)
	n_pos = float(y_true_cp.sum())
	n_neg = n - n_pos
	if n_pos == 0 or n_neg == 0:
		return float("nan")
	desc = cp.argsort(y_score_cp)[::-1]
	t = y_true_cp[desc]
	tps = cp.cumsum(t)
	fps = cp.arange(1, n + 1, dtype=cp.float32) - tps
	tpr = cp.concatenate([cp.zeros(1, dtype=cp.float32), tps / cp.float32(n_pos)])
	fpr = cp.concatenate([cp.zeros(1, dtype=cp.float32), fps / cp.float32(n_neg)])
	return float(cp.trapz(tpr, fpr))


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────


def _make_final_mask(pad_mask: np.ndarray, presence_mask: np.ndarray) -> np.ndarray:
	final = pad_mask * presence_mask
	if float(final.sum()) <= 0.0:
		final = pad_mask.copy()
	return final.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Model setup.
# ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def _infer_vivit(
 model: nn.Module,
 full_feats: np.ndarray,
 device: torch.device,
) -> np.ndarray:
	"""Vision: use ALL clips in a single forward pass → sigmoid → (C,)."""
	n_total = full_feats.shape[0]
	mask_np = np.ones(n_total, dtype=np.float32)
	x = torch.from_numpy(full_feats).unsqueeze(0).to(device, dtype=torch.float32)
	m = torch.from_numpy(mask_np).unsqueeze(0).to(device, dtype=torch.float32)
	with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
		logits = model(x, m)  # (1, C)
	return torch.sigmoid(logits.float().squeeze(0)).cpu().numpy()


@torch.inference_mode()
def _infer_ast(
 model: nn.Module,
 full_feats: np.ndarray,
 device: torch.device,
) -> np.ndarray:
	"""Audio(AST): use ALL clips in a single forward pass → sigmoid → (C,)."""
	n_total = full_feats.shape[0]
	mask_np = np.ones(n_total, dtype=np.float32)
	x = torch.from_numpy(full_feats.astype(np.float32, copy=False)).unsqueeze(0).to(device, dtype=torch.float32)
	m = torch.from_numpy(mask_np).unsqueeze(0).to(device, dtype=torch.float32)
	with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
		logits = model(x, m)  # (1, C)
	return torch.sigmoid(logits.float().squeeze(0)).cpu().numpy()


@torch.inference_mode()
def _infer_text(
 model: nn.Module,
 full_emb: np.ndarray,
 full_presence: np.ndarray,
 device: torch.device,
) -> np.ndarray:
	"""OCR/STT: use ALL clips in a single forward pass → sigmoid → (C,)."""
	n_total = full_emb.shape[0]
	pad_mask = np.ones(n_total, dtype=np.float32)
	crop_mask = _make_final_mask(pad_mask, full_presence)
	x = torch.from_numpy(full_emb.astype(np.float32, copy=False)).unsqueeze(0).to(device, dtype=torch.float32)
	m = torch.from_numpy(crop_mask).unsqueeze(0).to(device, dtype=torch.float32)
	with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
		logits = model(x, m)  # (1, C)
	return torch.sigmoid(logits.float().squeeze(0)).cpu().numpy()


def collect_predictions(
 df: pd.DataFrame,
 device: torch.device,
) -> tuple[
 dict[str, dict[str, Optional[np.ndarray]]],  # predictions[model_key][file_name]
 dict[str, np.ndarray],                         # labels[file_name]
 dict[str, str],                                 # splits[file_name]
 list[str],                                      # ordered file_names
]:
	"""Helper function for collect predictions."""

	labels: dict[str, np.ndarray] = {}
	splits: dict[str, str] = {}
	all_file_names: list[str] = []

	for _, row in df.iterrows():
		fn = Path(_safe_str(row.get("file_name"))).name
		if not fn:
			continue
		y = np.array([_to_float01(row.get(c)) for c in CLASS_NAME], dtype=np.float32)
		labels[fn] = y
		splits[fn] = str(row.get("split", "")).strip()
		all_file_names.append(fn)


	fn_to_row_idx: dict[str, int] = {}
	for i, row in df.iterrows():
		fn = Path(_safe_str(row.get("file_name"))).name
		if fn:
			fn_to_row_idx[fn] = i

	predictions: dict[str, dict[str, Optional[np.ndarray]]] = {}

	for reg in MODEL_REGISTRY:
		key = reg["key"]
		predictions[key] = {}
		print(f"\n{'='*60}")
		print(f"[{reg['label']}] Running model inference...")
		print(f"{'='*60}")

		# Load the model.
		try:
			model, cfg = load_model(reg, device)
		except FileNotFoundError as e:
			print(f"  Checkpoint unavailable: {e}")
			for fn in all_file_names:
				predictions[key][fn] = None
			continue

		dtype = reg["dataset_type"]
		done = 0
		skipped = 0

		for fn in tqdm(all_file_names, desc=f"  {reg['label']}", leave=True):
			row_idx = fn_to_row_idx.get(fn)
			if row_idx is None:
				predictions[key][fn] = None
				skipped += 1
				continue
			row = df.iloc[row_idx]

			try:
				if dtype == "vivit":
					if not _to_bool(row.get("vivit_complete")):
						predictions[key][fn] = None
						skipped += 1
						continue
					manifest_path = Path(_safe_str(row.get("vivit_manifest")))
					feat_paths = _read_vivit_manifest_feature_paths(manifest_path)
					if not feat_paths:
						predictions[key][fn] = None
						skipped += 1
						continue
					full_feats = _load_1d_npy_stack(feat_paths)
					probs = _infer_vivit(model, full_feats, device)

				elif dtype in ("ast_original", "ast_vocal", "ast_non_vocal"):
					audio_type_map = {
					 "ast_original": "original",
					 "ast_vocal": "vocal",
					 "ast_non_vocal": "non-vocal",
					}
					manifest_path = Path(_safe_str(row.get("ast_manifest")))
					feat_paths = _read_ast_feature_paths(manifest_path, audio_type_map[dtype])
					if not feat_paths:
						predictions[key][fn] = None
						skipped += 1
						continue
					if not all(p.exists() for p in feat_paths):
						predictions[key][fn] = None
						skipped += 1
						continue
					full_feats = _load_1d_npy_stack(feat_paths)
					probs = _infer_ast(model, full_feats, device)

				elif dtype == "ocr":
					if not _to_bool(row.get("ocr_complete")):
						predictions[key][fn] = None
						skipped += 1
						continue
					emb_path = Path(_safe_str(row.get("ocr_emb")))
					mask_path = Path(_safe_str(row.get("ocr_mask")))
					if not emb_path.exists() or not mask_path.exists():
						predictions[key][fn] = None
						skipped += 1
						continue
					emb, presence = _load_2d_emb_and_mask(emb_path, mask_path)
					if emb.shape[0] == 0 or presence.sum() <= 0:
						predictions[key][fn] = None
						skipped += 1
						continue
					probs = _infer_text(model, emb, presence, device)

				elif dtype == "stt":
					if not _to_bool(row.get("stt_complete")):
						predictions[key][fn] = None
						skipped += 1
						continue
					emb_path = Path(_safe_str(row.get("stt_emb")))
					mask_path = Path(_safe_str(row.get("stt_mask")))
					if not emb_path.exists() or not mask_path.exists():
						predictions[key][fn] = None
						skipped += 1
						continue
					emb, presence = _load_2d_emb_and_mask(emb_path, mask_path)
					if emb.shape[0] == 0 or presence.sum() <= 0:
						predictions[key][fn] = None
						skipped += 1
						continue
					probs = _infer_text(model, emb, presence, device)
				else:
					predictions[key][fn] = None
					skipped += 1
					continue

				predictions[key][fn] = probs
				done += 1
			except Exception as e:
				predictions[key][fn] = None
				skipped += 1
				if done < 3:
					print(f"    ⚠ {fn}: {e}")

		print(f"  Complete: {done}, skipped: {skipped}")
		del model
		torch.cuda.empty_cache()

	return predictions, labels, splits, all_file_names


# ──────────────────────────────────────────────────────────────
# Compute values.
# ──────────────────────────────────────────────────────────────

def compute_metrics(
 y_true: np.ndarray,
 y_prob: np.ndarray,
 threshold: float = THRESHOLD,
) -> dict[str, Any]:
	"""Helper function for compute metrics."""
	if y_true.shape[0] == 0:
		nan_d = {c: float("nan") for c in CLASS_NAME}
		return {
		 "macro_auc": float("nan"),
		 "macro_recall": float("nan"),
		 "macro_f1": float("nan"),
		 "per_class_auc": dict(nan_d),
		 "per_class_recall": dict(nan_d),
		 "per_class_f1": dict(nan_d),
		 "n": 0,
		}

	y_pred = (y_prob >= threshold).astype(np.int32)

	# GPU AUC
	aucs = _gpu_roc_auc_scores(y_true, y_prob)
	per_class_auc = {CLASS_NAME[i]: aucs[i] for i in range(CLASS_NUM)}
	valid_aucs = [a for a in aucs if a == a]  # filter NaN
	macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

	# GPU Recall / F1
	per_recall, per_f1, macro_recall, macro_f1 = _gpu_recall_f1(y_true, y_pred)
	per_class_recall = {CLASS_NAME[i]: per_recall[i] for i in range(CLASS_NUM)}
	per_class_f1 = {CLASS_NAME[i]: per_f1[i] for i in range(CLASS_NUM)}

	return {
	 "macro_auc": macro_auc,
	 "macro_recall": macro_recall,
	 "macro_f1": macro_f1,
	 "per_class_auc": per_class_auc,
	 "per_class_recall": per_class_recall,
	 "per_class_f1": per_class_f1,
	 "n": int(y_true.shape[0]),
	}


def _metrics_for_splits(
 y_true_all: np.ndarray,
 y_prob_all: np.ndarray,
 split_flags: np.ndarray,  # 0=train, 1=val
) -> dict[str, dict[str, Any]]:
	"""Helper function for metrics for splits."""
	train_mask = split_flags == 0
	val_mask = split_flags == 1
	return {
	 "total": compute_metrics(y_true_all, y_prob_all),
	 "train": compute_metrics(y_true_all[train_mask], y_prob_all[train_mask]),
	 "val": compute_metrics(y_true_all[val_mask], y_prob_all[val_mask]),
	}


# ──────────────────────────────────────────────────────────────
# Ensemble evaluation.
# ──────────────────────────────────────────────────────────────

def _get_available_data(
 model_keys: list[str],
 predictions: dict[str, dict[str, Optional[np.ndarray]]],
 labels: dict[str, np.ndarray],
 splits: dict[str, str],
 all_file_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
	"""Helper function for get available data."""
	valid_fns: list[str] = []
	y_list: list[np.ndarray] = []
	split_list: list[int] = []
	sample_preds: list[list[Optional[np.ndarray]]] = []

	for fn in all_file_names:
		avail = [predictions[k].get(fn) for k in model_keys]
		if all(a is None for a in avail):
			continue
		valid_fns.append(fn)
		y_list.append(labels[fn])
		split_list.append(0 if splits[fn] == "train" else 1)
		sample_preds.append(avail)

	N = len(valid_fns)
	M = len(model_keys)
	y_true = np.stack(y_list, axis=0) if y_list else np.zeros((0, CLASS_NUM), dtype=np.float32)
	split_flags = np.array(split_list, dtype=np.int32)
	prob_tensor = np.zeros((M, N, CLASS_NUM), dtype=np.float64)
	avail_mask = np.zeros((M, N), dtype=np.float64)
	for i in range(N):
		for j in range(M):
			p = sample_preds[i][j]
			if p is not None:
				prob_tensor[j, i] = p.astype(np.float64)
				avail_mask[j, i] = 1.0
	return y_true, prob_tensor, avail_mask, split_flags, valid_fns


def ensemble_hard_voting(
 model_keys: list[str],
 predictions: dict[str, dict[str, Optional[np.ndarray]]],
 labels: dict[str, np.ndarray],
 splits: dict[str, str],
 all_file_names: list[str],
) -> dict[str, Any]:
	"""Helper function for ensemble hard voting."""
	y_true, prob_tensor, avail_mask, split_flags, valid_fns = _get_available_data(
	 model_keys, predictions, labels, splits, all_file_names
	)
	if y_true.shape[0] == 0:
		return {}

 # prob_tensor: (M, N, C), avail_mask: (M, N)
	binary_votes = (prob_tensor >= THRESHOLD).astype(np.float64)
	weighted_votes = binary_votes * avail_mask[:, :, None]  # zero out missing
	votes = weighted_votes.sum(axis=0)  # (N, C)
	counts = np.maximum(avail_mask.sum(axis=0)[:, None], 1.0)  # (N, 1)
	y_prob = (votes / counts).astype(np.float32)

	result = _metrics_for_splits(y_true, y_prob, split_flags)
	return {"metrics": result, "weights": "N/A"}


def ensemble_soft_voting(
 model_keys: list[str],
 predictions: dict[str, dict[str, Optional[np.ndarray]]],
 labels: dict[str, np.ndarray],
 splits: dict[str, str],
 all_file_names: list[str],
) -> dict[str, Any]:
	"""Helper function for ensemble soft voting."""
	y_true, prob_tensor, avail_mask, split_flags, valid_fns = _get_available_data(
	 model_keys, predictions, labels, splits, all_file_names
	)
	if y_true.shape[0] == 0:
		return {}

 # prob_tensor: (M, N, C), avail_mask: (M, N)
	weighted_prob = prob_tensor * avail_mask[:, :, None]  # (M, N, C)
	prob_sum = weighted_prob.sum(axis=0)  # (N, C)
	counts = np.maximum(avail_mask.sum(axis=0)[:, None], 1.0)  # (N, 1)
	y_prob = (prob_sum / counts).astype(np.float32)

	result = _metrics_for_splits(y_true, y_prob, split_flags)
	return {"metrics": result, "weights": "equal"}


def ensemble_weighted_soft_voting(
 model_keys: list[str],
 predictions: dict[str, dict[str, Optional[np.ndarray]]],
 labels: dict[str, np.ndarray],
 splits: dict[str, str],
 all_file_names: list[str],
) -> dict[str, Any]:
	"""Helper function for ensemble weighted soft voting."""
	y_true, prob_tensor, avail_mask, split_flags, valid_fns = _get_available_data(
	 model_keys, predictions, labels, splits, all_file_names
	)
	if y_true.shape[0] == 0:
		return {}

	n_models = len(model_keys)
	if n_models < 2:
		return ensemble_soft_voting(model_keys, predictions, labels, splits, all_file_names)

	val_mask_flag = split_flags == 1
	val_indices = np.where(val_mask_flag)[0]

	if len(val_indices) < 5:
		return ensemble_soft_voting(model_keys, predictions, labels, splits, all_file_names)


	val_probs = prob_tensor[:, val_indices, :]  # (M, n_val, C)
	val_avail = avail_mask[:, val_indices]       # (M, n_val)
	val_y = y_true[val_indices]


	if HAS_CUPY and _GPU_DEVICE.type == "cuda":
		val_probs_cp = cp.asarray(val_probs, dtype=cp.float64)
		val_avail_cp = cp.asarray(val_avail, dtype=cp.float64)
		val_y_cp = cp.asarray(val_y, dtype=cp.float32)

		def objective(w_raw: np.ndarray) -> float:
			w = cp.asarray(w_raw / (w_raw.sum() + 1e-12), dtype=cp.float64)
			w_3d = w[:, None, None]
			weighted = (w_3d * val_probs_cp * val_avail_cp[:, :, None]).sum(axis=0)
			w_sum = (w[:, None] * val_avail_cp).sum(axis=0)[:, None]
			w_sum = cp.maximum(w_sum, 1e-12)
			y_prob_val = (weighted / w_sum).astype(cp.float32)

			auc_list: list[float] = []
			for c in range(CLASS_NUM):
				auc = _cupy_roc_auc_single(val_y_cp[:, c], y_prob_val[:, c])
				if auc == auc:  # not NaN
					auc_list.append(auc)
			if not auc_list:
				return 1.0
			return 1.0 - float(np.mean(auc_list))
	else:
		def objective(w_raw: np.ndarray) -> float:
			w = w_raw / (w_raw.sum() + 1e-12)
			w_3d = w[:, None, None]
			weighted = (w_3d * val_probs * val_avail[:, :, None]).sum(axis=0)
			w_sum = (w[:, None] * val_avail).sum(axis=0)[:, None]
			w_sum = np.maximum(w_sum, 1e-12)
			y_prob_val = (weighted / w_sum).astype(np.float32)

			auc_list: list[float] = []
			for c in range(CLASS_NUM):
				try:
					if len(np.unique(val_y[:, c])) < 2:
						continue
					auc_list.append(roc_auc_score(val_y[:, c], y_prob_val[:, c]))
				except Exception:
					continue
			if not auc_list:
				return 1.0
			return 1.0 - float(np.mean(auc_list))

	bounds = [(0.01, 1.0)] * n_models
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		result_opt = differential_evolution(
		 objective,
		 bounds,
		 seed=SEED,
		 maxiter=200,
		 tol=1e-6,
		 polish=True,
		 init="sobol",
		 workers=1,
		)

	w_opt = result_opt.x / (result_opt.x.sum() + 1e-12)
	weight_dict = {model_keys[j]: round(float(w_opt[j]), 4) for j in range(n_models)}


	w_3d = w_opt[:, None, None]  # (M, 1, 1)
	prob_weighted = (w_3d * prob_tensor * avail_mask[:, :, None]).sum(axis=0)  # (N, C)
	w_total = (w_opt[:, None] * avail_mask).sum(axis=0)[:, None]  # (N, 1)
	w_total = np.maximum(w_total, 1e-12)
	y_prob_all = (prob_weighted / w_total).astype(np.float32)

	result = _metrics_for_splits(y_true, y_prob_all, split_flags)
	return {"metrics": result, "weights": str(weight_dict)}


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────


class _TorchLogisticRegression:
	"""Helper class for TorchLogisticRegression."""

	def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> None:
		self.C = C
		self.max_iter = max_iter
		self.random_state = random_state
		self._weight: torch.Tensor | None = None
		self._bias: torch.Tensor | None = None
		self.coef_: np.ndarray | None = None
		self.classes_ = np.array([0, 1])

	def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchLogisticRegression":
		torch.manual_seed(self.random_state)
		dev = _GPU_DEVICE
		X_t = torch.from_numpy(X.astype(np.float32)).to(dev)
		y_t = torch.from_numpy(y.astype(np.float32)).to(dev)

		n_feat = X_t.shape[1]
		weight = torch.zeros(n_feat, device=dev, requires_grad=True)
		bias = torch.zeros(1, device=dev, requires_grad=True)
		reg = 1.0 / max(self.C, 1e-12)

		optimizer = torch.optim.LBFGS(
		 [weight, bias], max_iter=self.max_iter, line_search_fn="strong_wolfe",
		)

		def closure() -> torch.Tensor:
			optimizer.zero_grad()
			logits = X_t @ weight + bias
			loss = nn.functional.binary_cross_entropy_with_logits(logits, y_t, reduction="mean")
			loss = loss + 0.5 * reg * (weight * weight).sum()
			loss.backward()
			return loss

		optimizer.step(closure)

		self._weight = weight.detach()
		self._bias = bias.detach()
		self.coef_ = weight.detach().cpu().numpy().reshape(1, -1)
		return self

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		X_t = torch.from_numpy(X.astype(np.float32)).to(_GPU_DEVICE)
		with torch.no_grad():
			prob_1 = torch.sigmoid(X_t @ self._weight + self._bias).cpu().numpy()
		return np.column_stack([1.0 - prob_1, prob_1]).astype(np.float32)

	def predict(self, X: np.ndarray) -> np.ndarray:
		return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

	def get_params(self, deep: bool = True) -> dict[str, Any]:
		return {"C": self.C, "max_iter": self.max_iter, "random_state": self.random_state}


def ensemble_stacking(
 model_keys: list[str],
 predictions: dict[str, dict[str, Optional[np.ndarray]]],
 labels: dict[str, np.ndarray],
 splits: dict[str, str],
 all_file_names: list[str],
 meta_learner_name: str = "LR",
) -> dict[str, Any]:
	"""Helper function for ensemble stacking."""
	y_true, prob_tensor, avail_mask, split_flags, valid_fns = _get_available_data(
	 model_keys, predictions, labels, splits, all_file_names
	)
	if y_true.shape[0] == 0:
		return {}

	n_models = len(model_keys)
	n = y_true.shape[0]


	X = np.full((n, n_models * CLASS_NUM), 0.5, dtype=np.float32)
	for j in range(n_models):
		mask_j = avail_mask[j] > 0  # (N,) boolean
		X[mask_j, j * CLASS_NUM : (j + 1) * CLASS_NUM] = prob_tensor[j, mask_j].astype(np.float32)

	train_mask = split_flags == 0
	val_mask = split_flags == 1
	X_train, y_train = X[train_mask], y_true[train_mask]
	X_val, y_val = X[val_mask], y_true[val_mask]

	if X_train.shape[0] < 10 or X_val.shape[0] < 5:
		return {}


	_use_cuda = torch.cuda.is_available()
	if meta_learner_name == "LR":
		if _use_cuda:
			base = _TorchLogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
		else:
			base = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=SEED)
	elif meta_learner_name == "RF":
		base = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED, n_jobs=-1)
	elif meta_learner_name == "XGB":
		if not HAS_XGB:
			return {}
		xgb_params: dict[str, Any] = dict(
		 n_estimators=100, max_depth=3, learning_rate=0.1,
		 random_state=SEED, eval_metric="logloss", verbosity=0,
		)
		if _use_cuda:
			xgb_params["device"] = "cuda"
			xgb_params["tree_method"] = "hist"
		base = XGBClassifier(**xgb_params)
	else:
		return {}


	estimators_: list[Any] = []
	if isinstance(base, _TorchLogisticRegression):

		try:
			for c_idx in range(CLASS_NUM):
				est = _TorchLogisticRegression(**base.get_params())
				est.fit(X_train, y_train[:, c_idx].astype(np.float32))
				estimators_.append(est)
		except Exception as e:
			print(f"    Stacking {meta_learner_name} fit failed: {e}")
			return {}
	else:
		meta = MultiOutputClassifier(base, n_jobs=1)
		try:
			meta.fit(X_train, y_train.astype(int))
			estimators_ = list(meta.estimators_)
		except Exception as e:
			print(f"    Stacking {meta_learner_name} fit failed: {e}")
			return {}


	try:
		val_proba = np.zeros((X_val.shape[0], CLASS_NUM), dtype=np.float32)
		for c_idx, est in enumerate(estimators_):
			if hasattr(est, "predict_proba"):
				pp = est.predict_proba(X_val)
				if pp.ndim == 2 and pp.shape[1] == 2:
					val_proba[:, c_idx] = pp[:, 1].astype(np.float32)
				elif pp.ndim == 2 and pp.shape[1] == 1:
					val_proba[:, c_idx] = pp[:, 0].astype(np.float32)
				else:
					val_proba[:, c_idx] = pp.ravel().astype(np.float32)
			else:
				val_proba[:, c_idx] = est.predict(X_val).astype(np.float32)
	except Exception as e:
		print(f"    Stacking {meta_learner_name} prediction failed: {e}")
		return {}


	try:
		from sklearn.model_selection import StratifiedKFold
		train_proba = np.zeros((X_train.shape[0], CLASS_NUM), dtype=np.float32)
		n_cv = min(5, X_train.shape[0])

		if isinstance(base, _TorchLogisticRegression):

			for c_idx in range(CLASS_NUM):
				kf = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=SEED)
				y_col = y_train[:, c_idx].astype(int)
				for tr_idx, va_idx in kf.split(X_train, y_col):
					est = _TorchLogisticRegression(**base.get_params())
					est.fit(X_train[tr_idx], y_col[tr_idx].astype(np.float32))
					pp = est.predict_proba(X_train[va_idx])
					train_proba[va_idx, c_idx] = pp[:, 1].astype(np.float32)
		else:
			for c_idx in range(CLASS_NUM):
				base_clone = type(base)(**base.get_params())
				cv_pred = cross_val_predict(
				 base_clone, X_train, y_train[:, c_idx].astype(int),
				 cv=n_cv, method="predict_proba", n_jobs=1,
				)
				if cv_pred.ndim == 2 and cv_pred.shape[1] == 2:
					train_proba[:, c_idx] = cv_pred[:, 1].astype(np.float32)
				elif cv_pred.ndim == 2 and cv_pred.shape[1] == 1:
					train_proba[:, c_idx] = cv_pred[:, 0].astype(np.float32)
				else:
					train_proba[:, c_idx] = cv_pred.ravel().astype(np.float32)
	except Exception:
	 # fallback: in-sample
		train_proba = np.zeros((X_train.shape[0], CLASS_NUM), dtype=np.float32)
		for c_idx, est in enumerate(estimators_):
			if hasattr(est, "predict_proba"):
				pp = est.predict_proba(X_train)
				if pp.ndim == 2 and pp.shape[1] == 2:
					train_proba[:, c_idx] = pp[:, 1]
				else:
					train_proba[:, c_idx] = pp.ravel()
			else:
				train_proba[:, c_idx] = est.predict(X_train)


	y_prob_all = np.zeros((n, CLASS_NUM), dtype=np.float32)
	y_prob_all[train_mask] = train_proba
	y_prob_all[val_mask] = val_proba

	result = _metrics_for_splits(y_true, y_prob_all, split_flags)

	# Extract required data.
	weight_info = {}
	for c_idx, est in enumerate(estimators_):
		cname = CLASS_NAME[c_idx]
		if hasattr(est, "coef_"):
			weight_info[cname] = est.coef_.ravel().tolist()
		elif hasattr(est, "feature_importances_"):
			weight_info[cname] = est.feature_importances_.tolist()

	return {"metrics": result, "weights": str(weight_info) if weight_info else "N/A"}


# ──────────────────────────────────────────────────────────────
# Save output.
# ──────────────────────────────────────────────────────────────

def _build_best_criteria() -> list[dict[str, str]]:
	"""Helper function for build best criteria."""
	criteria: list[dict[str, str]] = []
	# 1-3) Macro AUC per dataset split
	for prefix in ("Total", "Train", "Val"):
		col = f"{prefix}_Macro_AUC"
		criteria.append({
		 "column": col,
		 "filename": f"best_{prefix.lower()}_macro_auc.pth",
		 "description": f"Best {prefix} Macro AUC",
		})
 # 4-21) Per-class AUC per dataset split × 6 classes
	for prefix in ("Total", "Train", "Val"):
		for cname in CLASS_NAME:
			col = f"{prefix}_AUC_{cname}"
			criteria.append({
			 "column": col,
			 "filename": f"best_{prefix.lower()}_auc_{cname}.pth",
			 "description": f"Best {prefix} AUC for {cname}",
			})
	return criteria


def save_best_ensemble_pth(result_df: pd.DataFrame, save_dir: Path) -> None:
	"""Helper function for save best ensemble pth."""
	pth_dir = save_dir / "best_pth"
	pth_dir.mkdir(parents=True, exist_ok=True)

	criteria = _build_best_criteria()
	saved_count = 0

	print(f"\n{'='*60}")
	print(f"Saving best ensemble .pth files for {len(criteria)} criteria")
	print(f"{'='*60}")

	for crit in criteria:
		col = crit["column"]
		fname = crit["filename"]
		desc = crit["description"]

		if col not in result_df.columns:
			print(f"  Column '{col}' is missing; skipping")
			continue

		series = result_df[col].dropna()
		if series.empty:
			print(f"  Column '{col}' has no valid data; skipping")
			continue

		best_idx = series.idxmax()
		best_row = result_df.loc[best_idx]
		best_score = float(best_row[col])


		metrics_row = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))
		               for k, v in best_row.to_dict().items()}

		pth_data = {
		 "criterion": col,
		 "description": desc,
		 "best_score": best_score,
		 "ensemble_method": str(best_row.get("Ensemble_Method", "")),
		 "model_combination": str(best_row.get("Model_Combination", "")),
		 "model_weights": str(best_row.get("Model_Weights", "")),
		 "metrics_row": metrics_row,
		 "class_names": CLASS_NAME,
		 "threshold": THRESHOLD,
		 "seed": SEED,
		}

		pth_path = pth_dir / fname
		torch.save(pth_data, str(pth_path))
		saved_count += 1
		print(f"  ✔ [{desc}] score={best_score:.4f} | "
		      f"{best_row.get('Ensemble_Method', '')} | "
		      f"{best_row.get('Model_Combination', '')} → {fname}")

	print(f"\n  Saved: {saved_count}/{len(criteria)} -> {pth_dir}")


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────

def _flatten_metrics_to_row(
 no: int,
 method: str,
 model_combo: str,
 weights: str,
 metrics_dict: dict[str, dict[str, Any]],
) -> dict[str, Any]:
	"""metrics_dict = {"total": {...}, "train": {...}, "val": {...}} → flat row dict."""
	row: dict[str, Any] = {
	 "No": no,
	 "Ensemble_Method": method,
	 "Model_Combination": model_combo,
	 "Model_Weights": weights,
	}

	for dataset_prefix, split_key in [("Total", "total"), ("Train", "train"), ("Val", "val")]:
		m = metrics_dict.get(split_key, {})
		row[f"{dataset_prefix}_N"] = m.get("n", 0)
		row[f"{dataset_prefix}_Macro_AUC"] = m.get("macro_auc", float("nan"))
		row[f"{dataset_prefix}_Macro_Recall"] = m.get("macro_recall", float("nan"))
		row[f"{dataset_prefix}_Macro_F1"] = m.get("macro_f1", float("nan"))

		pc_auc = m.get("per_class_auc", {})
		pc_recall = m.get("per_class_recall", {})
		pc_f1 = m.get("per_class_f1", {})
		for c in CLASS_NAME:
			row[f"{dataset_prefix}_AUC_{c}"] = pc_auc.get(c, float("nan"))
			row[f"{dataset_prefix}_Recall_{c}"] = pc_recall.get(c, float("nan"))
			row[f"{dataset_prefix}_F1_{c}"] = pc_f1.get(c, float("nan"))

	return row


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────

def main() -> None:
	seed_everything(SEED)
	warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel\.delayed.*")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	print(f"GPU acceleration: Metrics={'PyTorch CUDA' if device.type == 'cuda' else 'CPU'} | "
	      f"DE={'CuPy' if HAS_CUPY and device.type == 'cuda' else 'NumPy'} | "
	      f"Stacking LR={'PyTorch CUDA' if device.type == 'cuda' else 'sklearn'}")

	# Load input.
	if not DATA_CSV.exists():
		raise FileNotFoundError(f"CSV not found: {DATA_CSV}")
	df = _read_csv_df(DATA_CSV)
	print(f"CSV loaded: {len(df)} rows from {DATA_CSV}")


	predictions, labels, splits, all_file_names = collect_predictions(df, device)

	# Model setup.
	model_keys = [r["key"] for r in MODEL_REGISTRY]
	model_labels = {r["key"]: r["label"] for r in MODEL_REGISTRY}

	# Model setup.
	available_keys = []
	for k in model_keys:
		avail_count = sum(1 for fn in all_file_names if predictions[k].get(fn) is not None)
		print(f"  {model_labels[k]}: predictions available for {avail_count}/{len(all_file_names)} samples")
		if avail_count > 0:
			available_keys.append(k)

	if not available_keys:
		print("No models are available.")
		return

	print(f"\nAvailable models: {len(available_keys)} -> {', '.join(available_keys)}")

	# Create required output.
	all_combos: list[tuple[str, ...]] = []
	for r in range(1, len(available_keys) + 1):
		for combo in itertools.combinations(available_keys, r):
			all_combos.append(combo)
	print(f"Total combinations: {len(all_combos)}")

	# Ensemble evaluation.
	rows: list[dict[str, Any]] = []
	no = 0

	# ─── Individual ───
	print(f"\n{'='*60}")
	print("Phase 1: Individual model evaluation")
	print(f"{'='*60}")
	for key in tqdm(available_keys, desc="Individual"):
		combo = (key,)
		combo_label = model_labels[key]
		res = ensemble_soft_voting(list(combo), predictions, labels, splits, all_file_names)
		if res and "metrics" in res:
			no += 1
			rows.append(_flatten_metrics_to_row(no, "Individual", combo_label, "N/A", res["metrics"]))

 # ─── Hard Voting ───
	print(f"\n{'='*60}")
	print("Phase 2: Hard Voting")
	print(f"{'='*60}")
	for combo in tqdm(all_combos, desc="Hard Voting"):
		if len(combo) < 2:
			continue
		combo_label = "+".join(model_labels[k] for k in combo)
		res = ensemble_hard_voting(list(combo), predictions, labels, splits, all_file_names)
		if res and "metrics" in res:
			no += 1
			rows.append(_flatten_metrics_to_row(no, "Hard_Voting", combo_label, "N/A", res["metrics"]))

 # ─── Soft Voting ───
	print(f"\n{'='*60}")
	print("Phase 3: Soft Voting")
	print(f"{'='*60}")
	for combo in tqdm(all_combos, desc="Soft Voting"):
		if len(combo) < 2:
			continue
		combo_label = "+".join(model_labels[k] for k in combo)
		res = ensemble_soft_voting(list(combo), predictions, labels, splits, all_file_names)
		if res and "metrics" in res:
			no += 1
			rows.append(_flatten_metrics_to_row(no, "Soft_Voting", combo_label, res.get("weights", "equal"), res["metrics"]))

 # ─── Weighted Soft Voting ───
	print(f"\n{'='*60}")
	print("Phase 4: Weighted Soft Voting (scipy optimize)")
	print(f"{'='*60}")
	for combo in tqdm(all_combos, desc="Weighted Soft"):
		if len(combo) < 2:
			continue
		combo_label = "+".join(model_labels[k] for k in combo)
		res = ensemble_weighted_soft_voting(list(combo), predictions, labels, splits, all_file_names)
		if res and "metrics" in res:
			no += 1
			rows.append(_flatten_metrics_to_row(no, "Weighted_Soft_Voting", combo_label, res.get("weights", "N/A"), res["metrics"]))

 # ─── Stacking ───
	stacking_learners = ["LR", "RF"]
	if HAS_XGB:
		stacking_learners.append("XGB")
	else:
		print("⚠ XGBoost not installed, skipping Stacking_XGB")

	for learner in stacking_learners:
		print(f"\n{'='*60}")
		print(f"Phase 5: Stacking ({learner})")
		print(f"{'='*60}")
		for combo in tqdm(all_combos, desc=f"Stacking_{learner}"):
			if len(combo) < 2:
				continue
			combo_label = "+".join(model_labels[k] for k in combo)
			res = ensemble_stacking(list(combo), predictions, labels, splits, all_file_names, meta_learner_name=learner)
			if res and "metrics" in res:
				no += 1
				rows.append(_flatten_metrics_to_row(no, f"Stacking_{learner}", combo_label, res.get("weights", "N/A"), res["metrics"]))

 # Create required output.
	if not rows:
		print("No results were produced.")
		return

	result_df = pd.DataFrame(rows)
	# Sort values.
	result_df = result_df.sort_values("Val_Macro_AUC", ascending=False).reset_index(drop=True)
	result_df["No"] = range(1, len(result_df) + 1)

	# Save output.
	ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
	excel_path = ENSEMBLE_DIR / "ensemble_results.xlsx"

	with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
		result_df.to_excel(writer, sheet_name="All_Results", index=False)


		top20 = result_df.head(20).copy()
		summary_cols = [
		 "No", "Ensemble_Method", "Model_Combination", "Model_Weights",
		 "Total_N", "Total_Macro_AUC", "Total_Macro_Recall", "Total_Macro_F1",
		 "Train_N", "Train_Macro_AUC", "Train_Macro_Recall", "Train_Macro_F1",
		 "Val_N", "Val_Macro_AUC", "Val_Macro_Recall", "Val_Macro_F1",
		]
		for c in CLASS_NAME:
			summary_cols.append(f"Val_AUC_{c}")
		existing_cols = [c for c in summary_cols if c in top20.columns]
		top20[existing_cols].to_excel(writer, sheet_name="Top20_Summary", index=False)


		top_class_rows: list[dict[str, Any]] = []
		for dataset_prefix in ("Total", "Train", "Val"):
			for cname in CLASS_NAME:
				for metric in ("AUC", "Recall", "F1"):
					col = f"{dataset_prefix}_{metric}_{cname}"
					if col not in result_df.columns:
						continue
					series = result_df[col].dropna()
					if series.empty:
						continue
					best_idx = series.idxmax()
					best_row = result_df.loc[best_idx]
					macro_auc_col = f"{dataset_prefix}_Macro_AUC"
					top_class_rows.append({
					 "Dataset": dataset_prefix,
					 "Top_Class_Name": cname,
					 "Metric_Type": metric,
					 "Raw_No": int(best_row["No"]),
					 "Score": float(best_row[col]),
					 "Macro_AUC": float(best_row[macro_auc_col]) if macro_auc_col in best_row else float("nan"),
					 "Ensemble_Method": best_row["Ensemble_Method"],
					 "Model_Combination": best_row["Model_Combination"],
					 "Model_Weights": best_row["Model_Weights"],
					})

		if top_class_rows:
			top_class_df = pd.DataFrame(top_class_rows)
			top_class_df.to_excel(writer, sheet_name="Top_Class_Summary", index=False)

	print(f"\n{'='*60}")
	print(f"Excel output saved: {excel_path}")
	print(f"Total scenarios: {len(result_df)}")
	print(f"{'='*60}")

	# Save output.
	save_best_ensemble_pth(result_df, ENSEMBLE_DIR)


	print("\n[Top 10 - Val Macro AUC]")
	top10 = result_df.head(10)
	for _, r in top10.iterrows():
		print(
		 f"  #{int(r['No']):3d} | {r['Ensemble_Method']:25s} | "
		 f"{r['Model_Combination']:60s} | "
		 f"Val AUC={r['Val_Macro_AUC']:.4f} | "
		 f"Val F1={r['Val_Macro_F1']:.4f} | "
		 f"Val Recall={r['Val_Macro_Recall']:.4f}"
		)


if __name__ == "__main__":
	main()
