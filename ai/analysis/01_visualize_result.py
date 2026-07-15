

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_visualize_result.py  –  6-Modal MIL Ensemble Interactive Dashboard
=====================================================================
Loads best_pth ensemble settings and visualizes UMAP (2D/3D) scatter plots
with class-specific decision boundaries. Clicking a point shows the selected
instance's raw data (frames/audio/text) in the right panel.

Run: python ai/analysis/01_visualize_result.py --fold 1
"""
from __future__ import annotations

import argparse
import ast as ast_module
import base64
import csv
import hashlib
import io
import json
import pickle
import random
import re
import warnings
import zipfile
from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
import torch
import torch.nn as nn
import umap
from dash import dcc, html, ctx, ALL
from dash.dependencies import Input, Output, State
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RUNTIME_PARSER = argparse.ArgumentParser(description="Visualize one cross-validation fold.")
RUNTIME_PARSER.add_argument("--fold", type=int, required=True, choices=range(1, 6))
RUNTIME_PARSER.add_argument("--port", type=int, default=8050)
RUNTIME_ARGS, _ = RUNTIME_PARSER.parse_known_args()

# ──────────────────────────────────────────────────────────────
# Runtime paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_BASE = REPO_ROOT / "ai" / "training" / "outputs"
CV_OUTPUT_ROOT = OUTPUT_BASE / "cv"
FOLD_DIR = CV_OUTPUT_ROOT / f"fold_{RUNTIME_ARGS.fold:02d}"
DATA_CSV = FOLD_DIR / "data.csv"
BEST_PTH_DIR = FOLD_DIR / "ensemble" / "best_pth"
MIL_OUTPUT_BASE = FOLD_DIR

CLASS_NUM = 6
CLASS_NAME = [
	"sexual_content", "violence", "fear",
	"inappropriate_language", "drugs", "crime",
]
CLASS_DISPLAY_NAME = {
    "sexual_content": "Sexual Content",
    "violence": "Violence",
    "fear": "Fear",
    "inappropriate_language": "Inappropriate Language",
    "drugs": "Drugs",
    "crime": "Crime"
}
CLASS_COLORS = [
	"#e6194b", "#3cb44b", "#4363d8",
	"#f58231", "#911eb4", "#42d4f4",
]
SEED = 42
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = SCRIPT_DIR / "cache" / f"fold_{RUNTIME_ARGS.fold:02d}"

MODALITY_OPTIONS = [
	{"key": "all", "label": "Selected Ensemble"},
	{"key": "vision", "label": "Vision"},
	{"key": "original_audio", "label": "Original Audio"},
	{"key": "vocal_audio", "label": "Vocal Audio"},
	{"key": "non_vocal_audio", "label": "Non-Vocal Audio"},
	{"key": "ocr", "label": "OCR"},
	{"key": "stt", "label": "STT"},
]

MODEL_REGISTRY = [
	{"key": "vision", "label": "Vision", "ckpt_subdir": "vision_mil",
	 "ckpt_file": "best_vision_mil_model.pth", "activation": "gelu",
	 "dataset_type": "vivit", "default_embed_dim": 1024,
	 "default_attn_dim": 256, "default_dropout": 0.2},
	{"key": "original_audio", "label": "Original Audio",
	 "ckpt_subdir": "original_audio_mil",
	 "ckpt_file": "best_original_audio_mil_model.pth", "activation": "relu",
	 "dataset_type": "ast_original", "default_embed_dim": 1536,
	 "default_attn_dim": 384, "default_dropout": 0.3},
	{"key": "vocal_audio", "label": "Vocal Audio",
	 "ckpt_subdir": "vocal_audio_mil",
	 "ckpt_file": "best_vocal_audio_mil_model.pth", "activation": "relu",
	 "dataset_type": "ast_vocal", "default_embed_dim": 1536,
	 "default_attn_dim": 384, "default_dropout": 0.3},
	{"key": "non_vocal_audio", "label": "Non-Vocal Audio",
	 "ckpt_subdir": "non_vocal_audio_mil",
	 "ckpt_file": "best_non_vocal_audio_mil_model.pth", "activation": "relu",
	 "dataset_type": "ast_non_vocal", "default_embed_dim": 1536,
	 "default_attn_dim": 384, "default_dropout": 0.3},
	{"key": "ocr", "label": "OCR", "ckpt_subdir": "ocr_mil",
	 "ckpt_file": "best_ocr_mil_model.pth", "activation": "relu",
	 "dataset_type": "ocr", "default_embed_dim": 1536,
	 "default_attn_dim": 384, "default_dropout": 0.3},
	{"key": "stt", "label": "STT", "ckpt_subdir": "stt_mil",
	 "ckpt_file": "best_stt_mil_model.pth", "activation": "relu",
	 "dataset_type": "stt", "default_embed_dim": 1536,
	 "default_attn_dim": 384, "default_dropout": 0.3},
]

MODEL_KEY_TO_LABEL = {r["key"]: r["label"] for r in MODEL_REGISTRY}


# ──────────────────────────────────────────────────────────────
# Disk cache helpers
# ──────────────────────────────────────────────────────────────

def _compute_cache_fingerprint() -> str:
	"""Compute fingerprint from DATA_CSV + model checkpoint mtimes."""
	h = hashlib.md5()
	def _add_file(p: Path) -> None:
		if p.exists():
			h.update(str(p).encode())
			h.update(str(p.stat().st_mtime).encode())
	_add_file(DATA_CSV)
	_add_file(BEST_PTH_DIR / "best_val_macro_auc.pth")
	for reg in MODEL_REGISTRY:
		_add_file(MIL_OUTPUT_BASE / reg["ckpt_subdir"] / reg["ckpt_file"])
	return h.hexdigest()[:12]


def _clean_stale_caches(prefix: str, current_fp: str) -> None:
	"""Remove old cache files that don't match the current fingerprint."""
	for f in CACHE_DIR.glob(f"{prefix}_*.pkl"):
		if f.stem != f"{prefix}_{current_fp}":
			try:
				f.unlink()
				print(f"  Removed stale cache: {f.name}")
			except Exception:
				pass


# ──────────────────────────────────────────────────────────────
# Utility helpers (from 08_ensemble.py)
# ──────────────────────────────────────────────────────────────

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
	try:
		return float(1.0 if float(s) > 0.0 else 0.0)
	except Exception:
		return 0.0


def _to_bool(v: Any) -> bool:
	if isinstance(v, bool):
		return v
	s = _safe_str(v).strip().lower()
	return s in ("1", "true", "t", "yes", "y")


def _read_csv_df(path: Path) -> pd.DataFrame:
	for enc in ("utf-8-sig", "utf-8", "cp949"):
		try:
			return pd.read_csv(path, encoding=enc)
		except Exception:
			continue
	return pd.read_csv(path)



def _majority_value(values: list[str]) -> str:
	if not values:
		return "Unknown"
	counts: dict[str, int] = {}
	for value in values:
		counts[value] = counts.get(value, 0) + 1
	return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# ──────────────────────────────────────────────────────────────
# GatedAttentionMIL with embedding extraction
# ──────────────────────────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
	def __init__(self, *, in_dim: int, embed_dim: int, attn_dim: int,
	             num_classes: int = CLASS_NUM, dropout: float = 0.1,
	             activation: str = "relu") -> None:
		super().__init__()
		act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True)
		self.instance_encoder = nn.Sequential(
			nn.Linear(int(in_dim), int(embed_dim)), act,
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
		h = self.instance_encoder(x)
		v = torch.tanh(self.attn_v(h))
		u = torch.sigmoid(self.attn_u(h))
		a_logits = self.attn_w(v * u).squeeze(-1)
		a = self._masked_softmax(a_logits, mask, dim=1)
		z = (a.unsqueeze(-1) * h).sum(dim=1)
		return self.classifier(z)

	def forward_with_embeddings(self, x: torch.Tensor, mask: torch.Tensor):
		"""Returns (logits, instance_emb, attn_weights, bag_emb)."""
		h = self.instance_encoder(x)           # (1, T, D)
		v = torch.tanh(self.attn_v(h))
		u = torch.sigmoid(self.attn_u(h))
		a_logits = self.attn_w(v * u).squeeze(-1)  # (1, T)
		a = self._masked_softmax(a_logits, mask, dim=1)  # (1, T)
		z = (a.unsqueeze(-1) * h).sum(dim=1)   # (1, D)
		logits = self.classifier(z)             # (1, C)
		return logits, h, a, z


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────

def load_model(reg: dict, device: torch.device):
	ckpt_path = MIL_OUTPUT_BASE / reg["ckpt_subdir"] / reg["ckpt_file"]
	if not ckpt_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
	ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
	state_dict = ckpt["model"]
	cfg = ckpt.get("config") or ckpt.get("run_cfg") or {}
	feat_dim = cfg.get("feat_dim")
	if feat_dim is None:
		feat_dim = state_dict["instance_encoder.0.weight"].shape[1]
	model = GatedAttentionMIL(
		in_dim=int(feat_dim),
		embed_dim=int(cfg.get("embed_dim", reg["default_embed_dim"])),
		attn_dim=int(cfg.get("attn_dim", reg["default_attn_dim"])),
		num_classes=CLASS_NUM,
		dropout=float(cfg.get("dropout", reg["default_dropout"])),
		activation=reg["activation"],
	)
	model.load_state_dict(state_dict, strict=True)
	model.to(device).eval()
	return model, cfg


# ──────────────────────────────────────────────────────────────
# Feature loading helpers (from 08_ensemble.py)
# ──────────────────────────────────────────────────────────────

def _read_vivit_manifest_feature_paths(manifest_path: Path) -> list[Path]:
	if not manifest_path.exists():
		return []
	rows: list[dict[str, str]] = []
	for enc in ("utf-8-sig", "utf-8"):
		try:
			with manifest_path.open("r", encoding=enc, newline="") as f:
				rows = list(csv.DictReader(f))
			break
		except UnicodeDecodeError:
			continue
	if not rows:
		return []
	rows = sorted(rows, key=lambda r: int(float(_safe_str(r.get("chunk_idx", "999999")))))
	return [p for r in rows if (p := Path(_safe_str(r.get("feature_path")))).exists()]


def _resolve_feature_path(manifest_path: Path, raw: str) -> Path:
	raw = raw.strip()
	if not raw:
		return Path("")
	p = Path(raw)
	return p if p.is_absolute() else (manifest_path.parent / p).resolve()


def _read_ast_feature_paths(manifest_path: Path, audio_type: str) -> list[Path]:
	if not manifest_path.exists():
		return []
	rows: list[dict[str, str]] = []
	for enc in ("utf-8-sig", "utf-8"):
		try:
			with manifest_path.open("r", encoding=enc, newline="") as f:
				rows = [{k: _safe_str(v) for k, v in r.items()} for r in csv.DictReader(f)]
			break
		except UnicodeDecodeError:
			continue
	result: list[tuple[int, Path]] = []
	for r in rows:
		if r.get("audio_type", "") != audio_type:
			continue
		p = _resolve_feature_path(manifest_path, r.get("feature_path", ""))
		if not p or not p.exists():
			continue
		try:
			idx = int(float(r.get("chunk_idx", "")))
		except (ValueError, TypeError):
			idx = 999999
		result.append((idx, p))
	result.sort(key=lambda t: t[0])
	return [p for _, p in result]


def _load_1d_npy_stack(paths: list[Path]) -> np.ndarray:
	feats = []
	for p in paths:
		arr = np.load(str(p), allow_pickle=False)
		arr = np.asarray(arr)
		if arr.ndim == 2 and arr.shape[0] == 1:
			arr = arr[0]
		feats.append(arr.astype(np.float32, copy=False))
	return np.stack(feats, axis=0)


def _load_2d_emb_and_mask(emb_path: Path, mask_path: Path):
	emb = np.load(str(emb_path), allow_pickle=False).astype(np.float32, copy=False)
	mask = np.load(str(mask_path), allow_pickle=False).astype(np.float32, copy=False)
	return emb, (mask > 0).astype(np.float32)


def _make_final_mask(pad_mask: np.ndarray, presence_mask: np.ndarray) -> np.ndarray:
	final = pad_mask * presence_mask
	if float(final.sum()) <= 0.0:
		final = pad_mask.copy()
	return final.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Data preparation – embeddings + predictions
# ──────────────────────────────────────────────────────────────

def _load_features_for_sample(row: pd.Series, modality_key: str):
	"""Returns raw features (np.ndarray) for a single video+modality, or None."""
	dtype_map = {
		"vision": "vivit", "original_audio": "ast_original",
		"vocal_audio": "ast_vocal", "non_vocal_audio": "ast_non_vocal",
		"ocr": "ocr", "stt": "stt",
	}
	dtype = dtype_map[modality_key]

	if dtype == "vivit":
		if not _to_bool(row.get("vivit_complete")):
			return None
		paths = _read_vivit_manifest_feature_paths(Path(_safe_str(row.get("vivit_manifest"))))
		return _load_1d_npy_stack(paths) if paths else None

	elif dtype.startswith("ast_"):
		atype = {"ast_original": "original", "ast_vocal": "vocal", "ast_non_vocal": "non-vocal"}[dtype]
		paths = _read_ast_feature_paths(Path(_safe_str(row.get("ast_manifest"))), atype)
		if not paths or not all(p.exists() for p in paths):
			return None
		return _load_1d_npy_stack(paths)

	else:  # ocr or stt
		complete_col = f"{dtype}_complete"
		if not _to_bool(row.get(complete_col)):
			return None
		emb_path = Path(_safe_str(row.get(f"{dtype}_emb")))
		mask_path = Path(_safe_str(row.get(f"{dtype}_mask")))
		if not emb_path.exists() or not mask_path.exists():
			return None
		emb, presence = _load_2d_emb_and_mask(emb_path, mask_path)
		if emb.shape[0] == 0:
			return None
		return (emb, presence)


@torch.inference_mode()
def _extract_embeddings(model: nn.Module, raw_feats, modality_key: str, device: torch.device):
	"""Run model forward_with_embeddings. Returns dict with instance_emb, attn, bag_emb, probs."""
	if modality_key in ("ocr", "stt"):
		emb, presence = raw_feats
		n = emb.shape[0]
		pad_mask = np.ones(n, dtype=np.float32)
		crop_mask = _make_final_mask(pad_mask, presence)
		x = torch.from_numpy(emb).unsqueeze(0).to(device, dtype=torch.float32)
		m = torch.from_numpy(crop_mask).unsqueeze(0).to(device, dtype=torch.float32)
	else:
		feats = raw_feats
		n = feats.shape[0]
		mask_np = np.ones(n, dtype=np.float32)
		x = torch.from_numpy(feats.astype(np.float32, copy=False)).unsqueeze(0).to(device, dtype=torch.float32)
		m = torch.from_numpy(mask_np).unsqueeze(0).to(device, dtype=torch.float32)

	with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
		logits, h, a, z = model.forward_with_embeddings(x, m)
		instance_logits = model.classifier(h)

	probs = torch.sigmoid(logits.float().squeeze(0)).cpu().numpy()
	instance_probs = torch.sigmoid(instance_logits.float().squeeze(0)).cpu().numpy()
	instance_emb = h.squeeze(0).float().cpu().numpy()   # (T, D)
	attn_weights = a.squeeze(0).float().cpu().numpy()    # (T,)
	bag_emb = z.squeeze(0).float().cpu().numpy()         # (D,)
	return {
		"instance_emb": instance_emb, "instance_probs": instance_probs,
		"attn_weights": attn_weights, "bag_emb": bag_emb,
		"probs": probs, "n_instances": n,
	}


# ──────────────────────────────────────────────────────────────
# Data loading orchestrator
# ──────────────────────────────────────────────────────────────

def prepare_all_data():
	"""Load CSV, models, extract embeddings, load ensemble pth configs."""
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	fp = _compute_cache_fingerprint()
	embed_cache_path = CACHE_DIR / f"embeddings_{fp}.pkl"
	if embed_cache_path.exists():
		print(f"Loading embedding cache: {embed_cache_path.name}")
		with open(embed_cache_path, "rb") as _cf:
			return pickle.load(_cf)
	print("=" * 60)
	print("Loading data and models...")
	print("=" * 60)

	# 1. CSV
	df = _read_csv_df(DATA_CSV)
	print(f"CSV loaded: {len(df)} rows")

	# 2. Metadata
	file_names: list[str] = []
	labels: dict[str, np.ndarray] = {}
	splits: dict[str, str] = {}
	fn_to_idx: dict[str, int] = {}
	for i, row in df.iterrows():
		fn = Path(_safe_str(row.get("file_name"))).name
		if not fn:
			continue
		file_names.append(fn)
		labels[fn] = np.array([_to_float01(row.get(c)) for c in CLASS_NAME], dtype=np.float32)
		splits[fn] = str(row.get("split", "")).strip()
		fn_to_idx[fn] = i

	# 3. Load 6 MIL models & extract embeddings
	cache: dict[str, dict[str, Any]] = {}  # cache[modality_key][file_name]
	models_loaded: dict[str, nn.Module] = {}

	for reg in MODEL_REGISTRY:
		key = reg["key"]
		cache[key] = {}
		try:
			model, _ = load_model(reg, DEVICE)
			models_loaded[key] = model
		except FileNotFoundError as e:
			print(f"  ⚠ {reg['label']}: {e}")
			continue

		done, skipped = 0, 0
		for fn in tqdm(file_names, desc=f"  {reg['label']}", leave=True):
			idx = fn_to_idx.get(fn)
			if idx is None:
				continue
			row = df.iloc[idx]
			raw = _load_features_for_sample(row, key)
			if raw is None:
				skipped += 1
				continue
			emb_data = _extract_embeddings(model, raw, key, DEVICE)
			cache[key][fn] = emb_data
			done += 1
		print(f"    Done: {done}, skipped: {skipped}")
		del model
		models_loaded.pop(key, None)
		torch.cuda.empty_cache()

	# 4. Load best_pth ensemble configs
	ensemble_configs: dict[str, dict] = {}
	if BEST_PTH_DIR.exists():
		for pth_file in sorted(BEST_PTH_DIR.glob("*.pth")):
			data = torch.load(str(pth_file), map_location="cpu", weights_only=False)
			ensemble_configs[pth_file.stem] = data
	print(f"Loaded ensemble configs: {len(ensemble_configs)}")

	result = (df, file_names, labels, splits, fn_to_idx, cache, ensemble_configs)
	_clean_stale_caches("embeddings", fp)
	print(f"Saving embedding cache: {embed_cache_path.name}")
	with open(embed_cache_path, "wb") as _cf:
		pickle.dump(result, _cf)
	return result


# ──────────────────────────────────────────────────────────────
# Ensemble prediction computation
# ──────────────────────────────────────────────────────────────

def _parse_model_combination(combo_str: str) -> list[str]:
	"""Parse 'Vision+Original Audio+STT' → ['vision', 'original_audio', 'stt']."""
	label_to_key = {r["label"]: r["key"] for r in MODEL_REGISTRY}
	parts = [p.strip() for p in combo_str.split("+")]
	return [label_to_key[p] for p in parts if p in label_to_key]


def _parse_model_weights(weights_raw, model_keys: list[str]) -> Optional[dict[str, float]]:
	"""Parse weight dict (string or dict) → {key: float} or None for equal weights.
	Returns None for stacking (complex per-class weights) → falls back to soft voting."""
	if not weights_raw:
		return None
	if isinstance(weights_raw, str):
		if weights_raw.strip() in ("N/A", "equal", ""):
			return None
		try:
			weights_raw = ast_module.literal_eval(weights_raw)
		except Exception:
			return None
	if isinstance(weights_raw, dict):
		try:
			result = {k: float(v) for k, v in weights_raw.items() if k in model_keys and isinstance(v, (int, float))}
			return result if result else None
		except (TypeError, ValueError):
			return None
	return None


def compute_ensemble_predictions(
	config: dict, cache: dict, file_names: list[str],
) -> dict[str, Optional[np.ndarray]]:
	"""Compute per-video ensemble probabilities from a best_pth config."""
	combo = config.get("model_combination", "")
	method = config.get("ensemble_method", "")
	weights_str = config.get("model_weights", "")
	model_keys = _parse_model_combination(combo)
	if not model_keys:
		return {}

	weight_map = _parse_model_weights(weights_str, model_keys)
	preds: dict[str, Optional[np.ndarray]] = {}

	for fn in file_names:
		avail_probs = []
		avail_keys = []
		for k in model_keys:
			emb_data = cache.get(k, {}).get(fn)
			if emb_data is not None:
				avail_probs.append(emb_data["probs"])
				avail_keys.append(k)

		if not avail_probs:
			preds[fn] = None
			continue

		probs_arr = np.stack(avail_probs, axis=0)  # (M, C)

		if "Hard" in method:
			votes = (probs_arr >= THRESHOLD).astype(np.float32).mean(axis=0)
			preds[fn] = votes
		elif weight_map and "Weighted" in method:
			w = np.array([weight_map.get(k, 1.0) for k in avail_keys], dtype=np.float64)
			w = w / (w.sum() + 1e-12)
			preds[fn] = (w[:, None] * probs_arr).sum(axis=0).astype(np.float32)
		else:  # Soft voting / Individual / Stacking fallback
			preds[fn] = probs_arr.mean(axis=0).astype(np.float32)

	return preds


def compute_ensemble_instance_predictions(
	config: dict, cache: dict, file_names: list[str],
) -> dict[str, dict[int, np.ndarray]]:
	"""Compute chunk-aligned ensemble probabilities for instances."""
	combo = config.get("model_combination", "")
	method = config.get("ensemble_method", "")
	weights_str = config.get("model_weights", "")
	model_keys = _parse_model_combination(combo)
	if not model_keys:
		return {}

	weight_map = _parse_model_weights(weights_str, model_keys)
	result: dict[str, dict[int, np.ndarray]] = {}
	for fn in file_names:
		chunk_probs: dict[int, list[np.ndarray]] = {}
		chunk_keys: dict[int, list[str]] = {}
		for k in model_keys:
			emb_data = cache.get(k, {}).get(fn)
			if emb_data is None or "instance_probs" not in emb_data:
				continue
			inst_probs = emb_data["instance_probs"]
			for ci in range(int(emb_data.get("n_instances", len(inst_probs)))):
				if ci >= len(inst_probs):
					continue
				chunk_probs.setdefault(ci, []).append(inst_probs[ci])
				chunk_keys.setdefault(ci, []).append(k)

		fn_result: dict[int, np.ndarray] = {}
		for ci, probs_list in chunk_probs.items():
			probs_arr = np.stack(probs_list, axis=0)
			avail_keys = chunk_keys[ci]
			if "Hard" in method:
				fn_result[ci] = (probs_arr >= THRESHOLD).astype(np.float32).mean(axis=0)
			elif weight_map and "Weighted" in method:
				w = np.array([weight_map.get(k, 1.0) for k in avail_keys], dtype=np.float64)
				w = w / (w.sum() + 1e-12)
				fn_result[ci] = (w[:, None] * probs_arr).sum(axis=0).astype(np.float32)
			else:
				fn_result[ci] = probs_arr.mean(axis=0).astype(np.float32)
		if fn_result:
			result[fn] = fn_result
	return result


def _predicted_label_names(pred_vec: Optional[np.ndarray]) -> list[str]:
	if pred_vec is None or np.isnan(pred_vec).any():
		return []
	return [CLASS_NAME[i] for i in range(CLASS_NUM) if float(pred_vec[i]) >= THRESHOLD]


def _correctness_from_gt(pred_vec: Optional[np.ndarray], gt_vec: np.ndarray) -> tuple[str, list[str]]:
	pred_names = _predicted_label_names(pred_vec)
	if not pred_names:
		return "Neutral", []
	gt_names = {CLASS_NAME[i] for i in range(CLASS_NUM) if float(gt_vec[i]) > 0.0}
	false_positive = [name for name in pred_names if name not in gt_names]
	return ("Incorrect" if false_positive else "Correct"), false_positive


# ──────────────────────────────────────────────────────────────
# UMAP precomputation
# ──────────────────────────────────────────────────────────────

def precompute_umaps(cache: dict, file_names: list[str], ensemble_configs: dict,
                     labels: dict):
	"""Precompute UMAP projections for all modality × view × dim combos."""
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	fp = _compute_cache_fingerprint()
	umap_cache_path = CACHE_DIR / f"umap_{fp}.pkl"
	if umap_cache_path.exists():
		print(f"Loading UMAP cache: {umap_cache_path.name}")
		with open(umap_cache_path, "rb") as _cf:
			return pickle.load(_cf)
	print("\nPrecomputing UMAP projections...")
	umap_cache: dict[str, dict] = {}

	def _fit_umap(embeddings: np.ndarray, n_dim: int) -> np.ndarray:
		if embeddings.shape[0] < 5:
			return np.zeros((embeddings.shape[0], n_dim), dtype=np.float32)
		n_neighbors = min(15, embeddings.shape[0] - 1)
		reducer = umap.UMAP(n_components=n_dim, random_state=SEED, n_neighbors=n_neighbors, min_dist=0.1)
		return reducer.fit_transform(embeddings).astype(np.float32)

	# Per-modality Bag and Instance UMAPs
	for reg in MODEL_REGISTRY:
		key = reg["key"]
		# Collect bag and instance embeddings
		bag_embs, bag_fns = [], []
		inst_embs, inst_meta = [], []  # meta: (file_name, chunk_idx)

		for fn in file_names:
			data = cache.get(key, {}).get(fn)
			if data is None:
				continue
			bag_embs.append(data["bag_emb"])
			bag_fns.append(fn)
			for ci in range(data["n_instances"]):
				inst_embs.append(data["instance_emb"][ci])
				inst_meta.append((fn, ci))

		modality_data: dict[str, Any] = {"bag_fns": bag_fns, "inst_meta": inst_meta}

		if bag_embs:
			bag_arr = np.stack(bag_embs, axis=0)
			modality_data["bag_2d"] = _fit_umap(bag_arr, 2)
			modality_data["bag_3d"] = _fit_umap(bag_arr, 3)
		else:
			modality_data["bag_2d"] = np.zeros((0, 2), dtype=np.float32)
			modality_data["bag_3d"] = np.zeros((0, 3), dtype=np.float32)

		if inst_embs:
			modality_data["inst_meta"] = inst_meta
			inst_arr = np.stack(inst_embs, axis=0)
			modality_data["inst_2d"] = _fit_umap(inst_arr, 2)
			modality_data["inst_3d"] = _fit_umap(inst_arr, 3)
		else:
			modality_data["inst_2d"] = np.zeros((0, 2), dtype=np.float32)
			modality_data["inst_3d"] = np.zeros((0, 3), dtype=np.float32)

		umap_cache[key] = modality_data
		print(f"  {reg['label']}: Bag={len(bag_fns)}, Instance={len(inst_meta)}")

	# "all" modality: Bag and Instance view. Embeddings are concatenated across
	# modalities; missing modalities are zero-padded and excluded from ensemble scores.
	modality_dims = {}  # key -> embed_dim
	for reg in MODEL_REGISTRY:
		for fn in file_names:
			data = cache.get(reg["key"], {}).get(fn)
			if data is not None:
				modality_dims[reg["key"]] = data["bag_emb"].shape[0]
				break

	all_bag_embs, all_bag_fns, all_bag_padded = [], [], []
	for fn in file_names:
		parts = []
		has_any = False
		is_padded = False
		for reg in MODEL_REGISTRY:
			data = cache.get(reg["key"], {}).get(fn)
			dim = modality_dims.get(reg["key"], 0)
			if data is not None:
				parts.append(data["bag_emb"])
				has_any = True
			elif dim > 0:
				parts.append(np.zeros(dim, dtype=np.float32))
				is_padded = True
		if has_any and parts:
			all_bag_embs.append(np.concatenate(parts))
			all_bag_fns.append(fn)
			all_bag_padded.append(is_padded)

	all_inst_embs, all_inst_meta, all_inst_padded = [], [], []
	for fn in file_names:
		chunk_ids: set[int] = set()
		for reg in MODEL_REGISTRY:
			data = cache.get(reg["key"], {}).get(fn)
			if data is None:
				continue
			chunk_ids.update(range(int(data.get("n_instances", 0))))
		for ci in sorted(chunk_ids):
			parts = []
			has_any = False
			is_padded = False
			for reg in MODEL_REGISTRY:
				key = reg["key"]
				data = cache.get(key, {}).get(fn)
				dim = modality_dims.get(key, 0)
				if data is not None and ci < int(data.get("n_instances", 0)):
					parts.append(data["instance_emb"][ci])
					has_any = True
				elif dim > 0:
					parts.append(np.zeros(dim, dtype=np.float32))
					is_padded = True
			if has_any and parts:
				all_inst_embs.append(np.concatenate(parts))
				all_inst_meta.append((fn, ci))
				all_inst_padded.append(is_padded)

	all_data: dict[str, Any] = {
		"bag_fns": all_bag_fns,
		"inst_meta": all_inst_meta,
		"bag_padded": all_bag_padded,
		"inst_padded": all_inst_padded,
	}
	if all_bag_embs:
		all_arr = np.stack(all_bag_embs, axis=0)
		all_data["bag_2d"] = _fit_umap(all_arr, 2)
		all_data["bag_3d"] = _fit_umap(all_arr, 3)
	else:
		all_data["bag_2d"] = np.zeros((0, 2), dtype=np.float32)
		all_data["bag_3d"] = np.zeros((0, 3), dtype=np.float32)

	if all_inst_embs:
		all_inst_arr = np.stack(all_inst_embs, axis=0)
		all_data["inst_2d"] = _fit_umap(all_inst_arr, 2)
		all_data["inst_3d"] = _fit_umap(all_inst_arr, 3)
	else:
		all_data["inst_2d"] = np.zeros((0, 2), dtype=np.float32)
		all_data["inst_3d"] = np.zeros((0, 3), dtype=np.float32)

	umap_cache["all"] = all_data
	n_padded = sum(all_bag_padded)
	n_inst_padded = sum(all_inst_padded)
	print(
		f"  All: Bag={len(all_bag_fns)} (complete={len(all_bag_fns)-n_padded}, padded={n_padded}), "
		f"Instance={len(all_inst_meta)} (complete={len(all_inst_meta)-n_inst_padded}, padded={n_inst_padded})"
	)

	print("UMAP precomputation complete.")
	_clean_stale_caches("umap", fp)
	print(f"Saving UMAP cache: {umap_cache_path.name}")
	with open(umap_cache_path, "wb") as _cf:
		pickle.dump(umap_cache, _cf)
	return umap_cache


# ──────────────────────────────────────────────────────────────
# Detail panel helpers
# ──────────────────────────────────────────────────────────────

def _load_frames_b64(frames_dir: str, chunk_idx: int, max_frames: int = 32) -> list[dict]:
	"""Load 32 frames for a chunk as base64 strings."""
	d = Path(frames_dir)
	if not d.exists():
		return []
	# Files are named {n}.jpg (1-indexed, no zero-padding)
	all_files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".png")]
	all_files.sort(key=lambda f: int(re.sub(r"\D", "", f.stem) or "0"))
	start = chunk_idx * 32
	end = start + max_frames
	selected = all_files[start:end]
	result = []
	for f in selected:
		try:
			data = f.read_bytes()
			ext = f.suffix.lower().lstrip(".")
			mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
			result.append({
				"frame_no": int(f.stem),
				"src": f"data:{mime};base64,{base64.b64encode(data).decode()}"
			})
		except Exception:
			continue
	return result


def _load_audio_b64(wav_path: str, chunk_idx: int, sr: int = 44100) -> Optional[str]:
	"""Extract 8-second audio segment and return as base64 WAV."""
	p = Path(wav_path)
	if not p.exists():
		return None
	try:
		samples_per_chunk = sr * 8
		start = chunk_idx * samples_per_chunk
		wav, file_sr = sf.read(str(p), dtype="float32", always_2d=False,
		                       start=start, stop=start + samples_per_chunk)
		if wav.ndim == 2:
			wav = wav.mean(axis=1)
		buf = io.BytesIO()
		sf.write(buf, wav, file_sr, format="WAV")
		buf.seek(0)
		return f"data:audio/wav;base64,{base64.b64encode(buf.read()).decode()}"
	except Exception:
		return None


def _load_text_for_chunk(text_json_path: str, chunk_idx: int) -> str:
	"""Load text for a specific chunk from stt_8s_text.json or ocr_8s_text.json."""
	p = Path(text_json_path)
	if not p.exists():
		return "(Text file not found)"
	try:
		with open(p, "r", encoding="utf-8") as f:
			data = json.load(f)
		texts = data.get("texts", [])
		if 0 <= chunk_idx < len(texts):
			t = texts[chunk_idx]
			return t if t and t.strip() else "(Empty text)"
		return f"(Chunk index {chunk_idx} out of range; total={len(texts)})"
	except Exception as e:
		return f"(Load error: {e})"


def build_detail_panel(fn: str, chunk_idx: Optional[int], df: pd.DataFrame,
                       fn_to_idx: dict, cache: dict, show_modality: str,
                       ensemble_preds: dict[str, Optional[np.ndarray]],
                       ensemble_instance_preds: dict[str, dict[int, np.ndarray]]) -> list:
	"""Build right-side detail panel content."""
	idx = fn_to_idx.get(fn)
	if idx is None:
		return [html.P("Data not found.")]
	row = df.iloc[idx]

	parts: list = []
	video_id = _safe_str(row.get("video_id", fn))
	parts.append(html.H5(f"{video_id}", className="mb-2"))

	store_data = {
		"video_id": video_id,
		"chunk_idx": chunk_idx,
		"frames_dir": "",
		"wav_paths": {}
	}

	# Prediction summary for the clicked bag/instance.
	gt_vec = np.array([_to_float01(row.get(c)) for c in CLASS_NAME], dtype=np.float32)
	pred_vec = _point_prediction(
		show_modality, "instance" if chunk_idx is not None else "bag", fn,
		int(chunk_idx) if chunk_idx is not None else -1,
		cache, ensemble_preds, ensemble_instance_preds,
	)
	gt_names = [CLASS_NAME[i] for i in range(CLASS_NUM) if gt_vec[i] > 0]
	pred_names = _predicted_label_names(pred_vec)
	correctness, false_positive = _correctness_from_gt(pred_vec, gt_vec)
	source = "Selected Ensemble" if show_modality == "all" else MODEL_KEY_TO_LABEL.get(show_modality, show_modality)
	parts.append(dbc.Card(dbc.CardBody([
		html.H6("Prediction Detail"),
		html.P(f"Bag GT: {', '.join(gt_names) or 'None'}"),
		html.P(f"Instance Pred: {', '.join(pred_names) or 'None'}"),
		html.P(f"Correctness: {correctness}"),
		html.P(f"False-positive labels: {', '.join(false_positive) or 'None'}"),
		html.P(f"Chunk: {chunk_idx if chunk_idx is not None else 'Bag'}"),
		html.P(f"Source: {source}"),
		html.P(f"Split: {row.get('split', '?')}"),
	]), className="mb-2"))

	if chunk_idx is not None:
		parts.append(html.P(f"Time: {chunk_idx*8}s to {(chunk_idx+1)*8}s"))

	modalities_to_show = (
		[k for k in ["vision", "original_audio", "vocal_audio", "non_vocal_audio", "ocr", "stt"]]
		if show_modality == "all" else [show_modality]
	)

	for mod_key in modalities_to_show:
		if chunk_idx is None:
			# Bag view: show prediction summary
			emb_data = cache.get(mod_key, {}).get(fn)
			if emb_data is None:
				parts.append(dbc.Card(dbc.CardBody([
					html.H6(f"{MODEL_KEY_TO_LABEL.get(mod_key, mod_key)}"),
					html.P("No data", className="text-muted"),
				]), className="mb-2"))
				continue
			probs = emb_data["probs"]
			prob_strs = [f"{CLASS_NAME[i]}: {probs[i]:.3f}" for i in range(CLASS_NUM)]
			parts.append(dbc.Card(dbc.CardBody([
				html.H6(f"{MODEL_KEY_TO_LABEL.get(mod_key, mod_key)}"),
				html.P(f"Instances: {emb_data['n_instances']}"),
				html.P(" | ".join(prob_strs), style={"fontSize": "0.85em"}),
			]), className="mb-2"))
			continue

		# Instance view: show actual data
		mod_label = MODEL_KEY_TO_LABEL.get(mod_key, mod_key)

		if mod_key == "vision":
			frames_dir = _safe_str(row.get("frames_dir", ""))
			store_data["frames_dir"] = frames_dir
			images = _load_frames_b64(frames_dir, chunk_idx)
			# Save frames into detail-store so that modal can use them
			store_data["frames"] = images
			if images:
				img_grid = []
				for i in range(0, len(images), 8):
					row_imgs = [
						html.Img(
							src=img_info["src"],
							id={"type": "vision-frame-img", "index": img_info["frame_no"]},
							style={"width": "12%", "margin": "1px", "cursor": "pointer"},
							n_clicks=0,
						)
						for img_info in images[i:i+8]
					]
					img_grid.append(html.Div(row_imgs, style={"display": "flex"}))
				parts.append(dbc.Card(dbc.CardBody([
					html.Div([
						html.H6(f"{mod_label} (frames {chunk_idx*32+1} to {(chunk_idx+1)*32})", className="d-inline-block me-3"),
						dbc.Button("Download ZIP", id="btn-dl-img", size="sm", color="primary", className="mb-2 d-inline-block")
					]),
					html.Div(img_grid),
				]), className="mb-2"))
			else:
				parts.append(dbc.Card(dbc.CardBody([
					html.H6(f"{mod_label}"), html.P("No frames"),
				]), className="mb-2"))

		elif mod_key in ("original_audio", "vocal_audio", "non_vocal_audio"):
			wav_col_map = {
				"original_audio": "original_wav",
				"vocal_audio": "vocal_wav",
				"non_vocal_audio": "non_vocal_wav",
			}
			# Try column name, else construct from audio_dir
			wav_col = wav_col_map[mod_key]
			wav_path = _safe_str(row.get(wav_col, ""))
			if not wav_path or not Path(wav_path).exists():
				audio_dir = _safe_str(row.get("audio_dir", ""))
				name_map = {
					"original_audio": "original.wav",
					"vocal_audio": "vocal.wav",
					"non_vocal_audio": "non-vocal.wav",
				}
				wav_path = str(Path(audio_dir) / name_map[mod_key])
			store_data["wav_paths"][mod_key] = wav_path
			audio_b64 = _load_audio_b64(wav_path, chunk_idx)
			if audio_b64:
				parts.append(dbc.Card(dbc.CardBody([
					html.Div([
						html.H6(f"{mod_label} ({chunk_idx*8}s to {(chunk_idx+1)*8}s)", className="d-inline-block me-3"),
						dbc.Button("Download WAV", id={"type": "btn-dl-audio", "index": mod_key}, size="sm", color="primary", className="mb-2 d-inline-block")
					]),
					html.Audio(src=audio_b64, controls=True, style={"width": "100%"}),
				]), className="mb-2"))
			else:
				parts.append(dbc.Card(dbc.CardBody([
					html.H6(f"{mod_label}"),
					html.P("Audio file not found"),
				]), className="mb-2"))

		elif mod_key in ("stt", "ocr"):
			text_col = f"{mod_key}_text"
			text_path = _safe_str(row.get(text_col, ""))
			text_content = _load_text_for_chunk(text_path, chunk_idx)
			parts.append(dbc.Card(dbc.CardBody([
				html.H6(f"{mod_label} (#{chunk_idx})"),
				html.Pre(text_content, style={"whiteSpace": "pre-wrap", "maxHeight": "200px",
				                               "overflow": "auto", "fontSize": "0.9em"}),
			]), className="mb-2"))

	parts.append(dcc.Store(id="detail-store", data=store_data))
	return parts


# ──────────────────────────────────────────────────────────────
# Build UMAP figure
# ──────────────────────────────────────────────────────────────

def _point_prediction(
	modality: str,
	view: str,	fn: str,
	chunk_idx: int,
	cache: dict,
	ensemble_preds: dict[str, Optional[np.ndarray]],
	ensemble_instance_preds: dict[str, dict[int, np.ndarray]],
) -> Optional[np.ndarray]:
	if modality == "all":
		if view == "instance" and chunk_idx >= 0:
			return ensemble_instance_preds.get(fn, {}).get(chunk_idx)
		return ensemble_preds.get(fn)

	emb_data = cache.get(modality, {}).get(fn)
	if emb_data is None:
		return None
	if view == "instance" and chunk_idx >= 0:
		inst_probs = emb_data.get("instance_probs")
		if inst_probs is not None and chunk_idx < len(inst_probs):
			return inst_probs[chunk_idx]
		return None
	return emb_data.get("probs")


def build_umap_figure(
	modality: str, view: str, dim: str, dataset_filter: str,
	display_filter: str, class_checks: list[str],
	ensemble_preds: dict[str, Optional[np.ndarray]],
	ensemble_instance_preds: dict[str, dict[int, np.ndarray]],
	umap_cache: dict, labels: dict, splits: dict, cache: dict,
	show_boundary: bool = True, use_clustering: bool = True, n_clusters: int = 50,
) -> go.Figure:
	"""Build a UMAP plot using instance/bag predictions and subset correctness."""
	is_3d = (dim == "3d")
	is_instance = (view == "instance")

	mod_data = umap_cache.get(modality, {})
	if is_instance:
		coords_key = "inst_3d" if is_3d else "inst_2d"
		meta_key = "inst_meta"
	else:
		coords_key = "bag_3d" if is_3d else "bag_2d"
		meta_key = "bag_fns"

	coords = mod_data.get(coords_key, np.zeros((0, 3 if is_3d else 2)))
	meta = mod_data.get(meta_key, [])
	if coords.shape[0] == 0 or len(meta) == 0:
		fig = go.Figure()
		fig.update_layout(title="No data", template="plotly_white")
		return fig

	n = coords.shape[0]
	point_fns: list[str] = []
	point_chunks: list[int] = []
	point_labels = np.zeros((n, CLASS_NUM), dtype=np.float32)
	point_splits: list[str] = []
	point_preds = np.full((n, CLASS_NUM), np.nan, dtype=np.float32)
	point_correctness: list[str] = []
	point_false_positive: list[list[str]] = []

	for i in range(n):
		if is_instance:
			fn, ci = meta[i]
			ci = int(ci)
		else:
			fn = meta[i]
			ci = -1
		point_fns.append(fn)
		point_chunks.append(ci)
		gt_vec = labels.get(fn, np.zeros(CLASS_NUM, dtype=np.float32))
		point_labels[i] = gt_vec
		point_splits.append(splits.get(fn, ""))
		pred = _point_prediction(modality, view, fn, ci, cache, ensemble_preds, ensemble_instance_preds)
		if pred is not None:
			point_preds[i] = pred
		status, fp = _correctness_from_gt(pred, gt_vec)
		point_correctness.append(status)
		point_false_positive.append(fp)

	padded_key = "inst_padded" if is_instance else "bag_padded"
	padded = mod_data.get(padded_key, [])
	base_mask = np.ones(n, dtype=bool)
	for i in range(n):
		if padded and i < len(padded) and padded[i]:
			# Keep padded all-modality points visible, but do not use them for boundaries.
			pass
		if np.isnan(point_preds[i]).any():
			base_mask[i] = False

	db_mask = base_mask.copy()
	if padded:
		for i in range(n):
			if i < len(padded) and padded[i]:
				db_mask[i] = False
	db_coords = coords[db_mask]
	db_preds = point_preds[db_mask]

	axis_source = db_coords if len(db_coords) > 0 else coords
	axis_margin = 1.0
	axis_ranges: list[list[float]] = []
	for axis_idx in range(3 if is_3d else 2):
		axis_vals = axis_source[:, axis_idx]
		axis_min = float(np.nanmin(axis_vals))
		axis_max = float(np.nanmax(axis_vals))
		if axis_min == axis_max:
			axis_min -= axis_margin
			axis_max += axis_margin
		else:
			axis_min -= axis_margin
			axis_max += axis_margin
		axis_ranges.append([axis_min, axis_max])
	ui_revision_key = f"{modality}:{view}:{dim}"

	mask = base_mask.copy()
	if dataset_filter == "train":
		mask &= np.array([s == "train" for s in point_splits])
	elif dataset_filter == "val":
		mask &= np.array([s != "train" for s in point_splits])

	if display_filter in ("correct", "incorrect"):
		wanted = "Correct" if display_filter == "correct" else "Incorrect"
		mask &= np.array([status == wanted for status in point_correctness])

	selected_classes = set(class_checks or CLASS_NAME)
	for i in range(n):
		if not mask[i]:
			continue
		pred_names = set(_predicted_label_names(point_preds[i]))
		if not (pred_names & selected_classes):
			mask[i] = False

	idx_mask = np.where(mask)[0]
	if len(idx_mask) == 0:
		fig = go.Figure()
		fig.update_layout(title="No data matches the selected filters", template="plotly_white")
		return fig

	coords_f = coords[idx_mask]
	preds_f = point_preds[idx_mask]
	labels_f = point_labels[idx_mask]
	fns_f = [point_fns[i] for i in idx_mask]
	chunks_f = [point_chunks[i] for i in idx_mask]
	status_f = [point_correctness[i] for i in idx_mask]
	fp_f = [point_false_positive[i] for i in idx_mask]

	pred_label_color_map = {CLASS_NAME[i]: CLASS_COLORS[i] for i in range(CLASS_NUM)}
	fig = go.Figure()

	boundary_classes = class_checks or CLASS_NAME
	if show_boundary and len(db_coords) >= 5 and boundary_classes:
		for ci, cname in enumerate(CLASS_NAME):
			if cname not in boundary_classes:
				continue
			binary_labels = (db_preds[:, ci] >= THRESHOLD).astype(int)
			if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
				continue
			color = CLASS_COLORS[ci]
			r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
			if is_3d:
				knn3d = KNeighborsClassifier(n_neighbors=min(15, len(db_coords) - 1))
				knn3d.fit(db_coords[:, :3], binary_labels)
				margin = 1.0
				res = 20
				xs = np.linspace(db_coords[:, 0].min() - margin, db_coords[:, 0].max() + margin, res)
				ys = np.linspace(db_coords[:, 1].min() - margin, db_coords[:, 1].max() + margin, res)
				zs = np.linspace(db_coords[:, 2].min() - margin, db_coords[:, 2].max() + margin, res)
				xx3, yy3, zz3 = np.meshgrid(xs, ys, zs)
				grid3d = np.c_[xx3.ravel(), yy3.ravel(), zz3.ravel()]
				z3 = knn3d.predict_proba(grid3d)
				z3_vol = z3[:, 1] if z3.shape[1] == 2 else z3[:, 0]
				fig.add_trace(go.Isosurface(
					x=xx3.ravel(), y=yy3.ravel(), z=zz3.ravel(), value=z3_vol,
					isomin=0.5, isomax=1.0, surface_count=1, opacity=0.2,
					colorscale=[[0, f"rgba({r},{g},{b},0.1)"], [1, f"rgba({r},{g},{b},0.3)"]],
					showscale=False, caps=dict(x_show=False, y_show=False, z_show=False),
					name=f"Boundary: {cname}", hoverinfo="skip",
				))
			else:
				knn = KNeighborsClassifier(n_neighbors=min(15, len(db_coords) - 1))
				knn.fit(db_coords[:, :2], binary_labels)
				margin = 1.0
				x_min, x_max = db_coords[:, 0].min() - margin, db_coords[:, 0].max() + margin
				y_min, y_max = db_coords[:, 1].min() - margin, db_coords[:, 1].max() + margin
				step = max((x_max - x_min) / 100, 0.1)
				xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
				z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])
				z = (z[:, 1] if z.shape[1] == 2 else z[:, 0]).reshape(xx.shape)
				fig.add_trace(go.Contour(
					x=xx[0], y=yy[:, 0], z=z,
					colorscale=[[0, f"rgba({r},{g},{b},0)"], [1, f"rgba({r},{g},{b},0.2)"]],
					showscale=False, hoverinfo="skip", line_width=1,
					contours=dict(coloring="fill", start=0.5, end=0.5, size=0.1),
					name=f"Boundary: {cname}",
				))

	hover_text = []
	for i in range(len(idx_mask)):
		fn = fns_f[i]
		ci = chunks_f[i]
		pred_names = _predicted_label_names(preds_f[i])
		gt_names = [CLASS_NAME[j] for j in range(CLASS_NUM) if labels_f[i, j] > 0]
		chunk_str = f"<br>Chunk: {ci}" if ci >= 0 else ""
		fp_str = ", ".join(fp_f[i]) if fp_f[i] else "None"
		hover_text.append(
			f"{fn}{chunk_str}<br>Bag GT: {', '.join(gt_names) or 'None'}"
			f"<br>Instance Pred: {', '.join(pred_names) or 'None'}"
			f"<br>Correctness: {status_f[i]}<br>False-positive labels: {fp_str}"
		)

	custom_data = [[fns_f[i], chunks_f[i]] for i in range(len(idx_mask))]
	coords_disp = coords_f[:, :3] if is_3d else coords_f[:, :2]

	if use_clustering and len(coords_f) >= max(n_clusters, 2):
		km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto")
		km.fit(coords_disp)
		cluster_labels = km.labels_
		centers = km.cluster_centers_
		cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
		cluster_group = [""] * n_clusters
		cluster_rep_fn = [""] * n_clusters
		cluster_rep_chunk = [-1] * n_clusters
		for k in range(n_clusters):
			mask_k = cluster_labels == k
			if mask_k.sum() == 0:
				continue
			point_idxs_k = np.where(mask_k)[0].tolist()
			labels_in_cluster: list[str] = []
			for j in point_idxs_k:
				labels_in_cluster.extend(_predicted_label_names(preds_f[j]))
			cluster_group[k] = _majority_value([name for name in labels_in_cluster if name in selected_classes])
			idxs_k = np.where(mask_k)[0]
			dists_k = np.linalg.norm(coords_disp[idxs_k] - centers[k], axis=1)
			rep = idxs_k[np.argmin(dists_k)]
			cluster_rep_fn[k] = fns_f[rep]
			cluster_rep_chunk[k] = chunks_f[rep]
		max_sz = cluster_sizes.max() if cluster_sizes.max() > 0 else 1
		sizes_k = (8 + 22 * (cluster_sizes / max_sz)).tolist()
		for group_name in CLASS_NAME:
			if group_name not in selected_classes:
				continue
			ki = [k for k in range(n_clusters) if cluster_group[k] == group_name and cluster_sizes[k] > 0]
			if not ki:
				continue
			hover_cluster = [f"Cluster #{k}<br>Size: {cluster_sizes[k]}<br>Predicted label: {cluster_group[k]}" for k in ki]
			cluster_custom = [[cluster_rep_fn[k], cluster_rep_chunk[k]] for k in ki]
			if is_3d:
				fig.add_trace(go.Scatter3d(
					x=centers[ki, 0], y=centers[ki, 1], z=centers[ki, 2], mode="markers",
					marker=dict(color=pred_label_color_map[group_name], size=[sizes_k[k] for k in ki], opacity=0.85, line=dict(width=1, color="DarkSlateGrey")),
					text=hover_cluster, hoverinfo="text", customdata=cluster_custom, name=CLASS_DISPLAY_NAME.get(group_name, group_name),
					legendgroup=group_name, showlegend=True,
				))
			else:
				fig.add_trace(go.Scattergl(
					x=centers[ki, 0], y=centers[ki, 1], mode="markers",
					marker=dict(color=pred_label_color_map[group_name], size=[sizes_k[k] for k in ki], opacity=0.85, line=dict(width=1, color="DarkSlateGrey")),
					text=hover_cluster, hoverinfo="text", customdata=cluster_custom, name=CLASS_DISPLAY_NAME.get(group_name, group_name),
					legendgroup=group_name, showlegend=True,
				))
	else:
		for group_name in CLASS_NAME:
			if group_name not in selected_classes:
				continue
			group_idx = [i for i, pred_vec in enumerate(preds_f) if group_name in _predicted_label_names(pred_vec)]
			if not group_idx:
				continue
			c_coords = coords_f[group_idx]
			if is_3d:
				fig.add_trace(go.Scatter3d(
					x=c_coords[:, 0], y=c_coords[:, 1], z=c_coords[:, 2], mode="markers",
					marker=dict(color=pred_label_color_map[group_name], size=4, opacity=0.72, line=dict(width=0.5, color="DarkSlateGrey")),
					text=[hover_text[i] for i in group_idx], hoverinfo="text", customdata=[custom_data[i] for i in group_idx],
					name=CLASS_DISPLAY_NAME.get(group_name, group_name), legendgroup=group_name, showlegend=True,
				))
			else:
				fig.add_trace(go.Scattergl(
					x=c_coords[:, 0], y=c_coords[:, 1], mode="markers",
					marker=dict(color=pred_label_color_map[group_name], size=6, opacity=0.72, line=dict(width=0.5, color="DarkSlateGrey")),
					text=[hover_text[i] for i in group_idx], hoverinfo="text", customdata=[custom_data[i] for i in group_idx],
					name=CLASS_DISPLAY_NAME.get(group_name, group_name), legendgroup=group_name, showlegend=True,
				))

	if is_3d:
		fig.update_layout(scene=dict(
			xaxis=dict(title="UMAP 1", range=axis_ranges[0], autorange=False),
			yaxis=dict(title="UMAP 2", range=axis_ranges[1], autorange=False),
			zaxis=dict(title="UMAP 3", range=axis_ranges[2], autorange=False),
		))
	else:
		fig.update_layout(
			xaxis=dict(title="UMAP 1", range=axis_ranges[0], autorange=False),
			yaxis=dict(title="UMAP 2", range=axis_ranges[1], autorange=False),
		)
	fig.update_layout(
		template="plotly_white", height=750, clickmode="event+select",
		uirevision=ui_revision_key,
		selectionrevision=ui_revision_key,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
		margin=dict(l=40, r=20, t=60, b=40),
	)
	return fig


# ══════════════════════════════════════════════════════════════
# MAIN: Load data & launch app
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("6-Modal MIL Ensemble Interactive Dashboard")
print("=" * 60)

df, file_names, labels, splits, fn_to_idx, cache, ensemble_configs = prepare_all_data()
umap_cache = precompute_umaps(cache, file_names, ensemble_configs, labels)

# Pre-compute ensemble predictions for all configs
all_ensemble_preds: dict[str, dict[str, Optional[np.ndarray]]] = {}
all_ensemble_instance_preds: dict[str, dict[str, dict[int, np.ndarray]]] = {}
for cfg_name, cfg_data in ensemble_configs.items():
	all_ensemble_preds[cfg_name] = compute_ensemble_predictions(cfg_data, cache, file_names)
	all_ensemble_instance_preds[cfg_name] = compute_ensemble_instance_predictions(cfg_data, cache, file_names)

# Default config
default_config = "best_val_macro_auc" if "best_val_macro_auc" in ensemble_configs else (list(ensemble_configs.keys())[0] if ensemble_configs else "")

print(f"\nStarting dashboard... (http://127.0.0.1:8050)")


# ──────────────────────────────────────────────────────────────
# Dash App
# ──────────────────────────────────────────────────────────────

DARK_CSS = """
/* ── Dark mode ── */
body.dark-mode {
    background-color: #1a1a2e !important;
    color: #e0e0e0 !important;
}
body.dark-mode .container-fluid {
    background-color: #1a1a2e !important;
}
body.dark-mode .fw-bold,
body.dark-mode label,
body.dark-mode p,
body.dark-mode h4,
body.dark-mode h5,
body.dark-mode .text-muted {
    color: #e0e0e0 !important;
}
body.dark-mode .form-check-label,
body.dark-mode .form-control,
body.dark-mode .form-select {
    color: #e0e0e0 !important;
    background-color: #2d2d44 !important;
    border-color: #555 !important;
}
body.dark-mode input[type=number] {
    background-color: #2d2d44 !important;
    color: #e0e0e0 !important;
    border-color: #555 !important;
}
body.dark-mode .Select-control,
body.dark-mode .Select-menu-outer,
body.dark-mode .Select-option,
body.dark-mode .Select-value-label,
body.dark-mode .Select-placeholder,
body.dark-mode .VirtualizedSelectOption {
    background-color: #2d2d44 !important;
    color: #e0e0e0 !important;
}
body.dark-mode .Select-arrow { border-top-color: #aaa !important; }
body.dark-mode .dropdown-menu,
body.dark-mode .Select-menu { background-color: #2d2d44 !important; }
body.dark-mode .modal-content {
    background-color: #1e1e32 !important;
    color: #e0e0e0 !important;
    border-color: #444 !important;
}
body.dark-mode .modal-header,
body.dark-mode .modal-footer { border-color: #444 !important; }
body.dark-mode .modal-title { color: #e0e0e0 !important; }
body.dark-mode .btn-close { filter: invert(1); }
body.dark-mode .btn-secondary {
    background-color: #44445a !important;
    border-color: #666 !important;
    color: #e0e0e0 !important;
}
body.dark-mode .btn-secondary:hover {
    background-color: #55556e !important;
}
/* Plotly chart background */
body.dark-mode .js-plotly-plot .plotly,
body.dark-mode .js-plotly-plot .plotly .bg {
    background-color: #1a1a2e !important;
}
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>""" + DARK_CSS + """</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>
"""

# Build dropdown options
model_options = []
for name, cfg in ensemble_configs.items():
	desc = cfg.get("description", name)
	score = cfg.get("best_score", 0)
	label = str(desc)
	model_options.append({"label": f"{label} ({score:.4f})", "value": name})

app.layout = dbc.Container([
	# Title row with dark mode toggle
	dbc.Row([
		dbc.Col(html.H4("6-Modal MIL Ensemble Visualization", className="mt-2 mb-3"), width="auto"),
		dbc.Col(
			dbc.Button("Dark Mode", id="btn-dark-mode", color="secondary",
			           size="sm", className="mt-2 ms-2"),
			width="auto", className="ms-auto d-flex align-items-start",
		),
	], align="center"),

	# Control bar
	dbc.Row([
		dbc.Col([
			html.Label("Model", className="fw-bold"),
			dcc.Dropdown(id="dd-model", options=model_options,
			             value=default_config, clearable=False, style={"fontSize": "0.85em"}),
		], md=2),
		dbc.Col([
			html.Label("Dataset", className="fw-bold"),
			dcc.Dropdown(id="dd-dataset",
			             options=[{"label": "Total", "value": "total"},
			                      {"label": "Train", "value": "train"},
			                      {"label": "Val", "value": "val"}],
			             value="total", clearable=False),
		], md=1),
		dbc.Col([
			html.Label("Modality", className="fw-bold"),
			dcc.Dropdown(id="dd-modality",
			             options=[{"label": m["label"], "value": m["key"]} for m in MODALITY_OPTIONS],
			             value="all", clearable=False),
		], md=1),
		dbc.Col([
			html.Label("View", className="fw-bold"),
			dbc.RadioItems(id="radio-view",
			               options=[{"label": "Bag", "value": "bag"},
			                        {"label": "Instance", "value": "instance"}],
			               value="bag", inline=True),
		], md=1),
		dbc.Col([
			html.Label("Dimension", className="fw-bold"),
			dbc.RadioItems(id="radio-dim",
			               options=[{"label": "2D", "value": "2d"},
			                        {"label": "3D", "value": "3d"}],
			               value="2d", inline=True),
		], md=1),
		dbc.Col([
			html.Label("Filter", className="fw-bold"),
			dbc.RadioItems(id="radio-filter",
			               options=[{"label": "ALL", "value": "all"},
			                        {"label": "Correct", "value": "correct"},
			                        {"label": "Incorrect", "value": "incorrect"}],
			               value="all", inline=True),
		], md=2),
		dbc.Col([
			html.Label("Decision Boundary", className="fw-bold"),
			dbc.Switch(id="switch-boundary", label="Show", value=True, className="mt-1"),
		], md=1),
		dbc.Col([
			html.Label("Predicted Labels", className="fw-bold"),
			dbc.Checklist(id="check-classes",
			              options=[{"label": CLASS_DISPLAY_NAME.get(c, c), "value": c} for c in CLASS_NAME],
			              value=CLASS_NAME[:], inline=True,
			              style={"fontSize": "0.8em"}),
		], md=3),
	], className="mb-3"),

	# Clustering controls
	dbc.Row([
		dbc.Col([
			html.Label("Clustering", className="fw-bold"),
			dbc.Switch(id="switch-cluster", label="", value=True, className="mt-1"),
		], md=1),
		dbc.Col([
			html.Label("Clusters", className="fw-bold"),
			dbc.Input(id="input-n-cluster", type="number", value=50,
			          min=2, step=1, style={"width": "110px"}),
		], md=2),
		dbc.Col([
			html.Label("Export UMAP", className="fw-bold"),
			dbc.Button("Save PNG", id="btn-dl-umap-png", color="secondary",
			           size="sm", className="mt-1"),
			dcc.Download(id="download-umap-png"),
		], md=2),
		dbc.Col([], md=7),
	], className="mb-2"),

	# Main content
	dbc.Row([
		dbc.Col([
			html.Div(
				dcc.Graph(
					id="umap-graph",
					config={"scrollZoom": True, "responsive": True},
					responsive=True,
					style={"height": "100%", "width": "100%"},
				),
				id="umap-resize-box",
				style={
					"height": "750px",
					"width": "100%",
					"minHeight": "420px",
					"minWidth": "520px",
					"maxWidth": "calc(100vw - 32px)",
					"resize": "both",
					"overflow": "hidden",
					"border": "1px solid #e5e7eb",
					"borderRadius": "4px",
				},
			),
		], md=7, lg=7),
		dbc.Col([
			dcc.Download(id="download-image-zip"),
			dcc.Download(id="download-audio-wav"),
			dcc.Download(id="download-single-img"),
			dcc.Store(id="single-frame-store", data={"idx": -1}),
			dcc.Store(id="dark-mode-store", data=False),
			html.Div(id="detail-panel",
			         style={"maxHeight": "750px", "overflowY": "auto"},
			         children=[html.P("Click a point to inspect details.", className="text-muted mt-3")]),
		], md=5, lg=5),
	]),

	# Vision Frame Modal (Overlay)
	dbc.Modal([
		dbc.ModalHeader(dbc.ModalTitle(id="frame-modal-title")),
		dbc.ModalBody([
			html.Div(id="frame-modal-img-container", style={"textAlign": "center", "marginBottom": "15px"}),
			html.Div([
				dbc.Button("Previous (Left)", id="btn-prev-frame", color="secondary", className="me-2"),
				dbc.Button("Download", id="btn-dl-single-frame", color="primary", className="me-2"),
				dbc.Button("Next (Right)", id="btn-next-frame", color="secondary"),
			], style={"textAlign": "center"}),
		])
	], id="frame-modal", is_open=False, size="lg", keyboard=True),
], fluid=True, className="mt-2")

# Dark mode toggle: update body class + button label + store
app.clientside_callback(
	"""
	function(n_clicks, is_dark) {
		if (!n_clicks) { return [is_dark, is_dark ? 'Light Mode' : 'Dark Mode']; }
		var newDark = !is_dark;
		if (newDark) {
			document.body.classList.add('dark-mode');
		} else {
			document.body.classList.remove('dark-mode');
		}
		return [newDark, newDark ? 'Light Mode' : 'Dark Mode'];
	}
	""",
	Output("dark-mode-store", "data"),
	Output("btn-dark-mode", "children"),
	Input("btn-dark-mode", "n_clicks"),
	State("dark-mode-store", "data"),
	prevent_initial_call=False,
)

app.clientside_callback(
	"""
	function(id) {
		if (!window.frameModalKeydownListenerAdded) {
			document.addEventListener('keydown', function(e) {
				let modal = document.getElementById('frame-modal');
				if (e.key === 'ArrowLeft') {
					let btn = document.getElementById('btn-prev-frame');
					if(btn) btn.click();
				} else if (e.key === 'ArrowRight') {
					let btn = document.getElementById('btn-next-frame');
					if(btn) btn.click();
				}
			});
			window.frameModalKeydownListenerAdded = true;
		}
		return window.dash_clientside.no_update;
	}
	""",
	Output("btn-prev-frame", "id"),
	Input("btn-prev-frame", "id")
)

# ──────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────

@app.callback(
	Output("radio-view", "options"),
	Output("radio-view", "value"),
	Input("dd-modality", "value"),
	State("radio-view", "value"),
)
def update_view_toggle(modality, current_view):
	"""Enable both bag and instance views for every modality and the selected ensemble."""
	bag_opt = {"label": "Bag", "value": "bag"}
	inst_opt = {"label": "Instance", "value": "instance"}
	return [bag_opt, inst_opt], current_view or "bag"


@app.callback(
	Output("umap-graph", "figure"),
	Input("dd-model", "value"),
	Input("dd-dataset", "value"),
	Input("dd-modality", "value"),
	Input("radio-view", "value"),
	Input("radio-dim", "value"),
	Input("radio-filter", "value"),
	Input("check-classes", "value"),
	Input("switch-boundary", "value"),
	Input("switch-cluster", "value"),
	Input("input-n-cluster", "value"),
	Input("dark-mode-store", "data"),
)
def update_umap(model_name, dataset_filter, modality, view, dim, display_filter,
               class_checks, show_boundary, use_clustering, n_clusters, is_dark):
	ens_preds = all_ensemble_preds.get(model_name, {})
	ens_inst_preds = all_ensemble_instance_preds.get(model_name, {})
	fig = build_umap_figure(
		modality=modality, view=view, dim=dim,
		dataset_filter=dataset_filter, display_filter=display_filter,
		class_checks=class_checks or [],
		ensemble_preds=ens_preds,
		ensemble_instance_preds=ens_inst_preds,
		umap_cache=umap_cache,
		labels=labels, splits=splits, cache=cache,
		show_boundary=bool(show_boundary),
		use_clustering=bool(use_clustering),
		n_clusters=int(n_clusters) if n_clusters and int(n_clusters) >= 2 else 50,
	)
	if is_dark:
		bg = "#1a1a2e"
		text_color = "#e0e0e0"
		grid_color = "#444466"
		fig.update_layout(
			paper_bgcolor=bg,
			plot_bgcolor=bg,
			font=dict(color=text_color),
			legend=dict(bgcolor="rgba(30,30,50,0.85)", font=dict(color=text_color)),
			scene=dict(
				bgcolor=bg,
				xaxis=dict(backgroundcolor="#16213e", gridcolor=grid_color,
				           tickfont=dict(color=text_color), title=dict(font=dict(color=text_color))),
				yaxis=dict(backgroundcolor="#16213e", gridcolor=grid_color,
				           tickfont=dict(color=text_color), title=dict(font=dict(color=text_color))),
				zaxis=dict(backgroundcolor="#16213e", gridcolor=grid_color,
				           tickfont=dict(color=text_color), title=dict(font=dict(color=text_color))),
			),
			xaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color),
			           title=dict(font=dict(color=text_color))),
			yaxis=dict(gridcolor=grid_color, tickfont=dict(color=text_color),
			           title=dict(font=dict(color=text_color))),
		)
	return fig


@app.callback(
	Output("detail-panel", "children"),
	Input("umap-graph", "clickData"),
	State("dd-modality", "value"),
	State("radio-view", "value"),
	State("dd-model", "value"),
)
def on_point_click(click_data, modality, view, model_name):
	if not click_data:
		return [html.P("Click a point to inspect details.", className="text-muted mt-3")]

	point = click_data["points"][0]
	custom = point.get("customdata")

	# Guard: customdata must be a non-None sequence with at least 2 elements.
	# Clicks on Decision Boundary (Contour/Isosurface) or legend traces land here
	# with no customdata; show neutral message instead of error.
	if custom is None or not hasattr(custom, "__len__") or len(custom) < 2:
		return [html.P("Click a data point. Decision boundaries and legend items are not inspectable.",
		               className="text-muted mt-3")]

	fn = custom[0]
	if not fn:  # empty representative fn (empty cluster guard)
		return [html.P("Click a point to inspect details.", className="text-muted mt-3")]

	try:
		chunk_idx_raw = custom[1]
		chunk_idx = int(chunk_idx_raw) if int(chunk_idx_raw) >= 0 else None
	except (TypeError, ValueError):
		chunk_idx = None

	return build_detail_panel(fn, chunk_idx, df, fn_to_idx, cache, modality, all_ensemble_preds.get(model_name, {}), all_ensemble_instance_preds.get(model_name, {}))


@app.callback(
	Output("download-image-zip", "data"),
	Input("btn-dl-img", "n_clicks"),
	State("detail-store", "data"),
	prevent_initial_call=True
)
def download_image_zip(n_clicks, store_data):
	if not n_clicks or not store_data:
		raise dash.exceptions.PreventUpdate
	frames_dir = store_data.get("frames_dir")
	chunk_idx = store_data.get("chunk_idx")
	video_id = store_data.get("video_id", "unknown")
	if not frames_dir or chunk_idx is None:
		raise dash.exceptions.PreventUpdate

	d = Path(frames_dir)
	if not d.exists():
		raise dash.exceptions.PreventUpdate

	all_files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".png")]
	all_files.sort(key=lambda f: int(re.sub(r"\D", "", f.stem) or "0"))
	start = chunk_idx * 32
	end = start + 32
	selected = all_files[start:end]

	if not selected:
		raise dash.exceptions.PreventUpdate

	buf = io.BytesIO()
	with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
		for i, f in enumerate(selected, start=1):
			zf.writestr(f"{video_id}_{i}{f.suffix}", f.read_bytes())

	buf.seek(0)
	
	start_time = chunk_idx * 8
	end_time = start_time + 8
	file_name = f"{video_id}_{chunk_idx}_{start_time}_{end_time}.zip"
	return dcc.send_bytes(buf.getvalue(), file_name)


@app.callback(
	Output("download-audio-wav", "data"),
	Input({"type": "btn-dl-audio", "index": ALL}, "n_clicks"),
	State("detail-store", "data"),
	prevent_initial_call=True
)
def download_audio_wav(n_clicks_list, store_data):
	if not ctx.triggered or not store_data:
		raise dash.exceptions.PreventUpdate
	prop_id = ctx.triggered[0]["prop_id"]
	if "n_clicks" not in prop_id:
		raise dash.exceptions.PreventUpdate

	trig_dict = json.loads(prop_id.split(".")[0])
	mod_key = trig_dict["index"]

	chunk_idx = store_data.get("chunk_idx")
	video_id = store_data.get("video_id", "unknown")
	wav_path = store_data.get("wav_paths", {}).get(mod_key)

	if chunk_idx is None or not wav_path or not Path(wav_path).exists():
		raise dash.exceptions.PreventUpdate

	sr = 44100
	samples_per_chunk = sr * 8
	start = chunk_idx * samples_per_chunk
	try:
		wav, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=False,
		                       start=start, stop=start + samples_per_chunk)
		if wav.ndim == 2:
			wav = wav.mean(axis=1)

		buf = io.BytesIO()
		sf.write(buf, wav, file_sr, format="WAV")
		buf.seek(0)

		start_time = chunk_idx * 8
		end_time = start_time + 8
		file_name = f"{video_id}_{chunk_idx}_{start_time}_{end_time}.wav"
		return dcc.send_bytes(buf.getvalue(), file_name)
	except Exception:
		raise dash.exceptions.PreventUpdate


@app.callback(
	Output("frame-modal", "is_open"),
	Output("frame-modal-title", "children"),
	Output("frame-modal-img-container", "children"),
	Output("single-frame-store", "data"),
	Input({"type": "vision-frame-img", "index": ALL}, "n_clicks"),
	Input("btn-prev-frame", "n_clicks"),
	Input("btn-next-frame", "n_clicks"),
	State("detail-store", "data"),
	State("single-frame-store", "data"),
	prevent_initial_call=True
)
def handle_frame_modal(img_clicks, prev_clicks, next_clicks, detail_data, single_data):
	if not ctx.triggered or not detail_data or "frames" not in detail_data:
		raise dash.exceptions.PreventUpdate

	frames = detail_data["frames"]
	if not frames:
		raise dash.exceptions.PreventUpdate

	video_id = detail_data.get("video_id", "unknown")
	chunk_idx = detail_data.get("chunk_idx", 0)

	trig_id = ctx.triggered[0]["prop_id"]
	current_idx = single_data.get("idx", 0)

	if "vision-frame-img" in trig_id:
		# check if n_clicks > 0
		val = ctx.triggered[0]["value"]
		if val is None or val == 0:
			raise dash.exceptions.PreventUpdate
		trig_dict = json.loads(trig_id.split(".")[0])
		frame_no = trig_dict["index"]
		for i, f in enumerate(frames):
			if f["frame_no"] == frame_no:
				current_idx = i
				break
		is_open_out = True
	elif "btn-prev-frame" in trig_id:
		current_idx = max(0, current_idx - 1)
		is_open_out = dash.no_update
	elif "btn-next-frame" in trig_id:
		current_idx = min(len(frames) - 1, current_idx + 1)
		is_open_out = dash.no_update
	else:
		raise dash.exceptions.PreventUpdate

	frame_info = frames[current_idx]
	frame_no = frame_info["frame_no"]
	title = f"{video_id} - Chunk {chunk_idx} - Frame {frame_no}"
	img_elem = html.Img(src=frame_info["src"], style={"maxWidth": "100%", "maxHeight": "70vh"})

	return is_open_out, title, img_elem, {"idx": current_idx, "frame_no": frame_no}


@app.callback(
	Output("download-single-img", "data"),
	Input("btn-dl-single-frame", "n_clicks"),
	State("detail-store", "data"),
	State("single-frame-store", "data"),
	prevent_initial_call=True
)
def download_single_frame(n_clicks, detail_data, single_data):
	if not ctx.triggered or not detail_data or not single_data:
		raise dash.exceptions.PreventUpdate

	frames = detail_data.get("frames", [])
	idx = single_data.get("idx", -1)
	if idx < 0 or idx >= len(frames):
		raise dash.exceptions.PreventUpdate

	frame_info = frames[idx]
	src = frame_info["src"]
	frame_no = frame_info["frame_no"]
	video_id = detail_data.get("video_id", "unknown")
	chunk_idx = detail_data.get("chunk_idx", 0)

	header, b64_data = src.split(",", 1)
	ext = header.split(";")[0].split("/")[-1]
	if ext == "jpeg":
		ext = "jpg"

	raw_data = base64.b64decode(b64_data)
	file_name = f"{video_id}_{chunk_idx}_{frame_no}.{ext}"

	return dcc.send_bytes(raw_data, file_name)


@app.callback(
	Output("download-umap-png", "data"),
	Input("btn-dl-umap-png", "n_clicks"),
	State("umap-graph", "figure"),
	prevent_initial_call=True,
)
def download_umap_png(n_clicks, figure):
	if not n_clicks or not figure:
		raise dash.exceptions.PreventUpdate
	import plotly.io as pio
	fig_export = go.Figure(figure)
	transparent = "rgba(0,0,0,0)"
	fig_export.update_layout(
		paper_bgcolor=transparent,
		plot_bgcolor=transparent,
		showlegend=False,
		scene=dict(
			bgcolor=transparent,
			xaxis=dict(
				backgroundcolor=transparent,
				gridcolor="rgba(180,180,180,0.3)",
				showbackground=True,
			),
			yaxis=dict(
				backgroundcolor=transparent,
				gridcolor="rgba(180,180,180,0.3)",
				showbackground=True,
			),
			zaxis=dict(
				backgroundcolor=transparent,
				gridcolor="rgba(180,180,180,0.3)",
				showbackground=True,
			),
		),
	)
	img_bytes = pio.to_image(fig_export, format="png", scale=4)
	return dcc.send_bytes(img_bytes, "umap.png")


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
	app.run(debug=False, port=RUNTIME_ARGS.port)
