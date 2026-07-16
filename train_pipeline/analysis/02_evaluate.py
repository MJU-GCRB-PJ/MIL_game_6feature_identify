#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for evaluation summaries and paper figures."""
from __future__ import annotations

import argparse
import ast as ast_module
import csv
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Configuration.
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_BASE = REPO_ROOT / "train_pipeline" / "training" / "outputs"
CV_OUTPUT_ROOT = OUTPUT_BASE / "cv"
DATA_CSV = CV_OUTPUT_ROOT / "fold_01" / "data.csv"
BEST_PTH_DIR = CV_OUTPUT_ROOT / "fold_01" / "ensemble" / "best_pth"
PREFERRED_PTH_PATH = BEST_PTH_DIR / "best_val_macro_auc.pth"
MIL_OUTPUT_BASE = OUTPUT_BASE

RESULT_DIR = SCRIPT_DIR / "result" / "fold_01"

CLASS_NUM = 6
CLASS_NAME = [
    "sexual_content", "violence", "fear",
    "inappropriate_language", "drugs", "crime",
]

GT_COLS = CLASS_NAME + ["gambling"]

SEED = 42
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# GatedAttentionMIL
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


# ──────────────────────────────────────────────────────────────
# Model setup.
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


def _load_features_for_sample(row: pd.Series, modality_key: str):
    """Helper function for load features for sample."""
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


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def _infer_sample(
    model: nn.Module,
    raw_feats,
    modality_key: str,
    device: torch.device,
) -> np.ndarray:
    """Helper function for infer sample."""
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
        logits = model(x, m)

    probs = torch.sigmoid(logits.float().squeeze(0)).cpu().numpy()
    return probs.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Model setup.
# ──────────────────────────────────────────────────────────────

def run_all_model_inferences(
    df: pd.DataFrame,
    file_names: list[str],
    fn_to_idx: dict[str, int],
) -> dict[str, dict[str, Optional[np.ndarray]]]:
    """Helper function for run all model inferences."""
    predictions: dict[str, dict[str, Optional[np.ndarray]]] = {}

    for reg in MODEL_REGISTRY:
        key = reg["key"]
        predictions[key] = {}

        try:
            model, _ = load_model(reg, DEVICE)
        except FileNotFoundError as e:
            print(f"  [{reg['label']}] Checkpoint unavailable: {e}")
            for fn in file_names:
                predictions[key][fn] = None
            continue

        done, skipped = 0, 0
        for fn in tqdm(file_names, desc=f"  [{reg['label']}]", leave=True):
            idx = fn_to_idx.get(fn)
            if idx is None:
                predictions[key][fn] = None
                skipped += 1
                continue
            row = df.iloc[idx]
            try:
                raw = _load_features_for_sample(row, key)
                if raw is None:
                    predictions[key][fn] = None
                    skipped += 1
                    continue
                probs = _infer_sample(model, raw, key, DEVICE)
                predictions[key][fn] = probs
                done += 1
            except Exception as e:
                predictions[key][fn] = None
                skipped += 1
                if done < 3:
                    print(f"    ⚠ {fn}: {e}")

        print(f"    Complete: {done}, skipped: {skipped}")
        del model
        torch.cuda.empty_cache()

    return predictions


# ──────────────────────────────────────────────────────────────
# Ensemble evaluation.
# ──────────────────────────────────────────────────────────────

def _parse_model_combination(combo_str: str) -> list[str]:
    """'Vision+Original Audio+STT' → ['vision', 'original_audio', 'stt']"""
    label_to_key = {r["label"]: r["key"] for r in MODEL_REGISTRY}
    parts = [p.strip() for p in combo_str.split("+")]
    return [label_to_key[p] for p in parts if p in label_to_key]


def _parse_model_weights(weights_raw, model_keys: list[str]) -> Optional[dict[str, float]]:
    """Helper function for parse model weights."""
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
            result = {
                k: float(v) for k, v in weights_raw.items()
                if k in model_keys and isinstance(v, (int, float))
            }
            return result if result else None
        except (TypeError, ValueError):
            return None
    return None


def compute_ensemble_probs(
    config: dict,
    predictions: dict[str, dict[str, Optional[np.ndarray]]],
    file_names: list[str],
) -> dict[str, Optional[np.ndarray]]:
    """Helper function for compute ensemble probs."""
    combo = config.get("model_combination", "")
    method = config.get("ensemble_method", "")
    weights_str = config.get("model_weights", "")
    model_keys = _parse_model_combination(combo)
    if not model_keys:
        return {fn: None for fn in file_names}

    weight_map = _parse_model_weights(weights_str, model_keys)
    result: dict[str, Optional[np.ndarray]] = {}

    for fn in file_names:
        avail_probs = []
        avail_keys = []
        for k in model_keys:
            p = predictions.get(k, {}).get(fn)
            if p is not None:
                avail_probs.append(p)
                avail_keys.append(k)

        if not avail_probs:
            result[fn] = None
            continue

        probs_arr = np.stack(avail_probs, axis=0)  # (M, C)

        if "Hard" in method:
            votes = (probs_arr >= THRESHOLD).astype(np.float32).mean(axis=0)
            result[fn] = votes
        elif weight_map and "Weighted" in method:
            w = np.array([weight_map.get(k, 1.0) for k in avail_keys], dtype=np.float64)
            w = w / (w.sum() + 1e-12)
            result[fn] = (w[:, None] * probs_arr).sum(axis=0).astype(np.float32)
        else:  # Soft voting / Individual / Stacking fallback
            result[fn] = probs_arr.mean(axis=0).astype(np.float32)

    return result


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────

def _compute_corrected_and_coincidence(actual: list[str], pred: list[str]) -> tuple[str, float]:
    actual_set = set(actual)
    pred_set = set(pred)

    if pred_set == actual_set:
        corrected = "All"
    elif pred_set and pred_set < actual_set:
        corrected = "Parted"
    else:
        corrected = "Incorrect"

    if not actual_set:
        coincidence_rate = 100.0 if not pred_set else 0.0
    else:
        coincidence_rate = len(actual_set & pred_set) / len(actual_set) * 100.0
    return corrected, round(float(coincidence_rate), 2)


def build_sheet_df(
    df: pd.DataFrame,
    file_names: list[str],
    fn_to_idx: dict[str, int],
    fn_probs: dict[str, Optional[np.ndarray]],
) -> pd.DataFrame:
    """Helper function for build sheet df."""
    base_cols = ["split", "no", "file_name"]
    pred_cols = [f"pred_{c}" for c in CLASS_NAME]

    rows = []
    for fn in file_names:
        idx = fn_to_idx.get(fn)
        if idx is None:
            continue
        row = df.iloc[idx]


        rec: dict[str, Any] = {
            "split": _safe_str(row.get("split")),
            "no": row.get("no", ""),
            "file_name": fn,
        }


        for c in GT_COLS:
            rec[c] = row.get(c, "")


        actual = [c for c in CLASS_NAME if _to_float01(row.get(c)) > 0]
        rec["actual_class"] = ",".join(actual)


        probs = fn_probs.get(fn)
        pred: list[str] = []
        if probs is not None:
            pred = [CLASS_NAME[i] for i in range(CLASS_NUM) if probs[i] >= THRESHOLD]
            rec["pred_label"] = ",".join(pred)
            for i, c in enumerate(CLASS_NAME):
                rec[f"pred_{c}"] = round(float(probs[i]), 4)
        else:
            rec["pred_label"] = ""
            for c in CLASS_NAME:
                rec[f"pred_{c}"] = float("nan")

        corrected, coincidence_rate = _compute_corrected_and_coincidence(actual, pred)
        rec["corrected"] = corrected
        rec["coincidence_rate"] = coincidence_rate

        rows.append(rec)

    col_order = (
        base_cols
        + GT_COLS
        + ["actual_class", "pred_label", "corrected", "coincidence_rate"]
        + pred_cols
    )
    result_df = pd.DataFrame(rows)

    col_order = [c for c in col_order if c in result_df.columns]
    return result_df[col_order]


# ──────────────────────────────────────────────────────────────
# Sort values.
# ──────────────────────────────────────────────────────────────

def _sort_pth_files(pth_files: list[Path]) -> list[Path]:
    """Helper function for sort pth files."""
    split_order = {"val": 0, "train": 1, "total": 2}
    class_order = {c: i + 1 for i, c in enumerate(CLASS_NAME)}  # macro_auc=0

    def _sort_key(p: Path) -> tuple[int, int]:
        stem = p.stem  # e.g. "best_val_macro_auc" or "best_train_auc_crime"
        # split
        split_idx = 99
        for sp, si in split_order.items():
            if f"_{sp}_" in stem:
                split_idx = si
                break
        # class position
        if "macro_auc" in stem:
            class_idx = 0
        else:
            class_idx = 99
            for c, ci in class_order.items():
                if stem.endswith(f"_{c}"):
                    class_idx = ci
                    break
        return (split_idx, class_idx)

    return sorted(pth_files, key=_sort_key)


def _make_sheet_name(stem: str) -> str:
    """Helper function for make sheet name."""
    name = stem.removeprefix("best_")
    return name[:31]


def _load_target_pth_files(*, all_best: bool) -> list[Path]:
    if PREFERRED_PTH_PATH.exists() and not all_best:
        return [PREFERRED_PTH_PATH]
    if not BEST_PTH_DIR.exists():
        raise FileNotFoundError(f"best_pth directory not found: {BEST_PTH_DIR}")
    pth_files = list(BEST_PTH_DIR.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No best_pth files found: {BEST_PTH_DIR}")
    print(f"Using {len(pth_files)} best ensemble checkpoints.")
    return _sort_pth_files(pth_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument(
        "--all-best",
        action="store_true",
        help="Evaluate every criterion checkpoint instead of only best validation Macro AUC.",
    )
    parser.add_argument("--output-root", type=Path, default=CV_OUTPUT_ROOT)
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    global DATA_CSV, BEST_PTH_DIR, PREFERRED_PTH_PATH, MIL_OUTPUT_BASE, RESULT_DIR
    fold_dir = args.output_root / f"fold_{args.fold:02d}"
    DATA_CSV = fold_dir / "data.csv"
    BEST_PTH_DIR = fold_dir / "ensemble" / "best_pth"
    PREFERRED_PTH_PATH = BEST_PTH_DIR / "best_val_macro_auc.pth"
    MIL_OUTPUT_BASE = fold_dir
    RESULT_DIR = SCRIPT_DIR / "result" / f"fold_{args.fold:02d}"


# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    configure_runtime(args)
    print("=" * 60)
    print("6-Modal MIL Ensemble Batch Evaluation")
    print("=" * 60)

    # Load input.
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {DATA_CSV}")
    df = _read_csv_df(DATA_CSV)
    print(f"CSV loaded: {len(df)} rows")


    file_names: list[str] = []
    fn_to_idx: dict[str, int] = {}
    for i, row in df.iterrows():
        fn = Path(_safe_str(row.get("file_name"))).name
        if not fn:
            continue
        file_names.append(fn)
        fn_to_idx[fn] = i
    print(f"Valid files: {len(file_names)}")

    # Load input.
    pth_files = _load_target_pth_files(all_best=args.all_best)
    print(f"best_pth files: {len(pth_files)}")
    for p in pth_files:
        print(f"  {p.name}")

    # Model setup.
    print("\n" + "=" * 60)
    print("Starting model inference...")
    print("=" * 60)
    predictions = run_all_model_inferences(df, file_names, fn_to_idx)

    # Ensemble evaluation.
    print("\n" + "=" * 60)
    print("Building sheets for each ensemble configuration...")
    print("=" * 60)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "evaluate.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for pth_path in tqdm(pth_files, desc="Saving sheets"):
            config = torch.load(str(pth_path), map_location="cpu", weights_only=False)
            sheet_name = _make_sheet_name(pth_path.stem)

            fn_probs = compute_ensemble_probs(config, predictions, file_names)
            sheet_df = build_sheet_df(df, file_names, fn_to_idx, fn_probs)
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

            desc = config.get("description", pth_path.stem)
            score = config.get("best_score", float("nan"))
            combo = config.get("model_combination", "?")
            method = config.get("ensemble_method", "?")
            print(f"  [{sheet_name}] {desc} | score={score:.4f} | {method} | {combo}")

    print(f"\nSaved: {output_path}")
    print(f"  Sheet count: {len(pth_files)}")


if __name__ == "__main__":
    main()
