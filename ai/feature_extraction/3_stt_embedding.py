from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


DEFAULT_MODEL_NAME = "tencent/KaLM-Embedding-Gemma3-12B-2511"

DEFAULT_WINDOW_SIZE_SEC = 8.0
DEFAULT_WEIGHTING_METHOD = "overlap_seconds"  # overlap_seconds | overlap_ratio

# Batch processing.
DEFAULT_BATCH_SIZE = 16



DEFAULT_SPLIT_LONG_SEGMENTS_SEC = 12.0
DEFAULT_MIN_SUBSEG_SEC = 0.25
DEFAULT_MAX_TEXT_CHARS = 400

# Split or separate data.
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!\n。！？…])\s+|[\n]+")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    index_csv: Path
    model_root_dir: Path
    feature_root_dir: Path


def get_paths() -> Paths:

    script_dir = Path(__file__).resolve().parent  # repo/ai/feature_extraction
    repo_root = script_dir.parent.parent          # repo
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from ai.project_paths import FEATURE_ROOT, PREPROCESS_INDEX_CSV
    index_csv = PREPROCESS_INDEX_CSV
    model_root_dir = script_dir / "model"
    feature_root_dir = FEATURE_ROOT
    return Paths(
        repo_root=repo_root,
        index_csv=index_csv,
        model_root_dir=model_root_dir,
        feature_root_dir=feature_root_dir,
    )


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    return str(value)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_model_dir_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)


def _install_sentence_transformers() -> None:
    logging.info("sentence-transformers is not installed; attempting automatic installation.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])


def _optional_torch_kwargs(dtype: str, use_flash_attn2: bool) -> dict[str, Any]:
    """Helper function for optional torch kwargs."""
    try:
        import torch  # type: ignore
    except Exception:
        return {}

    dtype_map = {
        "auto": None,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype.lower(), None)

    kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    if use_flash_attn2:
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def load_model(
    paths: Paths,
    model_name: str,
    model_dir: Optional[Path] = None,
    dtype: str = "auto",
    use_flash_attn2: bool = False,
    max_seq_length: int = 512,
):
    """Helper function for load model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        _install_sentence_transformers()
        from sentence_transformers import SentenceTransformer

    ensure_dir(paths.model_root_dir)
    local_model_dir = model_dir if model_dir is not None else (paths.model_root_dir / _sanitize_model_dir_name(model_name))

    model_kwargs = _optional_torch_kwargs(dtype=dtype, use_flash_attn2=use_flash_attn2)

    if local_model_dir.exists() and any(local_model_dir.iterdir()):
        logging.info("Loading local embedding model: %s", local_model_dir)
        model = SentenceTransformer(str(local_model_dir), trust_remote_code=True, model_kwargs=model_kwargs)
    else:
        logging.info("Downloading model: %s", model_name)
        model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs=model_kwargs)
        ensure_dir(local_model_dir)
        model.save(str(local_model_dir))
        logging.info("Model saved locally: %s", local_model_dir)

    # Model setup.
    try:
        model.max_seq_length = int(max_seq_length)
    except Exception:
        pass

    return model, local_model_dir


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _load_index_df(index_csv: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(index_csv, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(index_csv)


def load_stt_segments(stt_json_path: Path) -> list[dict[str, Any]]:
    """Helper function for load stt segments."""
    if not stt_json_path.exists():
        raise FileNotFoundError(f"STT JSON file not found: {stt_json_path}")

    with stt_json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict):
        raw_segments = obj.get("segments", [])
    elif isinstance(obj, list):
        raw_segments = obj
    else:
        logging.warning("STT JSON root is not a dict or list; using an empty list: %s", stt_json_path)
        return []

    if not isinstance(raw_segments, list):
        logging.warning("segments is not a list; using an empty list: %s", stt_json_path)
        return []

    clean: list[dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            continue

        text = _safe_str(seg.get("text")).strip()
        if not text:
            continue

        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except Exception:
            logging.warning("Could not parse start/end for segment[%d]; excluding it: %s", idx, stt_json_path)
            continue

        start = max(0.0, start)
        end = max(0.0, end)
        if end <= start:
            continue

        clean.append({"start": start, "end": end, "text": text})

    clean.sort(key=lambda s: (float(s["start"]), float(s["end"])))
    return clean


def _split_text_to_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else [text]


def _chunk_by_length(text: str, max_chars: int) -> list[str]:
    """Helper function for chunk by length."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        cut = j
        if j < n:
            k = text.rfind(" ", i, j)
            if k != -1 and (k - i) >= int(max_chars * 0.5):
                cut = k
        part = text[i:cut].strip()
        if part:
            out.append(part)
        i = cut if cut > i else j
    return out


def explode_long_segments(
    segments: list[dict[str, Any]],
    split_long_segments_sec: float,
    min_subseg_sec: float,
    max_text_chars: int,
) -> list[dict[str, Any]]:
    """Helper function for explode long segments."""
    out: list[dict[str, Any]] = []

    for seg in segments:
        s = float(seg["start"])
        e = float(seg["end"])
        text = str(seg["text"]).strip()
        dur = e - s


        if dur <= split_long_segments_sec and len(text) <= max_text_chars:
            out.append(seg)
            continue

        sentences = _split_text_to_sentences(text)
        if not sentences:
            continue

        parts: list[str] = []
        for sent in sentences:
            parts.extend(_chunk_by_length(sent, max_text_chars))

        if not parts:
            continue

        lengths = np.asarray([max(1, len(p)) for p in parts], dtype=np.float32)
        total = float(lengths.sum())

        cur = s
        for i, p in enumerate(parts):
            frac = float(lengths[i] / total)
            sub_dur = max(min_subseg_sec, dur * frac)

            sub_start = cur
            sub_end = min(e, sub_start + sub_dur)
            if i == len(parts) - 1:
                sub_end = e

            if sub_end > sub_start:
                out.append({"start": sub_start, "end": sub_end, "text": p})

            cur = sub_end
            if cur >= e:
                break

    out.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    return out


def _encode_texts(model, texts: list[str], batch_size: int, normalize: bool) -> np.ndarray:
    """Helper function for encode texts."""
    if hasattr(model, "encode_document"):
        emb = model.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    else:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
    emb = np.asarray(emb)
    if emb.ndim != 2:
        raise ValueError(f"Invalid embedding dimensions: shape={emb.shape}")
    return emb.astype(np.float32)


def build_windows_with_text_and_map(
    *,
    units: list[dict[str, Any]],
    model,
    window_size: float,
    batch_size: int,
    weighting_method: str,
    normalize_embeddings: bool,
) -> Tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Helper function for build windows with text and map."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive: {window_size}")


    if not units:
        dim = int(model.get_sentence_embedding_dimension())
        empty_emb = np.zeros((0, dim), dtype=np.float32)
        empty_mask = np.zeros((0,), dtype=np.uint8)
        meta = {"D": dim, "max_end": 0.0, "T": 0, "n_units": 0, "n_non_empty_windows": 0}
        return empty_emb, empty_mask, meta, [], [], []


    stt_units: list[dict[str, Any]] = []
    for i, u in enumerate(units):
        stt_units.append(
            {
                "unit_id": int(i),
                "start": float(u["start"]),
                "end": float(u["end"]),
                "text": str(u["text"]),
            }
        )

    texts = [u["text"] for u in stt_units]
    starts = np.asarray([u["start"] for u in stt_units], dtype=np.float32)
    ends = np.asarray([u["end"] for u in stt_units], dtype=np.float32)
    durations = np.maximum(ends - starts, 1e-8)

    unit_emb = _encode_texts(model, texts, batch_size=batch_size, normalize=normalize_embeddings)
    dim = int(unit_emb.shape[1])

    max_end = float(ends.max(initial=0.0))
    T = int(math.ceil(max_end / window_size)) if max_end > 0 else 0

    window_emb = np.zeros((T, dim), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.uint8)

    window_map: list[dict[str, Any]] = []
    window_texts: list[str] = [""] * T

    for t in range(T):
        w_start = float(t * window_size)
        w_end = float((t + 1) * window_size)

        overlap = np.minimum(ends, w_end) - np.maximum(starts, w_start)
        overlap = np.maximum(overlap, 0.0)

        valid_idx = np.where(overlap > 0)[0]
        if valid_idx.size == 0:
            # Visualization.
            window_map.append(
                {
                    "t": int(t),
                    "start": w_start,
                    "end": w_end,
                    "unit_ids": [],
                    "overlap_seconds": [],
                    "weights": [],
                }
            )
            window_texts[t] = ""
            continue

        ov = overlap[valid_idx].astype(np.float32)

        if weighting_method == "overlap_ratio":
            weights = (ov / durations[valid_idx]).astype(np.float32)
        else:
            weights = ov  # overlap_seconds

        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            window_map.append(
                {
                    "t": int(t),
                    "start": w_start,
                    "end": w_end,
                    "unit_ids": [],
                    "overlap_seconds": [],
                    "weights": [],
                }
            )
            window_texts[t] = ""
            continue

        weighted_vec = (unit_emb[valid_idx] * weights[:, None]).sum(axis=0) / weight_sum
        window_emb[t] = weighted_vec.astype(np.float32)
        mask[t] = np.uint8(1)

        unit_ids = valid_idx.tolist()
        overlap_seconds_list = ov.tolist()
        weights_list = weights.tolist()


        # Visualization.
        joined_text = " ".join([texts[i] for i in unit_ids]).strip()

        window_map.append(
            {
                "t": int(t),
                "start": w_start,
                "end": w_end,
                "unit_ids": unit_ids,
                "overlap_seconds": overlap_seconds_list,
                "weights": weights_list,
            }
        )
        window_texts[t] = joined_text

    meta = {
        "D": dim,
        "max_end": max_end,
        "T": T,
        "n_units": int(len(stt_units)),
        "n_non_empty_windows": int(mask.sum()),
    }
    return window_emb, mask, meta, stt_units, window_map, window_texts


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_outputs(
    *,
    out_dir: Path,
    emb: np.ndarray,
    mask: np.ndarray,
    meta: dict[str, Any],
    stt_units: list[dict[str, Any]],
    window_map: list[dict[str, Any]],
    window_texts: list[str],
    overwrite: bool,
) -> bool:
    """Helper function for save outputs."""
    ensure_dir(out_dir)

    emb_path = out_dir / "stt_8s_emb.npy"
    mask_path = out_dir / "stt_8s_mask.npy"
    meta_path = out_dir / "stt_8s_meta.json"

    units_path = out_dir / "stt_units.jsonl"
    map_path = out_dir / "stt_8s_map.json"
    text_path = out_dir / "stt_8s_text.json"

    if emb_path.exists() and not overwrite:
        logging.info("Output already exists; skipping: %s", emb_path)
        return False

    # Save output.
    emb_tmp = out_dir / "stt_8s_emb.tmp.npy"
    mask_tmp = out_dir / "stt_8s_mask.tmp.npy"
    for p in (emb_tmp, mask_tmp):
        if p.exists():
            p.unlink()

    np.save(emb_tmp, emb.astype(np.float16, copy=False))
    np.save(mask_tmp, mask.astype(np.uint8, copy=False))
    emb_tmp.replace(emb_path)
    mask_tmp.replace(mask_path)

    _write_json_atomic(meta_path, meta)
    _write_jsonl_atomic(units_path, stt_units)
    _write_json_atomic(map_path, {"window_size": meta.get("window_size", DEFAULT_WINDOW_SIZE_SEC), "windows": window_map})
    _write_json_atomic(text_path, {"window_size": meta.get("window_size", DEFAULT_WINDOW_SIZE_SEC), "texts": window_texts})

    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 8-second STT text embeddings aligned for MIL time axis (with traceable texts).")
    p.add_argument("--index", type=str, default="", help="Path to the preprocessing index CSV")
    p.add_argument("--only-file", type=str, default="", help="Process only this file_name")
    p.add_argument("--limit", type=int, default=0, help="Process only first N rows (0=all)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--model-dir", type=str, default="", help="Local cache dir for the model (default: ai/feature_extraction/model/<sanitized>)")
    p.add_argument("--dtype", type=str, default="auto", help="auto|fp16|bf16|fp32 (applied when torch is available)")
    p.add_argument("--flash-attn2", action="store_true", help="Try flash_attention_2 when supported by torch and the environment")
    p.add_argument("--max-seq-len", type=int, default=512)

    p.add_argument("--window-size", type=float, default=DEFAULT_WINDOW_SIZE_SEC)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--weighting", type=str, default=DEFAULT_WEIGHTING_METHOD, choices=["overlap_seconds", "overlap_ratio"])
    p.add_argument("--normalize-emb", action="store_true", help="Normalize embeddings to unit length")

    p.add_argument("--split-long-sec", type=float, default=DEFAULT_SPLIT_LONG_SEGMENTS_SEC)
    p.add_argument("--min-subseg-sec", type=float, default=DEFAULT_MIN_SUBSEG_SEC)
    p.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS)

    p.add_argument("--out-root", type=str, default="", help="Feature root directory (default: /data/feature_extraction)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    paths = get_paths()
    if args.index:
        paths = Paths(
            repo_root=paths.repo_root,
            index_csv=Path(args.index),
            model_root_dir=paths.model_root_dir,
            feature_root_dir=paths.feature_root_dir,
        )
    if args.out_root:
        paths = Paths(
            repo_root=paths.repo_root,
            index_csv=paths.index_csv,
            model_root_dir=paths.model_root_dir,
            feature_root_dir=Path(args.out_root),
        )

    if not paths.index_csv.exists():
        raise FileNotFoundError(f"index.csv not found: {paths.index_csv}")

    df = _load_index_df(paths.index_csv)
    if df.empty:
        raise ValueError("index.csv contains no data.")

    for required in ("file_name", "stt_json"):
        if required not in df.columns:
            raise ValueError(f"index.csv must contain the `{required}` column.")

    if args.only_file:
        df = df[df["file_name"].astype(str) == args.only_file]
        if df.empty:
            raise ValueError(f"No row matches --only-file: {args.only_file}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    model_dir = Path(args.model_dir) if args.model_dir else None
    model, local_model_dir = load_model(
        paths=paths,
        model_name=args.model_name,
        model_dir=model_dir,
        dtype=args.dtype,
        use_flash_attn2=bool(args.flash_attn2),
        max_seq_length=int(args.max_seq_len),
    )

    logging.info("INDEX_CSV=%s", paths.index_csv)
    logging.info("FEATURE_ROOT=%s", paths.feature_root_dir)
    logging.info("MODEL_NAME=%s", args.model_name)
    logging.info("MODEL_LOCAL_DIR=%s", local_model_dir)
    logging.info("WINDOW_SIZE=%.3f", float(args.window_size))
    logging.info("WEIGHTING=%s", args.weighting)
    logging.info("SPLIT_LONG_SEC=%.3f, MAX_TEXT_CHARS=%d", float(args.split_long_sec), int(args.max_text_chars))

    processed = 0
    skipped = 0
    failed = 0

    rows = df.to_dict(orient="records")
    for row in tqdm(rows, total=len(rows), desc="STT embedding extraction"):
        file_name = _safe_str(row.get("file_name")).strip()
        stt_json_raw = _safe_str(row.get("stt_json")).strip()

        if not file_name or not stt_json_raw:
            failed += 1
            logging.warning("file_name or stt_json is empty; skipping: %s", row)
            continue

        stt_json_path = _resolve_path(paths.repo_root, stt_json_raw)
        out_dir = paths.feature_root_dir / file_name / "stt_feat"
        emb_path = out_dir / "stt_8s_emb.npy"

        if emb_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            segments = load_stt_segments(stt_json_path)
            n_before = len(segments)


            units = explode_long_segments(
                segments,
                split_long_segments_sec=float(args.split_long_sec),
                min_subseg_sec=float(args.min_subseg_sec),
                max_text_chars=int(args.max_text_chars),
            )
            n_after = len(units)

            emb, mask, extra, stt_units, window_map, window_texts = build_windows_with_text_and_map(
                units=units,
                model=model,
                window_size=float(args.window_size),
                batch_size=int(args.batch_size),
                weighting_method=str(args.weighting),
                normalize_embeddings=bool(args.normalize_emb),
            )

            meta = {
                "file_name": file_name,
                "source_stt_json": str(stt_json_path),
                "model_name": str(args.model_name),
                "model_local_dir": str(local_model_dir),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "window_size": float(args.window_size),
                "weighting": str(args.weighting),
                "batch_size": int(args.batch_size),
                "normalize_embeddings": bool(args.normalize_emb),
                "split_long_segments_sec": float(args.split_long_sec),
                "min_subseg_sec": float(args.min_subseg_sec),
                "max_text_chars": int(args.max_text_chars),
                "n_segments_before_split": int(n_before),
                "n_segments_after_split": int(n_after),
                **extra,
                "saved_files": [
                    "stt_8s_emb.npy",
                    "stt_8s_mask.npy",
                    "stt_8s_meta.json",
                    "stt_units.jsonl",
                    "stt_8s_map.json",
                    "stt_8s_text.json",
                ],
            }

            written = save_outputs(
                out_dir=out_dir,
                emb=emb,
                mask=mask,
                meta=meta,
                stt_units=stt_units,
                window_map=window_map,
                window_texts=window_texts,
                overwrite=bool(args.overwrite),
            )

            if written:
                processed += 1
                logging.info(
                    "Saved: %s | T=%d D=%d non_empty=%d units=%d | out=%s",
                    file_name,
                    int(meta.get("T", 0)),
                    int(meta.get("D", 0)),
                    int(meta.get("n_non_empty_windows", 0)),
                    int(meta.get("n_units", 0)),
                    out_dir,
                )
            else:
                skipped += 1

        except Exception as e:
            failed += 1
            logging.exception("Processing failed: file_name=%s stt_json=%s err=%s", file_name, stt_json_path, e)

    logging.info("Complete: processed=%d skipped=%d failed=%d total=%d", processed, skipped, failed, len(rows))


if __name__ == "__main__":
    main()
