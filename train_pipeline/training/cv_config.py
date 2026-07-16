"""Shared configuration for the paper-aligned cross-validation pipeline."""

from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent.parent
CV_OUTPUT_DIR = TRAINING_DIR / "outputs" / "cv"

N_FOLDS = 5
BASE_SEED = 42
EXPECTED_SAMPLE_COUNT = 663

CLASS_NAMES: tuple[str, ...] = (
    "sexual_content",
    "violence",
    "fear",
    "inappropriate_language",
    "drugs",
    "crime",
)

TARGET_MODEL_COMBINATION = (
    "Vision+Original Audio+Vocal Audio+Non-Vocal Audio+OCR+STT"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    output_dir_name: str
    script_name: str
    best_checkpoint: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("vision", "Vision", "vision_mil", "02_vision_mil.py", "best_vision_mil_model.pth"),
    ModelSpec(
        "original_audio",
        "Original Audio",
        "original_audio_mil",
        "03_original-audio_mil.py",
        "best_original_audio_mil_model.pth",
    ),
    ModelSpec(
        "vocal_audio",
        "Vocal Audio",
        "vocal_audio_mil",
        "04_vocal-audio_mil.py",
        "best_vocal_audio_mil_model.pth",
    ),
    ModelSpec(
        "non_vocal_audio",
        "Non-Vocal Audio",
        "non_vocal_audio_mil",
        "05_non-vocal-audio_mil.py",
        "best_non_vocal_audio_mil_model.pth",
    ),
    ModelSpec("ocr", "OCR", "ocr_mil", "06_ocr_mil.py", "best_ocr_mil_model.pth"),
    ModelSpec("stt", "STT", "stt_mil", "07_stt_mil.py", "best_stt_mil_model.pth"),
)
MODEL_SPEC_BY_KEY = {model.key: model for model in MODEL_SPECS}


def validate_fold(fold: int) -> int:
    fold = int(fold)
    if fold < 1 or fold > N_FOLDS:
        raise ValueError(f"fold must be in 1..{N_FOLDS}, got {fold}")
    return fold


def fold_dir(fold: int, output_root: Path = CV_OUTPUT_DIR) -> Path:
    return Path(output_root) / f"fold_{validate_fold(fold):02d}"


def fold_data_csv(fold: int, output_root: Path = CV_OUTPUT_DIR) -> Path:
    return fold_dir(fold, output_root) / "data.csv"


def model_output_dir(
    fold: int,
    model: ModelSpec,
    output_root: Path = CV_OUTPUT_DIR,
) -> Path:
    return fold_dir(fold, output_root) / model.output_dir_name


def fold_seed(fold: int, base_seed: int = BASE_SEED) -> int:
    return int(base_seed) + validate_fold(fold)


def resolve_requested_folds(
    *,
    fold: int | None = None,
    folds: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Resolve one, several, or all folds while preserving request order."""
    if fold is not None and folds is not None:
        raise ValueError("Use either --fold or --folds, not both")

    requested = [fold] if fold is not None else list(folds or range(1, N_FOLDS + 1))
    resolved: list[int] = []
    for value in requested:
        validated = validate_fold(value)
        if validated not in resolved:
            resolved.append(validated)
    if not resolved:
        raise ValueError("At least one fold is required")
    return resolved


def resolve_training_inputs(
    *,
    fold: int | None,
    data_csv: str | Path | None,
    output_dir: str | Path | None,
    model_key: str,
    seed: int | None,
    output_root: Path = CV_OUTPUT_DIR,
) -> tuple[int | None, Path, Path, int]:
    """Resolve canonical fold paths while retaining explicit-path compatibility."""
    if model_key not in MODEL_SPEC_BY_KEY:
        raise KeyError(f"Unknown model key: {model_key}")

    if fold is None:
        if not data_csv or not output_dir:
            raise ValueError("--fold is required unless both --csv and --output-dir are supplied")
        return None, Path(data_csv), Path(output_dir), BASE_SEED if seed is None else int(seed)

    fold = validate_fold(fold)
    model = MODEL_SPEC_BY_KEY[model_key]
    resolved_csv = Path(data_csv) if data_csv else fold_data_csv(fold, output_root)
    resolved_output = (
        Path(output_dir) if output_dir else model_output_dir(fold, model, output_root)
    )
    resolved_seed = fold_seed(fold) if seed is None else int(seed)
    return fold, resolved_csv, resolved_output, resolved_seed


def resolve_model_cv_runs(base_args: Any, *, model_key: str) -> list[Any]:
    """Build independent argument namespaces for a model's requested folds."""
    data_csv = getattr(base_args, "csv", "")
    output_dir = getattr(base_args, "output_dir", "")
    has_explicit_paths = bool(data_csv or output_dir)
    if has_explicit_paths and not (data_csv and output_dir):
        raise ValueError("--csv and --output-dir must be supplied together")

    fold = getattr(base_args, "fold", None)
    folds = getattr(base_args, "folds", None)
    if has_explicit_paths and fold is None and folds is None:
        requested: list[int | None] = [None]
    else:
        requested = resolve_requested_folds(fold=fold, folds=folds)
        if has_explicit_paths and len(requested) != 1:
            raise ValueError("Explicit --csv/--output-dir can only be used with one fold")

    output_root = Path(getattr(base_args, "output_root", CV_OUTPUT_DIR)).expanduser().resolve()
    runs: list[Any] = []
    for requested_fold in requested:
        run_args = copy.copy(base_args)
        resolved_fold, resolved_csv, resolved_output, resolved_seed = resolve_training_inputs(
            fold=requested_fold,
            data_csv=data_csv,
            output_dir=output_dir,
            model_key=model_key,
            seed=getattr(base_args, "seed", None),
            output_root=output_root,
        )
        run_args.fold = resolved_fold
        run_args.folds = None
        run_args.csv = str(resolved_csv)
        run_args.output_dir = str(resolved_output)
        run_args.output_root = output_root
        run_args.seed = resolved_seed
        runs.append(run_args)
    return runs


def run_model_cross_validation(
    base_args: Any,
    *,
    model_key: str,
    train_fold: Callable[[Any], None],
) -> None:
    """Run all requested folds inside the modality's own training script."""
    if model_key not in MODEL_SPEC_BY_KEY:
        raise KeyError(f"Unknown model key: {model_key}")
    try:
        runs = resolve_model_cv_runs(base_args, model_key=model_key)
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    spec = MODEL_SPEC_BY_KEY[model_key]
    for index, run_args in enumerate(runs, start=1):
        data_csv = Path(run_args.csv)
        output_dir = Path(run_args.output_dir)
        best_path = output_dir / spec.best_checkpoint
        metrics_path = output_dir / "metrics.json"
        fold_label = run_args.fold if run_args.fold is not None else "custom"

        if getattr(base_args, "skip_existing", False) and best_path.exists() and metrics_path.exists():
            print(f"[SKIP] model={model_key} fold={fold_label}: {best_path}")
            continue
        if not data_csv.exists():
            raise FileNotFoundError(
                f"Fold data not found: {data_csv}. Run 01_make_cv_splits.py first."
            )

        print("\n" + "=" * 80)
        print(
            f"CROSS-VALIDATION model={model_key} fold={fold_label} "
            f"({index}/{len(runs)}) seed={run_args.seed}"
        )
        print("=" * 80)
        train_fold(run_args)
        gc.collect()
