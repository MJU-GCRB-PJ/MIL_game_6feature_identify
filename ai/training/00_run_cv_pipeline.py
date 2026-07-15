"""Run the paper-aligned split, training, ensemble, and summary stages."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from cv_config import (
    CV_OUTPUT_DIR,
    EXPECTED_SAMPLE_COUNT,
    MODEL_SPECS,
    N_FOLDS,
    PROJECT_ROOT,
    TRAINING_DIR,
)


DEFAULT_FEATURE_ROOT = Path(
    os.environ.get("MIL_FEATURE_ROOT", "/data/feature_extraction")
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "data_list.xlsx")
    parser.add_argument(
        "--feature-index",
        type=Path,
        default=DEFAULT_FEATURE_ROOT / "feat_index.csv",
    )
    parser.add_argument("--output-root", type=Path, default=CV_OUTPUT_DIR)
    parser.add_argument("--expected-samples", type=int, default=EXPECTED_SAMPLE_COUNT)
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(range(1, N_FOLDS + 1))
    )
    parser.add_argument("--only", nargs="+", default=None, help="Train only these model keys.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--val-num-workers", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-splits", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print(f"[PIPELINE] {' '.join(command)}")
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    python = str(args.python)
    output_root = args.output_root.expanduser().resolve()

    model_by_key = {model.key: model for model in MODEL_SPECS}
    if args.only:
        unknown = sorted(set(args.only) - set(model_by_key))
        if unknown:
            raise ValueError(
                f"Unknown model keys: {unknown}. Available: {sorted(model_by_key)}"
            )
        selected_models = [model_by_key[key] for key in args.only]
    else:
        selected_models = list(MODEL_SPECS)

    if not args.skip_splits:
        run(
            [
                python,
                str(TRAINING_DIR / "01_make_cv_splits.py"),
                "--manifest",
                str(args.manifest),
                "--feature-index",
                str(args.feature_index),
                "--output-root",
                str(output_root),
                "--expected-samples",
                str(args.expected_samples),
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_training:
        for model in selected_models:
            command = [
                python,
                str(TRAINING_DIR / model.script_name),
                "--folds",
                *[str(fold) for fold in args.folds],
                "--output-root",
                str(output_root),
            ]
            if args.skip_existing:
                command.append("--skip-existing")
            if args.epochs is not None:
                command.extend(["--epochs", str(args.epochs)])
            if args.num_workers is not None:
                command.extend(["--num-workers", str(args.num_workers)])
            if args.val_num_workers is not None:
                command.extend(["--val-num-workers", str(args.val_num_workers)])
            run(command, dry_run=args.dry_run)

    if not args.skip_ensemble:
        command = [
            python,
            str(TRAINING_DIR / "08_ensemble.py"),
            "--folds",
            *[str(fold) for fold in args.folds],
            "--output-root",
            str(output_root),
        ]
        if args.skip_existing:
            command.append("--skip-existing")
        run(command, dry_run=args.dry_run)

    if not args.skip_summary:
        run(
            [
                python,
                str(TRAINING_DIR / "09_summarize_cv_results.py"),
                "--output-root",
                str(output_root),
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
