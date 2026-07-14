from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


N_FOLDS = 5
BASE_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
	key: str
	output_dir_name: str
	script_name: str
	best_ckpt: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
	ModelSpec("vision", "vision_mil", "02_vision_mil.py", "best_vision_mil_model.pth"),
	ModelSpec("original_audio", "original_audio_mil", "03_original-audio_mil.py", "best_original_audio_mil_model.pth"),
	ModelSpec("vocal_audio", "vocal_audio_mil", "04_vocal-audio_mil.py", "best_vocal_audio_mil_model.pth"),
	ModelSpec("non_vocal_audio", "non_vocal_audio_mil", "05_non-vocal-audio_mil.py", "best_non_vocal_audio_mil_model.pth"),
	ModelSpec("ocr", "ocr_mil", "06_ocr_mil.py", "best_ocr_mil_model.pth"),
	ModelSpec("stt", "stt_mil", "07_stt_mil.py", "best_stt_mil_model.pth"),
)


def get_paths() -> tuple[Path, Path, Path]:
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent.parent
	training_dir = repo_root / "ai" / "03_mil_training"
	output_root = script_dir / "outputs" / "kfold"
	return repo_root, training_dir, output_root


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run 5-fold MIL training sequentially.")
	p.add_argument("--folds", type=int, nargs="*", default=list(range(1, N_FOLDS + 1)), help="Fold numbers to run")
	p.add_argument("--only", type=str, nargs="*", default=None, help="Model keys to run, e.g. vision stt")
	p.add_argument("--skip-existing", action="store_true", help="Skip a model if its best checkpoint and metrics.json already exist")
	p.add_argument("--epochs", type=int, default=None, help="Override epochs for all model scripts")
	p.add_argument("--num-workers", type=int, default=None, help="Override train DataLoader workers for all model scripts")
	p.add_argument("--val-num-workers", type=int, default=None, help="Override val DataLoader workers for all model scripts")
	p.add_argument("--base-seed", type=int, default=BASE_SEED)
	p.add_argument("--python", type=str, default=sys.executable)
	return p.parse_args()


def select_models(only: list[str] | None) -> list[ModelSpec]:
	if not only:
		return list(MODEL_SPECS)
	allowed = {m.key: m for m in MODEL_SPECS}
	selected: list[ModelSpec] = []
	for key in only:
		if key not in allowed:
			raise KeyError(f"Unknown model key: {key}. Available: {sorted(allowed)}")
		selected.append(allowed[key])
	return selected


def ensure_fold_csv(output_root: Path, fold_no: int) -> Path:
	csv_path = output_root / f"{fold_no}_fold" / "feat_data-ration_list.csv"
	if not csv_path.exists():
		raise FileNotFoundError(
			f"Fold CSV not found: {csv_path}\n"
			"Run 10_make_5fold_split.py first."
		)
	return csv_path


def run_model(
	*,
	python_bin: str,
	training_dir: Path,
	output_root: Path,
	fold_no: int,
	spec: ModelSpec,
	base_seed: int,
	epochs: int | None,
	num_workers: int | None,
	val_num_workers: int | None,
	skip_existing: bool,
) -> None:
	fold_dir = output_root / f"{fold_no}_fold"
	csv_path = ensure_fold_csv(output_root, fold_no)
	model_out_dir = fold_dir / spec.output_dir_name
	best_path = model_out_dir / spec.best_ckpt
	metrics_path = model_out_dir / "metrics.json"

	if skip_existing and best_path.exists() and metrics_path.exists():
		print(f"[SKIP] fold={fold_no} model={spec.key} ({best_path})")
		return

	script_path = training_dir / spec.script_name
	if not script_path.exists():
		raise FileNotFoundError(f"Training script not found: {script_path}")

	cmd = [
		python_bin,
		str(script_path),
		"--csv",
		str(csv_path),
		"--output-dir",
		str(model_out_dir),
		"--seed",
		str(int(base_seed) + int(fold_no)),
	]
	if epochs is not None:
		cmd.extend(["--epochs", str(int(epochs))])
	if num_workers is not None:
		cmd.extend(["--num-workers", str(int(num_workers))])
	if val_num_workers is not None:
		cmd.extend(["--val-num-workers", str(int(val_num_workers))])

	print("\n" + "=" * 80)
	print(f"RUN fold={fold_no}/{N_FOLDS} model={spec.key}")
	print(" ".join(cmd))
	print("=" * 80)
	subprocess.run(cmd, cwd=str(training_dir), check=True)


def main() -> None:
	args = parse_args()
	_, training_dir, output_root = get_paths()
	models = select_models(args.only)

	for fold_no in args.folds:
		if int(fold_no) < 1 or int(fold_no) > N_FOLDS:
			raise ValueError(f"fold must be in 1..{N_FOLDS}, got {fold_no}")
		for spec in models:
			run_model(
				python_bin=str(args.python),
				training_dir=training_dir,
				output_root=output_root,
				fold_no=int(fold_no),
				spec=spec,
				base_seed=int(args.base_seed),
				epochs=args.epochs,
				num_workers=args.num_workers,
				val_num_workers=args.val_num_workers,
				skip_existing=bool(args.skip_existing),
			)


if __name__ == "__main__":
	main()
