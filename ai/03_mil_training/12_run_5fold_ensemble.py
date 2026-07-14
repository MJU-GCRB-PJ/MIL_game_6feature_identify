from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


N_FOLDS = 5


def get_paths() -> tuple[Path, Path, Path]:
	script_dir = Path(__file__).resolve().parent
	repo_root = script_dir.parent.parent
	ensemble_py = repo_root / "ai" / "03_mil_training" / "08_ensemble.py"
	output_root = script_dir / "outputs" / "kfold"
	return repo_root, ensemble_py, output_root


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Run ensemble evaluation for each 5-fold output directory.")
	p.add_argument("--folds", type=int, nargs="*", default=list(range(1, N_FOLDS + 1)))
	p.add_argument("--skip-existing", action="store_true", help="Skip fold if ensemble_results.xlsx already exists")
	return p.parse_args()


def load_ensemble_module(ensemble_py: Path) -> ModuleType:
	if not ensemble_py.exists():
		raise FileNotFoundError(f"08_ensemble.py not found: {ensemble_py}")
	spec = importlib.util.spec_from_file_location("kfold_ensemble_mod", ensemble_py)
	if spec is None or spec.loader is None:
		raise RuntimeError(f"Failed to load module spec: {ensemble_py}")
	module = importlib.util.module_from_spec(spec)
	sys.modules["kfold_ensemble_mod"] = module
	spec.loader.exec_module(module)
	return module


def configure_fold(module: ModuleType, output_root: Path, fold_no: int) -> None:
	fold_dir = output_root / f"{fold_no}_fold"
	data_csv = fold_dir / "feat_data-ration_list.csv"
	if not data_csv.exists():
		raise FileNotFoundError(f"Fold CSV not found: {data_csv}")

	module.TARGET_FOLDER = f"{fold_no}_fold"
	module.OUTPUT_BASE = fold_dir
	module.DATA_CSV = data_csv
	module.ENSEMBLE_DIR = fold_dir / "ensemble"


def main() -> None:
	args = parse_args()
	_, ensemble_py, output_root = get_paths()
	module = load_ensemble_module(ensemble_py)

	for fold_no in args.folds:
		if int(fold_no) < 1 or int(fold_no) > N_FOLDS:
			raise ValueError(f"fold must be in 1..{N_FOLDS}, got {fold_no}")

		fold_no = int(fold_no)
		excel_path = output_root / f"{fold_no}_fold" / "ensemble" / "ensemble_results.xlsx"
		if args.skip_existing and excel_path.exists():
			print(f"[SKIP] fold={fold_no} ensemble ({excel_path})")
			continue

		print("\n" + "=" * 80)
		print(f"RUN ensemble fold={fold_no}/{N_FOLDS}")
		print("=" * 80)
		configure_fold(module, output_root, fold_no)
		module.main()


if __name__ == "__main__":
	main()
