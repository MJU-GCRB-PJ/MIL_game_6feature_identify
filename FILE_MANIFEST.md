# File Manifest

## Canonical Data

- `data/data_list.xlsx`: authoritative sample metadata and labels.
- `data/raw_pre-processed/README.md`: extraction layout for shared gameplay,
  OCR, and STT archives.
- `data/raw_pre-processed/index.csv`: generated preprocessing index; ignored by
  Git because it contains machine-local absolute paths.

## Pipeline

- `ai/project_paths.py`: shared filesystem contract.
- `ai/data_manifest.py`: manifest validation and label normalization.
- `ai/01_preprocess/`: video, audio, optional OCR/STT, and index generation.
- `ai/02_feature_extraction/`: six-modality feature generation and feature index.
- `ai/03_mil_training/`: original split, six MIL trainers, ensemble, ablation,
  and integrated five-fold validation.
- `ai/04_analysis/01_visualize_result.py`: interactive result inspection.
- `ai/04_analysis/02_evaluate.py`: evaluation workbook generation.

## Generated Data

- All preprocessing output stays under `data/raw_pre-processed/`.
- Feature output stays under `/data/feature_extraction/` by default.
- Training output stays under `ai/03_mil_training/outputs/`.

Raw media, generated features, checkpoints, model caches, credentials, and
experiment outputs are intentionally excluded from version control.
