# MIL Game 6-Feature Identification

This repository reproduces the six-modality multiple-instance learning pipeline
used for game-content identification. The six input modalities are vision,
original audio, vocal audio, non-vocal audio, OCR text, and STT text.

## 1. Environment

Run all commands from the repository root.

```bash
conda env create -f environment.yml
conda activate mil-game-6feature
```

Install a CUDA-compatible PyTorch build when GPU acceleration is required.

## 2. Canonical Dataset Manifest

`data/data_list.xlsx` is the single source of truth for sample identity,
metadata, and labels. Every pipeline index is generated from its `file_name`
column. The source label columns are normalized as follows:

| `data_list.xlsx` column | Model column |
| --- | --- |
| `sexuality` | `sexual_content` |
| `violence` | `violence` |
| `fear/horror/threatening` | `fear` |
| `language` | `inappropriate_language` |
| `alcohol/tobacco/drug` | `drugs` |
| `crime/anti-societal or anti-governmental messages` | `crime` |

The gambling label remains in generated indexes for reference but is not one
of the six target classes used by the supplied MIL training scripts.

## 3. Shared Data Layout

Extract the distributed gameplay, OCR, and STT archives under the repository
root while preserving this layout:

```text
data/raw_pre-processed/
  game_play_raw_video/   Original gameplay MP4 files
  ocr_results/            One `<video_id>.jsonl` file per video
  stt_results/            One `<video_id>.json` file per video
```

`video_id` is the stem of `file_name` in `data/data_list.xlsx`. For example,
`1_TormentedSouls_19.mp4` uses `1_TormentedSouls_19.jsonl` and
`1_TormentedSouls_19.json`.

OCR and STT outputs are distributed because recreating them requires separate
large models and substantial processing time. Researchers can use the supplied
files directly and do not need to run the OCR or STT generation scripts.

The raw data root can be overridden with `MIL_RAW_PREPROCESSED_DIR`. Feature
outputs default to `/data/feature_extraction` as used by the paper pipeline and
can be overridden with `MIL_FEATURE_ROOT`.

## 4. Preprocessing

Create frames and audio from the supplied gameplay videos, then separate vocal
and non-vocal audio:

```bash
python ai/01_preprocess/1_raw_vid_preprocess.py
python ai/01_preprocess/2_sound_separate.py
```

The scripts write all generated data under `data/raw_pre-processed/`:

```text
frames/<video_id>/
audio/<video_id>/original.wav
audio/<video_id>/vocal.wav
audio/<video_id>/non-vocal.wav
logs/
```

Optional regeneration commands are available for OCR and STT:

```bash
python ai/01_preprocess/3_screen_ocr.py
# Alternative OCR backend:
python ai/01_preprocess/4_screen_olmocr.py
python ai/01_preprocess/5_stt.py
```

Build the preprocessing index after the distributed files are extracted and
the required frame/audio preprocessing is complete:

```bash
python ai/01_preprocess/6_index_maker.py
```

This writes `data/raw_pre-processed/index.csv` and `index.xlsx` directly from
`data/data_list.xlsx`.

## 5. Feature Extraction

Each extractor reads `data/raw_pre-processed/index.csv` and writes per-sample
artifacts under `/data/feature_extraction/<file_name>/`.

```bash
python ai/02_feature_extraction/1_vivit_feat.py
python ai/02_feature_extraction/2_ast_feat.py
python ai/02_feature_extraction/3_stt_embedding.py
python ai/02_feature_extraction/4_ocr_embedding.py
python ai/02_feature_extraction/5_feat_index_maker.py
```

The final command writes:

```text
/data/feature_extraction/feat_index.csv
/data/feature_extraction/feat_index.xlsx
```

## 6. MIL Training

Create the original balanced 80:20 train-validation split:

```bash
python ai/03_mil_training/01_train-validation_split.py
```

The split is written under `ai/03_mil_training/outputs/splits/`. Train the six
modality models with scripts `02_vision_mil.py` through `07_stt_mil.py`, then
run the ensemble and optional ablation:

```bash
python ai/03_mil_training/02_vision_mil.py
python ai/03_mil_training/03_original-audio_mil.py
python ai/03_mil_training/04_vocal-audio_mil.py
python ai/03_mil_training/05_non-vocal-audio_mil.py
python ai/03_mil_training/06_ocr_mil.py
python ai/03_mil_training/07_stt_mil.py
python ai/03_mil_training/08_ensemble.py
python ai/03_mil_training/09_ablation.py
```

## 7. Five-Fold Validation

The paper-aligned five-fold workflow is integrated into `03_mil_training`:

```bash
python ai/03_mil_training/10_make_5fold_split.py
python ai/03_mil_training/11_run_5fold_training.py --skip-existing
python ai/03_mil_training/12_run_5fold_ensemble.py --skip-existing
python ai/03_mil_training/13_final_validation.py
```

All fold splits, checkpoints, ensemble outputs, and final summaries are placed
under `ai/03_mil_training/outputs/kfold/`. The guarded notebook
`00_run_5fold_pipeline.ipynb` provides the same orchestration interactively.

## 8. Analysis

```bash
python ai/04_analysis/01_visualize_result.py
python ai/04_analysis/02_evaluate.py
```

Only the dashboard and evaluation scripts are retained. They read the standard
training output layout under `ai/03_mil_training/outputs/`.
