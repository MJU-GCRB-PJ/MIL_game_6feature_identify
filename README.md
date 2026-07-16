# Identifying Game Rating Content Descriptors in Gameplay Videos via Multimodal Multiple Instance Learning

This project implements the multimodal multiple instance learning (MIL) framework presented in the paper. It treats each gameplay video as a bag of 8-second vision, audio, and text instances, learns six GRAC Content Descriptors from game-level labels, and combines vision, original audio, vocal audio, non-vocal audio, OCR, and STT through five-fold cross-validation. In experiments on 663 games, the all-modality Weighted Soft Voting ensemble achieved a mean validation Macro AUC of 0.884 +/- 0.067 while producing candidate evidence segments for human review.

The purpose of this repository is to provide a reproducible end-to-end workflow for dataset preparation, multimodal feature extraction, fold-aware MIL training, ensemble evaluation, and result analysis, supporting researchers who want to reproduce or extend the study and helping human reviewers prioritize relevant segments in long gameplay videos rather than replacing their final rating decisions.

## 1. Requirements and Conda Environment

The full experiment is intended for Linux with an NVIDIA CUDA GPU. CPU-only execution is useful for manifest checks and tests, but preprocessing, feature extraction, 30 fold/model training runs, and fold ensembles are computationally expensive. The machine also needs `git`, `tar`, `sha256sum`, `ffmpeg`, and enough storage for the downloaded archives, extracted videos, frames, audio, features, and checkpoints.

Run every command below from the repository root:

```bash
cd /path/to/MIL_game_6feature_identify
export PROJECT_ROOT="$(pwd)"
```

Create and activate the supplied Conda environment:

```bash
conda env create --file environment.yml
conda activate mil-game-6feature
```

If the environment already exists, update it instead:

```bash
conda env update --name mil-game-6feature --file environment.yml --prune
conda activate mil-game-6feature
```

Verify the main runtime dependencies before starting expensive work:

```bash
python --version
ffmpeg -version | head -n 1
python -c "import torch, torchvision, torchaudio; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('torchaudio:', torchaudio.__version__); print('PyTorch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pip check
```

The environment pins `torch==2.10.0`, `torchvision==0.25.0`, and `torchaudio==2.10.0` as one compatible set. Do not upgrade only one member of this set. If a different PyTorch build is required for the machine, replace all three versions together and run `python -m pip check` again.

If CUDA is expected but reported as unavailable, install the PyTorch build that matches the machine's NVIDIA driver and CUDA runtime before continuing.

Run the complete import smoke test once after creating or updating the environment. It covers preprocessing, feature extraction, training, analysis, and the optional OCR/STT regeneration commands:

```bash
python - <<'PY'
import addict
import cv2
import dash
import dash_bootstrap_components
import easydict
import einops
import faster_whisper
import kaleido
import librosa
import matplotlib
import natsort
import numpy
import openpyxl
import pandas
import plotly
import scipy
import sentence_transformers
import sklearn
import soundfile
import soxr
import torch
import torchaudio
import torchvision
import tqdm
import transformers
import umap
import xgboost
import yt_dlp
import demucs

from PIL import Image
from transformers import (
    ASTModel,
    AutoFeatureExtractor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    VivitImageProcessor,
    VivitModel,
)

print("Environment import check: OK")
PY
```

The analysis dashboard's **Save PNG** action uses Plotly and Kaleido. Kaleido 1.x requires Chrome or Chromium in addition to the Python package. If neither browser is installed on the machine, install a compatible Chrome build once after activating the environment:

```bash
plotly_get_chrome
```

The distributed STT results can be used without regenerating them. When running `5_stt.py` on a machine whose CUDA libraries are not compatible with CTranslate2, use `--device cpu`; all other pipeline stages can continue to use the PyTorch CUDA device.

## 2. Canonical Dataset Manifest

`data/data_list.xlsx` is the single source of truth for the 663 sample identities, metadata, and labels. Do not replace it with a feature index or a locally edited spreadsheet. Every generated index is joined by `file_name`.

The source label columns are normalized as follows:

| `data_list.xlsx` column | Model column |
| --- | --- |
| `sexuality` | `sexual_content` |
| `violence` | `violence` |
| `fear/horror/threatening` | `fear` |
| `language` | `inappropriate_language` |
| `alcohol/tobacco/drug` | `drugs` |
| `crime/anti-societal or anti-governmental messages` | `crime` |

The gambling label remains in generated indexes for reference but is not one of the six target classes used by the supplied MIL models.

## 3. Downloading and Extracting the Shared Data

### 3.1 Request access and download

Open [6feature.newlearn.ai.kr](https://6feature.newlearn.ai.kr/). The first page asks for an authorization code. Request a code from the corresponding author or one of the paper's authors, including a brief description of the intended research use. A code is issued after the request has been reviewed.

Enter the issued code on the website to open the dataset download page. The authorization code is intended only for the approved researcher and must not be redistributed. Download every file shown on the page into one local directory, wait for all downloads to finish, and do not rename any file. In the commands below, `ARCHIVE_DIR` is this download directory.

### 3.2 Required release files

The directory must contain:

```text
SHA256SUMS.txt
game_play_raw_video.parts
game_play_raw_video.tar.part-0000
game_play_raw_video.tar.part-0001
...
game_play_raw_video.tar.part-0036
ocr_results.tar
stt_results.tar
```

There are exactly 37 gameplay parts. Parts `0000` through `0035` are 10 GiB each, and `0036` is the final smaller part. Together they contain one ordinary, uncompressed TAR stream of approximately 368 GiB. `ocr_results.tar` and `stt_results.tar` are also ordinary TAR files, not `tar.gz` files.

Keep all 37 parts and `game_play_raw_video.parts` in the same directory. Do not rename individual parts. If the archives and extracted videos are retained on the same filesystem, roughly 750 GiB is needed for those two copies alone. Frames, audio, and extracted features require substantial additional storage.

### 3.3 Verify and extract the downloaded archives

The following restoration script is based on the script used to prepare and verify the release. It performs these operations in order:

1. Verifies every downloaded file against `SHA256SUMS.txt`.
2. Confirms that every gameplay split listed in `game_play_raw_video.parts` exists.
3. Extracts the OCR and STT archives.
4. Concatenates the gameplay parts in manifest order and streams the resulting TAR directly into the destination directory.

Create `restore_paper_data.sh` with the following contents:

```bash
#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 ARCHIVE_DIR DEST" >&2
    exit 2
fi

ARCHIVE_DIR="$1"
DEST="$2"

if [[ ! -d "$ARCHIVE_DIR" ]]; then
    echo "ERROR: archive directory not found: $ARCHIVE_DIR" >&2
    exit 1
fi

mkdir -p "$DEST"
ARCHIVE_DIR="$(cd "$ARCHIVE_DIR" && pwd)"
DEST="$(cd "$DEST" && pwd)"

cd "$ARCHIVE_DIR"
sha256sum --check SHA256SUMS.txt

check_parts() {
    local dir="$1"
    local expected actual part
    local -a listed_parts

    mapfile -t listed_parts < "$dir.parts"
    expected="${#listed_parts[@]}"
    actual="$(find . -maxdepth 1 -type f -name "$dir.tar.part-*" | wc -l | tr -d ' ')"

    if [[ "$expected" -eq 0 || "$expected" != "$actual" ]]; then
        echo "ERROR: split files are incomplete for $dir." >&2
        echo "Expected: $expected" >&2
        echo "Found:    $actual" >&2
        exit 1
    fi

    for part in "${listed_parts[@]}"; do
        if [[ -z "$part" || ! -f "$part" ]]; then
            echo "ERROR: listed split file not found: $part" >&2
            exit 1
        fi
    done
}

extract_split_tar_stream() {
    local dir="$1"
    local -a listed_parts

    mapfile -t listed_parts < "$dir.parts"
    cat -- "${listed_parts[@]}" \
        | tar --extract --file=- --directory="$DEST" --no-same-owner
}

check_parts game_play_raw_video
tar --extract --file=stt_results.tar --directory="$DEST" --no-same-owner
tar --extract --file=ocr_results.tar --directory="$DEST" --no-same-owner
extract_split_tar_stream game_play_raw_video

echo "Restore completed: $DEST"
```

Run it from the repository root, supplying the directory that contains the downloads and the required project data destination:

```bash
cd /path/to/MIL_game_6feature_identify
export PROJECT_ROOT="$(pwd)"
export ARCHIVE_DIR="/path/to/downloaded/6feature-identify"

chmod +x restore_paper_data.sh
./restore_paper_data.sh \
    "$ARCHIVE_DIR" \
    "$PROJECT_ROOT/data/raw-pre-processed"
```

The checksum pass reads the complete dataset and can take some time. Every checksum must report `OK`. If a file is missing or a checksum reports `FAILED`, download that file again and rerun the script. Do not use the gzip option (`tar -z`) for these archives, and do not extract gameplay parts individually. Only `part-0000` contains the beginning of the TAR stream. The script streams all parts directly into `tar`, so it does not create an additional 368 GiB `game_play_raw_video.tar` file.

### 3.4 Verify the restored project layout

The three released directories must be placed under `data/raw-pre-processed/` in this repository. Verify the extracted file counts:

```bash
export RAW_DATA_DIR="$PROJECT_ROOT/data/raw-pre-processed"

test "$(find "$RAW_DATA_DIR/game_play_raw_video" -maxdepth 1 -type f -name '*.mp4' | wc -l)" -eq 663
test "$(find "$RAW_DATA_DIR/ocr_results" -maxdepth 1 -type f -name '*.jsonl' | wc -l)" -eq 663
test "$(find "$RAW_DATA_DIR/stt_results" -maxdepth 1 -type f -name '*.json' | wc -l)" -eq 663
```

The resulting project layout must be:

```text
data/raw-pre-processed/
  game_play_raw_video/   663 MP4 files
  ocr_results/           663 JSONL files
  stt_results/           663 JSON files
```

For each `file_name` in `data/data_list.xlsx`, OCR and STT use the same stem. For example, `1_TormentedSouls_19.mp4` corresponds to `1_TormentedSouls_19.jsonl` and `1_TormentedSouls_19.json`.

The `audio` and `frames` directories are intentionally not distributed. They are generated from the gameplay MP4 files in the next stage. If raw data must live on another disk, set `MIL_RAW_PREPROCESSED_DIR` to that absolute directory before extraction and before every subsequent command.

## 4. Preprocessing

Activate the environment and restore the runtime paths in every new shell:

```bash
cd /path/to/MIL_game_6feature_identify
conda activate mil-game-6feature
export PROJECT_ROOT="$(pwd)"
export MIL_RAW_PREPROCESSED_DIR="$PROJECT_ROOT/data/raw-pre-processed"
export MIL_FEATURE_ROOT="/path/to/feature_extraction"
mkdir -p "$MIL_FEATURE_ROOT"
```

`MIL_FEATURE_ROOT` must point to a writable filesystem with enough free space. The paper server used `/data/feature_extraction`; another researcher may use a different absolute path. Keep the same value for feature extraction and training because generated indexes contain absolute feature paths.

Generate 4 FPS frames and original audio, then separate vocal and non-vocal audio:

```bash
python train_pipeline/preprocessing/1_raw_vid_preprocess.py
python train_pipeline/preprocessing/2_sound_separate.py
```

These scripts add:

```text
data/raw-pre-processed/
  frames/<video_id>/
  audio/<video_id>/original.wav
  audio/<video_id>/vocal.wav
  audio/<video_id>/non-vocal.wav
  logs/
```

The distributed OCR and STT files can be used directly. The following commands are optional and should only be run when intentionally regenerating those modalities:

```bash
python train_pipeline/preprocessing/3_screen_ocr.py
# Alternative OCR backend:
python train_pipeline/preprocessing/4_screen_olmocr.py
python train_pipeline/preprocessing/5_stt.py
```

Build the preprocessing index after frames and all three audio forms are ready:

```bash
python train_pipeline/preprocessing/6_index_maker.py
```

This writes `index.csv` and `index.xlsx` under `$MIL_RAW_PREPROCESSED_DIR`. Review the printed counts before continuing.

## 5. Feature Extraction

Each extractor reads the preprocessing index and writes per-sample artifacts under `$MIL_FEATURE_ROOT/<file_name>/`:

```bash
python train_pipeline/feature_extraction/1_vivit_feat.py
python train_pipeline/feature_extraction/2_ast_feat.py
python train_pipeline/feature_extraction/3_stt_embedding.py
python train_pipeline/feature_extraction/4_ocr_embedding.py
python train_pipeline/feature_extraction/5_feat_index_maker.py
```

The final command validates the one-to-one manifest join and writes:

```text
$MIL_FEATURE_ROOT/feat_index.csv
$MIL_FEATURE_ROOT/feat_index.xlsx
```

Do not move the feature directories after creating the index. If they are moved, regenerate `feat_index.csv` so that its absolute paths remain valid.

## 6. Five-Fold Cross-Validation Training

Five-fold cross-validation is the canonical experiment. The split is deterministic (`seed=42`), covers all 663 samples exactly once as validation data, and uses model seed `42 + fold`. First confirm the command graph without starting training:

```bash
python train_pipeline/training/00_run_cv_pipeline.py --dry-run
```

Run the complete split, six-modality training, fold ensemble, and final summary pipeline:

```bash
python train_pipeline/training/00_run_cv_pipeline.py --skip-existing
```

`--skip-existing` makes the command resumable after interruption. It does not skip work on a new run. The default feature index is `$MIL_FEATURE_ROOT/feat_index.csv`; it can also be supplied explicitly with `--feature-index /absolute/path/to/feat_index.csv`.

The stages can be run independently:

```bash
python train_pipeline/training/01_make_cv_splits.py
python train_pipeline/training/02_vision_mil.py --skip-existing
python train_pipeline/training/03_original-audio_mil.py --skip-existing
python train_pipeline/training/04_vocal-audio_mil.py --skip-existing
python train_pipeline/training/05_non-vocal-audio_mil.py --skip-existing
python train_pipeline/training/06_ocr_mil.py --skip-existing
python train_pipeline/training/07_stt_mil.py --skip-existing
python train_pipeline/training/08_ensemble.py --skip-existing
python train_pipeline/training/09_summarize_cv_results.py
```

Each script from `02` through `07` owns its complete five-fold training loop; `08_ensemble.py` likewise owns its five-fold ensemble loop. No separate fold runner is required. Use `--fold` for one fold or `--folds` for a subset while debugging:

```bash
python train_pipeline/training/02_vision_mil.py --fold 1
python train_pipeline/training/03_original-audio_mil.py --folds 1 3 5
python train_pipeline/training/08_ensemble.py --fold 1
```

Every modality model consumes all available instances in each bag. The ensemble evaluates all 63 non-empty modality subsets. Weighted Soft Voting weights are optimized on each fold's validation partition, matching the paper experiment.

Generated files use this layout:

```text
train_pipeline/training/outputs/cv/
  fold_assignments.csv
  fold_assignments.xlsx
  fold_split_summary.csv
  fold_split_summary.xlsx
  fold_01/
    data.csv
    data.xlsx
    vision_mil/
    original_audio_mil/
    vocal_audio_mil/
    non_vocal_audio_mil/
    ocr_mil/
    stt_mil/
    ensemble/
  ...
  fold_05/
  summary/
    model_summary.xlsx
    ensemble_summary.xlsx
    summary.json
```

The summary reports per-fold values, five-fold means, sample standard deviations, and average modality weights where weights are defined.

## 7. Analysis

Analysis is fold-specific and requires completed checkpoints and ensemble outputs for the selected fold:

```bash
python train_pipeline/analysis/01_visualize_result.py --fold 1
python train_pipeline/analysis/02_evaluate.py --fold 1
python train_pipeline/analysis/03_ablation.py --fold 1 --ensemble-row 2
```

Use `--all-best` with `02_evaluate.py` to evaluate every saved ensemble criterion instead of only `best_val_macro_auc.pth`.

## 8. Verification and Common Failures

Run the manifest, schema, split, and summary regression tests before a full experiment:

```bash
python -m unittest discover -s tests -v
```

Expected split sizes are `133, 133, 133, 132, 132`. The tests also verify the class-positive counts and deterministic assignment used by the paper.

Common failures:

- `sha256sum` reports `FAILED`: re-download that exact archive or part.
- `tar` reports `Unexpected EOF`: one or more gameplay parts are missing, corrupt, renamed, or out of order. Re-run the checksum command.
- `feat_index.csv` is not found: activate the environment, export the same `MIL_FEATURE_ROOT` used during feature extraction, and rerun `5_feat_index_maker.py`.
- CUDA is unavailable: verify `nvidia-smi`, the NVIDIA driver, and the installed PyTorch CUDA build before running preprocessing or model scripts.
- A generated index reports fewer than 663 samples: inspect the missing-file flags and complete that preprocessing or feature-extraction stage before training.

## 9. Repository Map and Generated Files

The `train_pipeline` package uses role-based `snake_case` directory names, which keeps imports and filesystem paths stable as the pipeline evolves. Numeric prefixes are retained only on executable scripts to show the required order within each stage.

The main source locations are:

| Path | Role |
| --- | --- |
| `data/data_list.xlsx` | Authoritative sample metadata and labels |
| `train_pipeline/project_paths.py` | Shared filesystem paths and environment-variable overrides |
| `train_pipeline/data_manifest.py` | Manifest schema validation and label normalization |
| `train_pipeline/preprocessing/` | Frame/audio preprocessing, optional OCR/STT, and preprocessing index generation |
| `train_pipeline/feature_extraction/` | Six-modality feature extraction and feature index generation |
| `train_pipeline/training/` | Deterministic five-fold splitting, MIL training, fold ensembles, and cross-fold summaries |
| `train_pipeline/analysis/` | Interactive visualization, evaluation workbooks, and optional fold ablation |

Generated preprocessing data stays under `$MIL_RAW_PREPROCESSED_DIR`, feature data stays under `$MIL_FEATURE_ROOT`, and experiment output stays under `train_pipeline/training/outputs/cv/`. Generated indexes can contain machine-local absolute paths and should be regenerated after moving data.

Raw media, generated frames/audio/features, checkpoints, model caches, credentials, and experiment outputs are intentionally excluded from version control. Only source code, environment definitions, documentation, tests, and the canonical `data/data_list.xlsx` manifest should be committed.
