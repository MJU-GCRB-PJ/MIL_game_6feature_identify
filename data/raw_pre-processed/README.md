# Shared Raw and Preprocessed Data

Extract the distributed archives into this directory while preserving this layout:

```text
data/raw_pre-processed/
  game_play_raw_video/  Original gameplay MP4 files
  ocr_results/           Supplied per-video OCR JSONL files
  stt_results/           Supplied per-video STT JSON files
```

The preprocessing scripts create `frames/`, `audio/`, `logs/`, `index.csv`, and
`index.xlsx` in the same directory. OCR and STT are supplied to avoid requiring
other researchers to rerun the expensive model-dependent stages.
