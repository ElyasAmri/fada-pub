# Fetal Ultrasound VLM Training Dataset

Sonographer ground truth (GT) data for fine-tuning and evaluating Vision-Language Models on fetal ultrasound images.

## Contents

- `gt_train.jsonl` - GT training data (121,240 samples, 8 questions x ~15K images)
- `gt_val.jsonl` - GT validation data (15,152 samples)
- `gt_test.jsonl` - GT test data (15,088 samples)
- `gt_train_sharegpt.jsonl` - GT training in ShareGPT format (for frameworks that require it)
- `gt_val_sharegpt.jsonl` - GT validation in ShareGPT format

## Data Source

All files are derived from sonographer annotations in `data/Fetal Ultrasound Annotations Normalized.xlsx`
using dataset splits from `data/dataset_splits.json` (80/10/10 train/val/test).

Regenerate with:
```bash
./venv/Scripts/python.exe experiments/framework_comparison/generate_gt_training_data.py
```

Gemini pseudo-labels and legacy training formats have been archived to `data/archive/`.

## Image Paths

JSONL files reference images as relative paths (e.g., `Abdomen/Abdomen_001.png`).
Training scripts use `--data-root` to prepend the correct base directory at runtime.
