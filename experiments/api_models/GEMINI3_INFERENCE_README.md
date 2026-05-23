# Gemini 3 Flash Inference - Continuation Instructions

## Current Progress

- **Completed:** 6,313 / 19,019 images (33.2%)
- **Remaining:** 12,706 images (101,648 API requests)
- **Estimated completion:** ~10 daily runs

## Checkpoint Files

| File | Description |
|------|-------------|
| `checkpoint_gemini_gemini-3-flash-preview_shard0-19019.json` | **Active checkpoint** - use this to resume |
| `checkpoint_gemini_gemini-3-flash-preview_merged.json` | Merged checkpoint from all previous shards |
| `checkpoint_gemini_gemini-3-flash-preview.json` | Original checkpoint (2,497 images) |
| `checkpoint_gemini_gemini-3-flash-preview_shard*.json` | Historical shard checkpoints |

## Daily Run Command

Run this command once per day to process ~1,250 images (10,000 API requests):

```bash
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models gemini \
  --gemini-model gemini-3-flash-preview \
  --images-per-category 99999 \
  --start-index 0 \
  --end-index 19019 \
  --max-rpm 60 \
  --max-concurrent 10 \
  --max-requests 10000 \
  --resume checkpoint_gemini_gemini-3-flash-preview_shard0-19019.json
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--models gemini` | gemini | Use Gemini API |
| `--gemini-model` | gemini-3-flash-preview | Model identifier |
| `--images-per-category` | 99999 | Process all images (full dataset) |
| `--start-index` | 0 | Start of dataset range |
| `--end-index` | 19019 | End of dataset range |
| `--max-rpm` | 60 | Requests per minute limit |
| `--max-concurrent` | 10 | Concurrent requests |
| `--max-requests` | 10000 | Daily request limit (RPD protection) |
| `--resume` | checkpoint file | Resume from checkpoint |

## Check Progress

```bash
./venv/Scripts/python.exe -c "
import json
from pathlib import Path

cp_path = Path('experiments/api_models/results/checkpoint_gemini_gemini-3-flash-preview_shard0-19019.json')
with open(cp_path) as f:
    cp = json.load(f)

completed = len(cp.get('completed_images', {}))
total = 19019
print(f'Progress: {completed} / {total} ({completed/total*100:.1f}%)')
print(f'Remaining: {total - completed} images')
"
```

## Backup Checkpoints

To create a backup of all checkpoints:

```bash
./venv/Scripts/python.exe -c "
import zipfile
from pathlib import Path
from datetime import datetime

results_dir = Path('experiments/api_models/results')
backup_name = f'gemini3_flash_checkpoints_backup_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.zip'
backup_path = results_dir / backup_name

checkpoint_files = list(results_dir.glob('checkpoint_gemini_gemini-3-flash-preview*.json'))

with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for cp_file in checkpoint_files:
        zf.write(cp_file, cp_file.name)
        print(f'Added: {cp_file.name}')

print(f'Backup: {backup_path}')
"
```

## Troubleshooting

### Rate Limit Errors
If you hit daily rate limits (RPD), wait 24 hours and run again. The checkpoint saves progress automatically.

### Slow Start
The first batch after resuming may take longer to start (~1-2 minutes) due to model initialization.

### Script Appears Stuck
If no progress after 5 minutes, check the log file:
```bash
tail -20 experiments/api_models/results/vlm_test_*.log | tail -30
```

### Verify API is Working
Test with a small batch:
```bash
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models gemini \
  --gemini-model gemini-3-flash-preview \
  --images-per-category 1 \
  --max-rpm 60 \
  --max-concurrent 5
```

## Output Files

- **Checkpoints:** `experiments/api_models/results/checkpoint_*.json`
- **Logs:** `experiments/api_models/results/vlm_test_*.log`
- **Results:** `experiments/api_models/results/vlm_results_*.json`

## Notes

- Each image has 8 questions, so 10,000 requests = 1,250 images
- Processing rate: ~8-10 seconds per image with current settings
- Total time per daily run: ~3 hours for 1,250 images
