# RunPod VLM Dry Run

Test Linux-specific VLM models (GPTQ/AWQ) and gated models (Llama/MiniCPM) on RunPod infrastructure.

## What Gets Tested

1. **Qwen2.5-VL GPTQ models** (require Linux)
   - `hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4` (4GB VRAM)
   - `hfl/Qwen2.5-VL-7B-Instruct-GPTQ-Int4` (8GB VRAM)

2. **Qwen2.5-VL AWQ models** (require Linux)
   - `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` (20GB VRAM)

3. **Gated models** (require HuggingFace token)
   - `meta-llama/Llama-3.2-11B-Vision-Instruct` (26GB VRAM - skipped on 24GB GPU)
   - `openbmb/MiniCPM-V-2_6` (10GB VRAM)

## Prerequisites

1. RunPod API key in `.env.local`:
   ```bash
   RUNPOD_API_KEY=your_key_here
   ```

2. SSH key at `~/.ssh/id_rsa` (for SCP upload/download)

3. Python dependencies:
   ```bash
   pip install runpod python-dotenv
   ```

4. HuggingFace token (for gated models - will be prompted if needed)

## Usage

Run the dry run:
```bash
python experiments/runpod/run_dry_run.py
```

The script will:
1. Create a RunPod pod with RTX 4090 (24GB VRAM)
2. Install dependencies:
   - PyTorch, Transformers, Accelerate, BitsAndBytes
   - GPTQ Model, AutoAWQ, Optimum
3. Upload test script and test data
4. Run inference tests on all models
5. Download results to `outputs/runpod_dry_run/results.json`
6. Destroy the pod

## Cost

- **GPU**: RTX 4090 (24GB) @ ~$0.40/hr
- **Estimated runtime**: 60 minutes
- **Estimated cost**: ~$0.40

## Output

Results saved to `outputs/runpod_dry_run/results.json`:
```json
[
  {
    "model_name": "qwen2.5-vl-3b-gptq",
    "model_id": "hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4",
    "min_vram_gb": 4,
    "download_success": true,
    "inference_success": true,
    "load_time_seconds": 45.2,
    "inference_time_seconds": 3.8,
    "error_message": null
  },
  ...
]
```

## Files

- `run_dry_run.py`: Main orchestration script (runs locally)
- `remote_test.py`: Test script uploaded to pod (runs on RunPod)
- `runpod_instance.py`: RunPod SDK wrapper
- `test_subset.jsonl`: Test data (uploaded from `outputs/evaluation/`)

## Troubleshooting

### SSH Connection Failed
```bash
# Verify SSH key exists
ls -la ~/.ssh/id_rsa

# Test SSH connection manually
ssh -i ~/.ssh/id_rsa -p <port> root@<host>
```

### SCP Upload Failed
- Check firewall settings
- Verify RunPod pod has public SSH enabled
- Try increasing timeout in `scp_upload()`

### Model Download Failed
- Check HuggingFace token for gated models
- Verify disk space (80GB allocated)
- Check network connectivity

### OOM Errors
- Reduce batch size (already at 1)
- Use smaller models
- Upgrade to GPU with more VRAM (A100 40GB/80GB)

## Comparison with vast.ai

| Feature | RunPod | vast.ai |
|---------|--------|---------|
| API | Official SDK | SSH/REST |
| Setup | `runpod.create_pod()` | Search + rent + setup |
| Pricing | ~$0.40/hr RTX 4090 | ~$0.30/hr RTX 4090 |
| Reliability | High | Variable |
| Ease of use | Very easy | Manual |

RunPod is more expensive but significantly easier to use with official SDK support.
