# API Models Testing

Tests for cloud-based VLM APIs (OpenAI GPT-5.2, Gemini, Grok, Vertex AI MedGemma) on fetal ultrasound images.

## Main Script

- `test_api_vlm.py` - Async API VLM evaluation with rate limiting

## Usage

```bash
# Test single model
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --models gemini --images-per-category 3

# Test all models
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --models all --images-per-category 3

# Quiet mode (minimal output)
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --models gemini --quiet

# Specify model variant
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --models gemini --gemini-model gemini-3-flash-preview
```

## Options

- `--models`: openai, gemini, grok, vertex-ai, vllm, or all
- `--images-per-category`: Number of images per category (default: 3)
- `--max-rpm`: Rate limit requests per minute (default: 60)
- `--max-concurrent`: Max concurrent requests (default: 10)
- `--quiet`: Minimal output mode
- `--openai-model`, `--gemini-model`, `--grok-model`: Specify model variants
- `--vertex-ai-model`, `--vertex-ai-project`, `--vertex-ai-endpoint`: Vertex AI options
- `--vllm-url`, `--vllm-model`: vLLM server options (for Vast.ai/local deployment)

## Results

Results are saved to `results/` directory:
- `vlm_results_*.json` - Raw test results
- `vlm_test_*.log` - Detailed log files
- `checkpoint_*.json` - Checkpoint files for resuming

## Checkpoint/Resume

The script auto-saves checkpoints every 10 images. If you hit RPD (rate/quota) limits:

```bash
# Resume Gemini run
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --models gemini --gemini-model gemini-3-flash-preview --split test --resume checkpoint_gemini_gemini-3-flash-preview.json

# Resume OpenAI run
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --models openai --split test --resume checkpoint_openai_gpt-5.2-chat-latest.json

# Resume Grok run
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --models grok --split test --resume checkpoint_grok_grok-4.json
```

## Vertex AI MedGemma

Requires a deployed MedGemma endpoint on Vertex AI Model Garden.

### Setup

1. Deploy MedGemma from Model Garden (creates private endpoint)
2. Set environment variables:
   ```bash
   export VERTEX_AI_PROJECT_ID="your-project-id"
   export VERTEX_AI_ENDPOINT_ID="your-endpoint-id"
   ```

### Usage

```bash
# Quick test
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models vertex-ai \
  --images-per-category 3 \
  --max-concurrent 3

# Full test set
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models vertex-ai \
  --split test

# Resume from checkpoint
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models vertex-ai \
  --split test \
  --resume checkpoint_vertex_ai_medgemma-27b-mm-it.json
```

## vLLM / Vast.ai (MedGemma)

Run MedGemma on Vast.ai with vLLM for local inference.

### Setup on Vast.ai

1. Rent an A100 80GB instance with PyTorch template (100GB disk)
2. SSH into the instance and run:
   ```bash
   pip install -q vllm huggingface_hub openai

   # Login to HuggingFace (required for gated model)
   huggingface-cli login --token YOUR_HF_TOKEN

   # Start vLLM with MedGemma 27B Multimodal
   export HF_TOKEN=YOUR_HF_TOKEN && \
   nohup python -m vllm.entrypoints.openai.api_server \
       --model google/medgemma-27b-it \
       --trust-remote-code \
       --max-model-len 4096 \
       --gpu-memory-utilization 0.90 \
       --port 8000 \
       --host 0.0.0.0 \
       > /tmp/vllm.log 2>&1 &

   # Monitor loading (takes 5-10 min)
   tail -f /tmp/vllm.log
   ```

3. Create SSH tunnel from local machine:
   ```bash
   ssh -p PORT root@IP -L 8000:127.0.0.1:8000 -N -f
   ```

### Usage

```bash
# Test with vLLM (after SSH tunnel is active)
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models vllm \
  --images-per-category 3 \
  --max-concurrent 3

# Full test set
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models vllm \
  --split test

# Resume from checkpoint
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py \
  --models vllm \
  --split test \
  --resume checkpoint_vllm_google_medgemma-27b-it.json
```

### MedGemma Models

| Model | Type | VRAM | Notes |
|-------|------|------|-------|
| `google/medgemma-27b-it` | Multimodal | ~54GB | Can see images |
| `google/medgemma-27b-text-it` | Text-only | ~54GB | Cannot see images |
| `google/medgemma-4b-it` | Multimodal | ~8GB | Can see images |

## Archive

Old results and scripts are in `results/archive/`.
