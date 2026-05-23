# Experiments

This directory contains all experimental work for the FADA project, including VLM testing, notebooks, and external model integrations.

## Active Directories

### api_models/
API-based VLM inference infrastructure.
- `test_api_vlm.py` - Main async parallel VLM testing (Gemini, OpenAI, Grok)
- `batch_gemini_annotation.py` - Batch API processing for Gemini
- `results/` - Checkpoints, batch files, and results (DO NOT DELETE)

### evaluation/
VLM evaluation framework.
- `evaluate_vlm.py` - Main VLM evaluation pipeline
- `embedding_scorer.py` - Embedding similarity scoring
- `config.py` - Evaluation configuration

### vastai/
Unified Vast.ai CLI for remote GPU operations.
- `runner.py` - Main CLI interface
- `templates/` - Training and inference templates
- See `./venv/Scripts/python.exe -m experiments.vastai --help`

### external_models/
Third-party model integrations.
- `FetalCLIP/` - MBZUAI FetalCLIP (zero-shot, probing, few-shot)

### fine_tuning/
Model fine-tuning scripts.
- `train_qwen3vl_lora.py` - LoRA fine-tuning for Qwen3-VL
- `train_vastai.py` - Training on Vast.ai infrastructure

### notebooks/
Jupyter notebooks for training.
- `blip2_training/` - BLIP-2 VQA training (5 categories complete)

### mobile_vlm/
Edge VLM benchmarking infrastructure.
- `test_edge_models.py` - Benchmark Moondream2 and Qwen2.5-VL-3B
- Tracks inference latency, memory usage, tokens/sec
- MLflow integration for experiment tracking

### unsloth_vlm/
Unsloth-accelerated VLM training.
- `train_qwen3vl.py` - Qwen3-VL fine-tuning with Unsloth
- `evaluate.py` - Model evaluation pipeline

## Usage

Run scripts from project root:
```bash
./venv/Scripts/python.exe experiments/api_models/test_api_vlm.py --help
./venv/Scripts/python.exe experiments/vlm_testing/comprehensive/test_internvl2.py
```

## Notes

- Results in `api_models/results/` are critical - do not delete
- See `docs/experiments/` for detailed documentation
- Large model weights stored in `artifacts/` (gitignored)
- Old test scripts archived in `_archive/`
