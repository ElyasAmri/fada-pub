---
tags: [phase2, infrastructure]
---
**Phase**: 2 - Infrastructure

## Purpose
Cloud GPU infrastructure for training VLM models beyond local RTX 4070 constraints.

## Architecture

```
experiments/vastai/
+-- __main__.py     # Entry point
+-- runner.py       # Main CLI runner
+-- instance.py     # Instance management
+-- jobs.py         # Job orchestration
+-- presets.py      # GPU configurations
+-- transfer.py     # Data transfer
+-- templates/
    +-- run_inference.py   # Inference jobs
    +-- run_finetune.py    # Training jobs
```

## Usage

```bash
# Run via module
python -m experiments.vastai

# Commands
vastai list          # List available instances
vastai create        # Create new instance
vastai run           # Run training job
vastai transfer      # Upload/download data
```

## GPU Presets

| Preset | GPU | VRAM | Use Case |
|--------|-----|------|----------|
| cheap | RTX 3080 | 10GB | Inference |
| mid | RTX 4090 | 24GB | Training |
| high | A100 40GB | 40GB | Large models |

## Workflow

1. **Upload data** to cloud instance
2. **Create instance** with required GPU
3. **Run training** via templates
4. **Download results** (LoRA adapters)

## Status
- [x] CLI framework complete
- [x] Instance management
- [x] Data transfer utilities
- [ ] Full training deployment

## Links
- [[Fine-Tuning Approach]] - Training strategy
- [[VLM Models Tested]] - Models to fine-tune

