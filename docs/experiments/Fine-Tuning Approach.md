---
tags: [phase2, training]
---

**Phase**: 2 - Complete

## Strategy

Fine-tune top-performing VLMs on fetal ultrasound dataset using LoRA adapters for efficient training.

## Completed Results

| Model                 | Score     | Method               | Samples | Platform        |
| --------------------- | --------- | -------------------- | ------- | --------------- |
| Qwen2.5-VL-7B         | 81.1% [1] | Embedding similarity | 600     | RunPod RTX 3090 |
| Qwen3-VL-8B (Q7 only) | 82%       | Accuracy             | 50      | Local RTX 5090  |

[1] NOTE: 600-sample subset. Full test set (1,894 images) v3 score is embed_sim=0.5058. The 81.1% figure should be reproduced on the full test set for verification.

## Target Models (Priority Order)

1. **Qwen2.5-VL-7B** - 81.1% fine-tuned, best verified GT score
2. **Qwen3.5-35B-A3B** - 0.3650 primary (top zero-shot, 1,894 samples)
3. **gemma-3-12b-it** - 0.3629 primary (#2 zero-shot)
4. **InternVL3.5-4B** - 0.3491 primary (#3 zero-shot)

## Training Configuration

### LoRA Parameters (Current)

```python
lora_config = {
    "r": 16,
    "alpha": 32,
    "target_modules": "all_linear",
    "dropout": 0.05
}
```

### Training Setup

- **Quantization**: 8-bit (LoRA training)
- **Learning Rate**: 1e-4
- **Batch Size**: 1-4 (GPU constrained)
- **Epochs**: 1 for benchmarking, 3 for final

## Infrastructure

- **Local**: RTX 5090 (24GB VRAM) with Unsloth (verified for all 7 Qwen models)
- **Cloud**: RCCG (A100 at $1.35/h, H100 at $1.90/h)

### Unsloth-Compatible Models

- Qwen2-VL-2B, Qwen2-VL-7B
- Qwen2.5-VL-3B, Qwen2.5-VL-7B
- Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B

## Remaining Experiments

See `docs/experiments/unsloth_vlm_experiments.md` for 7 planned experiments:

1. Multi-task Q1-Q8 training
2. Combined comprehensive report generation
3. Layer ablation study
4. Model size comparison (2B vs 4B vs 8B)
5. Prompt engineering variations
6. Data augmentation impact
7. Class-balanced training (95% normal vs 5% abnormal)
