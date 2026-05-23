# Mobile VLM Benchmarking Architecture

Technical documentation for the edge VLM benchmarking infrastructure.

## Overview

This module provides comprehensive benchmarking for edge-deployable Vision-Language Models (VLMs) on fetal ultrasound analysis tasks. It focuses on deployment feasibility metrics: inference latency, memory usage, and model quality.

## Design Goals

1. **Reproducibility**: All benchmarks logged to MLflow with full configuration
2. **Flexibility**: Support multiple models, devices, and quantization levels
3. **Production-Ready**: Proper error handling, validation, and documentation
4. **Performance**: Efficient benchmarking with warmup runs and accurate timing
5. **Extensibility**: Easy to add new models and evaluation metrics

## Components

### 1. Core Benchmarking Script (`test_edge_models.py`)

Main benchmarking infrastructure with model-agnostic design.

#### Key Classes

**`ModelBenchmark`**
- Handles model loading, inference, and metric collection
- Supports multiple model types through polymorphic inference methods
- Tracks timing, memory, and generates performance metrics

Methods:
- `load_model()`: Load model with specified device/quantization
- `run_inference()`: Run single image inference with timing
- `get_metrics()`: Calculate aggregate performance metrics

#### Supported Models

**Moondream2** (1.6B parameters)
- Model ID: `vikhyatk/moondream2`
- Type: Compact edge VLM
- Interface: Custom `VisionLanguageModel` class
- Strengths: Small size, fast inference, low memory

**Qwen2.5-VL-3B** (3B parameters)
- Model ID: `Qwen/Qwen2.5-VL-3B-Instruct`
- Type: Compact multimodal LLM
- Interface: HuggingFace `Qwen2VLForConditionalGeneration`
- Strengths: Better accuracy, instruction following

#### Quantization Support

- **None**: Full precision (FP32 CPU, FP16 GPU)
- **INT8**: 8-bit quantization via bitsandbytes
- **INT4**: 4-bit quantization via bitsandbytes

Trade-offs:
- None: Best quality, highest memory
- INT8: 2x memory reduction, minimal quality loss
- INT4: 4x memory reduction, moderate quality loss

### 2. Test Data Generator (`create_test_data.py`)

Utility for preparing test images.

Modes:
1. **Copy from dataset**: Randomly sample from existing ultrasound images
2. **Generate dummy**: Create synthetic grayscale images for infrastructure testing

### 3. Batch Runner (`run_benchmarks.sh`)

Automated benchmarking across all configurations.

Features:
- Detects CUDA availability
- Runs comprehensive model/device/quantization matrix
- Adjusts sample count for CPU runs (slower)
- All results logged to single MLflow experiment

### 4. MLflow Integration

All benchmarks logged to experiment: `mobile_vlm_benchmark`

**Parameters Logged:**
- `model_name`: Model identifier
- `device`: cpu/cuda
- `quantization`: none/int8/int4
- `num_samples`: Number of test images
- `image_dir`: Test data location

**Metrics Logged:**
- `load_time_seconds`: Model initialization time
- `inference_latency_ms`: Average per-image inference time
- `memory_mb`: GPU memory usage (CUDA only)
- `tokens_per_second`: Estimated generation speed

**Artifacts:**
- `benchmark_results.json`: Complete results with per-sample details

## Workflow

### Standard Benchmark Run

```
1. Parse arguments (model, device, quantization, samples)
2. Find test images in specified directory
3. Initialize ModelBenchmark instance
4. Load model (timed, memory tracked)
5. Warmup run (excluded from metrics)
6. Benchmark runs on all test images
7. Calculate aggregate metrics
8. Log to MLflow
9. Display results
```

### Timing Methodology

1. **Load Time**: Wall-clock time from model instantiation to ready state
2. **Warmup**: Single inference run (excluded from metrics) to initialize CUDA/caches
3. **Inference Time**: Per-image wall-clock time including:
   - Image loading and preprocessing
   - Model forward pass
   - Decoding/post-processing
4. **Metrics**: Average over all non-warmup runs

### Memory Tracking

- **CUDA**: `torch.cuda.memory_allocated()` after model loading
- **CPU**: Not tracked (Python memory tracking unreliable)

## Extension Points

### Adding New Models

1. Add model config to `MODEL_CONFIGS`:
```python
"new-model": {
    "model_id": "org/model-name",
    "type": "new_type",
}
```

2. Implement loader method:
```python
def _load_new_type(self):
    # Model-specific loading logic
    pass
```

3. Implement inference method:
```python
def _inference_new_type(self, image: Image.Image, prompt: str) -> str:
    # Model-specific inference logic
    pass
```

### Adding New Metrics

1. Track during inference in `ModelBenchmark`
2. Add to `get_metrics()` return dictionary
3. Update `log_to_mlflow()` to log new metric

Example:
```python
# Track in run_inference
self.custom_metrics.append(custom_value)

# Return in get_metrics
"custom_metric": sum(self.custom_metrics) / len(self.custom_metrics)

# Log in log_to_mlflow
mlflow.log_metric("custom_metric", metrics["custom_metric"])
```

## Performance Baselines

Expected performance on RTX 4070 (12GB VRAM):

### Moondream2
| Config | Latency | Memory | Tokens/s |
|--------|---------|--------|----------|
| CUDA FP16 | ~500ms | ~1GB | ~100 |
| CUDA INT8 | ~400ms | ~600MB | ~125 |
| CUDA INT4 | ~350ms | ~400MB | ~140 |
| CPU FP32 | ~2.5s | N/A | ~20 |

### Qwen2.5-VL-3B
| Config | Latency | Memory | Tokens/s |
|--------|---------|--------|----------|
| CUDA FP16 | ~800ms | ~3GB | ~60 |
| CUDA INT8 | ~600ms | ~1.5GB | ~80 |
| CUDA INT4 | ~500ms | ~1GB | ~100 |
| CPU FP32 | ~6s | N/A | ~8 |

Note: Actual performance varies by hardware, input size, and generation length.

## Error Handling

### Model Loading Failures
- Missing packages: Clear error with installation instructions
- CUDA OOM: Suggestion to use quantization or CPU
- Invalid model name: List available models

### Inference Failures
- Missing images: Validate directory before benchmarking
- Corrupt images: Skip and continue (logged as warning)
- Generation errors: Caught per-image, doesn't stop benchmark

### MLflow Failures
- `--no-mlflow` flag to bypass logging
- Local file fallback if MLflow unavailable

## Testing Strategy

### Unit Tests (Recommended)
```python
# Test model config validation
# Test metric calculations
# Test error handling
```

### Integration Tests
```bash
# Quick smoke test with dummy data
./venv/Scripts/python.exe experiments/mobile_vlm/create_test_data.py \
  --output-dir /tmp/test --count 3

./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model moondream2 --device cpu --samples 3 \
  --image-dir /tmp/test --no-mlflow
```

### Full Benchmark Suite
```bash
bash experiments/mobile_vlm/run_benchmarks.sh data/test_images 10
```

## Future Enhancements

### Planned Features
1. **Quality Metrics**: Add BLEU/ROUGE scoring against ground truth
2. **Batch Inference**: Support batch processing for throughput testing
3. **Mobile Deployment**: ONNX export and mobile runtime benchmarking
4. **Fine-tuning Support**: Benchmark fine-tuned checkpoints
5. **Multi-GPU**: Support distributed inference benchmarking

### Model Additions
- SmolVLM (1.7B)
- LLaVA-Phi (3B)
- MobileVLM (3B)
- Nougat (OCR-focused)

### Metrics Additions
- VRAM peak usage (not just allocated)
- CPU usage percentage
- Power consumption (if available)
- First token latency
- Token generation throughput

## Dependencies

Core:
- `torch`: PyTorch deep learning framework
- `transformers`: HuggingFace model library
- `Pillow`: Image processing
- `mlflow`: Experiment tracking

Model-specific:
- `moondream`: Moondream2 package
- `bitsandbytes`: Quantization (optional)

## References

- [Moondream2 Documentation](https://github.com/vikhyat/moondream)
- [Qwen2-VL Documentation](https://huggingface.co/docs/transformers/model_doc/qwen2_vl)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
