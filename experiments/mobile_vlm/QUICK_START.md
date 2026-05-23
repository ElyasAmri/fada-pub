# Mobile VLM Quick Start

Fast setup and testing guide for edge VLM benchmarking.

## 1. Install Dependencies

```bash
pip install -r experiments/mobile_vlm/requirements.txt
```

Dependencies:
- moondream (Moondream2 model)
- transformers (Qwen2.5-VL)
- torch (PyTorch)
- mlflow (experiment tracking)

## 2. Prepare Test Images

### Option A: Use Existing Dataset
```bash
# Copy from existing ultrasound dataset
./venv/Scripts/python.exe experiments/mobile_vlm/create_test_data.py \
  --source-dir "data/Fetal Ultrasound" \
  --output-dir data/test_images \
  --count 10
```

### Option B: Create Dummy Images (for testing infrastructure)
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/create_test_data.py \
  --output-dir data/test_images \
  --count 10
```

### Option C: Use Your Own Images
```bash
# Ensure you have images in a directory
ls data/test_images/*.jpg
```

## 3. Run Single Benchmark

```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model moondream2 \
  --device cuda \
  --quantization none \
  --samples 10 \
  --image-dir data/test_images
```

## 4. Run All Benchmarks

```bash
bash experiments/mobile_vlm/run_benchmarks.sh data/test_images 10
```

This runs:
- Moondream2: CUDA (none/int8/int4), CPU (none/int8)
- Qwen2.5-VL-3B: CUDA (none/int8/int4)

## 5. View Results

Start MLflow UI:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open browser: http://localhost:5000

Navigate to experiment: `mobile_vlm_benchmark`

## Expected Output

```
======================================================================
Mobile VLM Benchmark: moondream2
======================================================================

Found 10 test images

Loading moondream2...
  Device: cuda
  Quantization: none
  Load time: 8.45s
  GPU memory: 1024.32 MB

Running warmup...
  Warmup time: 523.12ms

Benchmarking on 10 images...
  [1/10] Processing image1.jpg... 512.34ms
  [2/10] Processing image2.jpg... 498.76ms
  ...

======================================================================
Benchmark Results:
----------------------------------------------------------------------
Load time:          8.45s
Avg latency:        505.23ms
Memory usage:       1024.32 MB
Tokens/second:      98.76
======================================================================

Results logged to MLflow experiment: mobile_vlm_benchmark
Run name: moondream2_cuda_none
```

## Troubleshooting

### CUDA Out of Memory
Use quantization:
```bash
--quantization int8  # or int4
```

### Slow CPU Inference
Reduce samples:
```bash
--samples 3
```

### Missing moondream Package
```bash
pip install moondream
```

### Missing bitsandbytes (for quantization)
```bash
pip install bitsandbytes
```

## Performance Expectations

### Moondream2 (1.6B params)
- CUDA FP16: ~500ms/image, ~1GB VRAM
- CUDA INT8: ~400ms/image, ~600MB VRAM
- CPU: ~2-3s/image

### Qwen2.5-VL-3B (3B params)
- CUDA FP16: ~800ms/image, ~3GB VRAM
- CUDA INT8: ~600ms/image, ~1.5GB VRAM
- CPU: ~5-8s/image (not recommended)

## Next Steps

1. Compare results in MLflow UI
2. Select best model/quantization for deployment
3. Test on production hardware (mobile/edge device)
4. Fine-tune selected model if needed
