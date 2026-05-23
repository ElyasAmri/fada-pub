# Mobile VLM Benchmarking

Edge-deployable VLM benchmarking infrastructure for testing Moondream2 and Qwen2.5-VL-3B on fetal ultrasound analysis.

## Models Supported

- **Moondream2** (`vikhyatk/moondream2`) - 1.6B parameter edge VLM
- **Qwen2.5-VL-3B** (`Qwen/Qwen2.5-VL-3B-Instruct`) - 3B parameter compact VLM

## Setup

1. Install dependencies:
```bash
pip install -r experiments/mobile_vlm/requirements.txt
```

2. Prepare test images (choose one):
```bash
# Option A: Copy from existing dataset
./venv/Scripts/python.exe experiments/mobile_vlm/create_test_data.py \
  --source-dir "data/Fetal Ultrasound" --output-dir data/test_images --count 10

# Option B: Create dummy images for testing
./venv/Scripts/python.exe experiments/mobile_vlm/create_test_data.py \
  --output-dir data/test_images --count 10
```

See [QUICK_START.md](QUICK_START.md) for detailed setup guide.

## Usage

### Basic Benchmark
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model moondream2 \
  --device cuda \
  --quantization none \
  --samples 10 \
  --image-dir data/test_images
```

### CPU Testing with Quantization
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model qwen25-vl-3b \
  --device cpu \
  --quantization int8 \
  --samples 5 \
  --image-dir data/test_images
```

### GPU Testing with INT4 Quantization
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model moondream2 \
  --device cuda \
  --quantization int4 \
  --samples 20 \
  --image-dir data/test_images
```

## Arguments

- `--model`: Model to benchmark (`moondream2`, `qwen25-vl-3b`)
- `--device`: Device (`cpu`, `cuda`) - defaults to CUDA if available
- `--quantization`: Quantization level (`none`, `int8`, `int4`)
- `--samples`: Number of test samples (default: 10)
- `--image-dir`: Directory containing test images (required)
- `--no-mlflow`: Skip MLflow logging

## Metrics Tracked

### Performance
- **Load Time**: Model loading time (seconds)
- **Inference Latency**: Average inference time per image (ms)
- **Tokens/Second**: Estimated generation speed

### Resource Usage
- **Memory**: GPU memory usage (MB) for CUDA, N/A for CPU

### MLflow Integration
All benchmarks are logged to the `mobile_vlm_benchmark` MLflow experiment with:
- Parameters: model, device, quantization, sample count
- Metrics: load_time, latency, memory, tokens/sec
- Artifacts: `benchmark_results.json` with per-sample details

## Test Prompt

Uses Q7 fetal ultrasound prompt:
```
"Describe any abnormalities visible in this fetal ultrasound image."
```

## Output Structure

Results JSON contains:
```json
{
  "model": "moondream2",
  "device": "cuda",
  "quantization": "int8",
  "num_samples": 10,
  "metrics": {
    "load_time_seconds": 12.3,
    "inference_latency_ms": 450.2,
    "memory_mb": 2048.5,
    "tokens_per_second": 25.3
  },
  "per_sample_results": [
    {
      "image": "/path/to/image.jpg",
      "response": "The ultrasound shows...",
      "latency_ms": 445.2
    }
  ]
}
```

## View Results

Start MLflow UI:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Navigate to `http://localhost:5000` and select the `mobile_vlm_benchmark` experiment.

## Batch Benchmarking

Run comprehensive benchmarks across all configurations:
```bash
# From project root
bash experiments/mobile_vlm/run_benchmarks.sh data/test_images 10

# Custom sample count
bash experiments/mobile_vlm/run_benchmarks.sh data/test_images 20
```

This script automatically:
- Detects CUDA availability
- Runs all GPU benchmarks (if CUDA available)
- Runs CPU benchmarks with reduced samples
- Logs all results to MLflow

## Example Workflow

1. Benchmark Moondream2 baseline:
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model moondream2 --device cuda --quantization none \
  --samples 10 --image-dir data/test_images
```

2. Test INT8 quantization:
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model moondream2 --device cuda --quantization int8 \
  --samples 10 --image-dir data/test_images
```

3. Compare CPU performance:
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model moondream2 --device cpu --quantization none \
  --samples 5 --image-dir data/test_images
```

4. Benchmark Qwen2.5-VL-3B:
```bash
./venv/Scripts/python.exe experiments/mobile_vlm/test_edge_models.py \
  --model qwen25-vl-3b --device cuda --quantization none \
  --samples 10 --image-dir data/test_images
```

## Notes

- First run downloads models from HuggingFace (may take several minutes)
- Warmup run excluded from timing metrics
- INT4/INT8 quantization requires `bitsandbytes` package
- GPU benchmarks measure CUDA memory allocation
- CPU benchmarks focus on latency (no memory tracking)
