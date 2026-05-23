#!/bin/bash
# Mobile VLM Benchmark Runner
# Runs comprehensive benchmarks across models, devices, and quantization levels

set -e

# Configuration
PYTHON="./venv/Scripts/python.exe"
SCRIPT="experiments/mobile_vlm/test_edge_models.py"
IMAGE_DIR="${1:-data/test_images}"
SAMPLES="${2:-10}"

if [ ! -d "$IMAGE_DIR" ]; then
    echo "ERROR: Image directory not found: $IMAGE_DIR"
    echo "Usage: $0 [image_dir] [num_samples]"
    exit 1
fi

echo "=========================================="
echo "Mobile VLM Benchmark Suite"
echo "=========================================="
echo "Image directory: $IMAGE_DIR"
echo "Samples per run: $SAMPLES"
echo ""

# Function to run benchmark
run_benchmark() {
    local model=$1
    local device=$2
    local quant=$3

    echo ""
    echo "Running: $model / $device / $quant"
    echo "------------------------------------------"

    $PYTHON $SCRIPT \
        --model "$model" \
        --device "$device" \
        --quantization "$quant" \
        --samples "$SAMPLES" \
        --image-dir "$IMAGE_DIR"
}

# Check CUDA availability
if $PYTHON -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    HAS_CUDA=true
    echo "CUDA: Available"
else
    HAS_CUDA=false
    echo "CUDA: Not available (CPU only)"
fi

echo ""

# Benchmark configurations
if [ "$HAS_CUDA" = true ]; then
    # GPU benchmarks
    echo "Running GPU benchmarks..."

    # Moondream2 GPU
    run_benchmark "moondream2" "cuda" "none"
    run_benchmark "moondream2" "cuda" "int8"
    run_benchmark "moondream2" "cuda" "int4"

    # Qwen2.5-VL-3B GPU
    run_benchmark "qwen25-vl-3b" "cuda" "none"
    run_benchmark "qwen25-vl-3b" "cuda" "int8"
    run_benchmark "qwen25-vl-3b" "cuda" "int4"
fi

# CPU benchmarks (smaller sample size for speed)
CPU_SAMPLES=$((SAMPLES / 2))
if [ $CPU_SAMPLES -lt 3 ]; then
    CPU_SAMPLES=3
fi

echo ""
echo "Running CPU benchmarks (${CPU_SAMPLES} samples)..."

# Moondream2 CPU
run_benchmark "moondream2" "cpu" "none"
run_benchmark "moondream2" "cpu" "int8"

# Qwen2.5-VL-3B CPU (skip due to size)
# run_benchmark "qwen25-vl-3b" "cpu" "int8"

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
echo ""
echo "View results with: mlflow ui --host 0.0.0.0 --port 5000"
echo "Then navigate to: http://localhost:5000"
