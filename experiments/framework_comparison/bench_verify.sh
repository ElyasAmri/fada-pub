#!/bin/bash
export PATH=$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

echo "=== HuggingFace Login ==="
huggingface-cli login --token REDACTED_HF_TOKEN

echo "=== CUDA ==="
nvcc --version 2>/dev/null || echo "NO CUDA"

echo "=== PyTorch ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>/dev/null

echo "=== Unsloth ==="
python3 -c "import unsloth; print('unsloth OK')" 2>&1

echo "=== Done ==="
