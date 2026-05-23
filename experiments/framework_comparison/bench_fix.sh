#!/bin/bash
set -e
export PATH=$HOME/.local/bin:$PATH

echo "=== Reinstalling PyTorch for CUDA 12.1 (driver 535 compatible) ==="
pip3 install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Reinstalling unsloth ==="
pip3 install --force-reinstall unsloth

echo "=== HuggingFace login ==="
pip3 install huggingface_hub[cli]
$HOME/.local/bin/huggingface-cli login --token REDACTED_HF_TOKEN

echo "=== Verify ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python3 -c "from unsloth import FastModel; print('unsloth OK')"

echo "=== DONE ==="
