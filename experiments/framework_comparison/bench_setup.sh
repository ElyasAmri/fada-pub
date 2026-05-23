#!/bin/bash
set -e

echo "=== Installing CUDA 12.8 ==="
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -qq
sudo apt-get install -y -qq cuda-toolkit-12-8

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "=== Installing Python packages ==="
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip3 install unsloth transformers accelerate bitsandbytes peft datasets pillow sentencepiece protobuf
pip3 install huggingface_hub

echo "=== Logging into HuggingFace ==="
huggingface-cli login --token REDACTED_HF_TOKEN

echo "=== Verifying ==="
nvcc --version
python3 -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0))" 2>/dev/null || true

echo "=== Setup complete ==="
