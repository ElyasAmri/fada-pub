#!/bin/bash
# Vast.ai MedGemma Setup Script
# Run this after SSH into your Vast.ai instance

set -e

echo "=== MedGemma 27B Setup for Vast.ai ==="

# 1. Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2. Login to HuggingFace (required for MedGemma)
echo ""
echo "=== HuggingFace Login ==="
echo "MedGemma requires accepting the license agreement at:"
echo "https://huggingface.co/google/medgemma-27b-it"
echo ""
echo "Enter your HuggingFace token (with read access):"
read -s HF_TOKEN
export HF_TOKEN
huggingface-cli login --token $HF_TOKEN

# 3. Start vLLM server with MedGemma
echo ""
echo "=== Starting vLLM Server ==="
echo "This will download ~50GB model weights (takes 3-10 min depending on connection)"
echo ""

# Kill any existing vLLM processes
pkill -f "vllm.entrypoints" || true

# Start vLLM with MedGemma 27B multimodal (can see images)
# Using OpenAI-compatible API on port 8000
nohup python -m vllm.entrypoints.openai.api_server \
    --model google/medgemma-27b-it \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --port 8000 \
    --host 0.0.0.0 \
    > vllm_server.log 2>&1 &

echo "vLLM server starting in background..."
echo "Logs: tail -f vllm_server.log"
echo ""
echo "Waiting for server to be ready (downloading model if needed)..."

# Wait for server to be ready
for i in {1..120}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "=== Server Status ==="
curl -s http://localhost:8000/v1/models | python -m json.tool 2>/dev/null || echo "Server still starting... check: tail -f vllm_server.log"

echo ""
echo "=== Next Steps ==="
echo "1. Create SSH tunnel from local machine:"
echo "   ssh -p <PORT> root@<IP> -L 8000:127.0.0.1:8000 -N -f"
echo ""
echo "2. Run test:"
echo "   python experiments/api_models/test_api_vlm.py --models vllm --images-per-category 3"
