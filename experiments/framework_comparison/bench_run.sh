#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH

echo "=== GPU Benchmark Runner ==="
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')"

echo "=== Running benchmark ==="
cd /home/ubuntu/bench
python3 bench_train.py \
    --train-data /home/ubuntu/bench/bench_train_100.jsonl \
    --data-root /home/ubuntu/bench/images \
    --output-dir /home/ubuntu/bench/results

echo "=== Results ==="
cat /home/ubuntu/bench/results/bench_results.json
