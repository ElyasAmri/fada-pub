"""
RunPod VLM benchmarking runner.

Usage:
    python experiments/runpod/benchmark_runner.py --quick
    python experiments/runpod/benchmark_runner.py --models qwen2-vl-2b
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "vastai"))

from benchmark_config import (
    BENCHMARK_MODELS,
    QUICK_BENCHMARK_MODELS,
    ModelConfig,
)
from runpod_instance import RunPodInstance


# HuggingFace dataset info
HF_DATASET = "elyasamri/fetal-ultrasound-vlm"
HF_TRAIN_FILE = "excel_train.jsonl"
HF_VAL_FILE = "excel_val.jsonl"
HF_IMAGES_ARCHIVE = "fetal_ultrasound_images.tar.gz"

# Docker image
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def create_benchmark_script(model_config: ModelConfig, hf_token: str) -> str:
    """Generate the complete setup and training script."""
    return f'''#!/bin/bash
set -e

echo "=== Setting up benchmark for {model_config.name} ==="
echo "Started at: $(date)"

# Set HF token
export HF_TOKEN="{hf_token}"

# Install dependencies
echo "Installing dependencies..."
# Remove problematic blinker package
pip uninstall -y blinker 2>/dev/null || rm -rf /usr/lib/python3/dist-packages/blinker* 2>/dev/null || true
# Use transformers 4.46+ (has AutoModelForImageTextToText and set_submodule fix)
pip install -q "transformers>=4.46.0" accelerate peft bitsandbytes datasets mlflow qwen-vl-utils huggingface_hub

# Login to HF
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Create workspace
mkdir -p /workspace/data /workspace/output
cd /workspace

# Clone repository
echo "Cloning repository..."
git clone https://github.com/ElyasAmri/fada-v2.git
cd fada-v2

# Download dataset from HuggingFace
echo "Downloading dataset from HuggingFace..."
python << 'PYEOF'
from huggingface_hub import hf_hub_download
import tarfile
import os

data_dir = "/workspace/data"

print("Downloading training files...")
hf_hub_download(
    repo_id="{HF_DATASET}",
    filename="{HF_TRAIN_FILE}",
    repo_type="dataset",
    local_dir=data_dir
)
hf_hub_download(
    repo_id="{HF_DATASET}",
    filename="{HF_VAL_FILE}",
    repo_type="dataset",
    local_dir=data_dir
)

print("Downloading image archive...")
archive_path = hf_hub_download(
    repo_id="{HF_DATASET}",
    filename="{HF_IMAGES_ARCHIVE}",
    repo_type="dataset",
    local_dir=data_dir
)

print("Extracting images...")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(f"{{data_dir}}/images")

os.remove(archive_path)
print("Dataset ready!")
PYEOF

# Fix paths in JSONL
echo "Fixing image paths..."
sed -i 's|"images/|"/workspace/data/images/|g' /workspace/data/{HF_TRAIN_FILE}
sed -i 's|"images/|"/workspace/data/images/|g' /workspace/data/{HF_VAL_FILE}

# Run training (force --no-4bit to avoid bitsandbytes compatibility issues)
echo "Starting training for {model_config.name}..."
python experiments/fine_tuning/train_qwen3vl_lora.py \\
    --model {model_config.name} \\
    --train-data /workspace/data/{HF_TRAIN_FILE} \\
    --val-data /workspace/data/{HF_VAL_FILE} \\
    --epochs {model_config.epochs} \\
    --batch-size {model_config.batch_size} \\
    --gradient-accumulation {model_config.gradient_accumulation} \\
    {"--max-train-samples " + str(model_config.max_train_samples) if model_config.max_train_samples else ""} \\
    {"--max-val-samples " + str(model_config.max_val_samples) if model_config.max_val_samples else ""} \\
    --no-4bit \\
    --output-dir /workspace/output/{model_config.name}

echo "Training complete!"
echo "Finished at: $(date)"

# Save results summary
python << 'PYEOF'
import json
from pathlib import Path

output_dir = Path("/workspace/output/{model_config.name}")
config_file = output_dir / "config.json"

if config_file.exists():
    with open(config_file) as f:
        config = json.load(f)

    final_dir = output_dir / "final"
    adapter_files = list(final_dir.glob("*.safetensors")) if final_dir.exists() else []

    result = {{
        "model": "{model_config.name}",
        "model_id": "{model_config.model_id}",
        "config": config,
        "output_dir": str(final_dir),
        "adapter_size_mb": sum(f.stat().st_size for f in adapter_files) / 1e6 if adapter_files else 0,
    }}

    with open(f"/workspace/output/{model_config.name}_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
else:
    print("No config.json found - training may have failed")
PYEOF
'''


class RunPodBenchmarkRunner:
    """Manages RunPod instances for benchmarking."""

    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.instances: Dict[str, RunPodInstance] = {}
        self.results: Dict[str, dict] = {}
        self.output_dir = Path(__file__).parent / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def provision_instance(self, model_config: ModelConfig) -> Optional[RunPodInstance]:
        """Provision a RunPod instance for a model."""
        print(f"\n[{model_config.name}] Creating RTX 3090 pod...")

        instance = RunPodInstance()

        # Create pod (RTX 3090 - 24GB VRAM, good availability)
        pod_id = instance.create_pod(
            gpu_type="NVIDIA GeForce RTX 3090",
            image=DOCKER_IMAGE,
            disk_gb=50,
            name=f"benchmark-{model_config.name}"
        )

        if not pod_id:
            print(f"[{model_config.name}] Failed to create pod!")
            return None

        # Wait for ready
        if not instance.wait_for_ready(timeout=180):
            print(f"[{model_config.name}] Pod failed to start!")
            instance.destroy_pod()
            return None

        # Quick SSH test
        ret, out, err = instance.run_ssh("echo 'SSH OK'", timeout=30)
        if ret != 0:
            print(f"[{model_config.name}] SSH test failed: {err}")
            instance.destroy_pod()
            return None

        print(f"[{model_config.name}] Pod ready!")
        return instance

    def run_benchmark(self, model_config: ModelConfig, instance: RunPodInstance) -> dict:
        """Run benchmark on a provisioned instance."""
        print(f"\n[{model_config.name}] Starting benchmark...")

        # Generate and upload script
        script = create_benchmark_script(model_config, self.hf_token)
        script_b64 = base64.b64encode(script.encode()).decode()

        ret, _, stderr = instance.run_ssh(
            f"echo '{script_b64}' | base64 -d > /workspace/benchmark.sh && chmod +x /workspace/benchmark.sh",
            timeout=60
        )

        if ret != 0:
            return {"error": f"Failed to upload script: {stderr}"}

        # Run in background
        ret, _, stderr = instance.run_ssh(
            "nohup /workspace/benchmark.sh > /workspace/benchmark.log 2>&1 & echo $! > /workspace/benchmark.pid",
            timeout=30
        )

        if ret != 0:
            return {"error": f"Failed to start benchmark: {stderr}"}

        print(f"[{model_config.name}] Benchmark started, polling for completion...")

        # Poll for completion
        max_polls = 60  # 30 minutes max
        poll_interval = 30

        for poll_num in range(max_polls):
            time.sleep(poll_interval)

            ret, stdout, _ = instance.run_ssh(
                "if [ -f /workspace/benchmark.pid ]; then "
                "pid=$(cat /workspace/benchmark.pid); "
                "if ps -p $pid > /dev/null 2>&1; then echo 'RUNNING'; else echo 'DONE'; fi; "
                "else echo 'NO_PID'; fi",
                timeout=30
            )

            status = stdout.strip() if stdout else "UNKNOWN"
            elapsed = (poll_num + 1) * poll_interval // 60

            if status == "RUNNING":
                ret, last_line, _ = instance.run_ssh(
                    "tail -1 /workspace/benchmark.log 2>/dev/null || echo ''",
                    timeout=30
                )
                progress = ''.join(c if ord(c) < 128 else '?' for c in (last_line.strip()[:60] if last_line else ""))
                print(f"[{model_config.name}] Running ({elapsed}m)... {progress}")
            elif status == "DONE":
                print(f"[{model_config.name}] Completed after {elapsed}m")
                break
            else:
                print(f"[{model_config.name}] Status: {status} ({elapsed}m)")
        else:
            return {"error": f"Timeout after {max_polls * poll_interval // 60} minutes"}

        # Retrieve log
        ret, stdout, _ = instance.run_ssh("cat /workspace/benchmark.log 2>/dev/null", timeout=120)

        output_file = self.output_dir / f"{model_config.name}_output.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(stdout or "")
        except Exception:
            pass

        # Retrieve results
        ret, result_json, _ = instance.run_ssh(
            f"cat /workspace/output/{model_config.name}_result.json 2>/dev/null || echo '{{}}'",
            timeout=30
        )

        try:
            result = json.loads(result_json)
            if result:
                result["output_file"] = str(output_file)
                result["success"] = True
                return result
        except json.JSONDecodeError:
            pass

        return {"error": "Training completed but no results found", "output_file": str(output_file)}

    def run_all_benchmarks(self, models: List[ModelConfig], parallel: bool = True) -> Dict[str, dict]:
        """Run benchmarks for all models."""
        print(f"\n{'='*60}")
        print(f"RunPod Benchmarks: {len(models)} models")
        print(f"{'='*60}\n")

        start_time = time.time()

        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._run_single, m): m.name
                    for m in models
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        self.results[name] = future.result()
                        status = "SUCCESS" if self.results[name].get("success") else "FAILED"
                        print(f"\n[{name}] {status}")
                    except Exception as e:
                        err = ''.join(c if ord(c) < 128 else '?' for c in str(e))
                        self.results[name] = {"error": err}
        else:
            for model in models:
                self.results[model.name] = self._run_single(model)

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": time.time() - start_time,
            "platform": "runpod",
            "models": [asdict(m) for m in models],
            "results": self.results,
        }

        summary_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Complete! Total: {(time.time() - start_time)/60:.1f} min")
        print(f"Results: {summary_file}")
        print(f"{'='*60}\n")

        for name, result in self.results.items():
            status = "SUCCESS" if result.get("success") else "FAILED"
            info = f"adapter: {result.get('adapter_size_mb', 0):.1f}MB" if result.get("success") else result.get("error", "")[:50]
            print(f"{name}: {status} - {info}")

        return self.results

    def _run_single(self, model_config: ModelConfig) -> dict:
        """Run single benchmark."""
        instance = None
        try:
            instance = self.provision_instance(model_config)
            if not instance:
                return {"error": "Failed to provision"}

            self.instances[model_config.name] = instance
            return self.run_benchmark(model_config, instance)
        except Exception as e:
            return {"error": ''.join(c if ord(c) < 128 else '?' for c in str(e))}
        finally:
            if instance:
                print(f"[{model_config.name}] Destroying pod...")
                instance.destroy_pod()


def main():
    parser = argparse.ArgumentParser(description="RunPod VLM benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    parser.add_argument("--models", nargs="+", help="Specific models")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")

    args = parser.parse_args()

    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import HfApi
            hf_token = HfApi().token
        except Exception:
            pass

    if not hf_token:
        print("Error: HF_TOKEN required")
        return

    # Select models
    if args.quick:
        models = QUICK_BENCHMARK_MODELS
    elif args.models:
        models = [m for m in BENCHMARK_MODELS if m.name in args.models]
    else:
        models = BENCHMARK_MODELS

    print(f"Models: {[m.name for m in models]}")

    runner = RunPodBenchmarkRunner(hf_token)
    runner.run_all_benchmarks(models, parallel=not args.sequential)


if __name__ == "__main__":
    main()
