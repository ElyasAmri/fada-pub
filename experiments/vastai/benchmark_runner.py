"""
Multi-instance VLM benchmarking runner for vast.ai.

Architecture:
    1. Spawn N instances in parallel (one per model)
    2. Each instance downloads data from HuggingFace (~5 min)
    3. Each instance runs its assigned model benchmark
    4. Stop instances when done (keep for inspection)

Usage:
    # Run all benchmarks (one instance per model)
    python experiments/vastai/benchmark_runner.py

    # Quick test with 2 models
    python experiments/vastai/benchmark_runner.py --quick

    # Specific models only
    python experiments/vastai/benchmark_runner.py --models qwen2-vl-2b qwen2.5-vl-3b
"""

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from benchmark_config import (
    BENCHMARK_MODELS,
    QUICK_BENCHMARK_MODELS,
    ModelConfig,
    get_gpu_preset,
)
from instance import VastInstance

# HuggingFace dataset info
HF_DATASET = "elyasamri/fetal-ultrasound-vlm"
HF_TRAIN_FILE = "excel_train.jsonl"
HF_VAL_FILE = "excel_val.jsonl"
HF_IMAGES_ARCHIVE = "fetal_ultrasound_images.tar.gz"

# Docker image
DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


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
pip install -q transformers>=4.49.0 accelerate>=1.3.0 peft bitsandbytes datasets mlflow pillow qwen-vl-utils huggingface_hub

# Login to HF
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Create workspace
mkdir -p /workspace/data /workspace/output
cd /workspace

# Clone repository
echo "Cloning repository..."
git clone https://github.com/ElyasAmri/fada-v2.git
cd fada-v2

# Patch training script to add custom data collator for variable-sized images
echo "Patching training script for VLM collator..."
cat > /tmp/patch_vlm.py << 'PATCHSCRIPT'
import re

with open("experiments/fine_tuning/train_qwen3vl_lora.py", "r") as f:
    content = f.read()

collator_code = """
def collate_fn_for_vlm(batch):
    import torch
    collated = dict()
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            shapes = [v.shape for v in values]
            if len(set(shapes)) == 1:
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
        elif isinstance(values[0], (list, tuple)):
            collated[key] = values
        else:
            try:
                collated[key] = torch.tensor(values)
            except Exception:
                collated[key] = values
    return collated

"""

if "def collate_fn_for_vlm" not in content:
    content = content.replace(
        "def create_trainer(",
        collator_code + "def create_trainer("
    )
    content = content.replace(
        "processing_class=processor,",
        "processing_class=processor, data_collator=collate_fn_for_vlm,"
    )
    with open("experiments/fine_tuning/train_qwen3vl_lora.py", "w") as f:
        f.write(content)
    print("Training script patched successfully!")
else:
    print("Training script already patched.")
PATCHSCRIPT
python /tmp/patch_vlm.py

# Download dataset from HuggingFace
echo "Downloading dataset from HuggingFace..."
python << 'PYEOF'
from huggingface_hub import hf_hub_download
import tarfile
import os

data_dir = "/workspace/data"

# Download JSONL files
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

# Download and extract images
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

# Remove archive to save space
os.remove(archive_path)
print("Dataset ready!")
PYEOF

# Fix paths in JSONL (images/ -> /workspace/data/images/)
echo "Fixing image paths..."
sed -i 's|"images/|"/workspace/data/images/|g' /workspace/data/{HF_TRAIN_FILE}
sed -i 's|"images/|"/workspace/data/images/|g' /workspace/data/{HF_VAL_FILE}

# Verify setup
echo "Verifying setup..."
ls -la /workspace/data/
head -1 /workspace/data/{HF_TRAIN_FILE} | grep -o '"images": \\[[^]]*\\]' | head -c 100

# Run training
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
    {"--no-4bit" if not model_config.use_4bit else ""} \\
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


class BenchmarkRunner:
    """Manages parallel instances for benchmarking."""

    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.instances: Dict[str, VastInstance] = {}
        self.results: Dict[str, dict] = {}
        # Use directory relative to this script, not current working directory
        self.output_dir = Path(__file__).parent / "benchmark_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def provision_instance(self, model_config: ModelConfig) -> Optional[VastInstance]:
        """Provision an instance for a model."""
        gpu_preset = get_gpu_preset(model_config)
        print(f"\n[{model_config.name}] Searching for {gpu_preset['gpu_name']} instance...")

        instance = VastInstance()

        # Search for offers (US preferred for faster HuggingFace downloads)
        offers = instance.search_offers(
            gpu_name=gpu_preset["gpu_name"],
            min_vram=gpu_preset["min_vram"],
            max_price=gpu_preset["max_price"],
            min_reliability=0.95,
            min_download=500,
            geolocation="US",  # 25x faster HuggingFace downloads vs Asia
        )

        if not offers:
            print(f"[{model_config.name}] No suitable instances found!")
            return None

        # Select best offer
        offer = offers[0]
        print(f"[{model_config.name}] Selected: {offer.gpu_name} @ ${offer.price_per_hour:.3f}/hr")

        # Create instance
        instance_id = instance.create_instance(
            offer_id=offer.offer_id,
            image=DOCKER_IMAGE,
            disk_gb=gpu_preset["disk_gb"],
        )

        if not instance_id:
            print(f"[{model_config.name}] Failed to create instance!")
            return None

        print(f"[{model_config.name}] Created instance {instance_id}")

        # Wait for instance to be ready
        if not instance.wait_for_ready(timeout=600):
            print(f"[{model_config.name}] Instance failed to start!")
            instance.destroy()
            return None

        # Give instance time to fully boot (GPU drivers may take time to load)
        print(f"[{model_config.name}] Instance created, waiting for GPU drivers to load...")
        time.sleep(30)

        # Validate SSH with extended retries (GPU drivers can take time)
        validation = instance.validate_instance(max_retries=10, retry_delay=15)
        if not validation.get("valid", False):
            print(f"[{model_config.name}] SSH validation failed: {validation.get('errors', [])}")
            instance.destroy()
            return None

        print(f"[{model_config.name}] Instance ready at {instance.ssh_host}:{instance.ssh_port}")
        return instance

    def run_benchmark(self, model_config: ModelConfig, instance: VastInstance) -> dict:
        """Run benchmark on a provisioned instance (background execution with polling)."""
        print(f"\n[{model_config.name}] Starting benchmark...")

        # Generate script
        script = create_benchmark_script(model_config, self.hf_token)

        # Upload script using base64 to avoid heredoc parsing issues
        script_b64 = base64.b64encode(script.encode()).decode()

        ret, _, stderr = instance.run_ssh(
            f"echo '{script_b64}' | base64 -d > /workspace/benchmark.sh",
            timeout=120
        )

        if ret != 0:
            return {"error": f"Failed to upload script: {stderr}"}

        # Make executable and start in background
        ret, _, stderr = instance.run_ssh(
            "chmod +x /workspace/benchmark.sh",
            timeout=30
        )

        if ret != 0:
            return {"error": f"Failed to chmod script: {stderr}"}

        # Run benchmark in background with nohup and disown
        # Using bash -c to ensure proper background execution
        ret, stdout, stderr = instance.run_ssh(
            "bash -c 'nohup /workspace/benchmark.sh > /workspace/benchmark.log 2>&1 & "
            "echo $! > /workspace/benchmark.pid; disown'",
            timeout=60
        )

        if ret != 0:
            return {"error": f"Failed to start benchmark: {stderr}"}

        print(f"[{model_config.name}] Benchmark started in background, polling for completion...")

        # Poll for completion (check every 60 seconds, max 2 hours)
        max_polls = 120
        poll_interval = 60
        consecutive_failures = 0
        max_consecutive_failures = 5  # Allow temporary SSH issues

        for poll_num in range(max_polls):
            time.sleep(poll_interval)

            # Check if process is still running
            ret, stdout, stderr = instance.run_ssh(
                "if [ -f /workspace/benchmark.pid ]; then "
                "  pid=$(cat /workspace/benchmark.pid); "
                "  if ps -p $pid > /dev/null 2>&1; then "
                "    echo 'RUNNING'; "
                "  else "
                "    echo 'DONE'; "
                "  fi; "
                "else "
                "  echo 'NO_PID'; "
                "fi",
                timeout=60
            )

            status = stdout.strip() if stdout else "UNKNOWN"
            elapsed = (poll_num + 1) * poll_interval // 60

            if status == "RUNNING":
                consecutive_failures = 0
                # Show last line of log for progress (sanitized for Windows console)
                ret, last_line, _ = instance.run_ssh(
                    "tail -1 /workspace/benchmark.log 2>/dev/null || echo ''",
                    timeout=30
                )
                progress = (last_line.strip()[:60] + "...") if last_line and len(last_line.strip()) > 60 else (last_line.strip() if last_line else "")
                # Sanitize for Windows console
                progress = ''.join(c if ord(c) < 128 else '?' for c in progress)
                print(f"[{model_config.name}] Still running ({elapsed}m)... {progress}")
                continue
            elif status == "DONE":
                print(f"[{model_config.name}] Benchmark completed after {elapsed}m")
                break
            elif status == "NO_PID":
                # Process never started or PID file not found
                print(f"[{model_config.name}] Warning: PID file not found ({elapsed}m)")
                consecutive_failures += 1
            else:
                # SSH connection issue - retry a few times
                consecutive_failures += 1
                err_msg = stderr[:50] if stderr else status
                err_msg = ''.join(c if ord(c) < 128 else '?' for c in str(err_msg))
                print(f"[{model_config.name}] SSH issue ({elapsed}m, {consecutive_failures}/{max_consecutive_failures}): {err_msg}")

            if consecutive_failures >= max_consecutive_failures:
                print(f"[{model_config.name}] Too many consecutive failures, stopping poll")
                break
        else:
            return {"error": f"Benchmark timed out after {max_polls * poll_interval // 60} minutes"}

        # Retrieve full log
        ret, stdout, _ = instance.run_ssh(
            "cat /workspace/benchmark.log 2>/dev/null || echo ''",
            timeout=120
        )

        # Sanitize stdout for file writing (replace non-UTF8 characters)
        if stdout:
            stdout = stdout.encode('utf-8', errors='replace').decode('utf-8')

        # Save output
        output_file = self.output_dir / f"{model_config.name}_output.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(stdout or "")
        except Exception as write_err:
            sanitized_err = ''.join(c if ord(c) < 128 else '?' for c in str(write_err))
            print(f"[{model_config.name}] Warning: Could not write output file: {sanitized_err}")

        # Check for errors in output
        if stdout and ("error" in stdout.lower() or "failed" in stdout.lower() or "Traceback" in stdout):
            # Still try to get results in case training partially completed
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
            else:
                return {
                    "error": "Training completed but no results found",
                    "output_file": str(output_file),
                }
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse results",
                "output_file": str(output_file),
            }

    def run_all_benchmarks(
        self,
        models: List[ModelConfig],
        parallel: bool = True,
        max_workers: int = 4,
    ) -> Dict[str, dict]:
        """Run benchmarks for all models."""

        print(f"\n{'='*60}")
        print(f"Starting benchmarks for {len(models)} models")
        print(f"Parallel: {parallel}, Max workers: {max_workers}")
        print(f"{'='*60}\n")

        start_time = time.time()

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                for model_config in models:
                    future = executor.submit(self._run_single_benchmark, model_config)
                    futures[future] = model_config.name

                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        result = future.result()
                        self.results[model_name] = result
                        status = "SUCCESS" if result.get("success") else "FAILED"
                        print(f"\n[{model_name}] Completed: {status}")
                    except Exception as e:
                        # Sanitize error for Windows console compatibility
                        error_msg = ''.join(c if ord(c) < 128 else '?' for c in str(e))
                        self.results[model_name] = {"error": error_msg}
                        print(f"\n[{model_name}] Failed with exception: {error_msg}")
        else:
            for model_config in models:
                result = self._run_single_benchmark(model_config)
                self.results[model_config.name] = result

        total_time = time.time() - start_time

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "models": [asdict(m) for m in models],
            "results": self.results,
        }

        summary_file = self.output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"Benchmarks complete! Total time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {summary_file}")
        print(f"{'='*60}\n")

        return self.results

    def _run_single_benchmark(self, model_config: ModelConfig) -> dict:
        """Run a single benchmark (provision + train + stop)."""
        instance = None
        try:
            instance = self.provision_instance(model_config)
            if not instance:
                return {"error": "Failed to provision instance"}

            self.instances[model_config.name] = instance

            result = self.run_benchmark(model_config, instance)
            return result

        except Exception as e:
            # Sanitize error message for Windows console compatibility
            try:
                error_msg = str(e)
            except Exception:
                error_msg = repr(e)
            # Remove any non-ASCII characters to avoid Windows console issues
            error_msg = ''.join(c if ord(c) < 128 else '?' for c in error_msg)
            return {"error": error_msg}

        finally:
            # Stop instance (preserve for inspection)
            if instance and instance.instance_id:
                print(f"[{model_config.name}] Stopping instance {instance.instance_id}...")
                instance.stop_instance()

    def cleanup_instances(self, destroy: bool = False):
        """Stop or destroy all instances."""
        for name, instance in self.instances.items():
            if instance.instance_id:
                if destroy:
                    print(f"Destroying instance for {name}...")
                    instance.destroy()
                else:
                    print(f"Stopping instance for {name}...")
                    instance.stop_instance()


def main():
    parser = argparse.ArgumentParser(description="Run VLM benchmarks on vast.ai")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick benchmark with subset of data"
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Specific models to benchmark (e.g., qwen2-vl-2b qwen2.5-vl-3b)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run benchmarks sequentially instead of in parallel"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Maximum parallel instances"
    )
    parser.add_argument(
        "--hf-token", type=str,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--destroy-instances", action="store_true",
        help="Destroy instances instead of stopping them"
    )

    args = parser.parse_args()

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import HfApi
            hf_token = HfApi().token
        except Exception:
            pass

    if not hf_token:
        print("Error: HuggingFace token required. Set HF_TOKEN env var or use --hf-token")
        return

    # Select models
    if args.quick:
        models = QUICK_BENCHMARK_MODELS
    elif args.models:
        models = [m for m in BENCHMARK_MODELS if m.name in args.models]
        if not models:
            print(f"No matching models found. Available: {[m.name for m in BENCHMARK_MODELS]}")
            return
    else:
        models = BENCHMARK_MODELS

    print(f"Models to benchmark: {[m.name for m in models]}")

    runner = BenchmarkRunner(hf_token)

    try:
        results = runner.run_all_benchmarks(
            models=models,
            parallel=not args.sequential,
            max_workers=args.max_workers,
        )

        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*60)

        for model_name, result in results.items():
            if isinstance(result, dict):
                status = "SUCCESS" if result.get("success") else "FAILED"
                error = result.get("error", "")
                adapter_mb = result.get("adapter_size_mb", 0)
                info = f"adapter: {adapter_mb:.1f}MB" if adapter_mb else error
                print(f"{model_name}: {status} - {info}")

    except KeyboardInterrupt:
        print("\nInterrupted! Cleaning up...")
        runner.cleanup_instances(destroy=args.destroy_instances)


if __name__ == "__main__":
    main()
