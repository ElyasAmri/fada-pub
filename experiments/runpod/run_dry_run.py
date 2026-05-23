"""
RunPod VLM comprehensive dry run - tests ALL models.

Two-phase testing with different transformers versions:

Phase 1: transformers 4.45 (base environment)
- test_transformers_445.py - 18 models (InternVL2/3, Qwen2-VL, MiniCPM, Kimi, Llama, Phi)

Phase 2: transformers 4.48+ (isolated venv)
- test_transformers_448.py - 11 models (SmolVLM2, InternVL3.5, Qwen2.5-VL)

Total: 29 models tested

Usage:
    python experiments/runpod/run_dry_run.py
"""

import os
import time
import subprocess
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from runpod_instance import RunPodInstance

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env.local")


class RunPodDryRun:
    """Manages RunPod dry run execution."""

    def __init__(self):
        self.instance = RunPodInstance()
        self.project_root = project_root
        self.remote_workdir = "/workspace/test"

    def estimate_cost(self, runtime_minutes: int = 120) -> float:
        """Estimate cost for RTX 4090 at $0.40/hr."""
        return 0.40 * (runtime_minutes / 60)

    def scp_upload(self, local_path: Path, remote_path: str, timeout: int = 300) -> bool:
        """Upload file to pod via SCP."""
        if not self.instance.ssh_host or not self.instance.ssh_port:
            print("ERROR: SSH not available")
            return False

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-o", "LogLevel=ERROR",
            "-i", os.path.expanduser("~/.ssh/id_rsa"),
            "-P", str(self.instance.ssh_port),
            str(local_path),
            f"root@{self.instance.ssh_host}:{remote_path}"
        ]

        try:
            print(f"Uploading {local_path.name}...")
            result = subprocess.run(scp_cmd, capture_output=True, timeout=timeout)
            if result.returncode == 0:
                print(f"  OK: {local_path.name}")
                return True
            else:
                print(f"  ERROR: {result.stderr.decode('utf-8', errors='replace')}")
                return False
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def scp_download(self, remote_path: str, local_path: Path, timeout: int = 300) -> bool:
        """Download file from pod via SCP."""
        if not self.instance.ssh_host or not self.instance.ssh_port:
            return False

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-o", "LogLevel=ERROR",
            "-i", os.path.expanduser("~/.ssh/id_rsa"),
            "-P", str(self.instance.ssh_port),
            f"root@{self.instance.ssh_host}:{remote_path}",
            str(local_path)
        ]

        try:
            result = subprocess.run(scp_cmd, capture_output=True, timeout=timeout)
            return result.returncode == 0
        except Exception:
            return False

    def setup_remote_environment(self) -> bool:
        """Setup remote environment with transformers 4.45 (base)."""
        print("\n" + "="*80)
        print("Setting up remote environment (transformers 4.45)...")
        print("="*80)

        commands = [
            f"mkdir -p {self.remote_workdir}",
            # CRITICAL: Use transformers 4.45 - version 5.0 breaks all VLM models!
            "pip install -q transformers==4.45.0 accelerate bitsandbytes pillow tiktoken soundfile optimum huggingface_hub qwen-vl-utils",
            # Install VLM dependencies including missing ones (protobuf, librosa, vector_quantize_pytorch, vocos)
            "pip install -q num2words einops timm auto-gptq autoawq protobuf librosa vector_quantize_pytorch vocos flash-attn --no-build-isolation 2>/dev/null || pip install -q num2words einops timm auto-gptq autoawq protobuf librosa vector_quantize_pytorch vocos",
            "python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'",
            "python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'",
        ]

        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] {cmd[:70]}...")
            returncode, stdout, stderr = self.instance.run_ssh(cmd, timeout=600)

            stdout_safe = stdout.encode('ascii', 'replace').decode()
            stderr_safe = stderr.encode('ascii', 'replace').decode()

            if returncode != 0:
                print(f"  ERROR: {stderr_safe}")
                return False
            if stdout_safe.strip():
                print(f"  {stdout_safe.strip()}")
            print(f"  OK")

        return True

    def setup_new_transformers_venv(self) -> bool:
        """Setup isolated venv with transformers 4.48+ for newer models."""
        print("\n" + "="*80)
        print("Setting up isolated venv (transformers 4.48+)...")
        print("="*80)

        commands = [
            # Use --system-site-packages to inherit torch from base environment (avoids re-downloading)
            f"cd {self.remote_workdir} && python -m venv --system-site-packages venv_new_transformers",
            f"{self.remote_workdir}/venv_new_transformers/bin/pip install -q --upgrade pip",
            # Upgrade transformers to 4.48+ (gptqmodel skipped - requires CUDA compilation)
            f"{self.remote_workdir}/venv_new_transformers/bin/pip install -q 'transformers>=4.48.0' qwen-vl-utils",
            f"{self.remote_workdir}/venv_new_transformers/bin/python -c 'import torch; print(f\"PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}\")'",
            f"{self.remote_workdir}/venv_new_transformers/bin/python -c 'import transformers; print(f\"Transformers: {{transformers.__version__}}\")'",
        ]

        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] {cmd[:80]}...")
            returncode, stdout, stderr = self.instance.run_ssh(cmd, timeout=300)

            stdout_safe = stdout.encode('ascii', 'replace').decode()
            stderr_safe = stderr.encode('ascii', 'replace').decode()

            if returncode != 0:
                print(f"  ERROR: {stderr_safe[:300]}")
                # Don't fail completely - this is optional
                print(f"  WARNING: New transformers venv setup failed, skipping newer models")
                return False
            if stdout_safe.strip():
                print(f"  {stdout_safe.strip()}")
            print(f"  OK")

        return True

    def upload_test_files(self) -> bool:
        """Upload all test scripts and data."""
        print("\n" + "="*80)
        print("Uploading test files...")
        print("="*80)

        files = [
            (self.project_root / "experiments" / "runpod" / "test_transformers_445.py", f"{self.remote_workdir}/test_transformers_445.py"),
            (self.project_root / "experiments" / "runpod" / "test_transformers_448.py", f"{self.remote_workdir}/test_transformers_448.py"),
            (self.project_root / "experiments" / "runpod" / "test_gptq_awq.py", f"{self.remote_workdir}/test_gptq_awq.py"),
            (self.project_root / "experiments" / "runpod" / "test_legacy_models.py", f"{self.remote_workdir}/test_legacy_models.py"),
            (self.project_root / "outputs" / "evaluation" / "test_subset.jsonl", f"{self.remote_workdir}/test_subset.jsonl"),
        ]

        for local_path, remote_path in files:
            if not local_path.exists():
                print(f"WARNING: {local_path} not found, skipping")
                continue
            if not self.scp_upload(local_path, remote_path):
                return False

        return True

    def run_test_script(self, script_name: str, description: str, timeout: int = 3600) -> bool:
        """Run a single test script."""
        print(f"\n" + "="*80)
        print(f"Running {description}...")
        print("="*80)

        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        token_export = f"export HF_TOKEN={hf_token} && " if hf_token else ""

        cmd = f"{token_export}cd {self.remote_workdir} && python {script_name}"
        print(f"Command: python {script_name}")

        returncode, stdout, stderr = self.instance.run_ssh(cmd, timeout=timeout)

        stdout_safe = stdout.encode('ascii', 'replace').decode()
        stderr_safe = stderr.encode('ascii', 'replace').decode()

        print("\n" + "-"*80)
        print("OUTPUT:")
        print("-"*80)
        print(stdout_safe)

        if stderr_safe.strip():
            print("\nSTDERR (last 500 chars):")
            print(stderr_safe[-500:])

        return returncode == 0

    def run_new_transformers_test(self, timeout: int = 3600) -> bool:
        """Run test script in isolated venv with transformers 4.48+."""
        print(f"\n" + "="*80)
        print(f"Running transformers 4.48+ models (isolated venv)...")
        print("="*80)

        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        token_export = f"export HF_TOKEN={hf_token} && " if hf_token else ""

        # Use the isolated venv's Python
        python_path = f"{self.remote_workdir}/venv_new_transformers/bin/python"
        cmd = f"{token_export}cd {self.remote_workdir} && {python_path} test_transformers_448.py"
        print(f"Command: {python_path} test_transformers_448.py")

        returncode, stdout, stderr = self.instance.run_ssh(cmd, timeout=timeout)

        stdout_safe = stdout.encode('ascii', 'replace').decode()
        stderr_safe = stderr.encode('ascii', 'replace').decode()

        print("\n" + "-"*80)
        print("OUTPUT:")
        print("-"*80)
        print(stdout_safe)

        if stderr_safe.strip():
            print("\nSTDERR (last 500 chars):")
            print(stderr_safe[-500:])

        return returncode == 0

    def download_results(self) -> dict:
        """Download and merge all results."""
        print("\n" + "="*80)
        print("Downloading results...")
        print("="*80)

        local_dir = self.project_root / "outputs" / "runpod_dry_run"
        local_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        result_files = [
            ("results_main.json", "Main models (transformers 4.45)"),
            ("results_new_transformers.json", "New transformers 4.48+ models"),
            ("results_gptq.json", "GPTQ/AWQ models"),
            ("results_legacy.json", "Legacy models"),
        ]

        for remote_name, desc in result_files:
            local_path = local_dir / remote_name
            remote_path = f"{self.remote_workdir}/{remote_name}"

            if self.scp_download(remote_path, local_path):
                print(f"  Downloaded {remote_name}")
                try:
                    with open(local_path) as f:
                        results = json.load(f)
                        all_results.extend(results)
                except Exception:
                    print(f"  WARNING: Could not parse {remote_name}")
            else:
                print(f"  WARNING: {remote_name} not found")

        # Save merged results
        merged_path = local_dir / "all_results.json"
        with open(merged_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nMerged results saved to {merged_path}")

        return all_results

    def print_summary(self, results: list):
        """Print final summary."""
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        if not results:
            print("No results collected")
            return

        success = [r for r in results if r.get("inference_success") or r.get("success")]
        failed = [r for r in results if r.get("error_message") or r.get("error")]
        skipped = [r for r in results if r.get("skipped") or (r.get("error") and "GB" in str(r.get("error", "")))]

        print(f"\nTotal: {len(results)}")
        print(f"Success: {len(success)}")
        print(f"Failed: {len(failed)}")
        print(f"Skipped (VRAM): {len(skipped)}")

        if success:
            print(f"\n[SUCCESS] ({len(success)}):")
            for r in success:
                name = r.get("model_name") or r.get("name")
                print(f"  + {name}")

        if failed:
            print(f"\n[FAILED] ({len(failed)}):")
            for r in failed:
                name = r.get("model_name") or r.get("name")
                err = r.get("error_message") or r.get("error") or "Unknown"
                print(f"  - {name}: {err[:60]}")

    def run(self):
        """Execute full dry run workflow."""
        print("="*80)
        print("RunPod VLM Comprehensive Dry Run - ALL MODELS")
        print("="*80)

        estimated_cost = self.estimate_cost(runtime_minutes=120)
        print(f"\nEstimated cost: ${estimated_cost:.2f} (RTX 4090 @ $0.40/hr, ~2hr)")

        if os.environ.get("RUNPOD_CONFIRM", "yes").lower() == "no":
            print("\nCancelled (RUNPOD_CONFIRM=no)")
            return

        hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("\nWARNING: No HF_TOKEN found - gated models will fail")

        try:
            # Create pod
            print("\n" + "="*80)
            print("Creating RunPod instance...")
            print("="*80)

            env_vars = {}
            if hf_token:
                env_vars["HF_TOKEN"] = hf_token
                env_vars["HUGGING_FACE_HUB_TOKEN"] = hf_token

            pod_id = self.instance.create_pod(
                gpu_type="NVIDIA GeForce RTX 4090",
                image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                disk_gb=300,  # Increased to 300GB - need more space for all models
                name="vlm-full-test",
                env_vars=env_vars if env_vars else None
            )

            if not pod_id:
                print("ERROR: Failed to create pod")
                return

            print("\nWaiting for pod...")
            if not self.instance.wait_for_ready(timeout=300):
                print("ERROR: Pod not ready")
                return

            # Setup
            if not self.setup_remote_environment():
                print("ERROR: Setup failed")
                return

            # Upload
            if not self.upload_test_files():
                print("ERROR: Upload failed")
                return

            # Run all models with transformers 4.45 (5.0 breaks everything!)
            self.run_test_script("test_transformers_445.py", "All VLM models (transformers 4.45)", timeout=7200)

            # Setup and run transformers 4.48+ models in isolated venv
            if self.setup_new_transformers_venv():
                self.run_new_transformers_test(timeout=3600)
            else:
                print("\nSkipping transformers 4.48+ models (venv setup failed)")

            # Download and summarize
            results = self.download_results()
            self.print_summary(results)

            print("\n" + "="*80)
            print("DRY RUN COMPLETE")
            print("="*80)

        finally:
            print("\n" + "="*80)
            print("Cleaning up...")
            print("="*80)
            self.instance.destroy_pod()


def main():
    dry_run = RunPodDryRun()
    dry_run.run()


if __name__ == "__main__":
    main()
