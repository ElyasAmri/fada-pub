#!/usr/bin/env python3
"""
Launch Parallel MedGemma Inference

This script handles HF_TOKEN securely and launches parallel inference.
Run this directly - it will prompt for the token if not set.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_hf_token():
    """Get HuggingFace token from environment, cache, or prompt user"""
    # Check environment first
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"Using HF_TOKEN from environment (starts with {token[:8]}...)")
        return token

    # Check .env file in project root
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip('"\'')
                    print(f"Using HF_TOKEN from .env file (starts with {token[:8]}...)")
                    return token

    # Check HuggingFace cache (Linux/Mac: ~/.cache/huggingface/token, Windows: similar)
    cache_paths = [
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    for cache_path in cache_paths:
        if cache_path.exists():
            token = cache_path.read_text().strip()
            if token:
                print(f"Using HF_TOKEN from cache (starts with {token[:8]}...)")
                return token

    # Prompt user
    print("HF_TOKEN not found in environment, .env, or cache.")
    token = input("Enter your HuggingFace token: ").strip()

    if not token:
        print("Error: Token is required")
        sys.exit(1)

    return token


def main():
    # Get paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    parallel_script = script_dir / "run_parallel_vast.py"
    python_exe = project_root / "venv" / "Scripts" / "python.exe"
    vastai_exe = project_root / "venv" / "Scripts" / "vastai.exe"

    # Get current MedGemma progress
    checkpoint_file = script_dir / "results" / "checkpoint_vllm_google_medgemma-4b-it.json"
    start_from = 7870  # Default

    if checkpoint_file.exists():
        import json
        with open(checkpoint_file) as f:
            data = json.load(f)
            start_from = len(data.get("completed_images", {}))
        print(f"Found checkpoint: {start_from} images already completed")

    # Get token
    hf_token = get_hf_token()

    # Build command
    cmd = [
        str(python_exe),
        str(parallel_script),
        "--num-machines", "6",
        "--start-from", str(start_from),
        "--total-images", "19019",
        "--vastai-path", str(vastai_exe),
        "--hf-token", hf_token,
        "--max-price", "0.50",
        "--base-port", "8001"
    ]

    print("\n" + "=" * 60)
    print("Launching Parallel MedGemma Inference")
    print("=" * 60)
    print(f"Start from: {start_from}")
    print(f"Total images: 19019")
    print(f"Remaining: {19019 - start_from}")
    print(f"Machines: 6 x RTX 5090")
    print(f"Estimated time: ~4-5 hours")
    print(f"Estimated cost: ~$9")
    print("=" * 60 + "\n")

    # Confirm
    confirm = input("Proceed? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        sys.exit(0)

    # Run
    os.environ["HF_TOKEN"] = hf_token
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
