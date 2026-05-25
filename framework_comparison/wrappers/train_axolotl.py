"""
Axolotl training wrapper for framework comparison.

Generates an Axolotl YAML config and invokes `accelerate launch -m axolotl.cli.train`.
Uses ShareGPT format data. Outputs adapter in HF PEFT format.

Usage:
    python wrappers/train_axolotl.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train-data /path/to/train_sharegpt.jsonl \
        --data-root /path/to/images \
        --output-dir /path/to/output \
        --config /path/to/config.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Axolotl model type mapping
MODEL_TYPES = {
    "qwen2.5-vl": "AutoModelForVision2Seq",
    "qwen3-vl": "AutoModelForVision2Seq",
    "qwen3.5": "AutoModelForVision2Seq",
    "mistral": "AutoModelForCausalLM",
}


def detect_model_class(model_id: str) -> str:
    model_lower = model_id.lower()
    for key, cls in MODEL_TYPES.items():
        if key in model_lower:
            return cls
    return "AutoModelForVision2Seq"


def main():
    parser = argparse.ArgumentParser(description="Axolotl LoRA fine-tuning wrapper")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--train-data", required=True, help="Training JSONL path (ShareGPT format)")
    parser.add_argument("--val-data", default=None, help="Validation JSONL path (ShareGPT format)")
    parser.add_argument("--data-root", required=True, help="Root directory for image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--test-run", action="store_true", help="Quick test with 100 samples")
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent.parent / "config.json")
    with open(config_path) as f:
        config = json.load(f)

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    quant_cfg = config["quantization"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_class = detect_model_class(args.model)

    # Build datasets list
    datasets_config = [{
        "path": str(Path(args.train_data).resolve()),
        "type": "sharegpt",
        "conversation": "chatml",
    }]

    # Build Axolotl config
    ax_config = {
        "base_model": args.model,
        "model_type": model_class,
        "trust_remote_code": True,

        # LoRA
        "adapter": "lora",
        "lora_r": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "lora_dropout": lora_cfg["lora_dropout"],
        "lora_target_linear": True,

        # Quantization
        "load_in_4bit": quant_cfg["load_in_4bit"],
        "bnb_4bit_quant_type": quant_cfg["bnb_4bit_quant_type"],
        "bnb_4bit_use_double_quant": quant_cfg["bnb_4bit_use_double_quant"],

        # Data
        "datasets": datasets_config,
        "sequence_len": train_cfg["max_seq_length"],
        "sample_packing": False,  # Disable for VLM compatibility

        # Training
        "num_epochs": train_cfg["num_train_epochs"],
        "micro_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "lr_scheduler": train_cfg["lr_scheduler_type"],
        "optimizer": train_cfg["optim"],
        "weight_decay": train_cfg["weight_decay"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "logging_steps": train_cfg["logging_steps"],
        "save_strategy": train_cfg["save_strategy"],

        # Output
        "output_dir": str(output_dir / "adapter"),
        "bf16": "auto",
        "tf32": True,
        "seed": config["data"]["seed"],

        # Misc
        "flash_attention": True,
        "wandb_project": None,
        "hub_model_id": None,
    }

    if args.val_data and Path(args.val_data).exists():
        ax_config["val_set_size"] = 0.0  # We provide explicit val data
        datasets_config.append({
            "path": str(Path(args.val_data).resolve()),
            "type": "sharegpt",
            "conversation": "chatml",
            "split": "validation",
        })

    if args.test_run:
        ax_config["max_steps"] = 10
        ax_config["logging_steps"] = 1

    # Write Axolotl config
    ax_config_path = output_dir / "axolotl_config.yaml"
    with open(ax_config_path, "w") as f:
        yaml.dump(ax_config, f, default_flow_style=False)

    print(f"Axolotl config written to {ax_config_path}")
    print(f"Model class: {model_class}")
    print(f"Training {args.model}...")

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cmd = [
        "accelerate", "launch", "-m", "axolotl.cli.train",
        str(ax_config_path),
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    training_time = time.time() - start_time
    gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    if result.returncode != 0:
        manifest = {
            "model": args.model,
            "framework": "axolotl",
            "status": "failed",
            "error": f"axolotl exited with code {result.returncode}",
            "training_time_seconds": round(training_time, 1),
        }
        with open(output_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        sys.exit(1)

    # Read final loss from trainer state
    final_loss = None
    adapter_dir = output_dir / "adapter"
    for state_path in adapter_dir.rglob("trainer_state.json"):
        with open(state_path) as f:
            state = json.load(f)
        for entry in reversed(state.get("log_history", [])):
            if "loss" in entry:
                final_loss = entry["loss"]
                break
        break

    manifest = {
        "model": args.model,
        "framework": "axolotl",
        "status": "complete",
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "training_time_seconds": round(training_time, 1),
        "final_loss": final_loss,
        "gpu_memory_peak_gb": round(gpu_mem_peak / 1e9, 2),
        "adapter_path": str(adapter_dir),
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training complete in {training_time:.0f}s | Loss: {final_loss} | GPU peak: {gpu_mem_peak / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
