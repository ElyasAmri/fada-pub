"""
LLaMA Factory training wrapper for framework comparison.

Generates a LLaMA Factory YAML config and invokes `llamafactory-cli train`.
Uses ShareGPT format data. Outputs adapter in HF PEFT format.

Usage:
    python wrappers/train_llamafactory.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train-data /path/to/train_sharegpt.jsonl \
        --val-data /path/to/val_sharegpt.jsonl \
        --data-root /path/to/images \
        --output-dir /path/to/output \
        --config /path/to/config.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Map model IDs to LLaMA Factory templates using regex for precise matching.
# Each entry: (compiled_regex, template_name)
# More specific patterns MUST come before generic ones.
import re

MODEL_TEMPLATE_PATTERNS = [
    (re.compile(r'qwen2[\.\-]?5.*vl', re.I), "qwen2_vl"),
    (re.compile(r'qwen3.*vl', re.I), "qwen2_vl"),
    (re.compile(r'internvl', re.I), "intern_vl"),
    (re.compile(r'minicpm[\-_]o', re.I), "minicpm_o"),
    (re.compile(r'minicpm', re.I), "minicpm_v"),
    (re.compile(r'kimi', re.I), "kimi_vl"),
    (re.compile(r'llava[\-_]onevision', re.I), "llava_next_qwen"),
    (re.compile(r'llava[\-_]next', re.I), "llava_next"),
    (re.compile(r'llava', re.I), "llava"),
    (re.compile(r'llama[\-_]3[\.\-]2.*vision|mllama', re.I), "mllama"),
    (re.compile(r'pixtral', re.I), "pixtral"),
    (re.compile(r'mistral[\-_]small', re.I), "mistral_small"),
    (re.compile(r'mistral', re.I), "mistral"),
    (re.compile(r'medgemma', re.I), "gemma3"),
    (re.compile(r'paligemma', re.I), "paligemma"),
    (re.compile(r'gemma', re.I), "gemma3"),
]


def _classify_error(stderr: str) -> str:
    """Classify training error from stderr output."""
    s = stderr.lower()
    if "cuda out of memory" in s or "outofmemoryerror" in s:
        return "oom"
    if "no module named" in s or "modulenotfounderror" in s:
        return "missing_package"
    if "does not support image input" in s or "template" in s:
        return "template_mismatch"
    if "image features and image tokens do not match" in s:
        return "image_token_mismatch"
    if "no space left on device" in s:
        return "disk_full"
    if "not a valid model identifier" in s or "repository not found" in s:
        return "model_not_found"
    if "processor was not found" in s or "no attribute" in s:
        return "processor_error"
    return "unknown"


def detect_template(model_id: str) -> str:
    """Detect LLaMA Factory template from model ID using regex matching."""
    for pattern, template in MODEL_TEMPLATE_PATTERNS:
        if pattern.search(model_id):
            return template
    print(f"WARNING: No template matched for '{model_id}'. Using 'default' — this will likely fail.", file=sys.stderr)
    return "default"


def main():
    parser = argparse.ArgumentParser(description="LLaMA Factory LoRA fine-tuning wrapper")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--train-data", required=True, help="Training JSONL path (ShareGPT format)")
    parser.add_argument("--val-data", default=None, help="Validation JSONL path (ShareGPT format)")
    parser.add_argument("--data-root", required=True, help="Root directory for image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--test-run", action="store_true", help="Quick test with 100 samples")
    parser.add_argument("--keep-cache", action="store_true", help="Skip HF cache cleanup before training")
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent.parent / "config.json")
    with open(config_path) as f:
        config = json.load(f)

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    quant_cfg = config["quantization"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defensive disk check
    free_gb = shutil.disk_usage("/").free / 1e9
    if free_gb < 5:
        print(f"FATAL: Only {free_gb:.1f}GB free disk space. Aborting.", file=sys.stderr)
        sys.exit(1)

    template = detect_template(args.model)

    # Clear HF cache before downloading to avoid disk-full failures
    if not args.keep_cache:
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_cache.exists():
            import shutil
            cache_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file()) / 1e9
            if cache_size > 1.0:
                print(f"Clearing HF cache ({cache_size:.1f}GB) to free disk space...")
                for d in hf_cache.iterdir():
                    if d.name.startswith("models--"):
                        shutil.rmtree(d, ignore_errors=True)

    # Register dataset in a temporary dataset_info.json
    image_folder = str(Path(args.data_root).resolve())
    dataset_info = {
        "fada_train": {
            "file_name": str(Path(args.train_data).resolve()),
            "formatting": "sharegpt",
            "folder": image_folder,
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"},
        }
    }
    if args.val_data and Path(args.val_data).exists():
        dataset_info["fada_val"] = {
            "file_name": str(Path(args.val_data).resolve()),
            "formatting": "sharegpt",
            "folder": image_folder,
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"},
        }

    dataset_info_path = output_dir / "dataset_info.json"
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Build LLaMA Factory training config
    lf_config = {
        "stage": "sft",
        "model_name_or_path": args.model,
        "template": template,
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": lora_cfg["r"],
        "lora_alpha": lora_cfg["lora_alpha"],
        "lora_dropout": lora_cfg["lora_dropout"],
        "lora_target": "all" if lora_cfg["target_modules"] == "all-linear" else lora_cfg["target_modules"],
        "dataset": "fada_train",
        "dataset_dir": str(output_dir),
        "cutoff_len": train_cfg["max_seq_length"],
        "output_dir": str(output_dir / "adapter"),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "bf16": train_cfg["bf16"],
        "optim": train_cfg["optim"],
        "weight_decay": train_cfg["weight_decay"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "logging_steps": train_cfg["logging_steps"],
        "save_strategy": train_cfg["save_strategy"],
        "dataloader_num_workers": train_cfg["dataloader_num_workers"],
        "report_to": "none",
        "seed": config["data"]["seed"],
        # Quantization
        "quantization_bit": 4 if quant_cfg["load_in_4bit"] else None,
        "quantization_method": "bitsandbytes",
        "trust_remote_code": True,
    }

    if args.val_data and Path(args.val_data).exists():
        lf_config["eval_dataset"] = "fada_val"
        lf_config["eval_strategy"] = train_cfg["eval_strategy"]

    if train_cfg.get("max_samples"):
        lf_config["max_samples"] = train_cfg["max_samples"]

    if args.test_run:
        lf_config["max_samples"] = 100
        lf_config["num_train_epochs"] = 1
        lf_config["logging_steps"] = 1

    # Write config YAML
    lf_config_path = output_dir / "llamafactory_config.yaml"
    with open(lf_config_path, "w") as f:
        yaml.dump(lf_config, f, default_flow_style=False)

    print(f"LLaMA Factory config written to {lf_config_path}")
    print(f"Template: {template}")
    print(f"Training {args.model}...")

    # Run training
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cmd = ["llamafactory-cli", "train", str(lf_config_path)]
    result = subprocess.run(cmd, text=True, stderr=subprocess.PIPE)

    training_time = time.time() - start_time
    gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    if result.returncode != 0:
        stderr = result.stderr or ""
        error_category = _classify_error(stderr)
        manifest = {
            "model": args.model,
            "framework": "llamafactory",
            "status": "failed",
            "error": f"llamafactory-cli exited with code {result.returncode}",
            "error_category": error_category,
            "training_time_seconds": round(training_time, 1),
        }
        with open(output_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"FAILED [{error_category}]: {stderr[-500:]}", file=sys.stderr)
        sys.exit(1)

    # Read trainer_state.json for final loss
    final_loss = None
    trainer_state_path = output_dir / "adapter" / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            state = json.load(f)
        for entry in reversed(state.get("log_history", [])):
            if "loss" in entry:
                final_loss = entry["loss"]
                break

    # Validate adapter was saved
    adapter_dir = output_dir / "adapter"
    adapter_ok = (adapter_dir / "adapter_config.json").exists() and list(adapter_dir.glob("adapter_model*"))
    status = "complete" if adapter_ok else "failed"
    if not adapter_ok:
        print("ERROR: Adapter files missing after training!", file=sys.stderr)

    # Write manifest
    manifest = {
        "model": args.model,
        "framework": "llamafactory",
        "status": status,
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "training_time_seconds": round(training_time, 1),
        "final_loss": final_loss,
        "gpu_memory_peak_gb": round(gpu_mem_peak / 1e9, 2),
        "adapter_path": str(adapter_dir),
    }
    if not adapter_ok:
        manifest["error"] = "Adapter files not found after training"
        manifest["error_category"] = "adapter_save_failed"
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training complete in {training_time:.0f}s | Loss: {final_loss} | GPU peak: {gpu_mem_peak / 1e9:.1f}GB")
    if not adapter_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
