"""
Unsloth training wrapper for framework comparison.

Uses Unsloth's FastModel for optimized LoRA fine-tuning with 4-bit quantization.
Outputs adapter in HF PEFT format.

Usage:
    python wrappers/train_unsloth.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train-data /path/to/train.jsonl \
        --val-data /path/to/val.jsonl \
        --data-root /path/to/images \
        --output-dir /path/to/output \
        --config /path/to/config.json

    # Quick test
    python wrappers/train_unsloth.py --model Qwen/Qwen2.5-VL-7B-Instruct --test-run
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class VLMDataset(Dataset):
    """Dataset for VLM fine-tuning from HF chat format JSONL."""

    def __init__(self, jsonl_path, processor, max_samples=None, max_length=4096,
                 data_root=None, architecture="qwen"):
        self.processor = processor
        self.max_length = max_length
        self.data_root = Path(data_root) if data_root else None
        self.architecture = architecture
        self._missing_count = 0
        self.samples = []

        if self.architecture == "qwen":
            self._response_template_ids = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            self._end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            self._gemma_model_turn_ids = []
            self._gemma_turn_end_id = None
        elif self.architecture == "gemma":
            self._response_template_ids = []
            self._end_token_id = None
            self._gemma_model_turn_ids = self.processor.tokenizer.encode(
                "<start_of_turn>model\n", add_special_tokens=False,
            )
            end_ids = self.processor.tokenizer.encode("<end_of_turn>", add_special_tokens=False)
            self._gemma_turn_end_id = end_ids[0] if end_ids else None
        else:
            self._response_template_ids = []
            self._end_token_id = None
            self._gemma_model_turn_ids = []
            self._gemma_turn_end_id = None

        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = self._normalize_messages(sample["messages"])
        image_paths = sample.get("images", [])

        images = []
        for img_path in image_paths:
            if self.data_root and not os.path.isabs(img_path):
                full_path = (self.data_root / img_path).resolve()
                if not full_path.is_relative_to(self.data_root.resolve()):
                    raise ValueError(f"Path traversal detected: {img_path}")
                img_path = str(full_path)
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except FileNotFoundError:
                self._missing_count += 1
                if self._missing_count > 10:
                    raise RuntimeError(f"Too many missing images ({self._missing_count}). Check --data-root.")
                images.append(Image.new("RGB", (224, 224), color="gray"))

        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except ValueError as exc:
            if "does not have a chat template" not in str(exc):
                raise
            text = self._messages_to_plain_text(messages)
        inputs = self.processor(
            text=text, images=images if images else None,
            return_tensors="pt", padding=False,
            truncation=True, max_length=self.max_length,
        )
        # Remove batch dim but keep image_grid_thw as 2D (num_images, 3)
        keep_2d = {"image_grid_thw", "video_grid_thw"}
        squeezed = {}
        for k, v in inputs.items():
            if k in keep_2d:
                # (1, num_images, 3) -> (num_images, 3)
                squeezed[k] = v.squeeze(0) if v.dim() == 3 else v
            else:
                squeezed[k] = v.squeeze(0)
        inputs = squeezed

        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[-1]
        for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
            if key not in inputs or inputs[key].shape[-1] == seq_len:
                continue
            diff = seq_len - inputs[key].shape[-1]
            if diff > 0:
                inputs[key] = torch.nn.functional.pad(inputs[key], (0, diff), value=0)
            else:
                inputs[key] = inputs[key][..., :seq_len]

        labels = torch.full_like(input_ids, -100)

        if self._response_template_ids:
            template_len = len(self._response_template_ids)
            ids_list = input_ids.tolist()
            found_any_response = False
            i = 0
            while i <= len(ids_list) - template_len:
                if ids_list[i:i + template_len] == self._response_template_ids:
                    found_any_response = True
                    j = i + template_len
                    while j < len(ids_list):
                        labels[j] = input_ids[j]
                        if ids_list[j] == self._end_token_id:
                            break
                        j += 1
                    i = j + 1
                else:
                    i += 1
            if not found_any_response:
                labels = input_ids.clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
        else:
            labels = self._build_assistant_only_labels(input_ids)
            if labels is None:
                labels = input_ids.clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs

    def _normalize_messages(self, messages):
        if self.architecture != "gemma":
            return messages

        normalized = []
        system_parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                if isinstance(content, str) and content:
                    system_parts.append(content)
                continue

            message = dict(message)
            if role == "user" and system_parts:
                system_text = "System instruction: " + "\n".join(system_parts)
                if isinstance(content, list):
                    message["content"] = [{"type": "text", "text": system_text}, *content]
                elif isinstance(content, str):
                    message["content"] = f"{system_text}\n\n{content}"
                system_parts = []
            normalized.append(message)

        return normalized

    def _messages_to_plain_text(self, messages):
        parts = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if isinstance(content, list):
                text = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ).strip()
            else:
                text = str(content).strip()
            if text:
                parts.append(text if role == "user" else f"{role}: {text}")
        return "\n\n".join(parts)

    def _build_assistant_only_labels(self, input_ids):
        if self.architecture == "gemma":
            return self._build_gemma_turn_labels(input_ids)
        return None

    def _build_gemma_turn_labels(self, input_ids):
        if not self._gemma_model_turn_ids or self._gemma_turn_end_id is None:
            return None

        ids_list = input_ids.tolist()
        labels = torch.full_like(input_ids, -100)
        found_assistant_span = False

        marker_len = len(self._gemma_model_turn_ids)
        i = 0
        while i <= len(ids_list) - marker_len:
            if ids_list[i:i + marker_len] == self._gemma_model_turn_ids:
                start = i + marker_len
                end = start
                while end < len(ids_list) and ids_list[end] != self._gemma_turn_end_id:
                    end += 1
                if end > start:
                    labels[start:end] = input_ids[start:end]
                    found_assistant_span = True
                i = end + 1
            else:
                i += 1

        return labels if found_assistant_span else None

    def _build_assistant_only_labels_template_mask(self, messages, input_ids):
        tokenizer = self.processor.tokenizer
        try:
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            assistant_mask = tokenized.get("assistant_masks")
            if assistant_mask and len(assistant_mask) == len(input_ids):
                if sum(assistant_mask) > 0:
                    mask_tensor = torch.tensor(assistant_mask, dtype=torch.bool)
                    labels = torch.full_like(input_ids, -100)
                    labels[mask_tensor] = input_ids[mask_tensor]
                    return labels
        except Exception:
            pass
        return None


def collate_fn(batch, architecture: str = "qwen"):
    """Custom collator for variable-sized image tensors and dynamic padding."""
    # Qwen VL packs visual tokens per sample; those fields need concat behavior.
    # Gemma and other models use standard batched tensors for pixel values.
    concat_keys = {"image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    if architecture in {"qwen", "internvl"}:
        concat_keys.add("pixel_values")
    # Pad sequence tensors to longest in batch
    pad_keys = {"input_ids", "attention_mask", "labels", "mm_token_type_ids", "token_type_ids"}
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            if key in concat_keys:
                collated[key] = torch.cat(values, dim=0)
            elif key in pad_keys:
                max_len = max(v.shape[-1] for v in values)
                pad_value = -100 if key == "labels" else 0
                padded = []
                for v in values:
                    pad_size = max_len - v.shape[-1]
                    if pad_size > 0:
                        padded.append(torch.nn.functional.pad(v, (0, pad_size), value=pad_value))
                    else:
                        padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                shapes = {v.shape for v in values}
                collated[key] = torch.stack(values) if len(shapes) == 1 else torch.cat(values, dim=0)
        else:
            collated[key] = values
    return collated


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output directory."""
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def detect_architecture(model_id: str) -> str:
    model_lower = model_id.lower()
    if "qwen" in model_lower:
        return "qwen"
    if "internvl" in model_lower:
        return "internvl"
    if "minicpm" in model_lower:
        return "minicpm"
    if "paligemma" in model_lower:
        return "paligemma"
    if "gemma" in model_lower:
        return "gemma"
    return "generic"


def main():
    parser = argparse.ArgumentParser(description="Unsloth LoRA fine-tuning wrapper")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--train-data", required=True, help="Training JSONL path")
    parser.add_argument("--val-data", default=None, help="Validation JSONL path")
    parser.add_argument("--data-root", required=True, help="Root directory for image paths")
    parser.add_argument("--output-dir", required=True, help="Output directory for adapter")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--test-run", action="store_true", help="Quick test with 100 samples, 1 step")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically resume from latest checkpoint in output directory")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override dataloader_num_workers from config")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_train_epochs from config")
    parser.add_argument("--max-runtime-minutes", type=float, default=None,
                        help="Hard wall-clock budget for the full run")
    args = parser.parse_args()
    deadline_ts = None if args.max_runtime_minutes is None else time.time() + args.max_runtime_minutes * 60

    # Load config
    config_path = args.config or str(Path(__file__).parent.parent / "config.json")
    with open(config_path) as f:
        config = json.load(f)

    lora_cfg = config["lora"]
    train_cfg = config["training"]
    quant_cfg = config["quantization"]

    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        train_cfg["num_train_epochs"] = args.epochs

    if args.test_run:
        max_samples = 100
        train_cfg = {**train_cfg, "num_train_epochs": 1, "logging_steps": 1, "save_strategy": "no",
                     "eval_strategy": "no"}
    else:
        max_samples = train_cfg.get("max_samples", None)

    # Defensive disk check
    free_gb = shutil.disk_usage("/").free / 1e9
    if free_gb < 5:
        print(f"FATAL: Only {free_gb:.1f}GB free disk space. Aborting.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import unsloth
    from unsloth import FastModel
    if importlib.util.find_spec("flash_attn") is None:
        print("flash-attn is not installed in this environment; Unsloth will fall back to a slower attention path.")

    print(f"Loading model with Unsloth: {args.model}")
    model, processor = FastModel.from_pretrained(
        args.model,
        max_seq_length=train_cfg["max_seq_length"],
        load_in_4bit=quant_cfg["load_in_4bit"],
        dtype=torch.bfloat16,
    )

    # Apply LoRA via Unsloth
    model = FastModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load datasets
    architecture = detect_architecture(args.model)
    train_dataset = VLMDataset(
        args.train_data, processor, max_samples=max_samples,
        max_length=train_cfg["max_seq_length"], data_root=args.data_root,
        architecture=architecture,
    )
    eval_dataset = None
    if args.val_data and Path(args.val_data).exists():
        eval_dataset = VLMDataset(
            args.val_data, processor,
            max_samples=min(1000, max_samples) if max_samples else 1000,
            max_length=train_cfg["max_seq_length"], data_root=args.data_root,
            architecture=architecture,
        )

    if deadline_ts is not None and time.time() >= deadline_ts:
        raise TimeoutError("Runtime budget exhausted before training could start.")

    # Enable TF32 for H100 (faster matmuls with minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Training
    from transformers import TrainerCallback, TrainingArguments, Trainer

    class MaxRuntimeCallback(TrainerCallback):
        """Stop training once a wall-clock deadline is reached."""

        def __init__(self, deadline_ts: float):
            self.deadline_ts = deadline_ts
            self.triggered = False

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() >= self.deadline_ts:
                self.triggered = True
                control.should_training_stop = True
            return control

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        bf16=train_cfg["bf16"],
        tf32=True,
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg.get("eval_steps", 100),
        load_best_model_at_end=train_cfg["eval_strategy"] != "no",
        metric_for_best_model="eval_loss" if train_cfg["eval_strategy"] != "no" else None,
        dataloader_num_workers=args.num_workers if args.num_workers is not None else train_cfg["dataloader_num_workers"],
        dataloader_pin_memory=True,
        **({"dataloader_prefetch_factor": 2} if (args.num_workers if args.num_workers is not None else train_cfg["dataloader_num_workers"]) > 0 else {}),
        report_to="none",
        seed=config["data"]["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=lambda batch: collate_fn(batch, architecture=architecture),
    )
    runtime_callback = None
    if deadline_ts is not None:
        runtime_callback = MaxRuntimeCallback(deadline_ts)
        trainer.add_callback(runtime_callback)

    # Handle resume
    checkpoint = None
    if args.resume_from_checkpoint:
        checkpoint = args.resume_from_checkpoint
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"Resuming from checkpoint: {checkpoint}")
    elif args.auto_resume:
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint:
            print(f"Auto-resuming from {checkpoint}")
        else:
            print("Auto-resume: No checkpoints found, starting fresh")

    print("Starting training...")
    start_time = time.time()
    gpu_mem_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    training_time = time.time() - start_time
    gpu_mem_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    # Extract final loss
    final_loss = None
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break

    # Save adapter in PEFT format
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    # Save training log history (loss curves, eval metrics)
    log_history = trainer.state.log_history if trainer.state.log_history else []
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Training log saved: {len(log_history)} entries")

    # Write run manifest
    manifest = {
        "model": args.model,
        "framework": "unsloth",
        "status": "complete",
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "training_time_seconds": round(training_time, 1),
        "final_loss": final_loss,
        "gpu_memory_peak_gb": round(gpu_mem_peak / 1e9, 2),
        "train_samples": len(train_dataset),
        "val_samples": len(eval_dataset) if eval_dataset else 0,
        "adapter_path": str(adapter_dir),
        "max_runtime_minutes": args.max_runtime_minutes,
        "stopped_early_due_to_runtime_limit": runtime_callback.triggered if runtime_callback else False,
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training complete in {training_time:.0f}s | Loss: {final_loss} | GPU peak: {gpu_mem_peak / 1e9:.1f}GB")


if __name__ == "__main__":
    main()
