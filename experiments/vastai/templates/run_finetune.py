#!/usr/bin/env python3
"""
VLM Fine-Tuning Script for Vast.ai Remote Execution.

Runs LoRA fine-tuning on VLM models for fetal ultrasound VQA.
Designed to be uploaded and run on remote GPU instances.

Usage:
    python run_finetune.py --model OpenGVLab/InternVL3-2B --epochs 2
    python run_finetune.py --model Qwen/Qwen2.5-VL-3B-Instruct --batch-size 2
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


# Remote workspace paths
WORKSPACE = Path("/workspace/fada")
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "outputs"
MODELS_DIR = WORKSPACE / "models"

# System prompt for training
SYSTEM_PROMPT = """You are an expert fetal ultrasound analyst. Analyze the ultrasound image and answer the question thoroughly based on what you observe. Focus on anatomical structures, image quality, and any clinically relevant findings."""


class VLMDataset(Dataset):
    """Dataset for VLM fine-tuning from JSONL files."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        model_id: str = "",
        max_samples: Optional[int] = None,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.model_id = model_id.lower()
        self.max_length = max_length
        self.samples = []

        # Precompute Qwen chat template markers for label masking.
        # This script targets Qwen models (<|im_start|>/<|im_end|> format).
        self._response_template_ids = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

        if self.samples and not self._response_template_ids:
            print("WARNING: Could not encode '<|im_start|>assistant\\n' -- label masking may be incorrect.")

    def __len__(self):
        return len(self.samples)

    def _convert_path(self, windows_path: str) -> str:
        """Convert Windows path to Linux path."""
        match = re.search(r'Fetal Ultrasound[/\\](.+)$', windows_path)
        if match:
            relative_path = match.group(1).replace('\\', '/')
            return f"/workspace/data/Fetal Ultrasound/{relative_path}"
        return windows_path

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        image_paths = sample.get('images', [])

        # Load images
        images = []
        for img_path in image_paths:
            converted_path = self._convert_path(img_path)
            try:
                img = Image.open(converted_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {converted_path}: {e}")
                images.append(Image.new('RGB', (224, 224), color='gray'))

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process inputs
        inputs = self.processor(
            text=text,
            images=images if images else None,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels: mask everything except assistant response tokens.
        # For Qwen models, responses are between <|im_start|>assistant\n ... <|im_end|>
        input_ids = inputs['input_ids']
        labels = torch.full_like(input_ids, -100)

        if self._response_template_ids:
            template_len = len(self._response_template_ids)
            ids_list = input_ids.tolist()
            i = 0
            while i <= len(ids_list) - template_len:
                if ids_list[i:i + template_len] == self._response_template_ids:
                    response_start = i + template_len
                    j = response_start
                    while j < len(ids_list):
                        if ids_list[j] == self._end_token_id:
                            labels[j] = input_ids[j]
                            break
                        labels[j] = input_ids[j]
                        j += 1
                    i = j + 1
                else:
                    i += 1
        else:
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs['labels'] = labels

        return inputs


def load_model_and_processor(
    model_id: str,
    use_4bit: bool = True,
    use_flash_attn: bool = False,
):
    """Load model and processor with quantization."""
    from transformers import AutoProcessor, AutoModel, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    print(f"Loading model: {model_id}")

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model_lower = model_id.lower()

    # Determine model class based on model type
    if "internvl" in model_lower:
        print("Using AutoModel for InternVL...")
        model = AutoModel.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        )
    else:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
            )
        except ValueError:
            print("Falling back to AutoModel...")
            model = AutoModel.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, processor


def apply_lora(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Apply LoRA adapters to the model."""
    from peft import LoraConfig, get_peft_model

    # Determine target modules based on model architecture
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VLM on fetal ultrasound data")

    # Model arguments
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model ID')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--flash-attn', action='store_true', help='Use Flash Attention 2')

    # Data arguments
    parser.add_argument('--train-data', type=str, default='data/train.jsonl')
    parser.add_argument('--val-data', type=str, default='data/val.jsonl')
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--output-dir', type=str, default='outputs')

    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)

    # Other
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--max-length', type=int, default=2048)

    args = parser.parse_args()

    # Setup paths
    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split('/')[-1]
    run_dir = output_dir / f"{model_short}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VLM Fine-Tuning")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Train data: {train_path}")
    print(f"Val data: {val_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} gradient accumulation")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Output: {run_dir}")
    print(f"4-bit quantization: {not args.no_4bit}")

    # Check paths
    if not train_path.exists():
        print(f"Error: Training data not found: {train_path}")
        return 1

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nCUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available!")
        return 1

    # Load model and processor
    model, processor = load_model_and_processor(
        args.model,
        use_4bit=not args.no_4bit,
        use_flash_attn=args.flash_attn,
    )

    # Apply LoRA
    model = apply_lora(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VLMDataset(
        str(train_path),
        processor,
        model_id=args.model,
        max_samples=args.max_train_samples,
        max_length=args.max_length,
    )

    eval_dataset = None
    if val_path.exists():
        eval_dataset = VLMDataset(
            str(val_path),
            processor,
            model_id=args.model,
            max_samples=args.max_val_samples,
            max_length=args.max_length,
        )

    # Setup training with HuggingFace Trainer
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        bf16=True,
        fp16=False,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=args.checkpoint_interval,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.checkpoint_interval if eval_dataset else None,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    # Save config
    config = {
        'model': args.model,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation': args.gradient_accumulation,
        'learning_rate': args.learning_rate,
        'train_samples': len(train_dataset),
        'val_samples': len(eval_dataset) if eval_dataset else 0,
        'use_4bit': not args.no_4bit,
        'timestamp': timestamp,
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final adapter
    print("\nSaving adapter...")
    adapter_dir = run_dir / 'adapter'
    trainer.model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    print(f"\nTraining complete!")
    print(f"Adapter saved to: {adapter_dir}")

    # Save final metrics
    metrics = {
        'train_loss': trainer.state.log_history[-1].get('loss', None) if trainer.state.log_history else None,
        'completed_at': datetime.now().isoformat(),
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return 0


if __name__ == "__main__":
    exit(main())
