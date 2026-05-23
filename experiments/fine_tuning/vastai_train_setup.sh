#!/bin/bash
# Vast.ai VLM Fine-tuning Setup Script
# Run this after SSH into your Vast.ai instance
#
# Usage:
#   1. SSH into instance: ssh -p PORT root@IP
#   2. Run: bash vastai_train_setup.sh
#   3. Or with args: bash vastai_train_setup.sh --model qwen3-vl-8b --epochs 2

set -e

# Default values
MODEL="${MODEL:-qwen3-vl-8b}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"

# Parse command line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --lora-r) LORA_R="$2"; shift 2 ;;
        --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  Qwen3-VL Fine-tuning Setup for Vast.ai"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA r: $LORA_R"
echo "  LoRA alpha: $LORA_ALPHA"
echo ""

# 1. Check GPU
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# 2. Install dependencies
echo "[2/6] Installing dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers>=4.45.0 accelerate bitsandbytes peft trl
pip install -q datasets pillow pandas openpyxl
pip install -q wandb  # Optional: for logging

# Install latest transformers from git for Qwen3-VL support
pip install -q git+https://github.com/huggingface/transformers.git

echo "Dependencies installed."
echo ""

# 3. Setup workspace
echo "[3/6] Setting up workspace..."
WORKSPACE="/workspace/fada-finetune"
mkdir -p $WORKSPACE
cd $WORKSPACE

# Check if data exists
if [ ! -d "$WORKSPACE/data" ]; then
    echo "WARNING: Training data not found at $WORKSPACE/data"
    echo "Please upload training data using:"
    echo "  scp -P PORT -r data/vlm_training root@IP:$WORKSPACE/data"
    echo ""
fi

# 4. Create training script
echo "[4/6] Creating training script..."
cat > $WORKSPACE/train.py << 'TRAINING_SCRIPT'
"""
Qwen3-VL LoRA Fine-tuning Script for Vast.ai
Optimized for A100/H100 GPUs
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Model configurations
MODEL_CONFIGS = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
}


class VLMDataset(Dataset):
    """Dataset for VLM fine-tuning from JSONL files."""

    def __init__(
        self,
        jsonl_path: str,
        processor: AutoProcessor,
        max_samples: Optional[int] = None,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        image_paths = sample.get('images', [])

        # Load images
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
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

        # Create labels
        labels = inputs['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels

        return inputs


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL on Vast.ai")

    # Model arguments
    parser.add_argument('--model', type=str, default='qwen3-vl-8b',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--use-4bit', action='store_true',
                        help='Use 4-bit quantization (saves VRAM)')

    # Data arguments
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, default=None)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--output-dir', type=str, default='./output')

    # LoRA arguments
    parser.add_argument('--lora-r', type=int, default=32)
    parser.add_argument('--lora-alpha', type=int, default=64)
    parser.add_argument('--lora-dropout', type=float, default=0.05)

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Qwen3-VL Fine-tuning")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    # Load model and processor
    model_id = MODEL_CONFIGS[args.model]
    print(f"Loading {model_id}...")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VLMDataset(
        args.train_data,
        processor,
        max_samples=args.max_train_samples,
    )

    eval_dataset = None
    if args.val_data and Path(args.val_data).exists():
        eval_dataset = VLMDataset(
            args.val_data,
            processor,
            max_samples=args.max_val_samples,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        gradient_checkpointing=True,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        report_to="none",  # Set to "wandb" if using wandb
    )

    # Create trainer
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
        'model_id': model_id,
        'lora_config': {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
        },
        'training_args': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'gradient_accumulation': args.gradient_accumulation,
            'learning_rate': args.learning_rate,
        },
        'train_samples': len(train_dataset),
        'val_samples': len(eval_dataset) if eval_dataset else 0,
        'use_4bit': args.use_4bit,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving model...")
    trainer.save_model(str(output_dir / 'final'))
    processor.save_pretrained(str(output_dir / 'final'))

    print(f"\nTraining complete! Model saved to: {output_dir / 'final'}")
    print(f"\nTo download results:")
    print(f"  scp -P PORT -r root@IP:{output_dir} ./")


if __name__ == "__main__":
    main()
TRAINING_SCRIPT

echo "Training script created at $WORKSPACE/train.py"
echo ""

# 5. Create run script
echo "[5/6] Creating run script..."
cat > $WORKSPACE/run_training.sh << EOF
#!/bin/bash
# Run training with configured parameters

cd $WORKSPACE

python train.py \\
    --model $MODEL \\
    --train-data data/medgemma_4b_train.jsonl \\
    --val-data data/medgemma_4b_val.jsonl \\
    --epochs $EPOCHS \\
    --batch-size $BATCH_SIZE \\
    --gradient-accumulation $GRAD_ACCUM \\
    --learning-rate $LEARNING_RATE \\
    --lora-r $LORA_R \\
    --lora-alpha $LORA_ALPHA \\
    --output-dir ./output

echo ""
echo "Training complete!"
echo "Results saved to: $WORKSPACE/output/"
EOF

chmod +x $WORKSPACE/run_training.sh
echo "Run script created at $WORKSPACE/run_training.sh"
echo ""

# 6. Final instructions
echo "[6/6] Setup complete!"
echo ""
echo "=============================================="
echo "  Next Steps"
echo "=============================================="
echo ""
echo "1. Upload training data (from local machine):"
echo "   scp -P PORT -r data/vlm_training/* root@IP:$WORKSPACE/data/"
echo ""
echo "2. Start training:"
echo "   cd $WORKSPACE && ./run_training.sh"
echo ""
echo "   Or run manually with custom settings:"
echo "   python train.py --model qwen3-vl-8b --epochs 2 \\"
echo "       --train-data data/medgemma_4b_train.jsonl \\"
echo "       --val-data data/medgemma_4b_val.jsonl"
echo ""
echo "3. Monitor training:"
echo "   tail -f $WORKSPACE/output/*/logs/*"
echo ""
echo "4. Download results (from local machine):"
echo "   scp -P PORT -r root@IP:$WORKSPACE/output ./vastai_results/"
echo ""
echo "=============================================="
