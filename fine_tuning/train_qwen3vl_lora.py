"""
VLM LoRA Fine-tuning Script for Fetal Ultrasound VQA

Supports both Unsloth (optimized) and standard HuggingFace + PEFT training.
Designed for RTX 4070 (12GB) with 4-bit quantization.

Usage:
    # Quick test with small subset
    python experiments/fine_tuning/train_qwen3vl_lora.py --test-run

    # Full training
    python experiments/fine_tuning/train_qwen3vl_lora.py --epochs 3

    # With Unsloth (if installed)
    python experiments/fine_tuning/train_qwen3vl_lora.py --use-unsloth
"""

import os
import sys
import json
import argparse
import time
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from src.utils.mlflow_utils import setup_mlflow_experiment, MLflowTrainerCallback

# Try importing Unsloth, fall back to standard HF
try:
    from unsloth import FastModel
    UNSLOTH_AVAILABLE = True
    print("Unsloth available - will use optimized training")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Unsloth not available - using standard HuggingFace training")

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Try to import Qwen-specific model classes
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    QWEN25_AVAILABLE = True
except ImportError:
    QWEN25_AVAILABLE = False

try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False


# Model configurations - Verified HuggingFace model IDs
MODEL_CONFIGS = {
    # Qwen3-VL series (newest, Oct 2025)
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    # Qwen2.5-VL series (Jan 2025) - Note: smallest is 3B, not 2B
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    # Qwen2-VL series (older, but has 2B)
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    # InternVL2 family
    "internvl2-2b": "OpenGVLab/InternVL2-2B",
    "internvl2-4b": "OpenGVLab/InternVL2-4B",
    "internvl2-8b": "OpenGVLab/InternVL2-8B",
    # InternVL3 family
    "internvl3-1b": "OpenGVLab/InternVL3-1B",
    "internvl3-2b": "OpenGVLab/InternVL3-2B",
    "internvl3-8b": "OpenGVLab/InternVL3-8B",
    # MiniCPM-V family
    "minicpm-v-2.6": "openbmb/MiniCPM-V-2_6",
    "minicpm-v-4": "openbmb/MiniCPM-V-4",
    # Gemma 4 family
    "gemma-4-e2b-it": "google/gemma-4-E2B-it",
    "gemma-4-e4b-it": "google/gemma-4-E4B-it",
}

# LoRA target modules (identical across qwen/internvl/minicpm architectures)
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
GEMMA4_LORA_TARGET_MODULES = [
    "q_proj.linear",
    "k_proj.linear",
    "v_proj.linear",
    "o_proj.linear",
    "gate_proj.linear",
    "up_proj.linear",
    "down_proj.linear",
]

# Default LoRA configuration for RTX 4070
DEFAULT_LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Training defaults for RTX 4070 (12GB)
DEFAULT_TRAINING_ARGS = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": False,
    "bf16": True,
    "logging_steps": 10,
    "optim": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "save_strategy": "steps",
    "save_steps": 500,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "dataloader_num_workers": None,
}


class VLMDataset(Dataset):
    """Dataset for VLM fine-tuning from JSONL files."""

    def __init__(
        self,
        jsonl_path: str,
        processor: AutoProcessor,
        architecture: str = "qwen",
        max_samples: Optional[int] = None,
        max_length: int = 4096,
        data_root: Optional[str] = None,
    ):
        self.processor = processor
        self.architecture = architecture
        self.max_length = max_length
        self.data_root = Path(data_root) if data_root else None
        self._missing_count = 0
        self.samples = []

        # Qwen uses explicit assistant boundary tokens.
        # Non-Qwen architectures fall back to padding-only masking.
        if self.architecture == "qwen":
            self._response_template_ids = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            self._end_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        else:
            self._response_template_ids = []
            self._end_token_id = None

        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

        # Validate template tokens on first sample
        if self.samples and not self._response_template_ids:
            print("WARNING: Could not encode '<|im_start|>assistant\\n' -- label masking may be incorrect. "
                  "Falling back to padding-only masking.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        image_paths = sample.get('images', [])

        # Load images (prepend data_root for relative paths)
        images = []
        for img_path in image_paths:
            if self.data_root and not os.path.isabs(img_path):
                full_path = (self.data_root / img_path).resolve()
                if not full_path.is_relative_to(self.data_root.resolve()):
                    raise ValueError(f"Path traversal detected: {img_path}")
                img_path = str(full_path)
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
            except FileNotFoundError:
                self._missing_count += 1
                if self._missing_count > 10:
                    raise RuntimeError(
                        f"Too many missing images ({self._missing_count}). "
                        f"Check --data-root is correct. Last missing: {img_path}"
                    )
                print(f"Warning: Image not found: {img_path}")
                images.append(Image.new('RGB', (224, 224), color='gray'))
            except Exception as e:
                raise RuntimeError(f"Failed to load image {img_path}: {e}") from e

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
        # For Qwen models, assistant responses are between
        # <|im_start|>assistant\n ... <|im_end|>
        input_ids = inputs['input_ids']
        labels = torch.full_like(input_ids, -100)

        if self._response_template_ids:
            # Find all assistant response boundaries and unmask them
            template_len = len(self._response_template_ids)
            ids_list = input_ids.tolist()
            found_any_response = False
            i = 0
            while i <= len(ids_list) - template_len:
                # Check for <|im_start|>assistant\n marker
                if ids_list[i:i + template_len] == self._response_template_ids:
                    found_any_response = True
                    # Unmask from after the marker until <|im_end|>
                    response_start = i + template_len
                    j = response_start
                    while j < len(ids_list):
                        if ids_list[j] == self._end_token_id:
                            # Include the <|im_end|> token in labels
                            labels[j] = input_ids[j]
                            break
                        labels[j] = input_ids[j]
                        j += 1
                    i = j + 1
                else:
                    i += 1
            if not found_any_response:
                # Fallback in case template markers are not present in the tokenized sample.
                labels = input_ids.clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
        else:
            # Fallback: mask padding only (original behavior)
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs['labels'] = labels

        return inputs


def get_model_architecture(model_id: str) -> str:
    """
    Detect model architecture from model ID.

    Args:
        model_id: HuggingFace model ID or model name

    Returns:
        Architecture type: "qwen", "internvl", "minicpm", "gemma", or "generic"
    """
    model_lower = model_id.lower()
    if "qwen" in model_lower:
        return "qwen"
    elif "internvl" in model_lower:
        return "internvl"
    elif "minicpm" in model_lower:
        return "minicpm"
    elif "gemma" in model_lower:
        return "gemma"
    else:
        return "generic"


def create_model_and_processor(
    model_name: str,
    use_4bit: bool = True,
    use_unsloth: bool = False,
) -> tuple:
    """
    Load model and processor with optional quantization.

    Returns:
        (model, processor) tuple
    """
    model_id = MODEL_CONFIGS.get(model_name, model_name)
    architecture = get_model_architecture(model_id)
    print(f"Loading model: {model_id} (architecture: {architecture})")

    if use_unsloth and UNSLOTH_AVAILABLE:
        # Use Unsloth for optimized loading
        # TODO: Pin HuggingFace model revision for reproducibility (see #40)
        model, processor = FastModel.from_pretrained(
            model_id,
            max_seq_length=2048,
            load_in_4bit=use_4bit,
            dtype=torch.bfloat16,
        )
    else:
        # Standard HuggingFace loading
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

        # Use AutoModelForImageTextToText for best compatibility
        # It handles Qwen2-VL, Qwen2.5-VL, and Qwen3-VL
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if use_4bit:
            model = prepare_model_for_kbit_training(model)

    return model, processor


def apply_lora(
    model,
    model_id: str,
    lora_config: Optional[Dict[str, Any]] = None,
    use_unsloth: bool = False,
):
    """Apply LoRA adapters to the model."""
    config = lora_config or DEFAULT_LORA_CONFIG

    # All supported architectures use the same target modules
    architecture = get_model_architecture(model_id)
    config = config.copy()
    config['target_modules'] = (
        GEMMA4_LORA_TARGET_MODULES
        if architecture == "gemma"
        else LORA_TARGET_MODULES
    )
    print(f"Using {architecture} target modules: {config['target_modules']}")

    if use_unsloth and UNSLOTH_AVAILABLE:
        # Unsloth handles LoRA internally
        model = FastModel.get_peft_model(
            model,
            r=config['r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=config['target_modules'],
        )
    else:
        lora_config_obj = LoraConfig(**config)
        model = get_peft_model(model, lora_config_obj)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Find the latest checkpoint in output directory."""
    checkpoints = list(output_dir.glob("checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def collate_fn_for_vlm(batch):
    """
    Custom data collator for VLM training with variable-sized images.

    Handles tensors that can't be stacked due to different dimensions
    by keeping them as lists instead of stacking.
    """
    collated = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        # Check if all values are tensors
        if isinstance(values[0], torch.Tensor):
            # Try to stack - if shapes match
            shapes = [v.shape for v in values]
            if len(set(shapes)) == 1:
                # All same shape - stack normally
                collated[key] = torch.stack(values)
            else:
                # Different shapes - keep as list (model should handle this)
                # For pixel_values, image_grid_thw, etc.
                collated[key] = values
        elif isinstance(values[0], (list, tuple)):
            collated[key] = values
        else:
            # Try default stacking for other types
            try:
                collated[key] = torch.tensor(values)
            except (TypeError, ValueError):
                collated[key] = values

    return collated


def create_trainer(
    model,
    processor,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    output_dir: str,
    training_args: Optional[Dict[str, Any]] = None,
    callbacks: Optional[list] = None,
) -> Trainer:
    """Create HuggingFace Trainer."""
    args = DEFAULT_TRAINING_ARGS.copy()
    if training_args:
        args.update(training_args)

    args['output_dir'] = output_dir

    if args.get('dataloader_num_workers') is None:
        args['dataloader_num_workers'] = 4 if platform.system() == "Linux" else 0

    training_arguments = TrainingArguments(**args)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collate_fn_for_vlm,  # Custom collator for variable-sized images
        callbacks=callbacks,
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune VLMs for fetal ultrasound VQA"
    )

    # Model arguments
    parser.add_argument(
        '--model', type=str, default='qwen2-vl-2b',
        help='Model key from MODEL_CONFIGS or full HuggingFace model ID (default: qwen2-vl-2b)'
    )
    parser.add_argument(
        '--use-unsloth', action='store_true',
        help='Use Unsloth for optimized training (if available)'
    )
    parser.add_argument(
        '--no-4bit', action='store_true',
        help='Disable 4-bit quantization (requires more VRAM)'
    )

    # Data arguments
    parser.add_argument(
        '--train-data', type=str,
        default=str(PROJECT_ROOT / 'data/vlm_training/gt_train.jsonl'),
        help='Path to training JSONL file'
    )
    parser.add_argument(
        '--val-data', type=str,
        default=str(PROJECT_ROOT / 'data/vlm_training/gt_val.jsonl'),
        help='Path to validation JSONL file'
    )
    parser.add_argument(
        '--data-root', type=str,
        default=str(PROJECT_ROOT / 'data/Fetal Ultrasound'),
        help='Root directory for image paths in JSONL (prepended to relative paths)'
    )
    parser.add_argument(
        '--max-train-samples', type=int, default=None,
        help='Maximum training samples (for quick testing)'
    )
    parser.add_argument(
        '--max-val-samples', type=int, default=None,
        help='Maximum validation samples'
    )

    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='Per-device batch size'
    )
    parser.add_argument(
        '--gradient-accumulation', type=int, default=8,
        help='Gradient accumulation steps'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=2e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: auto-generated)'
    )

    # LoRA arguments
    parser.add_argument(
        '--lora-r', type=int, default=16,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora-alpha', type=int, default=32,
        help='LoRA alpha'
    )
    parser.add_argument(
        '--lora-dropout', type=float, default=0.05,
        help='LoRA dropout'
    )

    # Testing
    parser.add_argument(
        '--test-run', action='store_true',
        help='Quick test with 100 samples, 1 epoch'
    )

    # Resume arguments
    parser.add_argument(
        '--resume-from-checkpoint', type=str, default=None,
        help='Path to checkpoint directory to resume training from'
    )
    parser.add_argument(
        '--auto-resume', action='store_true',
        help='Automatically resume from latest checkpoint in output directory'
    )

    args = parser.parse_args()

    # Handle test run
    if args.test_run:
        args.max_train_samples = 100
        args.max_val_samples = 20
        args.epochs = 1
        print("\n*** TEST RUN MODE - Using minimal data ***\n")

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(PROJECT_ROOT / f"experiments/fine_tuning/runs/{args.model}_{timestamp}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Handle auto-resume
    checkpoint_to_resume = None
    if args.auto_resume:
        checkpoint_to_resume = find_latest_checkpoint(output_dir)
        if checkpoint_to_resume:
            print(f"\nAuto-resume: Found checkpoint at {checkpoint_to_resume}")
        else:
            print("\nAuto-resume: No checkpoints found, starting fresh")
    elif args.resume_from_checkpoint:
        checkpoint_to_resume = args.resume_from_checkpoint
        checkpoint_path = Path(checkpoint_to_resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_to_resume}")
        print(f"\nResuming from checkpoint: {checkpoint_to_resume}")

    # Setup MLflow experiment
    setup_mlflow_experiment("vlm_finetuning")

    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training will be slow!")
    else:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and processor
    use_unsloth = args.use_unsloth and UNSLOTH_AVAILABLE
    model, processor = create_model_and_processor(
        args.model,
        use_4bit=not args.no_4bit,
        use_unsloth=use_unsloth,
    )

    # Apply LoRA
    model_id = MODEL_CONFIGS.get(args.model, args.model)
    lora_config = {
        **DEFAULT_LORA_CONFIG,
        'r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
    }
    model = apply_lora(model, model_id, lora_config, use_unsloth=use_unsloth)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VLMDataset(
        args.train_data,
        processor,
        architecture=get_model_architecture(model_id),
        max_samples=args.max_train_samples,
        data_root=args.data_root,
    )

    eval_dataset = None
    if Path(args.val_data).exists():
        eval_dataset = VLMDataset(
            args.val_data,
            processor,
            architecture=get_model_architecture(model_id),
            max_samples=args.max_val_samples,
            data_root=args.data_root,
        )

    # Training arguments
    training_args = {
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation,
        'learning_rate': args.learning_rate,
    }

    # Save config
    config = {
        'model': args.model,
        'model_id': MODEL_CONFIGS.get(args.model, args.model),
        'lora_config': lora_config,
        'training_args': training_args,
        'train_samples': len(train_dataset),
        'val_samples': len(eval_dataset) if eval_dataset else 0,
        'use_unsloth': use_unsloth,
        'use_4bit': not args.no_4bit,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create run name based on model variant
    run_name = f"{args.model}_lora"
    if args.test_run:
        run_name += "_test"

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            'model': args.model,
            'model_id': MODEL_CONFIGS.get(args.model, args.model),
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'quantization': '4bit' if not args.no_4bit else 'none',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'gradient_accumulation': args.gradient_accumulation,
            'train_samples': len(train_dataset),
            'val_samples': len(eval_dataset) if eval_dataset else 0,
            'use_unsloth': use_unsloth,
            'test_run': args.test_run,
        })

        # Create MLflow callback
        mlflow_callback = MLflowTrainerCallback(log_every_n_steps=1)

        # Create trainer with MLflow callback
        trainer = create_trainer(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_dir),
            training_args=training_args,
            callbacks=[mlflow_callback],
        )

        # Train
        print("\nStarting training...")
        start_time = time.time()
        if checkpoint_to_resume:
            trainer.train(resume_from_checkpoint=checkpoint_to_resume)
        else:
            trainer.train()
        training_time = time.time() - start_time

        # Log final metrics
        final_metrics = {
            'training_time': training_time,
        }

        # Extract final loss from trainer state
        if trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if 'loss' in entry:
                    final_metrics['final_loss'] = entry['loss']
                    break

        mlflow.log_metrics(final_metrics)

        # Log config artifact
        mlflow.log_artifact(str(output_dir / 'config.json'))

        # Save final model
        print("\nSaving model...")
        trainer.save_model(str(output_dir / 'final'))
        processor.save_pretrained(str(output_dir / 'final'))

        print(f"\nTraining complete! Model saved to: {output_dir / 'final'}")
        print(f"Training time: {training_time:.2f} seconds")


if __name__ == "__main__":
    main()
