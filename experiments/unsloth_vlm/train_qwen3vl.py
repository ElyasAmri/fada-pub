"""
Fine-tune Qwen3-VL for Fetal Ultrasound Normality Assessment

This script uses Unsloth's FastVisionModel to fine-tune Qwen3-VL-8B
on the Q7 (Normality Assessment) task using LoRA/QLoRA.

Hardware: RTX 4070 (12GB VRAM)
Expected VRAM: ~10GB with 4-bit quantization
"""

import os
import sys
import time
from importlib.util import find_spec
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import torch
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from mlflow_callback import MLflowCallback
from prepare_dataset import prepare_dataset, DATA_ROOT, ANNOTATIONS_FILE


# Model configuration
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "qwen3vl_ultrasound"

# Training configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "num_train_epochs": 3,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "seed": 42,
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "target_modules": "all-linear",
    "use_rslora": False,
}


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


def find_latest_checkpoint(output_dir):
    """Find the latest checkpoint in output directory."""
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])


def load_model():
    """Load Qwen3-VL with 4-bit quantization."""
    print(f"Loading model: {MODEL_NAME}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Ensure CUDA is available - required for 4-bit quantization
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! 4-bit quantization requires a GPU.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    if find_spec("flash_attn") is None:
        print("flash-attn is not installed in this environment; Unsloth will fall back to a slower attention path.")

    # Flush output to ensure it appears
    import sys
    sys.stdout.flush()

    # TODO: Pin HuggingFace model revision for reproducibility (see #40)
    model, tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        device_map="auto",  # Explicitly use auto device mapping
    )

    # Verify model is on GPU
    print(f"Model device: {next(model.parameters()).device}")
    sys.stdout.flush()

    print("Model loaded successfully!")
    return model, tokenizer


def add_lora_adapters(model):
    """Add LoRA adapters for parameter-efficient fine-tuning."""
    print("Adding LoRA adapters...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        use_rslora=LORA_CONFIG["use_rslora"],
    )

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


def create_trainer(model, tokenizer, train_dataset, val_dataset, lazy_loading=False, deadline_ts=None):
    """Create SFTTrainer with vision data collator."""

    # Determine precision
    use_bf16 = is_bf16_supported()
    use_fp16 = not use_bf16

    print(f"Using {'bf16' if use_bf16 else 'fp16'} precision")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = SFTConfig(
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        seed=TRAINING_CONFIG["seed"],
        fp16=use_fp16,
        bf16=use_bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        output_dir=str(OUTPUT_DIR),
        report_to="none",  # Disable wandb/tensorboard for now
        remove_unused_columns=False,
        dataset_text_field="",  # Required but not used for vision
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Create data collator
    data_collator = UnslothVisionDataCollator(model, tokenizer)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    # Add MLflow callback for per-step logging
    trainer.add_callback(MLflowCallback())
    runtime_callback = None
    if deadline_ts is not None:
        runtime_callback = MaxRuntimeCallback(deadline_ts)
        trainer.add_callback(runtime_callback)

    return trainer, runtime_callback


def save_model(model, tokenizer, output_dir: Path):
    """Save LoRA adapters and tokenizer."""
    adapter_dir = output_dir / "lora_adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapters to {adapter_dir}")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print("Model saved successfully!")


def main():
    """Main training loop."""
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL for ultrasound analysis")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples for testing (None for all)")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["num_train_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["per_device_train_batch_size"],
                        help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=TRAINING_CONFIG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load model and data but don't train")
    parser.add_argument("--lazy-loading", action="store_true",
                        help="Use lazy loading to avoid OOM with large datasets")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Automatically resume from latest checkpoint in output directory")
    parser.add_argument("--max-runtime-minutes", type=float, default=None,
                        help="Hard wall-clock budget for the full run, including dataset prep and training")

    args = parser.parse_args()

    # Auto-enable lazy loading for full dataset
    use_lazy = args.lazy_loading or (args.max_samples is None)
    deadline_ts = None if args.max_runtime_minutes is None else time.time() + args.max_runtime_minutes * 60

    # Update config from args
    TRAINING_CONFIG["num_train_epochs"] = args.epochs
    TRAINING_CONFIG["per_device_train_batch_size"] = args.batch_size
    TRAINING_CONFIG["learning_rate"] = args.learning_rate

    print("=" * 60)
    print("Qwen3-VL Fine-tuning for Fetal Ultrasound Analysis")
    print("=" * 60)
    print(f"\nTask: Q7 - Normality Assessment")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {TRAINING_CONFIG['gradient_accumulation_steps']} = {args.batch_size * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Lazy loading: {use_lazy}")
    if args.max_runtime_minutes is not None:
        print(f"Runtime budget: {args.max_runtime_minutes:.1f} minutes")
    print()

    # Load dataset
    print("Loading dataset...")
    train_dataset, val_dataset = prepare_dataset(
        max_samples=args.max_samples,
        train_ratio=0.9,
        seed=TRAINING_CONFIG["seed"],
        lazy_loading=use_lazy,
        deadline_ts=deadline_ts,
    )

    # Load model
    model, tokenizer = load_model()

    # Add LoRA adapters
    model = add_lora_adapters(model)

    if deadline_ts is not None and time.time() >= deadline_ts:
        raise TimeoutError("Runtime budget exhausted before training could start.")

    if args.dry_run:
        print("\n[DRY RUN] Skipping training.")
        print("Model and data loaded successfully!")
        return

    # Set up MLflow experiment
    mlflow.set_experiment("unsloth_vlm_ultrasound")

    # Create run name based on config
    run_name = f"q7_baseline_{args.epochs}ep"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            "model": MODEL_NAME,
            "task": "Q7",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": TRAINING_CONFIG["gradient_accumulation_steps"],
            "effective_batch_size": args.batch_size * TRAINING_CONFIG["gradient_accumulation_steps"],
            "learning_rate": args.learning_rate,
            "max_seq_length": TRAINING_CONFIG["max_seq_length"],
            "lora_r": LORA_CONFIG["r"],
            "lora_alpha": LORA_CONFIG["lora_alpha"],
            "lora_dropout": LORA_CONFIG["lora_dropout"],
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "lazy_loading": use_lazy,
            "max_runtime_minutes": args.max_runtime_minutes,
        })

        # Create trainer
        print("\nCreating trainer...")
        trainer, runtime_callback = create_trainer(
            model,
            tokenizer,
            train_dataset,
            val_dataset,
            lazy_loading=use_lazy,
            deadline_ts=deadline_ts,
        )

        # Handle resume
        checkpoint = None
        if args.resume_from_checkpoint:
            checkpoint = args.resume_from_checkpoint
            if not Path(checkpoint).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            print(f"\nResuming from checkpoint: {checkpoint}")
        elif args.auto_resume:
            checkpoint = find_latest_checkpoint(OUTPUT_DIR)
            if checkpoint:
                print(f"\nAuto-resuming from {checkpoint}")
            else:
                print("\nAuto-resume: No checkpoints found, starting fresh")

        # Train
        print("\nStarting training...")
        print("=" * 60)

        start_time = time.time()
        if checkpoint:
            trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer_stats = trainer.train()
        training_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Final loss: {trainer_stats.training_loss:.4f}")
        print(f"Training time: {training_time:.1f}s")
        if runtime_callback and runtime_callback.triggered:
            print("Stopped early because the runtime budget was reached.")

        # Log final metrics
        mlflow.log_metrics({
            "final_loss": trainer_stats.training_loss,
            "total_steps": trainer_stats.global_step,
            "training_time_seconds": training_time,
        })
        if runtime_callback:
            mlflow.log_param("stopped_early_due_to_runtime_limit", runtime_callback.triggered)

        # Save model
        save_model(model, tokenizer, OUTPUT_DIR)

        # Log adapter path as parameter
        mlflow.log_param("adapter_path", str(OUTPUT_DIR / "lora_adapters"))

        print("\nDone!")


if __name__ == "__main__":
    main()
