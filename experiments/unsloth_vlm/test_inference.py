"""
Test inference with fine-tuned Qwen3-VL model for Fetal Ultrasound Analysis

This script loads the LoRA adapters and tests on sample images.
"""

import os
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
from unsloth import FastVisionModel

from prepare_dataset import DATA_ROOT, ANNOTATIONS_FILE
import pandas as pd


# Paths
MODEL_NAME = "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit"
LORA_PATH = Path(__file__).parent / "outputs" / "qwen3vl_ultrasound" / "lora_adapters"

# System prompt (same as training)
SYSTEM_PROMPT = """You are a medical imaging AI assistant specialized in analyzing fetal ultrasound images.
Your task is to assess whether the ultrasound image shows NORMAL or ABNORMAL findings.
Respond with only one word: either "Normal" or "Abnormal"."""


def load_model():
    """Load base model and apply LoRA adapters."""
    print(f"Loading fine-tuned model from: {LORA_PATH}")

    # Load directly from the saved LoRA adapters
    model, tokenizer = FastVisionModel.from_pretrained(
        str(LORA_PATH),
        load_in_4bit=True,
        use_gradient_checkpointing=False,
    )

    # Set to inference mode
    FastVisionModel.for_inference(model)

    return model, tokenizer


def get_sample_images(n=5):
    """Get random sample images from the dataset with their labels."""
    df = pd.read_excel(ANNOTATIONS_FILE)

    # Filter for images with Q7 answers
    q7_col = 'Q7: Normality Assessment'
    df = df[df[q7_col].notna()].copy()

    samples = []
    for _, row in df.sample(n=min(n, len(df)), random_state=42).iterrows():
        # Construct image path from folder and image name
        folder = row['Folder Name']
        image_name = row['Image Name']
        image_path = DATA_ROOT / folder / image_name

        if image_path.exists():
            q7 = str(row[q7_col]).strip().lower()
            label = "Normal" if 'normal' in q7 else "Abnormal"
            samples.append({
                'path': image_path,
                'label': label,
                'raw_q7': row[q7_col]
            })

    return samples


def run_inference(model, tokenizer, image_path):
    """Run inference on a single image."""
    # Load and resize image
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > 512:
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    # Create conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Is this fetal ultrasound image normal or abnormal?"}
        ]}
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            use_cache=True,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the model's answer (after the last assistant marker)
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    return response


def main():
    print("=" * 60)
    print("Qwen3-VL Fetal Ultrasound Inference Test")
    print("=" * 60)

    # Load model
    model, tokenizer = load_model()
    print("\nModel loaded successfully!\n")

    # Get sample images
    print("Loading sample images...")
    samples = get_sample_images(n=50)
    print(f"Found {len(samples)} test images\n")

    # Run inference
    print("-" * 60)
    correct = 0
    total = 0

    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Testing: {sample['path'].name}")
        print(f"  Ground truth: {sample['label']} (raw: {sample['raw_q7']})")

        try:
            prediction = run_inference(model, tokenizer, sample['path'])
            print(f"  Prediction: {prediction}")

            # Check if correct - extract just Normal/Abnormal from prediction
            pred_lower = prediction.lower().strip()
            pred_label = "abnormal" if "abnormal" in pred_lower else "normal" if "normal" in pred_lower else "unknown"
            gt_label = sample['label'].lower()

            if pred_label == gt_label:
                print("  Result: CORRECT")
                correct += 1
            else:
                print(f"  Result: INCORRECT (predicted {pred_label}, expected {gt_label})")
            total += 1

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.1f}% accuracy)")
    print("=" * 60)


if __name__ == "__main__":
    main()
