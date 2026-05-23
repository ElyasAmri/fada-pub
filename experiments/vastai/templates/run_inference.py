#!/usr/bin/env python3
"""
Generic VLM Inference Script for Vast.ai.

Runs inference on any supported VLM model and saves predictions.
Designed to be uploaded and run on remote instances.

Usage:
    python run_inference.py --model Qwen/Qwen2.5-VL-3B-Instruct --test-data data/test_subset.jsonl
    python run_inference.py --model OpenGVLab/InternVL3-2B --samples 20
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


# System prompt for fetal ultrasound analysis
SYSTEM_PROMPT = """You are an expert fetal ultrasound analyst. Analyze the ultrasound image and answer the question thoroughly based on what you observe. Focus on anatomical structures, image quality, and any clinically relevant findings."""

# Path conversion for vast.ai
IMAGES_BASE_PATH = "/workspace/data/Fetal Ultrasound"


def convert_windows_path(windows_path: str) -> str:
    """Convert Windows path to Linux path on vast.ai."""
    match = re.search(r'Fetal Ultrasound[/\\](.+)$', windows_path)
    if match:
        relative_path = match.group(1).replace('\\', '/')
        return f"{IMAGES_BASE_PATH}/{relative_path}"
    return windows_path


def extract_category_from_path(image_path: str) -> str:
    """Extract category name from image path."""
    parts = Path(image_path).parts
    for i, part in enumerate(parts):
        if part == "Fetal Ultrasound":
            if i + 1 < len(parts):
                return parts[i + 1]
    return "Unknown"


def load_model(model_id: str, adapter_path: str = None, use_4bit: bool = True):
    """
    Load VLM model with optional adapter.

    Supports:
    - Qwen2-VL / Qwen2.5-VL
    - InternVL2 / InternVL3
    - MiniCPM-V
    - SmolVLM
    - And other HuggingFace vision-language models
    """
    from transformers import AutoProcessor, AutoModel, AutoModelForImageTextToText, BitsAndBytesConfig

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

    # Determine which model class to use based on model name
    model_lower = model_id.lower()

    if "internvl" in model_lower:
        # InternVL models use custom AutoModel with trust_remote_code
        print("Using AutoModel for InternVL...")
        model = AutoModel.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        # Standard VLMs use AutoModelForImageTextToText
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        except ValueError:
            # Fall back to AutoModel if ImageTextToText doesn't support this model
            print("Falling back to AutoModel...")
            model = AutoModel.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

    # Load LoRA adapter if provided
    if adapter_path:
        from peft import PeftModel
        adapter_path = Path(adapter_path)
        if adapter_path.exists():
            print(f"Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
        else:
            print(f"Warning: Adapter not found at {adapter_path}, using base model")

    model.eval()
    print(f"Model loaded on {model.device}")
    return model, processor


def generate_response(
    model,
    processor,
    image_path: str,
    question: str,
    model_id: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Generate response for an image-question pair."""
    image = Image.open(image_path).convert('RGB')

    # InternVL models use a special .chat() method
    if "internvl" in model_id.lower() and hasattr(model, 'chat'):
        # InternVL-style chat
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'do_sample': temperature > 0,
        }
        if temperature > 0:
            generation_config['temperature'] = temperature

        prompt = f"{SYSTEM_PROMPT}\n\n{question}"
        response = model.chat(processor, image, prompt, generation_config)
        return response.strip()

    # Standard transformers-style generation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs['input_ids'].shape[1]
    response = processor.tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    )

    return response.strip()


def load_test_samples(test_path: Path, max_samples: int = None):
    """Load test samples from JSONL file."""
    samples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                if max_samples and len(samples) >= max_samples:
                    break
    return samples


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference on test data")

    parser.add_argument('--model', type=str, required=True, help='HuggingFace model ID')
    parser.add_argument('--adapter', type=str, default=None, help='Path to LoRA adapter')
    parser.add_argument('--test-data', type=str, default='data/test_subset.jsonl')
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.1)

    args = parser.parse_args()

    # Setup paths
    test_path = Path(args.test_data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split('/')[-1]

    print("=" * 60)
    print("VLM Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter or 'None'}")
    print(f"Test data: {test_path}")
    print(f"Max samples: {args.samples or 'All'}")
    print(f"4-bit quantization: {not args.no_4bit}")

    # Check paths
    if not test_path.exists():
        print(f"Error: Test data not found: {test_path}")
        return 1

    # Load test samples
    samples = load_test_samples(test_path, args.samples)
    print(f"Loaded {len(samples)} test samples")

    # Check GPU
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available!")

    # Load model
    model, processor = load_model(
        args.model,
        adapter_path=args.adapter,
        use_4bit=not args.no_4bit
    )

    # Run inference
    results = []
    checkpoint_path = output_dir / f"predictions_checkpoint_{model_short}_{timestamp}.jsonl"

    for idx, sample in enumerate(tqdm(samples, desc="Running inference")):
        original_image_path = sample['images'][0]
        image_path = convert_windows_path(original_image_path)
        messages = sample['messages']

        # Extract question from user message
        user_msg = messages[1]
        question = None
        for content in user_msg['content']:
            if isinstance(content, dict) and content.get('type') == 'text':
                question = content['text']
                break

        # Extract reference response from assistant message
        reference_response = messages[2]['content']

        # Extract category
        category = extract_category_from_path(original_image_path)

        # Generate prediction
        try:
            prediction = generate_response(
                model, processor, image_path, question,
                model_id=args.model,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            prediction = ""

        results.append({
            "sample_id": idx,
            "model": args.model,
            "image_path": original_image_path,
            "category": category,
            "question": question[:100] + "..." if len(question) > 100 else question,
            "reference_response": reference_response,
            "prediction": prediction
        })

        # Checkpoint
        if (idx + 1) % args.checkpoint_interval == 0:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"\n  Checkpoint saved: {idx + 1}/{len(samples)} samples")

    # Save final predictions
    predictions_path = output_dir / f"predictions_{model_short}_{timestamp}.jsonl"
    with open(predictions_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nInference complete!")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Total samples: {len(results)}")

    # Print summary stats
    successful = sum(1 for r in results if r['prediction'])
    print(f"Successful predictions: {successful}/{len(results)}")

    return 0


if __name__ == "__main__":
    exit(main())
