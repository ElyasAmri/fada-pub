"""
Evaluate a fine-tuned LoRA adapter using HF+PEFT direct inference.

Unlike vLLM, this preserves ALL LoRA weights including visual encoder.
Produces a checkpoint file compatible with the scoring pipeline.

Usage:
    python experiments/framework_comparison/eval_hf_peft.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --adapter /path/to/adapter \
        --output-dir /path/to/output

    # Quick test with 10 images
    python experiments/framework_comparison/eval_hf_peft.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --adapter /path/to/adapter \
        --output-dir /path/to/output \
        --max-images 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.questions import QUESTIONS

SYSTEM_PROMPT = (
    "You are an expert in fetal ultrasound imaging analysis. "
    "Provide accurate, detailed, and clinically relevant interpretations. "
    "Be precise and professional in your assessments."
)


def _is_gemma_processor(processor) -> bool:
    name = getattr(getattr(processor, "tokenizer", None), "name_or_path", "") or ""
    return "gemma" in name.lower()


def _decode_generated_text(processor, output_ids, input_len: int, is_gemma: bool) -> str:
    if not is_gemma:
        return processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=False)
    # Gemma 4 with thinking enabled emits a thought channel followed by <channel|>.
    if "<channel|>" in raw:
        raw = raw.split("<channel|>", 1)[1]
    return raw.strip()


def load_model(model_id: str, adapter_path: str, use_4bit: bool = True):
    """Load base model + LoRA adapter with HF+PEFT."""
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading base model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # Count adapter parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Adapter loaded: {trainable:,} trainable / {total:,} total params")

    return model, processor


def generate_response(model, processor, image_path: str, question: str) -> str:
    """Generate response for a single image-question pair."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    is_gemma = _is_gemma_processor(processor)
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if is_gemma:
        # Gemma 4 often gives "please provide image" on medical prompts when thinking is disabled.
        chat_kwargs["enable_thinking"] = True
    text = processor.apply_chat_template(messages, **chat_kwargs)

    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return _decode_generated_text(processor, output_ids, input_len, is_gemma)


def load_test_images(data_root: str, splits_path: str) -> list:
    """Load test split images."""
    with open(splits_path) as f:
        data = json.load(f)

    # Format: {splits: {test: {category: [relative_paths]}}}
    test_split = data.get("splits", data).get("test", {})

    test_images = []
    for category, paths in test_split.items():
        for rel_path in paths:
            img_path = Path(data_root) / rel_path
            if img_path.exists():
                test_images.append({
                    "image_path": str(img_path),
                    "relative_path": rel_path,
                    "category": category,
                })
    return test_images


def main():
    parser = argparse.ArgumentParser(description="HF+PEFT adapter evaluation")
    parser.add_argument("--model", required=True, help="Base model ID")
    parser.add_argument("--adapter", required=True, help="Adapter path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--data-root", default=None, help="Image data root")
    parser.add_argument("--splits", default=None, help="Dataset splits JSON")
    parser.add_argument("--max-images", type=int, default=0, help="Max images (0=all)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    data_root = args.data_root or str(project_root / "data" / "Fetal Ultrasound")
    splits_path = args.splits or str(project_root / "data" / "dataset_splits.json")

    # Load test images
    test_images = load_test_images(data_root, splits_path)
    if args.max_images > 0:
        test_images = test_images[:args.max_images]
    print(f"Test images: {len(test_images)}")

    # Load model
    model, processor = load_model(args.model, args.adapter, use_4bit=not args.no_4bit)

    # Output setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"checkpoint_hf-peft_{args.model.replace('/', '_')}.json"

    # Resume from checkpoint if exists
    completed = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            data = json.load(f)
        completed = data.get("completed_images", {})
        print(f"Resuming from checkpoint: {len(completed)} images done")

    # Run inference
    total_time = 0
    start_time = time.time()

    for i, img_info in enumerate(tqdm(test_images, desc="Evaluating")):
        rel_path = img_info["relative_path"]
        if rel_path in completed:
            continue

        image_results = {
            "image": Path(rel_path).name,
            "category": img_info["category"],
            "questions": [],
        }

        for q_idx, question in enumerate(QUESTIONS):
            t0 = time.time()
            try:
                response = generate_response(model, processor, img_info["image_path"], question)
                elapsed = time.time() - t0
            except Exception as e:
                response = f"ERROR: {e}"
                elapsed = time.time() - t0

            image_results["questions"].append({
                "question_idx": q_idx,
                "question": question[:80] + "...",
                "response": response,
                "time": elapsed,
            })

        completed[rel_path] = image_results

        # Save checkpoint periodically
        if (i + 1) % args.checkpoint_interval == 0:
            _save_checkpoint(checkpoint_path, args.model, completed, len(test_images))

    total_time = time.time() - start_time
    _save_checkpoint(checkpoint_path, args.model, completed, len(test_images))

    print(f"\nDone: {len(completed)} images in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"Checkpoint: {checkpoint_path}")


def _save_checkpoint(path, model_name, completed, total):
    data = {
        "model_name": f"HF-PEFT ({model_name})",
        "config": {"total_images": total},
        "completed_images": completed,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
