"""
Test script for models requiring transformers 4.48+.

Models tested (8 total):
- SmolVLM2 series (3 models) - needs 'smolvlm' architecture
- InternVL3.5 series (3 models) - needs Qwen3Config
- Qwen2.5-VL series (2 models) - needs newer transformers (GPTQ skipped)

This script is designed to run in an isolated venv with transformers 4.48+.
"""

import json
import time
import gc
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

# Models requiring transformers 4.48+
# NOTE: GPTQ/AWQ models skipped - gptqmodel requires CUDA compilation
NEW_TRANSFORMERS_MODELS = [
    # SmolVLM2 series - needs 'smolvlm' architecture
    {"name": "smolvlm2-256m", "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", "min_vram": 2},
    {"name": "smolvlm2-500m", "model_id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "min_vram": 3},
    {"name": "smolvlm2-2.2b", "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct", "min_vram": 6},

    # InternVL3.5 series - needs Qwen3Config
    {"name": "internvl3.5-1b", "model_id": "OpenGVLab/InternVL3_5-1B", "min_vram": 4},
    {"name": "internvl3.5-2b", "model_id": "OpenGVLab/InternVL3_5-2B", "min_vram": 6},
    {"name": "internvl3.5-8b", "model_id": "OpenGVLab/InternVL3_5-8B", "min_vram": 18},

    # Qwen2.5-VL series (non-quantized only - GPTQ needs gptqmodel)
    {"name": "qwen2.5-vl-3b", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct", "min_vram": 8},
    {"name": "qwen2.5-vl-7b", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct", "min_vram": 16},
]


def is_smolvlm_model(model_id: str) -> bool:
    return "SmolVLM" in model_id


def is_internvl_model(model_id: str) -> bool:
    return "InternVL" in model_id


def is_qwen_model(model_id: str) -> bool:
    return "Qwen" in model_id


def is_gptq_model(model_id: str) -> bool:
    return "GPTQ" in model_id or "gptq" in model_id


def is_awq_model(model_id: str) -> bool:
    return "AWQ" in model_id or "awq" in model_id


def is_quantized_model(model_id: str) -> bool:
    return is_gptq_model(model_id) or is_awq_model(model_id)


def run_smolvlm_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """SmolVLM2 inference."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_internvl_inference(model, tokenizer, image: Image.Image, prompt: str) -> str:
    """InternVL3.5 inference using chat method."""
    try:
        generation_config = {"max_new_tokens": 100, "do_sample": False}
        response = model.chat(tokenizer, image, prompt, generation_config=generation_config)
        return response
    except Exception as e:
        # Fallback with pixel values
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(model.device, model.dtype)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config={"max_new_tokens": 100})
        return response


def run_qwen25_vl_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """Qwen2.5-VL inference."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]


def load_test_sample(test_file: Path) -> Optional[Dict]:
    """Load first sample from test_subset.jsonl."""
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.loads(f.readline())
    except Exception as e:
        print(f"Error loading test sample: {e}")
        return None


def prepare_input(sample: Dict):
    """Prepare input for model inference."""
    messages = sample.get("messages", [])
    image_path = sample.get("images", [])[0] if sample.get("images") else None
    if not image_path:
        raise ValueError("No image path found")

    # Handle Windows paths on Linux (dry-run only)
    # WARNING: This replaces real ultrasound images with a gray placeholder.
    # Dry-run results validate model loading, NOT real image understanding.
    # For actual evaluation, use the standalone inference script with uploaded images.
    if image_path.startswith("C:"):
        print(f"  WARNING: Windows path detected ({image_path[:40]}...), using dummy gray image")
        image_path = "/workspace/test/test_image.png"

    image = Image.open(image_path).convert("RGB")

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", [])
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            prompt = " ".join(text_parts) if text_parts else "Describe this ultrasound image."
            return image, prompt

    return image, "Describe this ultrasound image."


def test_model(model_name: str, model_id: str, min_vram: int, image: Image.Image, prompt: str, max_vram_gb: int) -> Dict:
    """Test a single model."""
    result = {
        "model_name": model_name,
        "model_id": model_id,
        "min_vram_gb": min_vram,
        "download_success": False,
        "inference_success": False,
        "error_message": None,
        "load_time_seconds": None,
        "inference_time_seconds": None,
        "skipped": False,
        "skip_reason": None,
        "transformers_version": "4.48+",
    }

    if min_vram > max_vram_gb:
        result["skipped"] = True
        result["skip_reason"] = f"Requires {min_vram}GB VRAM (available: {max_vram_gb}GB)"
        print(f"SKIP: {model_name} - {result['skip_reason']}")
        return result

    print(f"\nTesting {model_name} ({model_id})...")

    try:
        # Setup quantization for non-quantized models
        # Skip BnB for: GPTQ/AWQ (already quantized), SmolVLM2, Qwen2.5-VL (BnB incompatible)
        bnb_config = None
        skip_bnb = is_quantized_model(model_id) or is_smolvlm_model(model_id) or "Qwen2.5-VL" in model_id
        if not skip_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        load_start = time.time()

        # Load processor/tokenizer
        if is_internvl_model(model_id):
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        else:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = None
        print(f"  Processor loaded")

        # Load model
        load_kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config

        if is_smolvlm_model(model_id):
            # SmolVLM2 needs AutoModelForImageTextToText for generate() method
            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        elif is_qwen_model(model_id):
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        else:
            model = AutoModel.from_pretrained(model_id, **load_kwargs)

        result["download_success"] = True
        result["load_time_seconds"] = round(time.time() - load_start, 2)
        print(f"  Model loaded in {result['load_time_seconds']}s")

        # Run inference
        inference_start = time.time()

        if is_smolvlm_model(model_id):
            response = run_smolvlm_inference(model, processor, image, prompt)
        elif is_internvl_model(model_id):
            response = run_internvl_inference(model, tokenizer, image, prompt)
        elif is_qwen_model(model_id):
            response = run_qwen25_vl_inference(model, processor, image, prompt)
        else:
            raise ValueError(f"No inference path for {model_id}")

        result["inference_time_seconds"] = round(time.time() - inference_start, 2)
        result["inference_success"] = True
        print(f"  Inference: {result['inference_time_seconds']}s, response: {len(response)} chars")
        print(f"  SUCCESS")

    except Exception as e:
        result["error_message"] = str(e)[:500]
        print(f"  ERROR: {result['error_message']}")

    finally:
        # Cleanup
        for var in ['model', 'processor', 'tokenizer']:
            if var in locals():
                del locals()[var]
        torch.cuda.empty_cache()
        gc.collect()

    return result


def main():
    """Run tests on models requiring transformers 4.48+."""
    # HF login
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("HuggingFace login OK")

    test_file = Path("test_subset.jsonl")
    output_file = Path("results_new_transformers.json")

    print("=" * 80)
    print("VLM Test - Models Requiring Transformers 4.48+")
    print("=" * 80)

    import transformers
    print(f"Transformers version: {transformers.__version__}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    gpu_name = torch.cuda.get_device_name(0)
    max_vram = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    print(f"GPU: {gpu_name}, VRAM: {max_vram}GB")

    # Load sample and prepare image
    sample = load_test_sample(test_file)
    if not sample:
        print("ERROR: No test sample")
        return

    # Create a dummy test image for dry-run validation
    # NOTE: This is a gray placeholder, NOT a real ultrasound image.
    # Dry-run only validates model loading and inference pipeline.
    # See GitHub issue #6 for details.
    test_image_path = Path("/workspace/test/test_image.png")
    if not test_image_path.exists():
        img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        img.save(test_image_path)
        print(f"WARNING: Created DUMMY gray test image at {test_image_path}")
        print(f"  Dry-run results validate model loading only, not real image understanding.")

    image, prompt = prepare_input(sample)
    print(f"Prompt: {prompt[:100]}...")

    # Test models
    results = []
    for i, m in enumerate(NEW_TRANSFORMERS_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"Model {i}/{len(NEW_TRANSFORMERS_MODELS)}")
        print("=" * 80)

        result = test_model(m["name"], m["model_id"], m["min_vram"], image, prompt, max_vram)
        results.append(result)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        time.sleep(2)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Transformers 4.48+ Models")
    print("=" * 80)
    success = [r for r in results if r["inference_success"]]
    failed = [r for r in results if r["error_message"]]
    skipped = [r for r in results if r["skipped"]]

    print(f"Total: {len(results)}, Success: {len(success)}, Failed: {len(failed)}, Skipped: {len(skipped)}")

    if success:
        print(f"\nSuccessful ({len(success)}):")
        for r in success:
            print(f"  - {r['model_name']}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  - {r['model_name']}: {r['error_message'][:80]}")


if __name__ == "__main__":
    main()
