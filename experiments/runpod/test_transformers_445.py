"""
Main VLM test script - tests models compatible with transformers 4.45.

Models tested here (18 models):
- InternVL3 series (5 models)
- InternVL2 series (3 models)
- Qwen2-VL series (2 models)
- MiniCPM-V series (3 models)
- Kimi-VL series (2 models)
- Llama-3.2-Vision (1 model)
- Phi-4-multimodal (1 model)

Models requiring transformers 4.48+ are tested by test_transformers_448.py:
- SmolVLM2 series (3 models)
- InternVL3.5 series (3 models)
- Qwen2.5-VL series (5 models)
"""

import json
import time
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

# Models to test with transformers 4.45 (excludes SmolVLM2, InternVL3.5, Qwen2.5-VL which need 4.48+)
ALL_MODELS = [
    # InternVL3 series (works with 4.45)
    {"name": "internvl3-1b", "model_id": "OpenGVLab/InternVL3-1B", "min_vram": 4},
    {"name": "internvl3-2b", "model_id": "OpenGVLab/InternVL3-2B", "min_vram": 6},
    {"name": "internvl3-8b", "model_id": "OpenGVLab/InternVL3-8B", "min_vram": 18},
    {"name": "internvl3-9b", "model_id": "OpenGVLab/InternVL3-9B", "min_vram": 20},
    {"name": "internvl3-14b-awq", "model_id": "OpenGVLab/InternVL3-14B-AWQ", "min_vram": 12},

    # InternVL2 series (works with 4.45)
    {"name": "internvl2-2b", "model_id": "OpenGVLab/InternVL2-2B", "min_vram": 6},
    {"name": "internvl2-4b", "model_id": "OpenGVLab/InternVL2-4B", "min_vram": 10},
    {"name": "internvl2-8b", "model_id": "OpenGVLab/InternVL2-8B", "min_vram": 18},

    # Qwen2-VL series (works with 4.45)
    {"name": "qwen2-vl-2b", "model_id": "Qwen/Qwen2-VL-2B-Instruct", "min_vram": 6},
    {"name": "qwen2-vl-7b", "model_id": "Qwen/Qwen2-VL-7B-Instruct", "min_vram": 16},

    # MiniCPM-V series (gated)
    {"name": "minicpm-v-2.6", "model_id": "openbmb/MiniCPM-V-2_6", "min_vram": 10},
    {"name": "minicpm-o-2.6", "model_id": "openbmb/MiniCPM-o-2_6", "min_vram": 10},
    {"name": "minicpm-v-4", "model_id": "openbmb/MiniCPM-V-4", "min_vram": 12},

    # Kimi-VL series
    {"name": "kimi-vl-a3b-instruct", "model_id": "moonshotai/Kimi-VL-A3B-Instruct", "min_vram": 10},
    {"name": "kimi-vl-a3b-thinking", "model_id": "moonshotai/Kimi-VL-A3B-Thinking", "min_vram": 10},

    # Llama 3.2 Vision (gated)
    {"name": "llama-3.2-11b-vision", "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct", "min_vram": 26},

    # Phi-4 multimodal
    {"name": "phi-4-multimodal", "model_id": "microsoft/Phi-4-multimodal-instruct", "min_vram": 32},
]

# Models requiring transformers 4.48+ (tested by test_transformers_448.py):
# - SmolVLM2 series (3 models): needs 'smolvlm' architecture
# - InternVL3.5 series (3 models): needs Qwen3Config
# - Qwen2.5-VL series (5 models): needs newer transformers


def is_qwen_model(model_id: str) -> bool:
    return "Qwen" in model_id


def is_internvl_model(model_id: str) -> bool:
    return "InternVL" in model_id


def is_smolvlm_model(model_id: str) -> bool:
    return "SmolVLM" in model_id


def is_minicpm_model(model_id: str) -> bool:
    return "MiniCPM" in model_id


def is_kimi_model(model_id: str) -> bool:
    return "Kimi" in model_id


def is_phi_model(model_id: str) -> bool:
    return "Phi" in model_id


def is_awq_model(model_id: str) -> bool:
    return "-AWQ" in model_id or "-awq" in model_id


def run_smolvlm_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """SmolVLM2 inference."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_internvl_inference(model, tokenizer, image: Image.Image, prompt: str) -> str:
    """InternVL inference using chat method."""
    # InternVL models use a special chat interface with tokenizer
    # Use load_image helper if available, otherwise process directly
    try:
        # Try the model's built-in chat method
        generation_config = {"max_new_tokens": 100, "do_sample": False}
        response = model.chat(tokenizer, image, prompt, generation_config=generation_config)
        return response
    except Exception as e:
        # Fallback: manual processing
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(model.config._name_or_path, trust_remote_code=True)
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(model.device, model.dtype)
        response = model.chat(tokenizer, pixel_values, prompt, generation_config={"max_new_tokens": 100})
        return response


def run_qwen_vl_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """Qwen2-VL/Qwen2.5-VL inference."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_minicpm_inference(model, tokenizer, image: Image.Image, prompt: str) -> str:
    """MiniCPM-V inference."""
    messages = [{"role": "user", "content": prompt}]
    with torch.no_grad():
        response = model.chat(image=image, msgs=messages, tokenizer=tokenizer, sampling=False, max_new_tokens=100)
    return response


def run_kimi_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """Kimi-VL inference."""
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_phi_inference(model, processor, image: Image.Image, prompt: str) -> str:
    """Phi-4 multimodal inference."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


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
    }

    if min_vram > max_vram_gb:
        result["skipped"] = True
        result["skip_reason"] = f"Requires {min_vram}GB VRAM (available: {max_vram_gb}GB)"
        print(f"SKIP: {model_name} - {result['skip_reason']}")
        return result

    print(f"\nTesting {model_name} ({model_id})...")

    try:
        # Setup quantization
        bnb_config = None
        if not is_awq_model(model_id):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        load_start = time.time()

        # Load processor/tokenizer
        if is_minicpm_model(model_id):
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            processor = None
        elif is_internvl_model(model_id):
            # InternVL needs both tokenizer and image processor
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        else:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            tokenizer = None
        print(f"  Processor loaded")

        # Load model
        load_kwargs = {"device_map": "auto", "trust_remote_code": True, "torch_dtype": torch.bfloat16}
        if bnb_config and not is_awq_model(model_id):
            load_kwargs["quantization_config"] = bnb_config

        if is_qwen_model(model_id):
            model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        elif is_minicpm_model(model_id):
            model = AutoModel.from_pretrained(model_id, **load_kwargs)
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
            response = run_qwen_vl_inference(model, processor, image, prompt)
        elif is_minicpm_model(model_id):
            response = run_minicpm_inference(model, tokenizer, image, prompt)
        elif is_kimi_model(model_id):
            response = run_kimi_inference(model, processor, image, prompt)
        elif is_phi_model(model_id):
            response = run_phi_inference(model, processor, image, prompt)
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
    """Run tests on all compatible models."""
    # HF login
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("HuggingFace login OK")

    test_file = Path("test_subset.jsonl")
    output_file = Path("results_main.json")

    print("=" * 80)
    print("VLM Test - All Compatible Models")
    print("=" * 80)

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

    # Test all models
    results = []
    for i, m in enumerate(ALL_MODELS, 1):
        print(f"\n{'='*80}")
        print(f"Model {i}/{len(ALL_MODELS)}")
        print("=" * 80)

        result = test_model(m["name"], m["model_id"], m["min_vram"], image, prompt, max_vram)
        results.append(result)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        time.sleep(2)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
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
