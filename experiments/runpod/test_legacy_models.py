"""
Legacy model test script - MiniCPM-V-2.6 and Llama 3.2 Vision.
Uses transformers 4.45 for compatibility with custom model code.
"""

import json
import time
import gc
import os
import subprocess
import sys
from pathlib import Path

# Models requiring older transformers
LEGACY_MODELS = [
    {"name": "minicpm-v-2.6", "model_id": "openbmb/MiniCPM-V-2_6", "min_vram": 10},
    {"name": "minicpm-o-2.6", "model_id": "openbmb/MiniCPM-o-2_6", "min_vram": 10},
    {"name": "llama-3.2-11b-vision", "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct", "min_vram": 26},
]


def setup_legacy_venv():
    """Create isolated venv with transformers 4.45."""
    venv_path = Path("/workspace/legacy_venv")

    if venv_path.exists():
        print("Legacy venv already exists")
        return venv_path

    print("Creating legacy venv with transformers 4.45...")

    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
    pip = venv_path / "bin" / "pip"

    packages = [
        "torch==2.4.0",
        "transformers==4.45.0",
        "accelerate",
        "bitsandbytes",
        "pillow",
        "soundfile",
        "huggingface_hub",
    ]

    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.run([str(pip), "install", "-q", pkg], check=True)

    print("Legacy venv setup complete")
    return venv_path


def run_legacy_test(venv_path: Path):
    """Run legacy model test in isolated venv."""
    python = venv_path / "bin" / "python"

    test_script = '''
import json
import time
import gc
import os
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import login

# HF login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print("HF login OK")

MODELS = [
    {"name": "minicpm-v-2.6", "model_id": "openbmb/MiniCPM-V-2_6", "min_vram": 10, "type": "minicpm"},
    {"name": "minicpm-o-2.6", "model_id": "openbmb/MiniCPM-o-2_6", "min_vram": 10, "type": "minicpm"},
    {"name": "llama-3.2-11b-vision", "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct", "min_vram": 26, "type": "llama"},
]

max_vram = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
print(f"GPU VRAM: {max_vram}GB")

img = Image.new("RGB", (224, 224), (128, 128, 128))
prompt = "Describe this image."

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

results = []
for m in MODELS:
    result = {"name": m["name"], "model_id": m["model_id"], "success": False, "error": None}

    if m["min_vram"] > max_vram:
        result["error"] = f"Needs {m['min_vram']}GB, have {max_vram}GB"
        results.append(result)
        print(f"SKIP {m['name']}: {result['error']}")
        continue

    print(f"Testing {m['name']}...")
    try:
        if m["type"] == "minicpm":
            tokenizer = AutoTokenizer.from_pretrained(m["model_id"], trust_remote_code=True)
            model = AutoModel.from_pretrained(
                m["model_id"],
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            print(f"  Model loaded")

            messages = [{"role": "user", "content": prompt}]
            with torch.no_grad():
                response = model.chat(image=img, msgs=messages, tokenizer=tokenizer, sampling=False, max_new_tokens=50)

        elif m["type"] == "llama":
            processor = AutoProcessor.from_pretrained(m["model_id"], trust_remote_code=True)
            model = MllamaForConditionalGeneration.from_pretrained(
                m["model_id"],
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            print(f"  Model loaded")

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=img, text=text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            response = processor.decode(outputs[0], skip_special_tokens=True)

        result["success"] = True
        print(f"  SUCCESS: {len(response)} chars")

    except Exception as e:
        result["error"] = str(e)[:200]
        print(f"  ERROR: {result['error']}")

    finally:
        if "model" in dir():
            del model
        torch.cuda.empty_cache()
        gc.collect()

    results.append(result)

with open("/workspace/test/results_legacy.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\nResults saved to results_legacy.json")
'''

    script_path = Path("/workspace/test/legacy_test_inner.py")
    script_path.write_text(test_script)

    env = os.environ.copy()
    result = subprocess.run([str(python), str(script_path)], env=env, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    print("=" * 80)
    print("Legacy Models Test (MiniCPM, Llama - transformers 4.45)")
    print("=" * 80)

    venv_path = setup_legacy_venv()
    success = run_legacy_test(venv_path)

    if success:
        print("\nLegacy test completed")
    else:
        print("\nLegacy test failed")


if __name__ == "__main__":
    main()
