"""
GPTQ/AWQ model test script - uses transformers 4.45 for compatibility.
Creates isolated venv with older transformers.
"""

import json
import time
import gc
import os
import subprocess
import sys
from pathlib import Path

# Models requiring GPTQ/AWQ with older transformers
GPTQ_AWQ_MODELS = [
    {"name": "qwen2.5-vl-3b-gptq", "model_id": "hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4", "min_vram": 4},
    {"name": "qwen2.5-vl-7b-gptq", "model_id": "hfl/Qwen2.5-VL-7B-Instruct-GPTQ-Int4", "min_vram": 8},
    {"name": "qwen2.5-vl-32b-awq", "model_id": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", "min_vram": 20},
]


def setup_gptq_venv():
    """Create isolated venv with transformers 4.45 for GPTQ compatibility."""
    venv_path = Path("/workspace/gptq_venv")

    if venv_path.exists():
        print("GPTQ venv already exists")
        return venv_path

    print("Creating GPTQ venv with transformers 4.45...")

    # Create venv
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    pip = venv_path / "bin" / "pip"

    # Install compatible packages
    packages = [
        "torch==2.4.0",
        "transformers==4.45.0",
        "accelerate",
        "bitsandbytes",
        "pillow",
        "auto-gptq",
        "autoawq",
        "huggingface_hub",
    ]

    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.run([str(pip), "install", "-q", pkg], check=True)

    print("GPTQ venv setup complete")
    return venv_path


def run_gptq_test(venv_path: Path):
    """Run GPTQ test in isolated venv."""
    python = venv_path / "bin" / "python"

    test_script = '''
import json
import time
import gc
import os
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from huggingface_hub import login

# HF login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print("HF login OK")

MODELS = [
    {"name": "qwen2.5-vl-3b-gptq", "model_id": "hfl/Qwen2.5-VL-3B-Instruct-GPTQ-Int4", "min_vram": 4},
    {"name": "qwen2.5-vl-7b-gptq", "model_id": "hfl/Qwen2.5-VL-7B-Instruct-GPTQ-Int4", "min_vram": 8},
    {"name": "qwen2.5-vl-32b-awq", "model_id": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ", "min_vram": 20},
]

max_vram = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
print(f"GPU VRAM: {max_vram}GB")

# Create test image
img = Image.new("RGB", (224, 224), (128, 128, 128))
prompt = "Describe this image."

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
        processor = AutoProcessor.from_pretrained(m["model_id"], trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            m["model_id"],
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print(f"  Model loaded")

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        result["success"] = True
        print(f"  SUCCESS: {len(response)} chars")

    except Exception as e:
        result["error"] = str(e)[:200]
        print(f"  ERROR: {result['error']}")

    finally:
        if "model" in dir():
            del model
        if "processor" in dir():
            del processor
        torch.cuda.empty_cache()
        gc.collect()

    results.append(result)

with open("/workspace/test/results_gptq.json", "w") as f:
    json.dump(results, f, indent=2)

print("\\nResults saved to results_gptq.json")
'''

    # Write and run test script
    script_path = Path("/workspace/test/gptq_test_inner.py")
    script_path.write_text(test_script)

    env = os.environ.copy()
    result = subprocess.run([str(python), str(script_path)], env=env, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    print("=" * 80)
    print("GPTQ/AWQ Models Test (transformers 4.45)")
    print("=" * 80)

    venv_path = setup_gptq_venv()
    success = run_gptq_test(venv_path)

    if success:
        print("\nGPTQ test completed")
    else:
        print("\nGPTQ test failed")


if __name__ == "__main__":
    main()
