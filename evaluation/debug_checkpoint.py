"""Debug checkpoint key format vs test_images key format."""
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.api_models.test_api_vlm import get_images_from_split

data_root = project_root / "data" / "Fetal Ultrasound"
test_images = get_images_from_split("test", data_root)

# Build keys the same way run_parallel_evaluation does
test_keys = set()
for img in test_images:
    key = f"{img['category']}/{img['name']}"
    test_keys.add(key)

# Load checkpoint
cp_path = project_root / "experiments" / "api_models" / "results" / "checkpoint_vllm_Qwen_Qwen3-VL-8B-Instruct.json"
cp = json.load(open(cp_path))
cp_keys = set(cp["completed_images"].keys())

print(f"Test images: {len(test_keys)}")
print(f"Checkpoint keys: {len(cp_keys)}")
print(f"Sample test keys: {sorted(test_keys)[:3]}")
print(f"Sample cp keys: {sorted(cp_keys)[:3]}")

missing = test_keys - cp_keys
extra = cp_keys - test_keys
print(f"\nIn test but NOT in checkpoint (should be 242): {len(missing)}")
print(f"In checkpoint but NOT in test: {len(extra)}")

if missing:
    print(f"\nFirst 5 missing: {sorted(missing)[:5]}")
if extra:
    print(f"\nFirst 5 extra: {sorted(extra)[:5]}")
