"""
Generate full ground truth training data from annotations + dataset splits.

Creates JSONL files in HF chat format with all 8 questions per image,
using the sonographer's ground truth annotations (not Gemini).

Output:
    data/vlm_training/gt_train.jsonl      (~121K samples)
    data/vlm_training/gt_val.jsonl        (~15K samples)
    data/vlm_training/gt_test.jsonl       (~15K samples)
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.questions import QUESTIONS, QUESTION_COLUMNS
from experiments.evaluation.config import EVAL_SYSTEM_PROMPT as SYSTEM_PROMPT

# Build column -> full question text mapping from canonical source
QUESTIONS_MAP = dict(zip(QUESTION_COLUMNS, QUESTIONS))


def create_sample(image_path: str, question_text: str, answer: str) -> dict:
    """Create a single training sample in HF chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question_text},
            ]},
            {"role": "assistant", "content": str(answer).strip()},
        ],
        "images": [image_path],
    }


def main():
    annotations_path = PROJECT_ROOT / "data" / "Fetal Ultrasound Annotations Normalized.xlsx"
    splits_path = PROJECT_ROOT / "data" / "dataset_splits.json"
    output_dir = PROJECT_ROOT / "data" / "vlm_training"

    print(f"Loading annotations from {annotations_path}")
    df = pd.read_excel(annotations_path)
    print(f"Total annotations: {len(df)}")

    # Build image -> row lookup
    # Image paths in splits are like "Abdomen/Abdomen_001.png"
    df["image_key"] = df["Folder Name"] + "/" + df["Image Name"]
    image_lookup = df.set_index("image_key")

    print(f"Loading splits from {splits_path}")
    with open(splits_path) as f:
        splits_data = json.load(f)

    for split_name in ["train", "val", "test"]:
        split_images = splits_data["splits"][split_name]
        output_path = output_dir / f"gt_{split_name}.jsonl"

        samples = []
        missing = 0

        for class_name, image_list in split_images.items():
            for image_path in image_list:
                if image_path not in image_lookup.index:
                    missing += 1
                    continue

                row = image_lookup.loc[image_path]
                # Handle duplicate index (multiple rows for same image)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                for q_col, q_text in QUESTIONS_MAP.items():
                    answer = row.get(q_col)
                    if pd.isna(answer) or str(answer).strip() == "":
                        continue
                    samples.append(create_sample(image_path, q_text, answer))

        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"{split_name}: {len(samples)} samples from {sum(len(v) for v in split_images.values())} images ({missing} missing annotations)")
        if missing > 0:
            print(f"WARNING: {missing} images have no annotations and were excluded")
            print(f"  NOTE: These images are unannotated by sonographer. This is expected.")

    print("Done!")


if __name__ == "__main__":
    main()
