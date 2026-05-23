"""
Convert HuggingFace chat format JSONL to ShareGPT format for LLaMA Factory and Axolotl.

HF chat format:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}, {"role": "assistant", "content": "..."}], "images": ["path/to/img.png"]}

ShareGPT format:
    {"conversations": [{"from": "system", "value": "..."}, {"from": "human", "value": "<image>\n..."}, {"from": "gpt", "value": "..."}], "images": ["path/to/img.png"]}

Usage:
    python experiments/framework_comparison/convert_to_sharegpt.py
    python experiments/framework_comparison/convert_to_sharegpt.py --input data/vlm_training/gt_val.jsonl --output data/vlm_training/gt_val_sharegpt.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

ROLE_MAP = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
}


def convert_message(msg: dict) -> dict:
    """Convert a single HF chat message to ShareGPT format."""
    role = ROLE_MAP[msg["role"]]
    content = msg["content"]

    if isinstance(content, list):
        # Multimodal content: extract text parts, prepend <image> for image parts
        text_parts = []
        has_image = False
        for part in content:
            if part["type"] == "image":
                has_image = True
            elif part["type"] == "text":
                text_parts.append(part["text"])
        value = "\n".join(text_parts)
        if has_image:
            value = "<image>\n" + value
    else:
        value = content

    return {"from": role, "value": value}


def convert_file(input_path: Path, output_path: Path) -> int:
    """Convert an entire JSONL file from HF chat to ShareGPT format."""
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = sample["messages"]

            conversations = [convert_message(m) for m in messages]

            out = {"conversations": conversations}
            if "images" in sample:
                out["images"] = sample["images"]

            fout.write(json.dumps(out) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert HF chat JSONL to ShareGPT format")
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "data/vlm_training/gt_train.jsonl",
        help="Input JSONL file in HF chat format",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL file in ShareGPT format (default: input with _sharegpt suffix)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_stem(args.input.stem + "_sharegpt")

    print(f"Converting {args.input} -> {args.output}")
    count = convert_file(args.input, args.output)
    print(f"Converted {count} samples")


if __name__ == "__main__":
    main()
