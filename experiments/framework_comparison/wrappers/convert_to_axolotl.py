"""Convert HF chat format JSONL to Axolotl multimodal format.

Input format (HF chat):
  {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}], "images": ["rel/path.jpg"]}

Output format (Axolotl):
  {"messages": [{"role": "user", "content": [{"type": "image", "path": "/abs/path.jpg"}, {"type": "text", "text": "..."}]}]}

Usage:
    python convert_to_axolotl.py --input data.jsonl --output axolotl_data.json --data-root /path/to/images
"""

import argparse
import json
import os
from pathlib import Path


def convert(input_path: str, output_path: str, data_root: str):
    data_root = str(Path(data_root).resolve())
    samples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            images = d.get("images", [])
            img_idx = 0

            new_messages = []
            for msg in d.get("messages", []):
                if isinstance(msg["content"], list):
                    new_content = []
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            # Replace with absolute path
                            if img_idx < len(images):
                                img_path = images[img_idx]
                                if not os.path.isabs(img_path):
                                    img_path = f"{data_root}/{img_path}"
                                new_content.append({"type": "image", "path": img_path})
                                img_idx += 1
                        elif item.get("type") == "text":
                            new_content.append({"type": "text", "text": item["text"]})
                        else:
                            new_content.append(item)
                    new_messages.append({"role": msg["role"], "content": new_content})
                else:
                    # String content — wrap in text type
                    new_messages.append({
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    })

            samples.append({"messages": new_messages})

    with open(output_path, "w", newline="\n") as f:
        json.dump(samples, f, ensure_ascii=False)

    print(f"Converted {len(samples)} samples: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    convert(args.input, args.output, args.data_root)
