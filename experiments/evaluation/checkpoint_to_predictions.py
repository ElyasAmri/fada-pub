"""
Convert test_api_vlm.py checkpoint file to predictions JSONL for scoring.

Usage:
    python -m experiments.evaluation.checkpoint_to_predictions \
        --checkpoint experiments/api_models/results/checkpoint_vllm_Qwen_Qwen3-VL-8B-Instruct.json \
        --output outputs/evaluation/predictions_qwen3vl8b.jsonl
"""

import argparse
import json
import sys
import logging
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config.questions import QUESTIONS

logger = logging.getLogger(__name__)


def convert(checkpoint_path: Path, output_path: Path) -> int:
    with open(checkpoint_path) as f:
        data = json.load(f)

    completed = data.get("completed_images", {})
    count = 0
    sample_id = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out:
        for image_key, result in completed.items():
            category = result.get("category", image_key.split("/")[0])
            image_name = result.get("image", image_key.split("/")[-1])
            image_path = f"{category}/{image_name}"

            for q in result.get("questions", []):
                raw_question_text = q.get("question", "")
                if "question_idx" in q:
                    q_idx = q["question_idx"]
                    question_text = QUESTIONS[q_idx] if q_idx < len(QUESTIONS) else raw_question_text
                else:
                    # question_idx absent: detect from question text to avoid silently
                    # miscategorizing all entries as Q1 (index 0)
                    if raw_question_text:
                        from experiments.evaluation.question_scorer import detect_question_index
                        try:
                            q_idx = detect_question_index(raw_question_text)
                            question_text = QUESTIONS[q_idx]
                        except ValueError:
                            logger.warning(
                                "Cannot detect question index for image %s, question text: %r -- skipping",
                                image_key, raw_question_text[:80],
                            )
                            continue
                    else:
                        logger.warning(
                            "Image %s has a question entry with no question_idx and no question text -- skipping",
                            image_key,
                        )
                        continue
                response = q.get("response", "")

                pred = {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "category": category,
                    "question": question_text,
                    "prediction": response,
                }
                out.write(json.dumps(pred) + "\n")
                count += 1
                sample_id += 1

    print(f"Converted {len(completed)} images -> {count} predictions to {output_path}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert checkpoint to predictions JSONL")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
