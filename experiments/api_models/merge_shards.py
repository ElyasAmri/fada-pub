#!/usr/bin/env python3
"""
Merge MedGemma shard checkpoints into a single checkpoint file.

Usage:
    python merge_shards.py
    python merge_shards.py --output merged_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def find_shard_checkpoints(results_dir: Path, model: str = "google_medgemma-4b-it") -> List[Path]:
    """Find all shard checkpoint files"""
    pattern = f"checkpoint_vllm_{model}_shard*.json"
    shards = list(results_dir.glob(pattern))
    shards.sort()
    return shards


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Load a checkpoint file"""
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def merge_checkpoints(main_checkpoint: Path, shard_checkpoints: List[Path],
                      output_path: Path) -> Dict:
    """
    Merge main checkpoint with shard checkpoints

    Args:
        main_checkpoint: Path to main checkpoint file
        shard_checkpoints: List of shard checkpoint paths
        output_path: Path to save merged checkpoint

    Returns:
        Statistics about the merge
    """
    merged = {
        "completed_images": {},
        "model": "google/medgemma-4b-it",
        "provider": "vllm"
    }

    stats = {
        "main_images": 0,
        "shard_images": {},
        "total_unique": 0,
        "duplicates": 0
    }

    # Load main checkpoint
    if main_checkpoint.exists():
        main_data = load_checkpoint(main_checkpoint)
        merged["completed_images"].update(main_data.get("completed_images", {}))
        stats["main_images"] = len(main_data.get("completed_images", {}))
        print(f"Loaded main checkpoint: {stats['main_images']} images")
    else:
        print(f"Main checkpoint not found: {main_checkpoint}")

    # Load shard checkpoints
    total_before_shards = len(merged["completed_images"])

    for shard_path in shard_checkpoints:
        shard_data = load_checkpoint(shard_path)
        shard_images = shard_data.get("completed_images", {})

        # Count new vs duplicate
        new_count = 0
        dup_count = 0
        for img_path, data in shard_images.items():
            if img_path not in merged["completed_images"]:
                merged["completed_images"][img_path] = data
                new_count += 1
            else:
                dup_count += 1

        stats["shard_images"][shard_path.name] = {
            "total": len(shard_images),
            "new": new_count,
            "duplicates": dup_count
        }
        stats["duplicates"] += dup_count

        print(f"Loaded {shard_path.name}: {len(shard_images)} images "
              f"({new_count} new, {dup_count} duplicates)")

    stats["total_unique"] = len(merged["completed_images"])

    # Save merged checkpoint
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged checkpoint saved to: {output_path}")
    print(f"Total unique images: {stats['total_unique']}")

    return stats


def calculate_accuracy(checkpoint_path: Path) -> Dict:
    """Calculate accuracy from a merged checkpoint"""
    data = load_checkpoint(checkpoint_path)
    completed = data.get("completed_images", {})

    total = 0
    correct = 0
    by_question = {}
    by_category = {}

    for img_path, img_data in completed.items():
        results = img_data.get("results", {})

        # Extract category from path
        parts = Path(img_path).parts
        category = parts[-2] if len(parts) >= 2 else "Unknown"

        if category not in by_category:
            by_category[category] = {"total": 0, "correct": 0}

        for q_num, q_data in results.items():
            if isinstance(q_data, dict):
                is_correct = q_data.get("correct", False)
                total += 1
                by_category[category]["total"] += 1

                if q_num not in by_question:
                    by_question[q_num] = {"total": 0, "correct": 0}
                by_question[q_num]["total"] += 1

                if is_correct:
                    correct += 1
                    by_category[category]["correct"] += 1
                    by_question[q_num]["correct"] += 1

    overall_acc = (correct / total * 100) if total > 0 else 0

    print(f"\n=== Accuracy Report ===")
    print(f"Overall: {correct}/{total} ({overall_acc:.2f}%)")

    print(f"\nBy Question:")
    for q_num in sorted(by_question.keys()):
        q_data = by_question[q_num]
        acc = (q_data["correct"] / q_data["total"] * 100) if q_data["total"] > 0 else 0
        print(f"  {q_num}: {q_data['correct']}/{q_data['total']} ({acc:.2f}%)")

    print(f"\nBy Category:")
    for cat in sorted(by_category.keys()):
        cat_data = by_category[cat]
        acc = (cat_data["correct"] / cat_data["total"] * 100) if cat_data["total"] > 0 else 0
        print(f"  {cat}: {cat_data['correct']}/{cat_data['total']} ({acc:.2f}%)")

    return {
        "overall": {"total": total, "correct": correct, "accuracy": overall_acc},
        "by_question": by_question,
        "by_category": by_category
    }


def main():
    parser = argparse.ArgumentParser(description="Merge MedGemma shard checkpoints")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing checkpoint files")
    parser.add_argument("--output", type=str, default="checkpoint_vllm_google_medgemma-4b-it_merged.json",
                        help="Output filename for merged checkpoint")
    parser.add_argument("--calculate-accuracy", action="store_true",
                        help="Calculate and display accuracy after merging")
    parser.add_argument("--model", type=str, default="google_medgemma-4b-it",
                        help="Model name pattern for finding checkpoints")
    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    output_path = results_dir / args.output

    # Find checkpoints
    main_checkpoint = results_dir / f"checkpoint_vllm_{args.model}.json"
    shard_checkpoints = find_shard_checkpoints(results_dir, args.model)

    print(f"=== MedGemma Shard Merger ===")
    print(f"Results directory: {results_dir}")
    print(f"Main checkpoint: {main_checkpoint}")
    print(f"Found {len(shard_checkpoints)} shard checkpoints")

    if not shard_checkpoints and not main_checkpoint.exists():
        print("No checkpoints found to merge!")
        return

    # Merge
    stats = merge_checkpoints(main_checkpoint, shard_checkpoints, output_path)

    # Calculate accuracy if requested
    if args.calculate_accuracy:
        calculate_accuracy(output_path)


if __name__ == "__main__":
    main()
