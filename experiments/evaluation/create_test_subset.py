"""
Create a stratified test subset from the full test data.

Usage:
    python experiments/evaluation/create_test_subset.py
    python experiments/evaluation/create_test_subset.py --samples-per-category 100
    python experiments/evaluation/create_test_subset.py --output custom_subset.jsonl
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .config import (
    FULL_TEST_DATA,
    STRATIFIED_TEST_DATA,
    CATEGORIES,
    DEFAULT_SAMPLES_PER_CATEGORY,
    OUTPUTS_DIR,
)


def extract_category_from_path(image_path: str) -> str:
    """
    Extract category name from image path.
    Path format: .../data/Fetal Ultrasound/{Category}/{image}.png
    """
    parts = Path(image_path).parts
    for i, part in enumerate(parts):
        if part == "Fetal Ultrasound":
            return parts[i + 1]
    raise ValueError(f"Could not extract category from {image_path}")


def load_jsonl(filepath: Path) -> list:
    """Load all samples from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def stratified_sample(
    samples: list,
    samples_per_category: int = DEFAULT_SAMPLES_PER_CATEGORY,
    seed: int = 42
) -> list:
    """
    Create stratified sample with equal representation per category.

    Returns subset with samples_per_category from each category.
    """
    random.seed(seed)

    # Group by category
    by_category = defaultdict(list)
    for sample in samples:
        image_path = sample['images'][0]
        category = extract_category_from_path(image_path)
        by_category[category].append(sample)

    print(f"\nCategory distribution in full test set:")
    print("-" * 50)
    for category in sorted(by_category.keys()):
        print(f"  {category:35s}: {len(by_category[category]):5d} samples")

    # Sample equally from each category
    stratified = []
    print(f"\nSampling {samples_per_category} per category:")
    print("-" * 50)

    for category in sorted(by_category.keys()):
        cat_samples = by_category[category]
        if len(cat_samples) >= samples_per_category:
            selected = random.sample(cat_samples, samples_per_category)
            print(f"  {category:35s}: {samples_per_category} selected")
        else:
            selected = cat_samples
            print(f"  {category:35s}: {len(cat_samples)} (all available, less than requested)")
        stratified.extend(selected)

    # Shuffle the final list
    random.shuffle(stratified)

    return stratified


def main():
    parser = argparse.ArgumentParser(description="Create stratified test subset")

    parser.add_argument(
        '--input', type=str, default=str(FULL_TEST_DATA),
        help='Path to full test JSONL file'
    )
    parser.add_argument(
        '--output', type=str, default=str(STRATIFIED_TEST_DATA),
        help='Path to output subset JSONL file'
    )
    parser.add_argument(
        '--samples-per-category', type=int, default=DEFAULT_SAMPLES_PER_CATEGORY,
        help=f'Number of samples per category (default: {DEFAULT_SAMPLES_PER_CATEGORY})'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Creating Stratified Test Subset")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Samples per category: {args.samples_per_category}")
    print(f"Random seed: {args.seed}")

    # Load full test set
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"\nLoading test data from {input_path}...")
    samples = load_jsonl(input_path)
    print(f"Loaded {len(samples)} total samples")

    # Create stratified subset
    subset = stratified_sample(
        samples,
        samples_per_category=args.samples_per_category,
        seed=args.seed
    )

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save subset
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in subset:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n" + "=" * 60)
    print(f"Created stratified test subset:")
    print(f"  Total samples: {len(subset)}")
    print(f"  Saved to: {output_path}")
    print("=" * 60)

    # Verify output
    verify_samples = load_jsonl(output_path)
    verify_categories = defaultdict(int)
    for s in verify_samples:
        cat = extract_category_from_path(s['images'][0])
        verify_categories[cat] += 1

    print("\nVerification - samples per category in output:")
    for cat in sorted(verify_categories.keys()):
        print(f"  {cat:35s}: {verify_categories[cat]}")

    return 0


if __name__ == "__main__":
    exit(main())
