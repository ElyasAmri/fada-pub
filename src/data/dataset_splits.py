"""
Dataset Split Manager - Persistent train/val/test splits for reproducibility

Creates and manages fixed dataset splits stored in JSON format.
Ensures the same images are always in the same split across runs.

Usage:
    # Generate splits (run once)
    python -m src.data.dataset_splits --generate

    # Load splits in code
    from src.data.dataset_splits import load_splits, get_split_images
    splits = load_splits()
    train_images = get_split_images('train')
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "Fetal Ultrasound"
SPLITS_FILE = PROJECT_ROOT / "data" / "dataset_splits.json"

# Split configuration
DEFAULT_CONFIG = {
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "random_seed": 42,
}

# Valid image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


def scan_images(data_root: Path = DATA_ROOT) -> Dict[str, List[str]]:
    """
    Scan all images in the dataset directory.

    Returns:
        Dict mapping category names to lists of image filenames
    """
    images_by_category = defaultdict(list)

    for category_dir in sorted(data_root.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        for img_path in sorted(category_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                # Store relative path from data root
                rel_path = f"{category_name}/{img_path.name}"
                images_by_category[category_name].append(rel_path)

    return dict(images_by_category)


def compute_dataset_hash(images_by_category: Dict[str, List[str]]) -> str:
    """
    Compute a hash of the dataset for integrity verification.

    Returns:
        SHA256 hash of all image paths
    """
    all_paths = []
    for category in sorted(images_by_category.keys()):
        all_paths.extend(sorted(images_by_category[category]))

    content = "\n".join(all_paths)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_stratified_splits(
    images_by_category: Dict[str, List[str]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42
) -> Dict[str, Dict[str, List[str]]]:
    """
    Create stratified train/val/test splits maintaining category proportions.

    Args:
        images_by_category: Dict mapping category to image paths
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Dict with 'train', 'val', 'test' keys, each containing category->images mapping
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    rng = np.random.default_rng(random_seed)

    splits = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }

    for category, images in images_by_category.items():
        n_images = len(images)

        if n_images < 10:
            # For very small categories, put all in train
            splits['train'][category] = images
            continue

        # Shuffle with fixed seed
        indices = rng.permutation(n_images)

        # Calculate split points
        n_test = max(1, int(n_images * test_ratio))
        n_val = max(1, int(n_images * val_ratio))
        n_train = n_images - n_test - n_val

        # Assign to splits
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        splits['train'][category] = [images[i] for i in sorted(train_idx)]
        splits['val'][category] = [images[i] for i in sorted(val_idx)]
        splits['test'][category] = [images[i] for i in sorted(test_idx)]

    # Convert defaultdicts to regular dicts
    return {k: dict(v) for k, v in splits.items()}


def generate_splits(
    config: Optional[Dict] = None,
    output_path: Path = SPLITS_FILE,
    force: bool = False
) -> Dict:
    """
    Generate and save dataset splits to JSON file.

    Args:
        config: Split configuration (uses DEFAULT_CONFIG if None)
        output_path: Path to save splits JSON
        force: Overwrite existing splits file

    Returns:
        The generated splits data
    """
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Splits file already exists: {output_path}\n"
            "Use --force to overwrite or load existing splits with load_splits()"
        )

    config = config or DEFAULT_CONFIG

    # Scan dataset
    print("Scanning dataset...")
    images_by_category = scan_images()

    total_images = sum(len(imgs) for imgs in images_by_category.values())
    print(f"Found {total_images} images across {len(images_by_category)} categories")

    # Create splits
    print("Creating stratified splits...")
    splits = create_stratified_splits(
        images_by_category,
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        random_seed=config['random_seed']
    )

    # Compute statistics
    stats = {}
    for split_name, split_data in splits.items():
        split_total = sum(len(imgs) for imgs in split_data.values())
        stats[split_name] = {
            'total': split_total,
            'by_category': {cat: len(imgs) for cat, imgs in split_data.items()}
        }

    # Build output structure
    output = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'dataset_hash': compute_dataset_hash(images_by_category),
            'total_images': total_images,
            'config': config
        },
        'statistics': stats,
        'splits': splits
    }

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved splits to: {output_path}")
    print_split_summary(output)

    return output


def load_splits(splits_path: Path = SPLITS_FILE) -> Dict:
    """
    Load dataset splits from JSON file.

    Args:
        splits_path: Path to splits JSON file

    Returns:
        Splits data dictionary

    Raises:
        FileNotFoundError: If splits file doesn't exist
    """
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}\n"
            "Generate splits first with: python -m src.data.dataset_splits --generate"
        )

    with open(splits_path, 'r') as f:
        return json.load(f)


def get_split_images(
    split: str,
    categories: Optional[List[str]] = None,
    splits_path: Path = SPLITS_FILE,
    absolute_paths: bool = False
) -> List[str]:
    """
    Get list of image paths for a specific split.

    Args:
        split: One of 'train', 'val', 'test'
        categories: Optional list of categories to filter (None = all)
        splits_path: Path to splits JSON file
        absolute_paths: Return absolute paths instead of relative

    Returns:
        List of image paths
    """
    data = load_splits(splits_path)

    if split not in data['splits']:
        raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test")

    split_data = data['splits'][split]
    images = []

    for category, category_images in split_data.items():
        if categories is None or category in categories:
            images.extend(category_images)

    if absolute_paths:
        images = [str(DATA_ROOT / img) for img in images]

    return images


def get_split_with_labels(
    split: str,
    splits_path: Path = SPLITS_FILE,
    absolute_paths: bool = False
) -> List[Tuple[str, str]]:
    """
    Get list of (image_path, category) tuples for a specific split.

    Args:
        split: One of 'train', 'val', 'test'
        splits_path: Path to splits JSON file
        absolute_paths: Return absolute paths instead of relative

    Returns:
        List of (image_path, category) tuples
    """
    data = load_splits(splits_path)

    if split not in data['splits']:
        raise ValueError(f"Invalid split '{split}'. Must be one of: train, val, test")

    split_data = data['splits'][split]
    results = []

    for category, category_images in split_data.items():
        for img in category_images:
            path = str(DATA_ROOT / img) if absolute_paths else img
            results.append((path, category))

    return results


def verify_splits(splits_path: Path = SPLITS_FILE) -> bool:
    """
    Verify that splits file matches current dataset.

    Returns:
        True if splits are valid, False otherwise
    """
    try:
        data = load_splits(splits_path)
        current_images = scan_images()
        current_hash = compute_dataset_hash(current_images)

        stored_hash = data['metadata']['dataset_hash']

        if current_hash != stored_hash:
            print(f"WARNING: Dataset has changed since splits were created")
            print(f"  Stored hash: {stored_hash}")
            print(f"  Current hash: {current_hash}")
            return False

        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def print_split_summary(data: Dict) -> None:
    """Print a summary of the splits."""
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    stats = data['statistics']
    total = data['metadata']['total_images']

    print(f"\nTotal images: {total}")
    print(f"Random seed: {data['metadata']['config']['random_seed']}")
    print(f"Dataset hash: {data['metadata']['dataset_hash']}")

    print("\nSplit sizes:")
    for split_name in ['train', 'val', 'test']:
        split_stats = stats[split_name]
        pct = (split_stats['total'] / total) * 100
        print(f"  {split_name:6s}: {split_stats['total']:5d} ({pct:5.1f}%)")

    print("\nPer-category breakdown:")
    print(f"  {'Category':<35s} {'Train':>6s} {'Val':>6s} {'Test':>6s} {'Total':>6s}")
    print("  " + "-" * 60)

    categories = sorted(stats['train']['by_category'].keys())
    for cat in categories:
        train_n = stats['train']['by_category'].get(cat, 0)
        val_n = stats['val']['by_category'].get(cat, 0)
        test_n = stats['test']['by_category'].get(cat, 0)
        total_n = train_n + val_n + test_n
        print(f"  {cat:<35s} {train_n:>6d} {val_n:>6d} {test_n:>6d} {total_n:>6d}")


def main():
    """Command-line interface for dataset splits."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage dataset train/val/test splits"
    )
    parser.add_argument(
        '--generate', action='store_true',
        help='Generate new splits file'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite existing splits file'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify splits match current dataset'
    )
    parser.add_argument(
        '--summary', action='store_true',
        help='Print summary of existing splits'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.80,
        help='Training set ratio (default: 0.80)'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.10,
        help='Validation set ratio (default: 0.10)'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.10,
        help='Test set ratio (default: 0.10)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    if args.generate:
        config = {
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'random_seed': args.seed
        }
        generate_splits(config=config, force=args.force)

    elif args.verify:
        if verify_splits():
            print("Splits are valid and match current dataset")
        else:
            print("Splits verification FAILED")
            exit(1)

    elif args.summary:
        try:
            data = load_splits()
            print_split_summary(data)
        except FileNotFoundError as e:
            print(e)
            exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
