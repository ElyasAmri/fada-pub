"""
Prepare Fetal Ultrasound Dataset for Qwen3-VL Fine-tuning

This script loads annotations from Excel, matches images, and converts to
the training format required by Unsloth's FastVisionModel.

Training focuses on Q7: Normality Assessment.
"""

import gc
import os
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from datasets import Dataset
from tqdm import tqdm


# Default paths
DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "Fetal Ultrasound"
ANNOTATIONS_FILE = DATA_ROOT.parent / "Fetal Ultrasound Annotations Normalized.xlsx"

# Q7 prompt for normality assessment
Q7_PROMPT = "Assess the normality of this fetal ultrasound image. Describe whether the visible structures appear normal or if there are any abnormalities."


def load_annotations(annotations_path: Path = ANNOTATIONS_FILE) -> pd.DataFrame:
    """Load and filter annotations from Excel file."""
    df = pd.read_excel(annotations_path)

    # Filter to only complete annotations with Q7 values
    df = df[df["Status"] == "complete"].copy()
    df = df.dropna(subset=["Q7: Normality Assessment"])

    print(f"Loaded {len(df)} complete annotations with Q7 values")
    return df


def get_image_path(folder_name: str, image_name: str, data_root: Path = DATA_ROOT) -> Optional[Path]:
    """Construct and validate image path."""
    # Try direct path
    image_path = data_root / folder_name / image_name
    if image_path.exists():
        return image_path

    # Try common variations
    variations = [
        data_root / folder_name / image_name.replace("_", " "),
        data_root / folder_name.replace("_", " ") / image_name,
    ]

    for path in variations:
        if path.exists():
            return path

    return None


def load_image(image_path: Path, max_size: int = 512) -> Optional[Image.Image]:
    """Load, resize, and preprocess image to save memory."""
    try:
        image = Image.open(image_path).convert("RGB")

        # Resize to max_size while maintaining aspect ratio
        # This significantly reduces memory usage (from ~2MB to ~100KB per image)
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return image
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def create_conversation(image: Image.Image, q7_answer: str) -> dict:
    """
    Create a conversation in the format expected by Unsloth's vision models.

    Format:
    {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": <PIL Image>},
                {"type": "text", "text": "<prompt>"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "<answer>"}
            ]}
        ]
    }
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": Q7_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": q7_answer.strip()}
                ]
            }
        ]
    }


def _deadline_reached(deadline_ts: Optional[float]) -> bool:
    return deadline_ts is not None and time.time() >= deadline_ts


def prepare_dataset(
    annotations_path: Path = ANNOTATIONS_FILE,
    data_root: Path = DATA_ROOT,
    train_ratio: float = 0.9,
    seed: int = 42,
    max_samples: Optional[int] = None,
    lazy_loading: bool = False,
    deadline_ts: Optional[float] = None,
) -> tuple[Dataset, Dataset]:
    """
    Prepare train and validation datasets.

    Args:
        annotations_path: Path to annotations Excel file
        data_root: Root directory containing image folders
        train_ratio: Ratio of data to use for training (rest for validation)
        seed: Random seed for reproducibility
        max_samples: Maximum samples to use (None for all)
        lazy_loading: If True, store paths and load images on-demand (memory efficient)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    random.seed(seed)

    # Load annotations
    df = load_annotations(annotations_path)

    # Limit samples if specified
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=seed)

    if lazy_loading:
        return _prepare_dataset_lazy(df, data_root, train_ratio, seed, deadline_ts=deadline_ts)
    else:
        return _prepare_dataset_eager(df, data_root, train_ratio, deadline_ts=deadline_ts)


def _prepare_dataset_eager(
    df: pd.DataFrame,
    data_root: Path,
    train_ratio: float,
    deadline_ts: Optional[float] = None,
) -> tuple[Dataset, Dataset]:
    """Load all images into memory (original behavior)."""
    conversations = []
    skipped = 0

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Loading images"), start=1):
        if _deadline_reached(deadline_ts):
            print(f"Dataset deadline reached after loading {len(conversations)} samples; continuing with partial dataset.")
            break
        folder_name = row["Folder Name"]
        image_name = row["Image Name"]
        q7_answer = row["Q7: Normality Assessment"]

        # Get image path
        image_path = get_image_path(folder_name, image_name, data_root)
        if image_path is None:
            skipped += 1
            continue

        # Load image
        image = load_image(image_path)
        if image is None:
            skipped += 1
            continue

        # Create conversation
        conv = create_conversation(image, q7_answer)
        conversations.append(conv)

        if idx % 50 == 0 and _deadline_reached(deadline_ts):
            print(f"Dataset deadline reached after loading {len(conversations)} samples; continuing with partial dataset.")
            break

    print(f"Loaded {len(conversations)} samples, skipped {skipped}")

    if not conversations:
        raise RuntimeError("Dataset preparation produced no samples before the runtime deadline.")

    # Shuffle and split
    random.shuffle(conversations)
    split_idx = int(len(conversations) * train_ratio)

    train_data = conversations[:split_idx]
    val_data = conversations[split_idx:]

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def _load_sample_with_image(sample: dict) -> dict:
    """Load image and create conversation format for a single sample."""
    image = load_image(Path(sample["image_path"]))
    if image is None:
        return None

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": Q7_PROMPT}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["q7_answer"]}
                ]
            }
        ]
    }


def _prepare_dataset_lazy(
    df: pd.DataFrame,
    data_root: Path,
    train_ratio: float,
    seed: int,
    deadline_ts: Optional[float] = None,
) -> tuple[Dataset, Dataset]:
    """
    Load images in batches to balance memory usage and speed.

    Uses chunked processing to avoid loading all 15K images at once while
    still being faster than pure on-demand loading.
    """
    samples = []
    skipped = 0

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Validating paths"), start=1):
        if _deadline_reached(deadline_ts):
            print(f"Dataset deadline reached after validating {len(samples)} samples; continuing with partial dataset.")
            break
        folder_name = row["Folder Name"]
        image_name = row["Image Name"]
        q7_answer = row["Q7: Normality Assessment"]

        # Get image path (validate it exists)
        image_path = get_image_path(folder_name, image_name, data_root)
        if image_path is None:
            skipped += 1
            continue

        samples.append({
            "image_path": str(image_path),
            "q7_answer": q7_answer.strip(),
        })

        if idx % 100 == 0 and _deadline_reached(deadline_ts):
            print(f"Dataset deadline reached after validating {len(samples)} samples; continuing with partial dataset.")
            break

    print(f"Validated {len(samples)} image paths, skipped {skipped}")

    if not samples:
        raise RuntimeError("Dataset preparation produced no valid samples before the runtime deadline.")

    # Shuffle and split
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)

    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train: {len(train_samples)}, Validation: {len(val_samples)}")

    # Load images in chunks to avoid OOM
    # Process in chunks of 2000 images at a time
    CHUNK_SIZE = 2000

    def load_in_chunks(sample_list):
        """Load images in chunks to avoid memory issues."""
        all_conversations = []
        total = len(sample_list)

        for start_idx in range(0, total, CHUNK_SIZE):
            if _deadline_reached(deadline_ts):
                print(f"Dataset deadline reached after loading {len(all_conversations)} conversations; continuing with partial dataset.")
                break
            end_idx = min(start_idx + CHUNK_SIZE, total)
            chunk = sample_list[start_idx:end_idx]
            print(f"  Loading images {start_idx+1}-{end_idx} of {total}...")

            for item_idx, sample in enumerate(tqdm(chunk, desc=f"  Chunk {start_idx//CHUNK_SIZE + 1}", leave=False), start=1):
                if _deadline_reached(deadline_ts):
                    print(f"Dataset deadline reached after loading {len(all_conversations)} conversations; continuing with partial dataset.")
                    return all_conversations
                conv = _load_sample_with_image(sample)
                if conv is not None:
                    all_conversations.append(conv)
                if item_idx % 50 == 0 and _deadline_reached(deadline_ts):
                    print(f"Dataset deadline reached after loading {len(all_conversations)} conversations; continuing with partial dataset.")
                    return all_conversations

        return all_conversations

    print("Loading training images in chunks...")
    train_conversations = load_in_chunks(train_samples)

    print("Loading validation images in chunks...")
    val_conversations = load_in_chunks(val_samples)

    print(f"Loaded {len(train_conversations)} train, {len(val_conversations)} val conversations")

    if not train_conversations:
        raise RuntimeError("Dataset preparation produced no training samples before the runtime deadline.")

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_conversations)
    val_dataset = Dataset.from_list(val_conversations)

    return train_dataset, val_dataset


def create_conversation_from_path(image_path: str, q7_answer: str) -> Optional[dict]:
    """Create conversation by loading image from path (for lazy loading)."""
    image = load_image(Path(image_path))
    if image is None:
        return None
    return create_conversation(image, q7_answer)


def prepare_dataset_streaming(
    annotations_path: Path = ANNOTATIONS_FILE,
    data_root: Path = DATA_ROOT,
    train_ratio: float = 0.9,
    seed: int = 42
):
    """
    Generator version for memory-efficient loading.
    Yields (split, conversation) tuples.

    Use this for very large datasets that don't fit in memory.
    """
    random.seed(seed)

    # Load annotations
    df = load_annotations(annotations_path)

    # Shuffle indices
    indices = list(range(len(df)))
    random.shuffle(indices)
    split_idx = int(len(indices) * train_ratio)

    for i, idx in enumerate(tqdm(indices, desc="Processing")):
        row = df.iloc[idx]
        folder_name = row["Folder Name"]
        image_name = row["Image Name"]
        q7_answer = row["Q7: Normality Assessment"]

        # Get and load image
        image_path = get_image_path(folder_name, image_name, data_root)
        if image_path is None:
            continue

        image = load_image(image_path)
        if image is None:
            continue

        # Determine split
        split = "train" if i < split_idx else "val"

        yield split, create_conversation(image, q7_answer)


def get_q7_distribution(annotations_path: Path = ANNOTATIONS_FILE) -> pd.Series:
    """Get distribution of Q7 answers for analysis."""
    df = load_annotations(annotations_path)
    return df["Q7: Normality Assessment"].value_counts()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare dataset for Qwen3-VL fine-tuning")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to load (for testing)")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Ratio of data for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--show-distribution", action="store_true",
                        help="Show Q7 value distribution")

    args = parser.parse_args()

    if args.show_distribution:
        print("\nQ7 Value Distribution:")
        print(get_q7_distribution().head(20))
    else:
        print(f"\nPreparing dataset (max_samples={args.max_samples})...")
        train_ds, val_ds = prepare_dataset(
            train_ratio=args.train_ratio,
            seed=args.seed,
            max_samples=args.max_samples
        )

        print(f"\nDataset prepared successfully!")
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")

        # Show sample
        if len(train_ds) > 0:
            print("\nSample conversation:")
            sample = train_ds[0]
            print(f"User prompt: {sample['messages'][0]['content'][1]['text'][:100]}...")
            print(f"Assistant: {sample['messages'][1]['content'][0]['text'][:100]}...")
