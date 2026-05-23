"""
Create Test Data for Mobile VLM Benchmarking

Generates sample images or copies from existing dataset for testing.
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


def create_dummy_images(output_dir: Path, count: int = 10):
    """
    Create dummy ultrasound-like images for testing.

    Args:
        output_dir: Output directory
        count: Number of images to create
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {count} dummy images in {output_dir}...")

    for i in range(count):
        # Create grayscale image (ultrasound-like)
        img = Image.new('L', (640, 480), color=0)
        draw = ImageDraw.Draw(img)

        # Add some random shapes to simulate ultrasound
        for _ in range(20):
            x1 = random.randint(0, 600)
            y1 = random.randint(0, 440)
            x2 = x1 + random.randint(20, 100)
            y2 = y1 + random.randint(20, 100)
            gray = random.randint(50, 200)
            draw.ellipse([x1, y1, x2, y2], fill=gray)

        # Add label
        try:
            draw.text((10, 10), f"Test Image {i+1}", fill=255)
        except Exception:
            pass  # Font not available, skip text

        # Save
        output_path = output_dir / f"test_image_{i+1:03d}.jpg"
        img.save(output_path, "JPEG")
        print(f"  Created: {output_path.name}")

    print(f"\nDone! Created {count} images in {output_dir}")


def copy_from_dataset(source_dir: Path, output_dir: Path, count: int = 10):
    """
    Copy images from existing dataset.

    Args:
        source_dir: Source directory with images
        output_dir: Output directory
        count: Number of images to copy
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(source_dir.rglob(ext))

    if not images:
        raise ValueError(f"No images found in {source_dir}")

    # Copy random selection
    selected = random.sample(images, min(count, len(images)))

    print(f"Copying {len(selected)} images from {source_dir} to {output_dir}...")

    for i, img_path in enumerate(selected, 1):
        output_path = output_dir / f"test_{i:03d}_{img_path.name}"
        shutil.copy2(img_path, output_path)
        print(f"  Copied: {img_path.name} -> {output_path.name}")

    print(f"\nDone! Copied {len(selected)} images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Create test data for mobile VLM benchmarking"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/test_images"),
        help="Output directory for test images",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of test images",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Copy from existing dataset instead of creating dummy images",
    )

    args = parser.parse_args()

    if args.source_dir:
        copy_from_dataset(args.source_dir, args.output_dir, args.count)
    else:
        create_dummy_images(args.output_dir, args.count)

    print(f"\nTest images ready! Use with:")
    print(f"  --image-dir {args.output_dir}")


if __name__ == "__main__":
    main()
