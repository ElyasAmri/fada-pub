"""
Fetal Ultrasound Dataset V2 - 12-class classification with stratified splits
No patient-aware splitting since no patient IDs exist in the data
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import logging
from collections import Counter

from src.config.constants import CLASSES, DISPLAY_NAMES, CLASS_DESCRIPTIONS, MIN_IMAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FetalUltrasoundDataset12Class(Dataset):
    """Dataset for 12-class fetal ultrasound classification"""

    # Import from centralized constants
    CLASSES = CLASSES
    DISPLAY_NAMES = DISPLAY_NAMES
    CLASS_DESCRIPTIONS = CLASS_DESCRIPTIONS

    def __init__(
        self,
        data_root: str,
        indices: Optional[List[int]] = None,
        transform=None
    ):
        """
        Initialize dataset

        Args:
            data_root: Path to 'Fetal Ultrasound' directory
            indices: Subset indices for train/val/test split
            transform: Image transformations
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self._fallback_count = 0

        # Verify data root exists
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {data_root}")

        # Load all image paths and labels
        self.samples = []
        self.labels = []
        self.load_dataset()

        # Apply subset indices if provided
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        logger.info(f"Loaded {len(self.samples)} images")
        logger.info(f"Class distribution: {self.get_class_distribution()}")

    def load_dataset(self) -> None:
        """Load all image paths and labels from directory structure"""
        for folder in self.data_root.iterdir():
            if not folder.is_dir():
                continue

            folder_name = folder.name

            # Use folder name directly as class (12-class approach)
            if folder_name not in self.CLASSES:
                logger.warning(f"Unknown folder: {folder_name}")
                continue

            class_idx = self.CLASSES.index(folder_name)

            # Load images from folder
            image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
            for img_path in folder.iterdir():
                if img_path.suffix in image_extensions:
                    try:
                        # Verify image can be opened
                        with Image.open(img_path) as img:
                            if img.size[0] < MIN_IMAGE_SIZE or img.size[1] < MIN_IMAGE_SIZE:
                                logger.warning(f"Skipping tiny image: {img_path}")
                                continue

                        self.samples.append(str(img_path))
                        self.labels.append(class_idx)

                    except Exception as e:
                        logger.error(f"Failed to verify image {img_path}: {e}")
                        continue

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across classes"""
        distribution = {}
        for label in self.labels:
            class_name = self.CLASSES[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance"""
        class_counts = Counter(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.CLASSES)

        # Calculate weight for each class: total_samples / (num_classes * class_count)
        weights = torch.zeros(num_classes)
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)

        return weights

    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Convert to numpy for albumentations
            image = np.array(image)

            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Default: convert to tensor
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            return image, label

        except FileNotFoundError:
            logger.error(f"Image file not found: {img_path}")
            # Return black image as fallback for missing files
            self._fallback_count += 1
            return self._get_fallback_image(), label

        except (IOError, OSError) as e:
            # Handles PIL.UnidentifiedImageError and other IO issues
            logger.error(f"Error reading image {img_path}: {e}")
            self._fallback_count += 1
            return self._get_fallback_image(), label

        except Exception as e:
            # Log unexpected errors but don't crash training
            logger.error(f"Unexpected error loading {img_path}: {type(e).__name__}: {e}")
            self._fallback_count += 1
            return self._get_fallback_image(), label

    @property
    def fallback_count(self):
        return self._fallback_count

    def _get_fallback_image(self) -> torch.Tensor:
        """Return a black fallback image for error cases."""
        if self.transform:
            black_image = np.zeros((224, 224, 3), dtype=np.uint8)
            transformed = self.transform(image=black_image)
            return transformed['image']
        else:
            return torch.zeros((3, 224, 224), dtype=torch.float32)


class FetalDataModule12Class:
    """Data module for 12-class classification with stratified splits"""

    def __init__(
        self,
        data_root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42,
        subset_fraction: float = 1.0
    ):
        """
        Initialize data module with stratified splitting

        Args:
            data_root: Path to data directory
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            val_split: Validation split ratio
            test_split: Test split ratio
            random_state: Random seed for reproducibility
            subset_fraction: Fraction of dataset to use (e.g., 0.1 for 10%)
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        self.subset_fraction = subset_fraction

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_weights = None

    def setup(self, train_transform: Optional[object] = None, val_transform: Optional[object] = None) -> None:
        """
        Setup datasets with stratified splits

        Args:
            train_transform: Transformations for training data
            val_transform: Transformations for validation/test data
        """
        # Load full dataset to get all samples
        full_dataset = FetalUltrasoundDataset12Class(self.data_root)

        # Get all labels for stratification
        all_labels = full_dataset.labels
        all_indices = list(range(len(all_labels)))

        # Apply subset sampling if requested
        if self.subset_fraction < 1.0:
            from sklearn.model_selection import train_test_split
            all_indices, _, all_labels, _ = train_test_split(
                all_indices, all_labels,
                train_size=self.subset_fraction,
                stratify=all_labels,
                random_state=self.random_state
            )
            logger.info(f"Using {self.subset_fraction*100:.0f}% of dataset: {len(all_indices)} samples")

        # First split: separate test set
        train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
            all_indices, all_labels,
            test_size=self.test_split,
            stratify=all_labels,
            random_state=self.random_state
        )

        # Second split: separate train and validation
        val_size_adjusted = self.val_split / (1 - self.test_split)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size_adjusted,
            stratify=train_val_labels,
            random_state=self.random_state
        )

        # Create datasets with appropriate indices
        self.train_dataset = FetalUltrasoundDataset12Class(
            self.data_root,
            indices=train_indices,
            transform=train_transform
        )

        self.val_dataset = FetalUltrasoundDataset12Class(
            self.data_root,
            indices=val_indices,
            transform=val_transform
        )

        self.test_dataset = FetalUltrasoundDataset12Class(
            self.data_root,
            indices=test_indices,
            transform=val_transform
        )

        # Calculate class weights from training data
        self.class_weights = self.train_dataset.get_class_weights()

        # Log split statistics
        self._log_split_statistics()

    def _log_split_statistics(self) -> None:
        """Log statistics about the data splits"""
        logger.info("=" * 60)
        logger.info("DATA SPLIT STATISTICS")
        logger.info("=" * 60)

        for split_name, dataset in [
            ('Train', self.train_dataset),
            ('Val', self.val_dataset),
            ('Test', self.test_dataset)
        ]:
            distribution = dataset.get_class_distribution()
            logger.info(f"\n{split_name} set: {len(dataset)} samples")

            # Check for missing classes
            missing_classes = []
            for class_name in FetalUltrasoundDataset12Class.CLASSES:
                if class_name not in distribution:
                    missing_classes.append(class_name)
                else:
                    count = distribution[class_name]
                    percentage = (count / len(dataset)) * 100
                    logger.info(f"  {class_name:30s}: {count:4d} ({percentage:5.1f}%)")

            if missing_classes:
                logger.warning(f"  Missing classes: {missing_classes}")

        logger.info("\nClass weights for balanced training:")
        for idx, class_name in enumerate(FetalUltrasoundDataset12Class.CLASSES):
            if self.class_weights is not None:
                logger.info(f"  {class_name:30s}: {self.class_weights[idx]:.3f}")

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )


if __name__ == "__main__":
    # Test the new dataset
    print("Testing 12-class dataset implementation...")

    data_module = FetalDataModule12Class(
        data_root='data/Fetal Ultrasound',
        batch_size=32,
        num_workers=0
    )

    from src.data.augmentation import get_training_augmentation, get_validation_augmentation

    data_module.setup(
        train_transform=get_training_augmentation(224),
        val_transform=get_validation_augmentation(224)
    )

    print(f"\nDataloader test:")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label values: {labels[:10].tolist()}")