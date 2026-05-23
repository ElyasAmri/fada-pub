"""
Data Augmentation Pipeline for Ultrasound Images
Implements heavy augmentation (10-20x) specifically designed for ultrasound
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Optional


def get_training_augmentation(image_size: int = 224) -> A.Compose:
    """
    Heavy augmentation pipeline for training (10-20x amplification)
    Specifically designed for ultrasound images

    Args:
        image_size: Target image size

    Returns:
        Albumentations composition
    """
    return A.Compose([
        # Resize to target size
        A.Resize(image_size, image_size),

        # === Geometric Transformations (safe for ultrasound) ===
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        # Vertical flip avoided - can change anatomical orientation
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),

        # Slight zoom/crop to simulate probe movement
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),  # Scale must be between 0 and 1
            ratio=(0.9, 1.1),
            p=0.5
        ),

        # === Intensity Transformations (ultrasound-specific) ===
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            brightness_by_max=True,
            p=0.5
        ),

        # Simulate ultrasound noise patterns
        A.GaussNoise(p=0.3),

        # Mild blur to simulate probe contact variations
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),

        # Gamma correction to simulate depth attenuation
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # === Ultrasound-Specific Augmentations ===
        # Speckle noise simulation
        A.MultiplicativeNoise(
            multiplier=(0.9, 1.1),
            per_channel=False,
            elementwise=True,
            p=0.3
        ),

        # Acoustic shadow simulation
        A.CoarseDropout(
            max_holes=3,
            max_height=30,
            max_width=30,
            fill_value=0,
            p=0.2
        ),

        # Elastic deformation (tissue movement)
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            p=0.2
        ),

        # === Advanced Augmentations ===
        A.OneOf([
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.05, p=0.5),
        ], p=0.3),

        # CLAHE for contrast enhancement (common in ultrasound processing)
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # Normalize to ImageNet statistics (for pretrained models)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: int = 224) -> A.Compose:
    """
    Minimal augmentation for validation/test (only necessary preprocessing)

    Args:
        image_size: Target image size

    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_heavy_augmentation_pipeline(image_size: int = 224) -> A.Compose:
    """
    Extra heavy augmentation for maximum data amplification (20x)

    Args:
        image_size: Target image size

    Returns:
        Albumentations composition
    """
    return A.Compose([
        # Resize to target size
        A.Resize(image_size, image_size),

        # === Aggressive Geometric Transformations ===
        A.RandomRotate90(p=0.7),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=45, p=0.6, border_mode=cv2.BORDER_CONSTANT),

        # More aggressive crop
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.7, 1.0),
            ratio=(0.85, 1.15),
            p=0.7
        ),

        # Perspective changes
        A.Perspective(scale=(0.05, 0.1), p=0.4),

        # === Aggressive Intensity Transformations ===
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            brightness_by_max=True,
            p=0.7
        ),

        # More noise
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0),
        ], p=0.5),

        # Variable blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.4),

        # More aggressive shadows
        A.CoarseDropout(
            max_holes=5,
            max_height=40,
            max_width=40,
            fill_value=0,
            p=0.4
        ),

        # Strong deformations
        A.OneOf([
            A.ElasticTransform(alpha=2, sigma=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.1, p=1.0),
        ], p=0.5),

        # Color augmentations (even though ultrasound is grayscale)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
            A.ChannelShuffle(p=1.0),
        ], p=0.3),

        # CLAHE variations
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.Equalize(p=1.0),
        ], p=0.4),

        # Random gamma
        A.RandomGamma(gamma_limit=(60, 140), p=0.5),

        # Normalize
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),

        ToTensorV2(),
    ])


def get_test_time_augmentation(image_size: int = 224) -> A.Compose:
    """
    Test-time augmentation for improved predictions
    Apply multiple augmentations and average predictions

    Args:
        image_size: Target image size

    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.RandomRotate90(p=1.0),
            A.Rotate(limit=15, p=1.0),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


class UltrasoundAugmentation:
    """Custom ultrasound-specific augmentations"""

    @staticmethod
    def add_acoustic_shadows(image: np.ndarray, num_shadows: int = 3) -> np.ndarray:
        """
        Add realistic acoustic shadows to ultrasound image

        Args:
            image: Input image
            num_shadows: Number of shadows to add

        Returns:
            Image with shadows
        """
        h, w = image.shape[:2]
        shadowed = image.copy()

        for _ in range(num_shadows):
            # Random shadow parameters
            x = np.random.randint(0, w)
            y = np.random.randint(0, h // 2)  # Shadows start from top half
            width = np.random.randint(10, 50)

            # Create triangular shadow mask
            shadow_mask = np.zeros((h, w), dtype=np.float32)

            # Define shadow polygon (triangular/fan shape)
            pts = np.array([
                [x, y],
                [max(0, x - width), h],
                [min(w, x + width), h]
            ], np.int32)

            cv2.fillPoly(shadow_mask, [pts], 1.0)

            # Apply shadow with gradual intensity
            shadow_intensity = np.random.uniform(0.3, 0.7)
            shadowed = shadowed * (1 - shadow_mask[:, :, None] * shadow_intensity)

        return shadowed.astype(np.uint8)

    @staticmethod
    def add_speckle_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Add speckle noise characteristic of ultrasound imaging

        Args:
            image: Input image
            intensity: Noise intensity

        Returns:
            Noisy image
        """
        h, w = image.shape[:2]

        # Generate multiplicative noise
        gauss = np.random.randn(h, w, 1)
        noisy = image + image * gauss * intensity
        noisy = np.clip(noisy, 0, 255)

        return noisy.astype(np.uint8)