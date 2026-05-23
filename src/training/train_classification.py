"""
FADA Classification Model Training Script

Comprehensive training pipeline with MLflow experiment tracking for
fetal ultrasound organ classification using EfficientNet-B0 and other backbones.

Features:
- Multiple backbone architectures (EfficientNet-B0, ResNet18, DenseNet121, MobileNetV2)
- Heavy data augmentation for small dataset scenarios
- Cross-validation support
- MLflow experiment tracking
- Focal loss for class imbalance
- Model checkpointing
- Comprehensive evaluation metrics
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mlflow_utils import (
    setup_mlflow_experiment,
    log_training_config,
    log_model_architecture,
    log_evaluation_results,
    log_confusion_matrix,
    log_training_curves,
    MLflowModelCheckpoint,
    log_gpu_metrics,
)
from src.models.classifier import create_model

import mlflow

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FetalUltrasoundDataset(Dataset):
    """
    Dataset for fetal ultrasound images organized in class folders.

    Expected structure:
        data/Fetal Ultrasound/
            Abdomen/
                image1.jpg
                image2.jpg
            Aorta/
                image1.jpg
            ...
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
        max_samples_per_class: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            root_dir: Root directory containing class folders
            transform: Optional transform to apply to images
            max_samples_per_class: Optional limit on samples per class (for testing)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class

        # Find all class directories (ignore xlsx files)
        self.class_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        if len(self.class_dirs) == 0:
            raise ValueError(f"No class directories found in {root_dir}")

        # Create class to index mapping
        self.class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(self.class_dirs)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # Load all image paths and labels
        self.samples = []
        self.labels = []

        for class_dir in self.class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            # Find all images in this class
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_paths.extend(list(class_dir.glob(ext)))

            # Limit samples if requested
            if max_samples_per_class and len(image_paths) > max_samples_per_class:
                image_paths = image_paths[:max_samples_per_class]

            for img_path in image_paths:
                self.samples.append(img_path)
                self.labels.append(class_idx)

        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_dirs)} classes")

        # Print class distribution
        class_counts = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        logger.info("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            logger.info(f"  {class_name}: {count} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get image and label at index."""
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)

        # Inverse frequency weighting
        weights = total_samples / (len(class_counts) * class_counts)

        return torch.FloatTensor(weights)


def get_transforms(augmentation_level: str = 'heavy') -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get train and validation transforms.

    Args:
        augmentation_level: 'light', 'medium', or 'heavy'

    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Standard normalization for ImageNet pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Training transform with augmentation
    if augmentation_level == 'light':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif augmentation_level == 'medium':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # heavy
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ])

    return train_transform, val_transform


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number

    Returns:
        Dictionary of metrics
    """
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': accuracy_score(all_labels, all_preds)
        })

    # Calculate epoch metrics
    metrics = {
        'train_loss': running_loss / len(dataloader),
        'train_accuracy': accuracy_score(all_labels, all_preds),
        'train_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
    }

    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number

    Returns:
        Dictionary of metrics
    """
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': accuracy_score(all_labels, all_preds)
        })

    # Calculate epoch metrics
    metrics = {
        'val_loss': running_loss / len(dataloader),
        'val_accuracy': accuracy_score(all_labels, all_preds),
        'val_f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'val_precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'val_recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'predictions': all_preds,
        'labels': all_labels,
    }

    return metrics


def train_fold(
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_classes: int,
    class_names: List[str],
    fold: int,
) -> Dict[str, float]:
    """
    Train a single fold.

    Args:
        args: Training arguments
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_classes: Number of classes
        class_names: List of class names
        fold: Fold number

    Returns:
        Dictionary of final metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Calculate class weights for imbalanced datasets
    class_weights = None
    if args.use_class_weights:
        class_weights = train_dataset.dataset.get_class_weights().to(device)
        logger.info(f"Using class weights: {class_weights}")

    # Create model and criterion
    model, criterion = create_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout_rate=args.dropout,
        class_weights=class_weights,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
    )
    model = model.to(device)

    # Log model architecture
    log_model_architecture(model, model_name=args.backbone)

    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )

    # Create learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

    # Create checkpoint handler
    checkpoint_handler = MLflowModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_dir=Path('checkpoints') / f'fold_{fold}',
        log_to_mlflow=True,
    )

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'learning_rate': [],
    }

    # Training loop
    best_val_accuracy = 0.0
    best_epoch = 0

    logger.info(f"Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)

        # Update learning rate
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['val_loss'])
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Update history
        history['train_loss'].append(train_metrics['train_loss'])
        history['train_accuracy'].append(train_metrics['train_accuracy'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['val_f1'].append(val_metrics['val_f1'])
        history['learning_rate'].append(current_lr)

        # Log to MLflow
        mlflow.log_metrics({
            'train_loss': train_metrics['train_loss'],
            'train_accuracy': train_metrics['train_accuracy'],
            'train_f1': train_metrics['train_f1'],
            'val_loss': val_metrics['val_loss'],
            'val_accuracy': val_metrics['val_accuracy'],
            'val_f1': val_metrics['val_f1'],
            'val_precision': val_metrics['val_precision'],
            'val_recall': val_metrics['val_recall'],
            'learning_rate': current_lr,
        }, step=epoch)

        # Log GPU metrics
        if epoch % 5 == 0:
            log_gpu_metrics()

        # Save checkpoint if best
        if checkpoint_handler.update(epoch, val_metrics['val_accuracy'], model):
            best_val_accuracy = val_metrics['val_accuracy']
            best_epoch = epoch
            logger.info(f"New best model at epoch {epoch} with val_accuracy={best_val_accuracy:.4f}")

        # Log epoch results
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Train Acc: {train_metrics['train_accuracy']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
            f"Val F1: {val_metrics['val_f1']:.4f}, "
            f"LR: {current_lr:.6f}"
        )

    # Log training curves
    log_training_curves(history)

    # Calculate final metrics on best model
    logger.info(f"Loading best model from epoch {best_epoch}...")
    checkpoint_path = checkpoint_handler.save_dir / 'best_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    final_metrics = validate_epoch(model, val_loader, criterion, device, args.epochs)

    # Calculate per-class metrics
    cm = confusion_matrix(final_metrics['labels'], final_metrics['predictions'])
    log_confusion_matrix(cm, class_names)

    # Classification report
    report = classification_report(
        final_metrics['labels'],
        final_metrics['predictions'],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Log per-class metrics
    per_class_metrics = {}
    for class_name in class_names:
        if class_name in report:
            per_class_metrics[f'class_{class_name}_precision'] = report[class_name]['precision']
            per_class_metrics[f'class_{class_name}_recall'] = report[class_name]['recall']
            per_class_metrics[f'class_{class_name}_f1'] = report[class_name]['f1-score']

    mlflow.log_metrics(per_class_metrics)

    # Log final metrics
    mlflow.log_metrics({
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val_accuracy,
        'final_val_accuracy': final_metrics['val_accuracy'],
        'final_val_f1': final_metrics['val_f1'],
        'final_val_precision': final_metrics['val_precision'],
        'final_val_recall': final_metrics['val_recall'],
    })

    logger.info(f"Training complete. Best val accuracy: {best_val_accuracy:.4f} at epoch {best_epoch}")

    return {
        'best_val_accuracy': best_val_accuracy,
        'best_epoch': best_epoch,
        'final_val_accuracy': final_metrics['val_accuracy'],
        'final_val_f1': final_metrics['val_f1'],
    }


def main():
    parser = argparse.ArgumentParser(description='Train FADA classification model')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/Fetal Ultrasound',
                        help='Path to data directory')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples per class (for testing)')

    # Model arguments
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet18', 'densenet121', 'mobilenet_v2'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Use pretrained weights (use --no-pretrained to train from scratch)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=20,
                        help='Step size for StepLR scheduler')

    # Augmentation arguments
    parser.add_argument('--augmentation', type=str, default='heavy',
                        choices=['light', 'medium', 'heavy'],
                        help='Augmentation level')

    # Loss arguments
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                        help='Use focal loss instead of cross entropy')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use class weights for imbalanced datasets')

    # Cross-validation arguments
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (None = all folds)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds for cross-validation')

    # System arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # MLflow arguments
    parser.add_argument('--experiment-name', type=str, default='fada_classification',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='MLflow run name')

    args = parser.parse_args()

    # Set random seed
    from src.utils.seed import set_all_seeds
    set_all_seeds(args.seed)

    # Setup MLflow experiment
    logger.info(f"Setting up MLflow experiment: {args.experiment_name}")
    setup_mlflow_experiment(args.experiment_name)

    # Prepare data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Get transforms
    train_transform, val_transform = get_transforms(args.augmentation)

    # Load full dataset
    full_dataset = FetalUltrasoundDataset(
        root_dir=data_dir,
        transform=train_transform,
        max_samples_per_class=args.max_samples,
    )

    num_classes = len(full_dataset.class_to_idx)
    class_names = [full_dataset.idx_to_class[i] for i in range(num_classes)]

    logger.info(f"Dataset loaded: {len(full_dataset)} samples, {num_classes} classes")
    logger.info(f"Classes: {class_names}")

    # Prepare cross-validation splits
    if args.fold is not None:
        # Train single fold
        folds_to_train = [args.fold]
    else:
        # Train all folds
        folds_to_train = list(range(args.n_folds))

    # Run training for each fold
    all_fold_results = []

    for fold_idx in folds_to_train:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"{'='*80}\n")

        # Create run name
        if args.run_name:
            run_name = f"{args.run_name}_fold{fold_idx}"
        else:
            run_name = f"{args.backbone}_fold{fold_idx}"

        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log configuration
            config = {
                'backbone': args.backbone,
                'pretrained': args.pretrained,
                'dropout': args.dropout,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'optimizer': args.optimizer,
                'scheduler': args.scheduler,
                'augmentation': args.augmentation,
                'use_focal_loss': args.use_focal_loss,
                'focal_gamma': args.focal_gamma,
                'use_class_weights': args.use_class_weights,
                'fold': fold_idx,
                'n_folds': args.n_folds,
                'num_classes': num_classes,
                'seed': args.seed,
            }
            log_training_config(config)

            # Create stratified K-fold splits
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

            # Get train/val indices for this fold
            all_indices = list(range(len(full_dataset)))
            all_labels = full_dataset.labels

            for current_fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
                if current_fold == fold_idx:
                    break

            # Create datasets for this fold
            train_dataset = Subset(full_dataset, train_idx)

            # Create validation dataset with validation transform
            val_dataset_base = FetalUltrasoundDataset(
                root_dir=data_dir,
                transform=val_transform,
                max_samples_per_class=args.max_samples,
            )
            val_dataset = Subset(val_dataset_base, val_idx)

            logger.info(f"Fold {fold_idx}: Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

            # Train this fold
            fold_results = train_fold(
                args=args,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_classes=num_classes,
                class_names=class_names,
                fold=fold_idx,
            )

            all_fold_results.append(fold_results)

    # Log cross-validation summary
    if len(all_fold_results) > 1:
        logger.info("\n" + "="*80)
        logger.info("Cross-Validation Summary")
        logger.info("="*80)

        avg_accuracy = np.mean([r['best_val_accuracy'] for r in all_fold_results])
        std_accuracy = np.std([r['best_val_accuracy'] for r in all_fold_results])

        logger.info(f"Average validation accuracy: {avg_accuracy:.4f} +/- {std_accuracy:.4f}")

        for i, result in enumerate(all_fold_results):
            logger.info(
                f"Fold {i}: {result['best_val_accuracy']:.4f} "
                f"(epoch {result['best_epoch']})"
            )

    logger.info("\nTraining complete!")


if __name__ == '__main__':
    main()
