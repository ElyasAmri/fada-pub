"""
Reusable Training Utilities for Deep Learning Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union
import mlflow
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    metrics: Optional[object] = None,
    gradient_clip: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    Train for one epoch with mixed precision support

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        metrics: Optional metrics tracker
        gradient_clip: Gradient clipping value
        scaler: Optional GradScaler for mixed precision

    Returns:
        Tuple of (loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update comprehensive metrics if provided
        if metrics is not None:
            with torch.no_grad():
                metrics.update(predicted, labels, torch.softmax(outputs, dim=1))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: Optional[object] = None,
    desc: str = 'Validation'
) -> Tuple[float, float]:
    """
    Validate the model

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        metrics: Optional metrics tracker
        desc: Description for progress bar

    Returns:
        Tuple of (loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update comprehensive metrics if provided
            if metrics is not None:
                metrics.update(predicted, labels, torch.softmax(outputs, dim=1))

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


class EarlyStopping:
    """Early stopping utility"""

    def __init__(self, patience: int = 10, mode: str = 'max', verbose: bool = True) -> None:
        """
        Args:
            patience: How many epochs to wait after last improvement
            mode: 'min' or 'max' for the metric
            verbose: Print messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode

        if mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def __call__(self, score: float) -> bool:
        """
        Check if should stop

        Args:
            score: Current metric value

        Returns:
            True if should stop
        """
        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                logger.info(f'EarlyStopping: Score improved to {score:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints"""

    def __init__(self, filepath: str, monitor: str = 'val_acc', mode: str = 'max', verbose: bool = True) -> None:
        """
        Args:
            filepath: Path to save checkpoint
            monitor: Metric to monitor
            mode: 'min' or 'max' for the metric
            verbose: Print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        if mode == 'min':
            self.best = float('inf')
        else:
            self.best = float('-inf')

    def __call__(self, score: float, model: nn.Module, optimizer: optim.Optimizer, epoch: int) -> bool:
        """
        Callable interface for checkpoint saving

        Args:
            score: Current metric value
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch

        Returns:
            True if saved
        """
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            self.monitor: score,
            'best_score': self.best
        }
        return self.save_checkpoint(state, score)

    def save_checkpoint(self, state: dict, score: float) -> bool:
        """
        Save checkpoint if improved

        Args:
            state: State dict to save
            score: Current metric value

        Returns:
            True if saved
        """
        if self.mode == 'min':
            improved = score < self.best
        else:
            improved = score > self.best

        if improved:
            self.best = score
            torch.save(state, self.filepath)
            if self.verbose:
                logger.info(f'Checkpoint saved: {self.monitor}={score:.4f}')
            return True
        return False


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """
    Create optimizer from config

    Args:
        model: PyTorch model
        config: Configuration dictionary with keys:
            - optimizer: 'adam', 'adamw', 'sgd'
            - learning_rate: float
            - weight_decay: float
            - momentum: float (for SGD)

    Returns:
        Optimizer instance
    """
    opt_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)

    if opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer: optim.Optimizer, config: dict) -> Optional[optim.lr_scheduler.LRScheduler]:
    """
    Create learning rate scheduler from config

    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary with keys:
            - scheduler: 'cosine', 'plateau', 'step'
            - Various scheduler-specific parameters

    Returns:
        Scheduler instance
    """
    scheduler_name = config.get('scheduler', 'cosine').lower()

    if scheduler_name == 'cosine':
        T_0 = config.get('T_0', 10)
        T_mult = config.get('T_mult', 2)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    elif scheduler_name == 'plateau':
        mode = config.get('mode', 'max')
        patience = config.get('scheduler_patience', 5)
        factor = config.get('factor', 0.5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)
    elif scheduler_name == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return None


def log_metrics(metrics: dict, step: Optional[int] = None, prefix: str = '') -> None:
    """
    Log metrics to MLflow

    Args:
        metrics: Dictionary of metrics
        step: Optional step number
        prefix: Optional prefix for metric names
    """
    logged_metrics = {}
    for key, value in metrics.items():
        if prefix:
            key = f"{prefix}_{key}"
        logged_metrics[key] = value

    if step is not None:
        mlflow.log_metrics(logged_metrics, step=step)
    else:
        mlflow.log_metrics(logged_metrics)