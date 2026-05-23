"""
Metrics Visualization - Plotting functions for model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix with detailed annotations

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize values to percentages
    """
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    if normalize:
        cm = cm.astype('float')
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_sum[cm_sum == 0] = 1
        cm = cm / cm_sum * 100

    plt.figure(figsize=(10, 8))

    if normalize:
        cm = np.nan_to_num(cm, nan=0.0)
        fmt = '.1f'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names,
               square=True,
               cbar_kws={'label': 'Percentage' if normalize else 'Count'})

    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    for i in range(num_classes):
        class_acc = cm[i, i] / cm[i].sum() * 100 if not normalize else cm[i, i]
        plt.text(num_classes + 0.3, i + 0.5, f'{class_acc:.1f}%',
                ha='left', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_per_class_metrics(
    class_names: List[str],
    per_class_metrics: Dict,
    weighted_f1: float,
    save_path: Optional[str] = None
) -> None:
    """
    Plot per-class precision, recall, and F1 scores

    Args:
        class_names: List of class names
        per_class_metrics: Dictionary with precision, recall, f1 per class
        weighted_f1: Weighted F1 score for reference line
        save_path: Path to save figure
    """
    precision_values = [per_class_metrics['precision'][c] for c in class_names]
    recall_values = [per_class_metrics['recall'][c] for c in class_names]
    f1_values = [per_class_metrics['f1'][c] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision_values, width, label='Precision')
    bars2 = ax.bar(x, recall_values, width, label='Recall')
    bars3 = ax.bar(x + width, f1_values, width, label='F1-Score')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])

    ax.axhline(y=weighted_f1, color='r', linestyle='--',
              label=f"Weighted F1: {weighted_f1:.3f}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class metrics saved to {save_path}")

    plt.show()
