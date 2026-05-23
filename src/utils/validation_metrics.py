"""
Comprehensive Validation Metrics for Medical Image Classification
Includes confusion matrix, per-class metrics, and statistical significance testing
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score
)
import torch
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

# Import extracted modules
from .metrics_visualization import plot_confusion_matrix, plot_per_class_metrics
from .statistical_testing import statistical_significance_test
from .metrics_export import print_classification_report, export_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveMetrics:
    """Calculate and visualize comprehensive metrics for model evaluation"""

    def __init__(self, class_names: List[str]) -> None:
        """
        Initialize metrics calculator

        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()

    def reset(self) -> None:
        """Reset all stored predictions"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(self, preds: torch.Tensor, labels: torch.Tensor, probs: Optional[torch.Tensor] = None) -> None:
        """
        Update with batch predictions

        Args:
            preds: Predicted class indices
            labels: True class indices
            probs: Optional prediction probabilities for AUC
        """
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.detach().cpu().numpy())

    def compute_metrics(self) -> Dict:
        """
        Compute all metrics

        Returns:
            Dictionary containing all computed metrics
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes)
        )

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )

        # Cohen's Kappa (agreement metric)
        kappa = cohen_kappa_score(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        # AUC if probabilities available
        auc_scores = None
        if self.all_probs:
            try:
                y_probs = np.array(self.all_probs)
                if self.num_classes == 2:
                    auc_scores = roc_auc_score(y_true, y_probs[:, 1])
                else:
                    # One-vs-rest AUC for multiclass
                    from sklearn.preprocessing import label_binarize
                    y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
                    auc_scores = {}
                    for i in range(self.num_classes):
                        try:
                            auc_scores[self.class_names[i]] = roc_auc_score(
                                y_true_binarized[:, i], y_probs[:, i]
                            )
                        except Exception:
                            auc_scores[self.class_names[i]] = None
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")

        metrics = {
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': cm,
            'per_class': {
                'precision': dict(zip(self.class_names, precision)),
                'recall': dict(zip(self.class_names, recall)),
                'f1': dict(zip(self.class_names, f1)),
                'support': dict(zip(self.class_names, support))
            },
            'weighted': {
                'precision': precision_weighted,
                'recall': recall_weighted,
                'f1': f1_weighted
            },
            'macro': {
                'precision': precision_macro,
                'recall': recall_macro,
                'f1': f1_macro
            },
            'auc': auc_scores
        }

        return metrics

    def plot_confusion_matrix(self, save_path: Optional[str] = None, normalize: bool = False) -> None:
        """
        Plot confusion matrix with detailed annotations

        Args:
            save_path: Path to save figure
            normalize: Whether to normalize values to percentages
        """
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        plot_confusion_matrix(y_true, y_pred, self.class_names, save_path, normalize)

    def plot_per_class_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot per-class precision, recall, and F1 scores

        Args:
            save_path: Path to save figure
        """
        metrics = self.compute_metrics()
        plot_per_class_metrics(
            self.class_names,
            metrics['per_class'],
            metrics['weighted']['f1'],
            save_path
        )

    def print_classification_report(self) -> None:
        """Print detailed classification report"""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        metrics = self.compute_metrics()
        print_classification_report(y_true, y_pred, self.class_names, metrics)

    def statistical_significance_test(
        self,
        other_preds: List[int],
        test_type: str = 'mcnemar'
    ) -> Dict:
        """
        Test statistical significance between two models

        Args:
            other_preds: Predictions from another model
            test_type: Type of test ('mcnemar' or 'wilcoxon')

        Returns:
            Dictionary with test results
        """
        y_true = np.array(self.all_labels)
        y_pred1 = np.array(self.all_preds)
        y_pred2 = np.array(other_preds)
        return statistical_significance_test(y_true, y_pred1, y_pred2, test_type)

    def export_results(self, save_path: str) -> pd.DataFrame:
        """
        Export all metrics to a CSV file

        Args:
            save_path: Path to save CSV
        """
        metrics = self.compute_metrics()
        return export_results(
            self.class_names,
            metrics,
            len(self.all_labels),
            save_path
        )
