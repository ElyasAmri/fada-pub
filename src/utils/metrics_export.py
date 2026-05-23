"""
Metrics Export - Reporting and export functions for model evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    metrics: Dict
) -> None:
    """
    Print detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        metrics: Pre-computed metrics dictionary
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")

    if metrics['auc'] is not None:
        if isinstance(metrics['auc'], dict):
            print("\nAUC Scores:")
            for class_name, auc in metrics['auc'].items():
                if auc is not None:
                    print(f"  {class_name}: {auc:.4f}")
        else:
            print(f"AUC Score: {metrics['auc']:.4f}")


def export_results(
    class_names: List[str],
    metrics: Dict,
    total_samples: int,
    save_path: str
) -> pd.DataFrame:
    """
    Export all metrics to a CSV file

    Args:
        class_names: List of class names
        metrics: Pre-computed metrics dictionary
        total_samples: Total number of samples
        save_path: Path to save CSV

    Returns:
        DataFrame with results
    """
    results = []

    for class_name in class_names:
        results.append({
            'Class': class_name,
            'Precision': metrics['per_class']['precision'][class_name],
            'Recall': metrics['per_class']['recall'][class_name],
            'F1-Score': metrics['per_class']['f1'][class_name],
            'Support': metrics['per_class']['support'][class_name],
            'AUC': metrics['auc'][class_name] if isinstance(metrics['auc'], dict) else None
        })

    results.append({
        'Class': 'Weighted Average',
        'Precision': metrics['weighted']['precision'],
        'Recall': metrics['weighted']['recall'],
        'F1-Score': metrics['weighted']['f1'],
        'Support': total_samples,
        'AUC': None
    })

    results.append({
        'Class': 'Macro Average',
        'Precision': metrics['macro']['precision'],
        'Recall': metrics['macro']['recall'],
        'F1-Score': metrics['macro']['f1'],
        'Support': total_samples,
        'AUC': None
    })

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    logger.info(f"Results exported to {save_path}")

    return df
