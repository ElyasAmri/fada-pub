"""
Statistical Testing - Significance tests for model comparison
"""

import numpy as np
from typing import List, Dict
from scipy import stats


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict:
    """
    Perform McNemar's test for paired nominal data

    Args:
        y_true: True labels
        y_pred1: Predictions from first model
        y_pred2: Predictions from second model

    Returns:
        Dictionary with test results
    """
    from statsmodels.stats.contingency_tables import mcnemar

    # Create contingency table
    correct1_correct2 = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    correct1_wrong2 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    wrong1_correct2 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    wrong1_wrong2 = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))

    table = [[correct1_correct2, correct1_wrong2],
            [wrong1_correct2, wrong1_wrong2]]
    result = mcnemar(table, exact=False, correction=True)

    return {
        'test': 'McNemar',
        'statistic': result.statistic,
        'p_value': result.pvalue,
        'contingency_table': table,
        'significant': result.pvalue < 0.05
    }


def wilcoxon_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict:
    """
    Perform Wilcoxon signed-rank test

    Args:
        y_true: True labels
        y_pred1: Predictions from first model
        y_pred2: Predictions from second model

    Returns:
        Dictionary with test results
    """
    acc1 = (y_pred1 == y_true).astype(int)
    acc2 = (y_pred2 == y_true).astype(int)
    statistic, p_value = stats.wilcoxon(acc1, acc2)

    return {
        'test': 'Wilcoxon signed-rank',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def statistical_significance_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    test_type: str = 'mcnemar'
) -> Dict:
    """
    Test statistical significance between two models

    Args:
        y_true: True labels
        y_pred1: Predictions from first model
        y_pred2: Predictions from second model
        test_type: Type of test ('mcnemar' or 'wilcoxon')

    Returns:
        Dictionary with test results
    """
    if test_type == 'mcnemar':
        return mcnemar_test(y_true, y_pred1, y_pred2)
    elif test_type == 'wilcoxon':
        return wilcoxon_test(y_true, y_pred1, y_pred2)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
