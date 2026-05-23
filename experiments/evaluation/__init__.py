"""
VLM Evaluation Pipeline

This package provides tools for evaluating fine-tuned VLM models against
ground truth annotations using embedding similarity.

Usage:
    1. Create stratified test subset:
       python experiments/evaluation/create_test_subset.py

    2. Run full evaluation:
       python experiments/evaluation/evaluate_vlm.py

    3. Or run inference separately (for vast.ai):
       python experiments/evaluation/evaluate_vlm.py --inference-only

    4. Then score locally:
       python experiments/evaluation/evaluate_vlm.py --score-only --predictions <path>
"""

from .config import (
    BASE_MODEL_ID,
    ADAPTER_PATH,
    CATEGORIES,
    DEFAULT_EMBEDDING_MODEL,
)
from .embedding_scorer import EmbeddingScorer
from .question_scorer import MultiMetricScorer
