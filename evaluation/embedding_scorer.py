"""
Embedding similarity scorer for comparing VLM outputs against ground truth.

Uses sentence-transformers to encode texts and compute cosine similarity.
"""

from typing import List, Dict, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")

from .config import DEFAULT_EMBEDDING_MODEL, FALLBACK_EMBEDDING_MODEL


class EmbeddingScorer:
    """Compute embedding similarity scores for text comparison."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cuda",
        batch_size: int = 32
    ):
        """
        Initialize the embedding scorer.

        Args:
            model_name: Name of the sentence-transformer model to use.
                - "all-mpnet-base-v2": Best quality (768 dim)
                - "all-MiniLM-L6-v2": Faster, smaller (384 dim)
            device: Device to run on ("cuda" or "cpu")
            batch_size: Batch size for encoding
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding scoring. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        print(f"Loading embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            print(f"Falling back to {FALLBACK_EMBEDDING_MODEL}")
            self.model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL, device=device)
            self.model_name = FALLBACK_EMBEDDING_MODEL

        print(f"Model loaded: {self.model_name} (embedding dim: {self.model.get_sentence_embedding_dimension()})")

    def encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

    def compute_similarity(
        self,
        predictions: List[str],
        references: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between predictions and reference responses.

        Args:
            predictions: List of model predictions
            references: List of reference responses

        Returns:
            numpy array of similarity scores (one per sample)
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
            )

        print("Encoding predictions...")
        pred_embeddings = self.encode_texts(predictions, show_progress)

        print("Encoding references...")
        gt_embeddings = self.encode_texts(references, show_progress)

        # Compute pairwise cosine similarities (dot product of normalized vectors)
        # Since embeddings are normalized, cosine_sim = dot product
        similarities = np.sum(pred_embeddings * gt_embeddings, axis=1)

        return similarities

    def compute_aggregate_metrics(
        self,
        similarities: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute aggregate metrics from similarity scores.

        Args:
            similarities: numpy array of similarity scores

        Returns:
            Dictionary with mean, std, median, min, max, percentiles
        """
        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "median_similarity": float(np.median(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "percentile_25": float(np.percentile(similarities, 25)),
            "percentile_75": float(np.percentile(similarities, 75)),
            "num_samples": int(len(similarities)),
        }

    def compute_category_metrics(
        self,
        similarities: np.ndarray,
        categories: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per category.

        Args:
            similarities: numpy array of similarity scores
            categories: List of category labels (one per sample)

        Returns:
            Dictionary mapping category name to its metrics
        """
        if len(similarities) != len(categories):
            raise ValueError(
                f"Length mismatch: {len(similarities)} scores vs {len(categories)} categories"
            )

        category_metrics = {}
        unique_categories = sorted(set(categories))

        for category in unique_categories:
            indices = [i for i, c in enumerate(categories) if c == category]
            cat_similarities = similarities[indices]
            category_metrics[category] = self.compute_aggregate_metrics(cat_similarities)

        return category_metrics


def quick_test():
    """Quick test of the embedding scorer."""
    print("\n" + "=" * 60)
    print("Quick Test: Embedding Scorer")
    print("=" * 60)

    scorer = EmbeddingScorer(device="cpu")  # Use CPU for quick test

    predictions = [
        "The fetal head shows normal development with clear brain structures visible.",
        "This is an image of the fetal abdomen showing stomach and liver.",
        "Unable to determine structures due to poor image quality."
    ]

    ground_truths = [
        "The fetal head appears normal with visible brain ventricles and midline structures.",
        "This ultrasound shows the fetal abdominal region with stomach bubble visible.",
        "The image quality is suboptimal, limiting anatomical assessment."
    ]

    similarities = scorer.compute_similarity(predictions, ground_truths)

    print("\nResults:")
    for i, (pred, gt, sim) in enumerate(zip(predictions, ground_truths, similarities)):
        print(f"\n  Sample {i+1}: similarity = {sim:.4f}")
        print(f"    Pred: {pred[:60]}...")
        print(f"    GT:   {gt[:60]}...")

    metrics = scorer.compute_aggregate_metrics(similarities)
    print(f"\nAggregate Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    quick_test()
