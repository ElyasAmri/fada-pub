"""
Multi-metric per-question scoring for VLM evaluation.

Scores VLM predictions against sonographer ground truth annotations using
question-appropriate metrics:
  Q1 (Anatomical Structures): synonym-expanded set F2 (recall-weighted)
  Q2 (Fetal Orientation): relaxed accuracy with presentation keyword fallback
  Q3 (Imaging Plane): relaxed accuracy with plane type keyword fallback
  Q4 (Biometric Measurements): keyword F1 on measurement types
  Q5 (Gestational Age): exact bin match with adjacent-bin partial credit
  Q6 (Image Quality): exact tier match (Good/Medium/Low) with adjacency
  Q7 (Normality Assessment): exact match with binary (normal/abnormal) fallback
  Q8 (Clinical Recommendations): BERTScore F1

All questions also get embedding cosine similarity as a secondary metric
for backward compatibility with existing evaluation scores.
"""

import re
import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.questions import QUESTIONS, QUESTION_COLUMNS, QUESTION_SHORT_NAMES
from src.data.normalize_annotations import AnnotationNormalizer, _normalize_q1_structures
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    ANNOTATIONS_PATH,
    BERTSCORE_MODEL,
    BERTSCORE_MODEL_CLINICAL,
    GA_BINS_ORDERED,
    QUALITY_TIERS,
    SCORING_PIPELINE_VERSION,
)
from .embedding_scorer import EmbeddingScorer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question detection
# ---------------------------------------------------------------------------

# Map question prefix (text before first colon, lowercased) -> index 0-7
_QUESTION_PREFIX_MAP: Dict[str, int] = {}
for _i, _q in enumerate(QUESTIONS):
    _prefix = _q.split(":")[0].strip().lower()
    _QUESTION_PREFIX_MAP[_prefix] = _i

# C1: JSONL variant prefixes (alternative phrasings used in JSONL training files)
_QUESTION_PREFIX_MAP["imaging plane identification"] = 2   # JSONL variant of Q3
_QUESTION_PREFIX_MAP["normality/abnormality determination"] = 6  # JSONL variant of Q7
_QUESTION_PREFIX_MAP["gestational age estimation"] = 4  # JSONL variant of Q5
_QUESTION_PREFIX_MAP["image quality assessment"] = 5  # JSONL variant of Q6


def detect_question_index(question_text: str) -> int:
    """Map question text to Q1-Q8 index (0-7) via the label before the colon.

    Raises ValueError if no match is found.
    """
    prefix = question_text.split(":")[0].strip().lower()
    idx = _QUESTION_PREFIX_MAP.get(prefix)
    if idx is not None:
        return idx
    # Fuzzy fallback: substring containment
    for key, idx in _QUESTION_PREFIX_MAP.items():
        if key in prefix or prefix in key:
            print(f"WARNING: Fuzzy question match for '{question_text[:40]}' -> index {idx}")
            return idx
    raise ValueError(f"Cannot detect question index from: {question_text[:80]}")


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

class GroundTruthLoader:
    """Loads normalized sonographer annotations and builds (folder, image) lookup."""

    def __init__(self, annotations_path: Path):
        self.annotations_path = annotations_path
        self._data: Dict[Tuple[str, str], Dict[str, str]] = {}

    def load(self) -> "GroundTruthLoader":
        """Load the Excel and build the lookup dict. Returns self for chaining."""
        logger.info("Loading ground truth from %s", self.annotations_path)
        df = pd.read_excel(self.annotations_path)

        for _, row in df.iterrows():
            folder = str(row["Folder Name"]).strip()
            image = str(row["Image Name"]).strip()
            answers: Dict[str, str] = {}
            for col in QUESTION_COLUMNS:
                if col in df.columns:
                    val = row[col]
                    answers[col] = str(val).strip() if pd.notna(val) else ""
            self._data[(folder, image)] = answers

        logger.info("Loaded %d ground truth entries", len(self._data))
        return self

    def get(self, folder_name: str, image_name: str) -> Optional[Dict[str, str]]:
        """Look up GT answers for a (folder, image) pair."""
        return self._data.get((folder_name, image_name))

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QuestionScore:
    """Per-sample scoring result."""
    question_index: int
    question_name: str
    primary_metric_name: str
    primary_score: float
    embedding_similarity: float
    normalized_prediction: str
    normalized_gt: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringRecord:
    """Intermediate record joining a prediction to its GT answer."""
    sample_id: int
    folder_name: str
    image_name: str
    category: str
    question_index: int
    question_column: str
    prediction: str
    ground_truth: str


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_set_f1(
    pred_set: set, gt_set: set, beta: float = 1.0
) -> Tuple[float, float, float]:
    """Compute precision, recall, F-beta between two sets.

    Args:
        beta: F-beta weight. beta=1 gives F1, beta=2 weights recall 2x.
    """
    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gt_set:
        return 0.0, 0.0, 0.0
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    beta2 = beta ** 2
    fbeta = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, fbeta


# ---------------------------------------------------------------------------
# Q1: Structure synonyms for fuzzy set matching
# ---------------------------------------------------------------------------

Q1_STRUCTURE_SYNONYMS: Dict[str, set] = {
    "abdomen": {"abdominal wall", "abdominal area", "belly"},
    "abdominal wall": {"abdomen", "abdominal area", "belly"},
    "abdominal area": {"abdomen", "abdominal wall", "belly"},
    "belly": {"abdomen", "abdominal wall", "abdominal area"},
    "head": {"fetal head", "fetal skull", "calvarium", "skull"},
    "fetal head": {"head", "calvarium", "skull"},
    "fetal skull": {"skull", "calvarium", "head"},
    "skull": {"fetal skull", "calvarium", "head", "fetal head", "calvarial bones"},
    "calvarium": {"skull", "fetal skull", "calvarial bones", "head", "fetal head"},
    "calvarial bones": {"calvarium", "skull"},
    "spine": {"vertebral column", "spinal column", "fetal spine", "vertebrae"},
    "spinal column": {"spine", "vertebral column", "fetal spine", "vertebrae"},
    "fetal spine": {"spine", "vertebral column", "spinal column"},
    "vertebral column": {"spine", "fetal spine", "vertebrae", "spinal column"},
    "vertebrae": {"spine", "vertebral column", "spinal column"},
    "brain": {"cerebrum", "intracranial structures"},
    "intracranial structures": {"brain", "cerebrum"},
    "cerebrum": {"brain", "intracranial structures"},
    "heart": {"cardiac structures", "fetal heart"},
    "fetal heart": {"heart", "cardiac structures"},
    "cardiac structures": {"heart", "fetal heart"},
    "ivc": {"inferior vena cava"},
    "inferior vena cava": {"ivc"},
    "stomach bubble": {"stomach", "gastric bubble"},
    "stomach": {"stomach bubble", "gastric bubble"},
    "gastric bubble": {"stomach", "stomach bubble"},
    "umbilical cord": {"umbilical"},
    "umbilical": {"umbilical cord"},
    "thalami": {"thalamus"},
    "thalamus": {"thalami"},
    "falx": {"falx cerebri", "interhemispheric fissure"},
    "falx cerebri": {"falx", "interhemispheric fissure"},
    "interhemispheric fissure": {"falx", "falx cerebri"},
    "choroid plexus": {"choroid"},
    "choroid": {"choroid plexus"},
    "ventricle": {"lateral ventricle"},
    "lateral ventricle": {"ventricle"},
    "cerebellum": {"cerebellar vermis"},
    "cerebellar vermis": {"cerebellum"},
    "femur": {"femoral bone", "thigh bone"},
    "femoral bone": {"femur", "thigh bone"},
    "thigh bone": {"femur", "femoral bone"},
    "placenta": {"placental tissue"},
    "placental tissue": {"placenta"},
    "amniotic fluid": {"liquor amnii"},
    "liquor amnii": {"amniotic fluid"},
    "diaphragm": {"diaphragmatic dome"},
    "diaphragmatic dome": {"diaphragm"},
}


def _expand_with_synonyms(structure_set: set) -> set:
    """Expand a set of structures with their synonyms."""
    expanded = set(structure_set)
    for s in structure_set:
        if s in Q1_STRUCTURE_SYNONYMS:
            expanded.update(Q1_STRUCTURE_SYNONYMS[s])
    return expanded


def _get_git_hash() -> str:
    """Return the current git commit hash (first 40 chars) or 'unknown'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).strip().decode()
    except Exception:
        return "unknown"


def _file_sha256(path) -> str:
    """Return the first 16 hex chars of the SHA-256 of a file, or 'unknown'."""
    import hashlib
    try:
        h = hashlib.sha256(Path(path).read_bytes())
        return h.hexdigest()[:16]
    except Exception:
        return "unknown"


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a score array.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(scores, size=len(scores), replace=True))
        for _ in range(n_bootstrap)
    ])
    mean = float(np.mean(scores))
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, ci_lower, ci_upper


def _extract_presentation_keyword(text: str) -> Optional[str]:
    """Extract presentation type from orientation text."""
    tl = text.lower()
    if "cephalic" in tl or "vertex" in tl or "head down" in tl or "head-down" in tl or "head first" in tl:
        return "cephalic"
    if "frank breech" in tl or "complete breech" in tl or "footling" in tl:
        return "breech"
    if "feet first" in tl or "foot first" in tl:
        return "breech"
    if "breech" in tl:
        return "breech"
    if "transverse" in tl:
        return "transverse"
    if "oblique" in tl:
        return "oblique"
    if "longitudinal" in tl:
        return "longitudinal"
    return None


def _extract_plane_keyword(text: str) -> Optional[str]:
    """Extract dominant plane type from imaging plane text."""
    tl = text.lower()
    # Check specific compound planes first
    for compound, label in [
        ("trans-thalamic", "trans-thalamic"), ("transthalamic", "trans-thalamic"),
        ("trans-cerebellar", "trans-cerebellar"), ("transcerebellar", "trans-cerebellar"),
        ("trans-ventricular", "trans-ventricular"), ("transventricular", "trans-ventricular"),
        ("trans-abdominal", "trans-abdominal"), ("transabdominal", "trans-abdominal"),
    ]:
        if compound in tl:
            return label
    if "4-chamber" in tl or "4 chamber" in tl or "four chamber" in tl:
        return "4-chamber"
    if "mid-sagittal" in tl or "mid sagittal" in tl or "midsagittal" in tl:
        return "mid-sagittal"
    if "para-sagittal" in tl or "parasagittal" in tl:
        return "para-sagittal"
    if "sagittal" in tl:
        return "sagittal"
    if "coronal" in tl:
        return "coronal"
    if "transverse" in tl or "axial" in tl or "cross-sectional" in tl or "cross sectional" in tl:
        return "axial"
    if "longitudinal" in tl:
        return "longitudinal"
    if "lateral" in tl:
        return "lateral"
    if "anterior" in tl:
        return "anterior"
    if "posterior" in tl:
        return "posterior"
    return None


# Q4 measurement keyword extraction
_Q4_KEYWORDS: Dict[str, str] = {
    "abdominal circumference": "AC",
    "head circumference": "HC",
    "biparietal diameter": "BPD",
    "crown rump length": "CRL",
    "nuchal translucency": "NT",
    "femur length": "FL",
    "cervical length": "CL",
    "cerebellar": "CEREBELLAR",
    "cisterna magna": "CISTERNA_MAGNA",
    "lateral ventricle": "LV",
    "nuchal fold": "NF",
    "angle of progression": "AOP",
    "aortic": "AORTIC",
    "ac": "AC",
    "bpd": "BPD",
    "fl": "FL",
    "hc": "HC",
    "nt": "NT",
    "crl": "CRL",
    "cl": "CL",
    "inferior vena cava": "IVC",
    "ivc": "IVC",
    "frontomaxillary facial angle": "FMF",
    "frontomaxillary": "FMF",
    "fmf angle": "FMF",
    "intracranial translucency": "IT",
    "occipitofrontal diameter": "OFD",
    "occipitofrontal": "OFD",
    "ofd": "OFD",
    "transcerebellar diameter": "TCD",
    "tcd": "TCD",
    "aortic diameter": "AORTIC",
}

# Pre-sorted by key length descending to match longest first
_Q4_KEYWORDS_SORTED = sorted(_Q4_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True)


def _extract_q4_keywords(text: str) -> set:
    """Extract measurement keyword set from Q4 text."""
    tl = text.lower()
    found = set()
    for key, canonical in _Q4_KEYWORDS_SORTED:
        if re.search(r"\b" + re.escape(key) + r"\b", tl):
            found.add(canonical)
    return found


def _ga_bin_index(bin_str: str) -> Optional[int]:
    """Get the ordered index of a GA bin for adjacency checking."""
    normalized = bin_str.strip().lower()
    for i, b in enumerate(GA_BINS_ORDERED):
        if b.lower() == normalized:
            return i
    return None


def _extract_quality_tier(text: str) -> Optional[str]:
    """Reduce quality text to good/medium/low tier."""
    tl = text.lower()
    if any(w in tl for w in ("good", "high", "excellent")):
        return "good"
    if any(w in tl for w in ("medium", "acceptable", "moderate", "adequate")):
        return "medium"
    if any(w in tl for w in ("low", "bad", "poor", "suboptimal", "dark")):
        return "low"
    return None


def _extract_normality_binary(text: str) -> Optional[str]:
    """Reduce normality text to normal/abnormal binary."""
    tl = text.lower()
    # Check abnormal first ("normal" is a substring of "abnormal")
    if re.search(
        r"\babnormal\b|\bincreased\s+nt\b|\bthicken\w*\s+nt\b"
        r"|\bspina\s+bifida\b|\bvsd\b|\bdilated\b|\bhematoma\b|\bprevia\b",
        tl,
    ):
        return "abnormal"
    if re.search(r"\bnormal\b|\bwithin\s+normal\b", tl):
        return "normal"
    return None


# ---------------------------------------------------------------------------
# Lazy-loaded BERTScore
# ---------------------------------------------------------------------------

_bertscore_fn = None


def _compute_bertscore_batch(
    predictions: List[str], references: List[str], model_type: str = BERTSCORE_MODEL,
    num_layers: Optional[int] = None,
) -> List[float]:
    """Batch compute BERTScore F1. Raises if bert_score is unavailable."""
    global _bertscore_fn
    if _bertscore_fn is None:
        from bert_score import score as bertscore
        _bertscore_fn = bertscore
    kwargs = {"model_type": model_type, "verbose": False}
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    _, _, F1 = _bertscore_fn(predictions, references, **kwargs)
    return [float(f) for f in F1]


# ---------------------------------------------------------------------------
# Per-question scoring functions
# ---------------------------------------------------------------------------

class _QuestionScorers:
    """Stateless per-question scoring implementations."""

    def __init__(self, normalizer: AnnotationNormalizer):
        self.normalizer = normalizer

    def score_q1_structures(self, pred: str, gt: str) -> QuestionScore:
        """Q1: Synonym-expanded set F2 (recall-weighted) on anatomical structures."""
        col = QUESTION_COLUMNS[0]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        pred_set = {s.strip().lower() for s in norm_pred.split(",") if s.strip()}
        gt_set = {s.strip().lower() for s in norm_gt.split(",") if s.strip()}

        # Expand both sets with synonyms for matching
        pred_expanded = _expand_with_synonyms(pred_set)
        gt_expanded = _expand_with_synonyms(gt_set)

        # Count matches: a pred item matches if it or any synonym is in gt_expanded
        tp_pred = sum(1 for s in pred_set if s in gt_expanded)
        tp_gt = sum(1 for s in gt_set if s in pred_expanded)

        precision = tp_pred / len(pred_set) if pred_set else 0.0
        recall = tp_gt / len(gt_set) if gt_set else 0.0

        # F-beta with beta=2 (recall-weighted)
        beta = 2.0
        beta2 = beta ** 2
        f2 = (
            (1 + beta2) * precision * recall / (beta2 * precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return QuestionScore(
            question_index=0,
            question_name=QUESTION_SHORT_NAMES[0],
            primary_metric_name="set_f2_synonym",
            primary_score=f2,
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
            details={
                "precision": precision,
                "recall": recall,
                "pred_structures": sorted(pred_set),
                "gt_structures": sorted(gt_set),
            },
        )

    def score_q2_orientation(self, pred: str, gt: str) -> QuestionScore:
        """Q2: Relaxed accuracy on fetal orientation."""
        col = QUESTION_COLUMNS[1]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        if norm_pred.strip().lower() == norm_gt.strip().lower():
            score = 1.0
        else:
            pred_kw = _extract_presentation_keyword(norm_pred)
            gt_kw = _extract_presentation_keyword(norm_gt)
            score = 0.5 if (pred_kw and gt_kw and pred_kw == gt_kw) else 0.0  # Partial credit=0.5; see sensitivity analysis in docs

        return QuestionScore(
            question_index=1,
            question_name=QUESTION_SHORT_NAMES[1],
            primary_metric_name="relaxed_accuracy",
            primary_score=score,
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
        )

    def score_q3_plane(self, pred: str, gt: str) -> QuestionScore:
        """Q3: Relaxed accuracy on imaging plane."""
        col = QUESTION_COLUMNS[2]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        if norm_pred.strip().lower() == norm_gt.strip().lower():
            score = 1.0
        else:
            pred_kw = _extract_plane_keyword(norm_pred)
            gt_kw = _extract_plane_keyword(norm_gt)
            score = 0.5 if (pred_kw and gt_kw and pred_kw == gt_kw) else 0.0  # Partial credit=0.5; see sensitivity analysis in docs

        return QuestionScore(
            question_index=2,
            question_name=QUESTION_SHORT_NAMES[2],
            primary_metric_name="relaxed_accuracy",
            primary_score=score,
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
        )

    def score_q4_measurements(self, pred: str, gt: str) -> QuestionScore:
        """Q4: Keyword F1 on measurement types."""
        col = QUESTION_COLUMNS[3]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        pred_kw = _extract_q4_keywords(norm_pred)
        gt_kw = _extract_q4_keywords(norm_gt)
        precision, recall, f1 = _compute_set_f1(pred_kw, gt_kw)

        return QuestionScore(
            question_index=3,
            question_name=QUESTION_SHORT_NAMES[3],
            primary_metric_name="keyword_f1",
            primary_score=f1,
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
            details={
                "precision": precision,
                "recall": recall,
                "pred_keywords": sorted(pred_kw),
                "gt_keywords": sorted(gt_kw),
            },
        )

    def score_q5_gestational_age(self, pred: str, gt: str) -> QuestionScore:
        """Q5: Exact bin match on gestational age."""
        col = QUESTION_COLUMNS[4]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        pred_bin = norm_pred.strip().lower()
        gt_bin = norm_gt.strip().lower()

        if pred_bin == gt_bin:
            score = 1.0
        else:
            pred_idx = _ga_bin_index(pred_bin)
            gt_idx = _ga_bin_index(gt_bin)
            if pred_idx is not None and gt_idx is not None and abs(pred_idx - gt_idx) == 1:
                score = 0.5  # Partial credit=0.5; see sensitivity analysis in docs
            else:
                score = 0.0

        return QuestionScore(
            question_index=4,
            question_name=QUESTION_SHORT_NAMES[4],
            primary_metric_name="exact_bin_match",
            primary_score=score,
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
            details={"pred_bin": pred_bin, "gt_bin": gt_bin},
        )

    def score_q6_quality(self, pred: str, gt: str) -> QuestionScore:
        """Q6: Exact tier match on image quality."""
        col = QUESTION_COLUMNS[5]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        pred_tier = _extract_quality_tier(norm_pred)
        gt_tier = _extract_quality_tier(norm_gt)

        if pred_tier and gt_tier:
            if pred_tier == gt_tier:
                score = 1.0
            elif abs(QUALITY_TIERS.get(pred_tier, -99) - QUALITY_TIERS.get(gt_tier, -99)) == 1:
                score = 0.5  # Partial credit=0.5; see sensitivity analysis in docs
            else:
                score = 0.0
        else:
            score = 0.0

        return QuestionScore(
            question_index=5,
            question_name=QUESTION_SHORT_NAMES[5],
            primary_metric_name="exact_tier_match",
            primary_score=score,
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
            details={"pred_tier": pred_tier, "gt_tier": gt_tier},
        )

    def score_q7_normality(self, pred: str, gt: str) -> QuestionScore:
        """Q7: Exact match with binary normal/abnormal fallback."""
        col = QUESTION_COLUMNS[6]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        details: Dict[str, Any] = {}

        if norm_pred.strip().lower() == norm_gt.strip().lower():
            score = 1.0
        else:
            pred_binary = _extract_normality_binary(norm_pred)
            gt_binary = _extract_normality_binary(norm_gt)
            details["pred_binary"] = pred_binary
            details["gt_binary"] = gt_binary
            score = 0.5 if (pred_binary and gt_binary and pred_binary == gt_binary) else 0.0  # Partial credit=0.5; see sensitivity analysis in docs

        # Confusion matrix component
        pred_binary = _extract_normality_binary(norm_pred)
        gt_binary = _extract_normality_binary(norm_gt)
        if pred_binary == "abnormal" and gt_binary == "abnormal":
            details["cm"] = "TP"
        elif pred_binary == "normal" and gt_binary == "normal":
            details["cm"] = "TN"
        elif pred_binary == "abnormal" and gt_binary == "normal":
            details["cm"] = "FP"
        elif pred_binary == "normal" and gt_binary == "abnormal":
            details["cm"] = "FN"

        return QuestionScore(
            question_index=6,
            question_name=QUESTION_SHORT_NAMES[6],
            primary_metric_name="exact_match_with_binary",
            primary_score=score,
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
            details=details,
        )

    def score_q8_placeholder(self, pred: str, gt: str) -> QuestionScore:
        """Q8: Placeholder -- primary_score filled by batch BERTScore later."""
        col = QUESTION_COLUMNS[7]
        norm_pred = self.normalizer.normalize_single(col, pred)
        norm_gt = self.normalizer.normalize_single(col, gt)

        return QuestionScore(
            question_index=7,
            question_name=QUESTION_SHORT_NAMES[7],
            primary_metric_name="bertscore_f1",
            primary_score=0.0,  # filled by batch step
            embedding_similarity=0.0,
            normalized_prediction=norm_pred,
            normalized_gt=norm_gt,
        )


# ---------------------------------------------------------------------------
# Multi-metric scorer (orchestrator)
# ---------------------------------------------------------------------------

class MultiMetricScorer:
    """Orchestrates multi-metric per-question scoring of VLM predictions
    against sonographer ground truth annotations."""

    def __init__(
        self,
        annotations_path: Path = ANNOTATIONS_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cuda",
    ):
        self.annotations_path = annotations_path
        self.gt_loader = GroundTruthLoader(annotations_path)
        self.normalizer = AnnotationNormalizer()
        self.scorers = _QuestionScorers(self.normalizer)
        self.embedding_model = embedding_model
        self.device = device
        self._embedding_scorer: Optional[EmbeddingScorer] = None

    def _get_embedding_scorer(self) -> EmbeddingScorer:
        if self._embedding_scorer is None:
            self._embedding_scorer = EmbeddingScorer(
                model_name=self.embedding_model, device=self.device
            )
        return self._embedding_scorer

    @staticmethod
    def _extract_image_key(image_path: str) -> Tuple[str, str]:
        """Extract (folder_name, image_name) from an image path.

        Handles both Windows backslash and Unix forward-slash paths.
        """
        normalized = image_path.replace("\\", "/").rstrip("/")
        parts = normalized.split("/")
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return "", parts[-1] if parts else ""

    def score_predictions(
        self, predictions: List[Dict], predictions_file: str = ""
    ) -> Dict[str, Any]:
        """Score all predictions against sonographer ground truth.

        Args:
            predictions: List of dicts with keys: sample_id, image_path,
                         category, question, prediction.
            predictions_file: Path string for metadata (optional).

        Returns:
            Structured results dict with overall, per_question, per_category.
        """
        if len(self.gt_loader) == 0:
            self.gt_loader.load()

        # Pre-pass: count error and empty predictions
        error_prediction_count = sum(
            1 for p in predictions
            if not p.get("prediction") or p["prediction"].startswith("ERROR:")
        )

        # 1. Build scoring records (join predictions to GT)
        records: List[ScoringRecord] = []
        unmatched: List[Dict] = []

        for pred in predictions:
            folder, image = self._extract_image_key(pred["image_path"])
            gt = self.gt_loader.get(folder, image)
            if gt is None:
                unmatched.append(pred)
                continue

            try:
                q_idx = detect_question_index(pred["question"])
            except ValueError:
                unmatched.append(pred)
                continue

            q_col = QUESTION_COLUMNS[q_idx]
            gt_answer = gt.get(q_col, "")
            if not gt_answer:
                unmatched.append(pred)
                continue

            records.append(ScoringRecord(
                sample_id=pred.get("sample_id", -1),
                folder_name=folder,
                image_name=image,
                category=pred.get("category", ""),
                question_index=q_idx,
                question_column=q_col,
                prediction=pred["prediction"],
                ground_truth=gt_answer,
            ))

        logger.info("Matched %d predictions, %d unmatched", len(records), len(unmatched))
        if not records:
            return {"error": "No predictions matched to ground truth",
                    "num_unmatched": len(unmatched)}

        # 2. Batch embedding similarity
        embedding_scorer = self._get_embedding_scorer()
        similarities = embedding_scorer.compute_similarity(
            [r.prediction for r in records],
            [r.ground_truth for r in records],
        )

        # 3. Per-question scoring (Q1-Q7 fast, Q8 placeholder)
        scorer_dispatch = {
            0: self.scorers.score_q1_structures,
            1: self.scorers.score_q2_orientation,
            2: self.scorers.score_q3_plane,
            3: self.scorers.score_q4_measurements,
            4: self.scorers.score_q5_gestational_age,
            5: self.scorers.score_q6_quality,
            6: self.scorers.score_q7_normality,
            7: self.scorers.score_q8_placeholder,
        }

        scores: List[QuestionScore] = []
        q8_indices: List[int] = []

        for i, record in enumerate(records):
            scorer_fn = scorer_dispatch[record.question_index]
            qs = scorer_fn(record.prediction, record.ground_truth)
            qs.embedding_similarity = float(similarities[i])
            scores.append(qs)

            if record.question_index == 7:
                q8_indices.append(i)

        # 4. Batch BERTScore for Q8
        if q8_indices:
            q8_preds = [scores[i].normalized_prediction for i in q8_indices]
            q8_refs = [scores[i].normalized_gt for i in q8_indices]
            try:
                bert_scores = _compute_bertscore_batch(q8_preds, q8_refs)
                for j, idx in enumerate(q8_indices):
                    scores[idx].primary_score = bert_scores[j]
            except Exception as e:
                logger.error("BERTScore computation failed, falling back to 0.0: %s", e)
                for idx in q8_indices:
                    scores[idx].primary_score = 0.0
                    scores[idx].details["bertscore_error"] = str(e)

            # L3: Clinical BERTScore as secondary metric
            try:
                clinical_scores = _compute_bertscore_batch(
                    q8_preds, q8_refs, model_type=BERTSCORE_MODEL_CLINICAL,
                    num_layers=12,  # Bio_ClinicalBERT is BERT-base (12 layers)
                )
                for j, idx in enumerate(q8_indices):
                    scores[idx].details["bertscore_clinical_f1"] = clinical_scores[j]
            except Exception as e:
                logger.warning("Clinical BERTScore failed (non-fatal): %s", e)

        # 5. Aggregate
        result = self._aggregate_results(
            scores, records, unmatched, predictions_file
        )
        result["metadata"]["error_prediction_count"] = error_prediction_count
        # L4: error_rate in overall for direct access
        result["overall"]["error_rate"] = error_prediction_count / len(predictions) if predictions else 0
        return result

    def _aggregate_results(
        self,
        scores: List[QuestionScore],
        records: List[ScoringRecord],
        unmatched: List[Dict],
        predictions_file: str,
    ) -> Dict[str, Any]:
        """Build the structured output dict."""
        from datetime import datetime

        primary_scores = np.array([s.primary_score for s in scores])
        embedding_sims = np.array([s.embedding_similarity for s in scores])

        # C3: Z-score normalized aggregation across heterogeneous metrics
        per_q_means: Dict[int, np.ndarray] = {}
        for q_idx in range(8):
            q_scores_arr = np.array([s.primary_score for s in scores if s.question_index == q_idx])
            if len(q_scores_arr) > 0:
                per_q_means[q_idx] = q_scores_arr

        if len(per_q_means) >= 2:
            # Normalize each question's scores, then average across all
            all_normalized: List[float] = []
            for q_arr in per_q_means.values():
                mu, sigma = np.mean(q_arr), np.std(q_arr)
                if sigma > 0:
                    all_normalized.extend(((q_arr - mu) / sigma).tolist())
                else:
                    all_normalized.extend(np.zeros_like(q_arr).tolist())
            normalized_mean = float(np.mean(all_normalized)) if all_normalized else 0.0
        else:
            normalized_mean = float(np.mean(primary_scores))

        # C2: Bootstrap CI for overall primary score
        overall_mean, overall_ci_lower, overall_ci_upper = bootstrap_confidence_interval(primary_scores)

        result: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "predictions_file": predictions_file,
                "annotations_file": str(self.annotations_path),
                "num_matched": len(records),
                "num_unmatched": len(unmatched),
                "scoring_pipeline_version": SCORING_PIPELINE_VERSION,
                "git_commit_hash": _get_git_hash(),
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "bertscore_model": BERTSCORE_MODEL,
                "bertscore_model_clinical": BERTSCORE_MODEL_CLINICAL,
                "annotations_file_hash": _file_sha256(self.annotations_path),
            },
            "overall": {
                "_WARNING": "primary_score_mean averages heterogeneous metrics (F2, accuracy, F1, BERTScore); prefer per-question scores or primary_score_mean_normalized",
                "primary_score_mean": float(np.mean(primary_scores)),
                "primary_score_mean_normalized": normalized_mean,
                "primary_score_ci_lower": overall_ci_lower,
                "primary_score_ci_upper": overall_ci_upper,
                "embedding_similarity_mean": float(np.mean(embedding_sims)),
                "num_samples": len(scores),
            },
            "per_question": {},
            "per_category": {},
        }

        # Per question
        for q_idx in range(8):
            q_scores = [s for s in scores if s.question_index == q_idx]
            if not q_scores:
                continue

            q_name = f"Q{q_idx + 1}: {QUESTION_SHORT_NAMES[q_idx]}"
            q_primary = np.array([s.primary_score for s in q_scores])
            q_embed = np.array([s.embedding_similarity for s in q_scores])

            # Normalization rate: fraction of predictions that normalized
            # to something different from the raw prediction
            norm_count = sum(
                1 for s, r in zip(
                    (s for s in scores if s.question_index == q_idx),
                    (r for r in records if r.question_index == q_idx),
                )
                if s.normalized_prediction != r.prediction
            )

            # H1: Bootstrap CI per question
            q_mean, q_ci_lower, q_ci_upper = bootstrap_confidence_interval(q_primary)

            entry: Dict[str, Any] = {
                "metric_name": q_scores[0].primary_metric_name,
                "primary_mean": float(np.mean(q_primary)),
                "primary_std": float(np.std(q_primary)),
                "primary_ci_lower": q_ci_lower,
                "primary_ci_upper": q_ci_upper,
                "embedding_similarity_mean": float(np.mean(q_embed)),
                "normalization_rate": norm_count / len(q_scores) if q_scores else 0.0,
                "num_samples": len(q_scores),
            }

            # Q7 confusion matrix
            if q_idx == 6:
                cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
                for s in q_scores:
                    cm_key = s.details.get("cm")
                    if cm_key and cm_key in cm:
                        cm[cm_key] += 1
                total_pos = cm["TP"] + cm["FN"]
                total_neg = cm["TN"] + cm["FP"]
                cm["sensitivity"] = cm["TP"] / total_pos if total_pos > 0 else 0.0
                cm["specificity"] = cm["TN"] / total_neg if total_neg > 0 else 0.0
                # M1: Balanced accuracy for Q7
                cm["balanced_accuracy"] = (cm["sensitivity"] + cm["specificity"]) / 2
                cm["classified_count"] = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
                cm["cm_coverage"] = cm["classified_count"] / len(q_scores) if q_scores else 0.0
                entry["confusion_matrix"] = cm
                entry["balanced_accuracy"] = cm["balanced_accuracy"]

            # Q8: aggregate clinical BERTScore if available
            if q_idx == 7:
                clinical_vals = [s.details.get("bertscore_clinical_f1") for s in q_scores
                                 if s.details.get("bertscore_clinical_f1") is not None]
                if clinical_vals:
                    entry["bertscore_clinical_mean"] = float(np.mean(clinical_vals))

            result["per_question"][q_name] = entry

        # Per category
        for cat in sorted(set(r.category for r in records)):
            cat_indices = [i for i, r in enumerate(records) if r.category == cat]
            cat_scores = [scores[i] for i in cat_indices]

            cat_primary = np.array([s.primary_score for s in cat_scores])
            cat_embed = np.array([s.embedding_similarity for s in cat_scores])

            cat_entry: Dict[str, Any] = {
                "primary_score_mean": float(np.mean(cat_primary)),
                "embedding_similarity_mean": float(np.mean(cat_embed)),
                "num_samples": len(cat_scores),
                "per_question": {},
            }

            for q_idx in range(8):
                q_cat = [s for s in cat_scores if s.question_index == q_idx]
                if not q_cat:
                    continue
                q_name = f"Q{q_idx + 1}: {QUESTION_SHORT_NAMES[q_idx]}"
                cat_entry["per_question"][q_name] = {
                    "primary_mean": float(np.mean([s.primary_score for s in q_cat])),
                    "embedding_similarity_mean": float(
                        np.mean([s.embedding_similarity for s in q_cat])
                    ),
                    "num_samples": len(q_cat),
                }

            result["per_category"][cat] = cat_entry

        # M2: Macro-average across categories
        if result["per_category"]:
            cat_means = [c["primary_score_mean"] for c in result["per_category"].values()]
            result["overall"]["category_macro_mean"] = float(np.mean(cat_means))

        return result


def print_report(results: Dict[str, Any]) -> None:
    """Print a human-readable summary of scoring results."""
    meta = results.get("metadata", {})
    overall = results.get("overall", {})

    print("=" * 70)
    print("Multi-Metric Scoring Report")
    print("=" * 70)
    print(f"  Matched: {meta.get('num_matched', 0)}  |  "
          f"Unmatched: {meta.get('num_unmatched', 0)}")
    print(f"  Overall primary score:       {overall.get('primary_score_mean', 0):.4f}")
    print(f"  Overall embedding similarity: {overall.get('embedding_similarity_mean', 0):.4f}")
    print()

    print("Per-Question Scores:")
    print("-" * 70)
    for q_name, q_data in results.get("per_question", {}).items():
        metric = q_data.get("metric_name", "?")
        primary = q_data.get("primary_mean", 0)
        embed = q_data.get("embedding_similarity_mean", 0)
        n = q_data.get("num_samples", 0)
        print(f"  {q_name}")
        print(f"    {metric}: {primary:.4f}  |  embed_sim: {embed:.4f}  |  n={n}")

        # Q7 confusion matrix
        cm = q_data.get("confusion_matrix")
        if cm:
            print(f"    CM: TP={cm['TP']} TN={cm['TN']} FP={cm['FP']} FN={cm['FN']}"
                  f"  sens={cm['sensitivity']:.3f} spec={cm['specificity']:.3f}")

    print()
    print("Per-Category Scores (top-level):")
    print("-" * 70)
    for cat, cat_data in results.get("per_category", {}).items():
        primary = cat_data.get("primary_score_mean", 0)
        embed = cat_data.get("embedding_similarity_mean", 0)
        n = cat_data.get("num_samples", 0)
        print(f"  {cat:35s}  primary={primary:.4f}  embed={embed:.4f}  n={n}")
