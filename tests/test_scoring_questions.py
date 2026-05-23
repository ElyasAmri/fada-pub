"""Tests for per-question scoring functions (Q1-Q7).

Tests the _QuestionScorers class methods that produce QuestionScore results.
Q8 (BERTScore) is tested separately as it requires model loading.
"""

import pytest

from src.data.normalize_annotations import AnnotationNormalizer
from experiments.evaluation.question_scorer import _QuestionScorers, QuestionScore


@pytest.fixture
def scorers():
    return _QuestionScorers(AnnotationNormalizer())


# ---------------------------------------------------------------------------
# Q1: Anatomical Structures (set F2 with synonyms)
# ---------------------------------------------------------------------------

class TestQ1Structures:
    def test_exact_match(self, scorers):
        result = scorers.score_q1_structures(
            "skull, spine, heart", "skull, spine, heart"
        )
        assert result.primary_score == 1.0
        assert result.question_index == 0

    def test_synonym_match(self, scorers):
        """Belly should match abdomen via synonyms."""
        result = scorers.score_q1_structures("belly", "abdomen")
        assert result.primary_score > 0.0

    def test_symmetric_synonym_match(self, scorers):
        """S2 fix: abdomen->belly and belly->abdomen should score equally."""
        score_a = scorers.score_q1_structures("belly", "abdomen")
        score_b = scorers.score_q1_structures("abdomen", "belly")
        assert score_a.primary_score == score_b.primary_score

    def test_no_match(self, scorers):
        result = scorers.score_q1_structures("skull", "femur")
        assert result.primary_score == 0.0

    def test_partial_match(self, scorers):
        result = scorers.score_q1_structures(
            "skull, spine", "skull, spine, heart"
        )
        assert 0.0 < result.primary_score < 1.0

    def test_empty_pred(self, scorers):
        result = scorers.score_q1_structures("", "skull, spine")
        assert result.primary_score == 0.0

    def test_returns_question_score(self, scorers):
        result = scorers.score_q1_structures("skull", "skull")
        assert isinstance(result, QuestionScore)
        assert result.primary_metric_name == "set_f2_synonym"

    def test_gt_is_normalized(self, scorers):
        """S4 fix: GT should be normalized, not used raw."""
        result = scorers.score_q1_structures("skull", "  Skull  ")
        assert result.primary_score == 1.0


# ---------------------------------------------------------------------------
# Q2: Fetal Orientation (relaxed accuracy)
# ---------------------------------------------------------------------------

class TestQ2Orientation:
    def test_exact_match(self, scorers):
        result = scorers.score_q2_orientation("cephalic presentation", "cephalic presentation")
        assert result.primary_score == 1.0

    def test_keyword_match(self, scorers):
        result = scorers.score_q2_orientation(
            "The fetus is in cephalic position",
            "Vertex presentation observed"
        )
        assert result.primary_score == 0.5

    def test_no_match(self, scorers):
        result = scorers.score_q2_orientation("cephalic", "breech")
        assert result.primary_score == 0.0

    def test_question_index(self, scorers):
        result = scorers.score_q2_orientation("x", "y")
        assert result.question_index == 1


# ---------------------------------------------------------------------------
# Q3: Imaging Plane (relaxed accuracy)
# ---------------------------------------------------------------------------

class TestQ3Plane:
    def test_exact_match(self, scorers):
        result = scorers.score_q3_plane("sagittal plane", "sagittal plane")
        assert result.primary_score == 1.0

    def test_keyword_match(self, scorers):
        result = scorers.score_q3_plane(
            "trans-thalamic view of the brain",
            "Transthalamic plane showing midline"
        )
        assert result.primary_score == 0.5

    def test_no_match(self, scorers):
        result = scorers.score_q3_plane("sagittal", "coronal")
        assert result.primary_score == 0.0


# ---------------------------------------------------------------------------
# Q4: Biometric Measurements (keyword F1)
# ---------------------------------------------------------------------------

class TestQ4Measurements:
    def test_exact_keywords(self, scorers):
        result = scorers.score_q4_measurements(
            "biparietal diameter", "biparietal diameter"
        )
        assert result.primary_score == 1.0

    def test_multiple_measurements(self, scorers):
        result = scorers.score_q4_measurements(
            "HC and BPD measurements",
            "head circumference and biparietal diameter"
        )
        assert result.primary_score == 1.0

    def test_partial_match(self, scorers):
        result = scorers.score_q4_measurements(
            "BPD measurement",
            "BPD and HC measurements"
        )
        assert 0.0 < result.primary_score < 1.0

    def test_no_match(self, scorers):
        result = scorers.score_q4_measurements(
            "no measurements", "BPD measurement"
        )
        assert result.primary_score == 0.0


# ---------------------------------------------------------------------------
# Q5: Gestational Age (exact bin match)
# ---------------------------------------------------------------------------

class TestQ5GestationalAge:
    def test_exact_bin(self, scorers):
        result = scorers.score_q5_gestational_age("20-25 weeks", "20-25 weeks")
        assert result.primary_score == 1.0

    def test_adjacent_bin(self, scorers):
        result = scorers.score_q5_gestational_age("20-25 weeks", "25-30 weeks")
        assert result.primary_score == 0.5

    def test_non_adjacent_bin(self, scorers):
        result = scorers.score_q5_gestational_age("8-13 weeks", "25-30 weeks")
        assert result.primary_score == 0.0

    def test_unknown_bin(self, scorers):
        result = scorers.score_q5_gestational_age("unknown", "20-25 weeks")
        assert result.primary_score == 0.0

    def test_adjacent_15_20_to_20_25(self, scorers):
        """Adjacent bins should get 0.5 partial credit."""
        result = scorers.score_q5_gestational_age("15-20 weeks", "20-25 weeks")
        assert result.primary_score == 0.5

    def test_35_38_adjacent(self, scorers):
        """35-38 adjacent to 30-35 should get 0.5."""
        result = scorers.score_q5_gestational_age("35-38 weeks", "30-35 weeks")
        assert result.primary_score == 0.5


# ---------------------------------------------------------------------------
# Q6: Image Quality (exact tier match)
# ---------------------------------------------------------------------------

class TestQ6Quality:
    def test_exact_tier(self, scorers):
        result = scorers.score_q6_quality("good quality", "good quality image")
        assert result.primary_score == 1.0

    def test_adjacent_tier(self, scorers):
        result = scorers.score_q6_quality("good quality", "medium quality")
        assert result.primary_score == 0.5

    def test_non_adjacent_tier(self, scorers):
        result = scorers.score_q6_quality("good quality", "poor quality")
        assert result.primary_score == 0.0

    def test_unrecognized_pred_scores_zero(self, scorers):
        """S1 fix: unrecognized prediction should not get partial credit."""
        result = scorers.score_q6_quality("blurry unclear", "low quality")
        assert result.primary_score == 0.0

    def test_sentinel_no_false_partial_credit(self, scorers):
        """S1 fix: sentinel -99 should never be adjacent to any valid tier."""
        result = scorers.score_q6_quality("xyz not a tier", "low quality")
        assert result.primary_score == 0.0


# ---------------------------------------------------------------------------
# Q7: Normality Assessment (exact + binary fallback)
# ---------------------------------------------------------------------------

class TestQ7Normality:
    def test_exact_match(self, scorers):
        result = scorers.score_q7_normality("normal", "normal")
        assert result.primary_score == 1.0

    def test_binary_match(self, scorers):
        result = scorers.score_q7_normality(
            "appears normal, no abnormalities",
            "within normal limits"
        )
        # Both are "normal" binary, but text differs -> 0.5
        assert result.primary_score >= 0.5

    def test_binary_mismatch(self, scorers):
        result = scorers.score_q7_normality("abnormal findings", "normal appearance")
        assert result.primary_score == 0.0

    def test_confusion_matrix_tp(self, scorers):
        result = scorers.score_q7_normality("abnormal", "abnormal")
        assert result.details.get("cm") == "TP"

    def test_confusion_matrix_tn(self, scorers):
        result = scorers.score_q7_normality("normal", "normal")
        assert result.details.get("cm") == "TN"

    def test_confusion_matrix_fp(self, scorers):
        result = scorers.score_q7_normality("abnormal finding", "normal appearance")
        assert result.details.get("cm") == "FP"

    def test_confusion_matrix_fn(self, scorers):
        result = scorers.score_q7_normality("normal appearance", "abnormal finding")
        assert result.details.get("cm") == "FN"


# ---------------------------------------------------------------------------
# Q8: Placeholder (BERTScore filled later)
# ---------------------------------------------------------------------------

class TestQ8Placeholder:
    def test_returns_zero_primary(self, scorers):
        result = scorers.score_q8_placeholder("some recommendation", "some other rec")
        assert result.primary_score == 0.0
        assert result.primary_metric_name == "bertscore_f1"
        assert result.question_index == 7
