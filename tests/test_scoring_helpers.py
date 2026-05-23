"""Tests for scoring pipeline helper functions.

Tests the pure functions used by question scorers: synonym expansion,
keyword extraction, bin indexing, tier extraction, and set F-beta.
"""

import pytest

from experiments.evaluation.question_scorer import (
    _compute_set_f1,
    _expand_with_synonyms,
    _extract_presentation_keyword,
    _extract_plane_keyword,
    _extract_q4_keywords,
    _extract_quality_tier,
    _extract_normality_binary,
    _ga_bin_index,
    detect_question_index,
    Q1_STRUCTURE_SYNONYMS,
)
from experiments.evaluation.config import GA_BINS_ORDERED, QUALITY_TIERS


# ---------------------------------------------------------------------------
# _compute_set_f1
# ---------------------------------------------------------------------------

class TestComputeSetF1:
    def test_both_empty(self):
        p, r, f = _compute_set_f1(set(), set())
        assert (p, r, f) == (1.0, 1.0, 1.0)

    def test_pred_empty(self):
        p, r, f = _compute_set_f1(set(), {"a"})
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_gt_empty(self):
        p, r, f = _compute_set_f1({"a"}, set())
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_perfect_match(self):
        p, r, f = _compute_set_f1({"a", "b"}, {"a", "b"})
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_partial_overlap(self):
        p, r, f = _compute_set_f1({"a", "b"}, {"b", "c"})
        assert p == 0.5
        assert r == 0.5

    def test_no_overlap(self):
        p, r, f = _compute_set_f1({"a"}, {"b"})
        assert (p, r, f) == (0.0, 0.0, 0.0)

    def test_f2_weights_recall(self):
        # pred has 1 of 1 correct, but misses 1 gt item
        _, _, f1 = _compute_set_f1({"a"}, {"a", "b"}, beta=1.0)
        _, _, f2 = _compute_set_f1({"a"}, {"a", "b"}, beta=2.0)
        # F2 should penalize missing recall more
        assert f2 < f1


# ---------------------------------------------------------------------------
# Q1: Synonym expansion
# ---------------------------------------------------------------------------

class TestSynonymExpansion:
    def test_no_synonyms(self):
        result = _expand_with_synonyms({"xyz_unknown"})
        assert result == {"xyz_unknown"}

    def test_abdomen_expands(self):
        result = _expand_with_synonyms({"abdomen"})
        assert "abdominal wall" in result
        assert "belly" in result

    def test_bidirectional_abdomen_belly(self):
        """S2 fix: belly->abdomen and abdomen->belly both work."""
        belly_expanded = _expand_with_synonyms({"belly"})
        abdomen_expanded = _expand_with_synonyms({"abdomen"})
        assert "abdomen" in belly_expanded
        assert "belly" in abdomen_expanded

    def test_bidirectional_all_entries(self):
        """Verify every synonym relationship is bidirectional."""
        for term, synonyms in Q1_STRUCTURE_SYNONYMS.items():
            for syn in synonyms:
                assert syn in Q1_STRUCTURE_SYNONYMS, (
                    f"'{syn}' (synonym of '{term}') has no entry in Q1_STRUCTURE_SYNONYMS"
                )
                assert term in Q1_STRUCTURE_SYNONYMS[syn], (
                    f"'{term}' not in synonyms of '{syn}' (asymmetric)"
                )

    def test_skull_head_group(self):
        skull_exp = _expand_with_synonyms({"skull"})
        assert "head" in skull_exp
        assert "fetal head" in skull_exp

    def test_heart_cardiac(self):
        heart_exp = _expand_with_synonyms({"heart"})
        assert "cardiac structures" in heart_exp
        cardiac_exp = _expand_with_synonyms({"cardiac structures"})
        assert "heart" in cardiac_exp


# ---------------------------------------------------------------------------
# Q2: Presentation keyword extraction
# ---------------------------------------------------------------------------

class TestPresentationKeyword:
    @pytest.mark.parametrize("text,expected", [
        ("cephalic presentation", "cephalic"),
        ("vertex presentation", "cephalic"),
        ("head down position", "cephalic"),
        ("frank breech", "breech"),
        ("breech presentation", "breech"),
        ("feet first", "breech"),
        ("transverse lie", "transverse"),
        ("oblique lie", "oblique"),
        ("longitudinal lie", "longitudinal"),
    ])
    def test_known_presentations(self, text, expected):
        assert _extract_presentation_keyword(text) == expected

    def test_no_match(self):
        assert _extract_presentation_keyword("unknown position") is None

    def test_case_insensitive(self):
        assert _extract_presentation_keyword("CEPHALIC") == "cephalic"


# ---------------------------------------------------------------------------
# Q3: Plane keyword extraction
# ---------------------------------------------------------------------------

class TestPlaneKeyword:
    @pytest.mark.parametrize("text,expected", [
        ("trans-thalamic plane", "trans-thalamic"),
        ("transthalamic view", "trans-thalamic"),
        ("trans-cerebellar plane", "trans-cerebellar"),
        ("sagittal view", "sagittal"),
        ("mid-sagittal plane", "mid-sagittal"),
        ("coronal section", "coronal"),
        ("axial view", "axial"),
        ("transverse section", "axial"),
        ("cross-sectional view", "axial"),
        ("4-chamber view", "4-chamber"),
        ("four chamber heart view", "4-chamber"),
    ])
    def test_known_planes(self, text, expected):
        assert _extract_plane_keyword(text) == expected

    def test_compound_before_simple(self):
        """trans-thalamic should match before transverse/axial."""
        assert _extract_plane_keyword("trans-thalamic axial") == "trans-thalamic"

    def test_no_match(self):
        assert _extract_plane_keyword("some random text") is None


# ---------------------------------------------------------------------------
# Q4: Measurement keyword extraction
# ---------------------------------------------------------------------------

class TestQ4Keywords:
    def test_single_keyword(self):
        result = _extract_q4_keywords("biparietal diameter measurement")
        assert "BPD" in result

    def test_multiple_keywords(self):
        result = _extract_q4_keywords("head circumference and femur length")
        assert "HC" in result
        assert "FL" in result

    def test_abbreviations(self):
        result = _extract_q4_keywords("BPD and HC measurements")
        assert "BPD" in result
        assert "HC" in result

    def test_no_keywords(self):
        result = _extract_q4_keywords("no measurements visible")
        assert len(result) == 0

    def test_case_insensitive(self):
        result = _extract_q4_keywords("ABDOMINAL CIRCUMFERENCE")
        assert "AC" in result


# ---------------------------------------------------------------------------
# Q5: GA bin indexing
# ---------------------------------------------------------------------------

class TestGABinIndex:
    def test_all_bins_have_index(self):
        for i, bin_name in enumerate(GA_BINS_ORDERED):
            assert _ga_bin_index(bin_name) == i

    def test_unknown_bin(self):
        assert _ga_bin_index("unknown") is None

    def test_case_insensitive(self):
        assert _ga_bin_index("8-13 WEEKS") == 0

    def test_no_gap_at_13_15(self):
        """S3 fix: 13-15 weeks bin exists."""
        assert _ga_bin_index("13-15 weeks") is not None
        idx_8_13 = _ga_bin_index("8-13 weeks")
        idx_13_15 = _ga_bin_index("13-15 weeks")
        assert idx_13_15 == idx_8_13 + 1

    def test_terminal_bin_exists(self):
        """S3 fix: 38+ weeks terminal bin exists."""
        assert _ga_bin_index("38+ weeks") is not None
        assert _ga_bin_index("38+ weeks") == len(GA_BINS_ORDERED) - 1

    def test_adjacent_bins_are_contiguous(self):
        """Every consecutive pair of bins should have adjacent indices."""
        for i in range(len(GA_BINS_ORDERED) - 1):
            idx_a = _ga_bin_index(GA_BINS_ORDERED[i])
            idx_b = _ga_bin_index(GA_BINS_ORDERED[i + 1])
            assert idx_b == idx_a + 1


# ---------------------------------------------------------------------------
# Q6: Quality tier extraction
# ---------------------------------------------------------------------------

class TestQualityTier:
    @pytest.mark.parametrize("text,expected", [
        ("good quality", "good"),
        ("high quality image", "good"),
        ("excellent resolution", "good"),
        ("medium quality", "medium"),
        ("acceptable quality", "medium"),
        ("moderate quality", "medium"),
        ("low quality", "low"),
        ("poor image quality", "low"),
        ("suboptimal resolution", "low"),
    ])
    def test_known_tiers(self, text, expected):
        assert _extract_quality_tier(text) == expected

    def test_no_match(self):
        assert _extract_quality_tier("some text without quality words") is None

    def test_sentinel_not_adjacent_to_low(self):
        """S1 fix: unknown tier sentinel should not be adjacent to low=0."""
        sentinel = QUALITY_TIERS.get("nonexistent_tier", -99)
        low_val = QUALITY_TIERS["low"]
        assert abs(sentinel - low_val) > 1, "Sentinel is adjacent to 'low', will cause false partial credit"


# ---------------------------------------------------------------------------
# Q7: Normality binary extraction
# ---------------------------------------------------------------------------

class TestNormalityBinary:
    def test_normal(self):
        assert _extract_normality_binary("normal appearance") == "normal"

    def test_abnormal(self):
        assert _extract_normality_binary("abnormal findings") == "abnormal"

    def test_abnormal_before_normal(self):
        """'abnormal' should be detected even though it contains 'normal'."""
        assert _extract_normality_binary("abnormal") == "abnormal"

    def test_within_normal(self):
        assert _extract_normality_binary("within normal limits") == "normal"

    def test_increased_nt(self):
        assert _extract_normality_binary("increased NT measurement") == "abnormal"

    def test_no_match(self):
        assert _extract_normality_binary("cannot determine from image") is None


# ---------------------------------------------------------------------------
# Question detection
# ---------------------------------------------------------------------------

class TestDetectQuestionIndex:
    def test_all_canonical_questions(self):
        from src.config.questions import QUESTIONS
        for i, q in enumerate(QUESTIONS):
            assert detect_question_index(q) == i

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            detect_question_index("Completely unrelated question text with no match at all xyz")

    def test_partial_match(self):
        idx = detect_question_index("Anatomical Structures: list them")
        assert idx == 0


# ---------------------------------------------------------------------------
# Config consistency
# ---------------------------------------------------------------------------

class TestConfigConsistency:
    def test_quality_tiers_ordered(self):
        assert QUALITY_TIERS["low"] < QUALITY_TIERS["medium"] < QUALITY_TIERS["good"]

    def test_ga_bins_count(self):
        assert len(GA_BINS_ORDERED) == 8

    def test_ga_bins_cover_full_range(self):
        first = GA_BINS_ORDERED[0]
        last = GA_BINS_ORDERED[-1]
        assert "8" in first
        assert "38+" in last
