"""Tests for configuration constants and data integrity.

Validates that config files are internally consistent and match
the actual dataset structure.
"""

import pytest

from src.config.constants import (
    CLASSES,
    DISPLAY_NAMES,
    CLASS_DESCRIPTIONS,
    VQA_MODEL_MAPPING,
)
from src.config.questions import QUESTIONS, QUESTION_COLUMNS, QUESTION_SHORT_NAMES
from experiments.evaluation.config import CATEGORIES, GA_BINS_ORDERED, QUALITY_TIERS


# ---------------------------------------------------------------------------
# Constants consistency
# ---------------------------------------------------------------------------

class TestClassConstants:
    def test_classes_count(self):
        """Q2 fix: should have 14 classes including CRL-View and NT-View."""
        assert len(CLASSES) == 14

    def test_crl_view_present(self):
        assert "CRL-View" in CLASSES

    def test_nt_view_present(self):
        assert "NT-View" in CLASSES

    def test_classes_sorted(self):
        assert CLASSES == sorted(CLASSES)

    def test_display_names_complete(self):
        for cls in CLASSES:
            assert cls in DISPLAY_NAMES, f"Missing DISPLAY_NAMES entry for '{cls}'"

    def test_descriptions_complete(self):
        for cls in CLASSES:
            assert cls in CLASS_DESCRIPTIONS, f"Missing CLASS_DESCRIPTIONS entry for '{cls}'"

    def test_vqa_mapping_complete(self):
        for cls in CLASSES:
            assert cls in VQA_MODEL_MAPPING, f"Missing VQA_MODEL_MAPPING entry for '{cls}'"

    def test_classes_match_categories(self):
        """CLASSES in constants.py should match CATEGORIES in eval config."""
        assert set(CLASSES) == set(CATEGORIES)


class TestQuestionConstants:
    def test_8_questions(self):
        assert len(QUESTIONS) == 8

    def test_8_columns(self):
        assert len(QUESTION_COLUMNS) == 8

    def test_8_short_names(self):
        assert len(QUESTION_SHORT_NAMES) == 8

    def test_columns_have_q_prefix(self):
        for i, col in enumerate(QUESTION_COLUMNS):
            assert col.startswith(f"Q{i+1}:"), f"Column {col} doesn't start with Q{i+1}:"


class TestEvalConfig:
    def test_categories_14(self):
        assert len(CATEGORIES) == 14

    def test_ga_bins_8(self):
        assert len(GA_BINS_ORDERED) == 8

    def test_ga_bins_no_gaps(self):
        """S3 fix: no gap between 8-13 and 15-20 (13-15 fills it)."""
        bin_names = [b.lower() for b in GA_BINS_ORDERED]
        assert "13-15 weeks" in bin_names

    def test_quality_tiers_3(self):
        assert len(QUALITY_TIERS) == 3
        assert set(QUALITY_TIERS.keys()) == {"good", "medium", "low"}
