"""Tests for evaluation helper functions outside the scorer."""

import pytest

from experiments.evaluation.evaluate_vlm import extract_category_from_path
from experiments.evaluation.question_scorer import MultiMetricScorer


class TestExtractCategoryFromPath:
    def test_standard_path(self):
        result = extract_category_from_path("data/Fetal Ultrasound/Abdomen/img.png")
        assert result == "Abdomen"

    def test_windows_path(self):
        result = extract_category_from_path("data\\Fetal Ultrasound\\Thorax\\img.png")
        assert result == "Thorax"

    def test_no_fetal_ultrasound(self):
        result = extract_category_from_path("some/other/path/img.png")
        assert result == "Unknown"

    def test_fetal_ultrasound_at_end(self):
        """H-I5 fix: should not IndexError when Fetal Ultrasound is last."""
        result = extract_category_from_path("data/Fetal Ultrasound")
        assert result == "Unknown"


class TestExtractImageKey:
    def test_unix_path(self):
        folder, image = MultiMetricScorer._extract_image_key("Abdomen/img_001.png")
        assert folder == "Abdomen"
        assert image == "img_001.png"

    def test_windows_path(self):
        folder, image = MultiMetricScorer._extract_image_key("Abdomen\\img_001.png")
        assert folder == "Abdomen"
        assert image == "img_001.png"

    def test_deep_path(self):
        folder, image = MultiMetricScorer._extract_image_key(
            "data/Fetal Ultrasound/Thorax/thorax_001.png"
        )
        assert folder == "Thorax"
        assert image == "thorax_001.png"

    def test_single_component(self):
        folder, image = MultiMetricScorer._extract_image_key("img.png")
        assert folder == ""
        assert image == "img.png"
