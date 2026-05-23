"""
Unified Vast.ai Management for VLM Testing and Fine-Tuning.

Usage:
    python -m experiments.vastai status
    python -m experiments.vastai test InternVL3-2B --samples 20
    python -m experiments.vastai batch-test Model1 Model2 Model3
    python -m experiments.vastai finetune Qwen2.5-VL-3B --data train.jsonl
    python -m experiments.vastai destroy job-abc123
"""

from .jobs import JobDatabase
from .instance import VastInstance
from .presets import get_preset_for_model, PRESETS

__all__ = ["JobDatabase", "VastInstance", "get_preset_for_model", "PRESETS"]
