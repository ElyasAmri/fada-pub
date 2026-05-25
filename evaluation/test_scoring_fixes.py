"""Test different scoring fix variants for Q1-Q3 and compare results.

Runs the top model (gemma-3-12b-it) through each variant and reports scores.
"""
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from src.config.questions import QUESTION_COLUMNS
from src.data.normalize_annotations import AnnotationNormalizer
from experiments.evaluation.question_scorer import (
    MultiMetricScorer, detect_question_index, _compute_set_f1,
    _extract_presentation_keyword, _extract_plane_keyword,
    Q1_STRUCTURE_SYNONYMS, _expand_with_synonyms,
)
from experiments.evaluation.config import ANNOTATIONS_PATH


# -----------------------------------------------------------------------
# Q2 Fix: Scan orientation keywords
# -----------------------------------------------------------------------
def _extract_scan_orientation(text: str) -> Optional[str]:
    """Extract scan orientation type (axial/longitudinal/etc)."""
    tl = text.lower()
    if "axial" in tl or "transverse" in tl or "cross-section" in tl or "cross section" in tl:
        return "axial"
    if "longitudinal" in tl or "sagittal" in tl:
        return "longitudinal"
    if "coronal" in tl:
        return "coronal"
    if "oblique" in tl:
        return "oblique"
    return None


# -----------------------------------------------------------------------
# Q3 Fix: Better plane matching
# -----------------------------------------------------------------------
def _extract_plane_keyword_v2(text: str) -> Optional[str]:
    """Extract plane keyword with broader synonyms."""
    tl = text.lower()
    # Specific compounds first
    for compound, label in [
        ("trans-thalamic", "axial-brain"), ("transthalamic", "axial-brain"),
        ("trans-cerebellar", "axial-brain"), ("transcerebellar", "axial-brain"),
        ("trans-ventricular", "axial-brain"), ("transventricular", "axial-brain"),
        ("trans-abdominal", "axial-abdomen"), ("transabdominal", "axial-abdomen"),
    ]:
        if compound in tl:
            return label
    if "4-chamber" in tl or "4 chamber" in tl or "four chamber" in tl:
        return "4-chamber"
    if "mid-sagittal" in tl or "mid sagittal" in tl or "midsagittal" in tl:
        return "mid-sagittal"
    # Fix GT misspellings
    if "mid sagital" in tl or "mid dagital" in tl or "mid saggital" in tl or "midsagital" in tl:
        return "mid-sagittal"
    if "para-sagittal" in tl or "parasagittal" in tl:
        return "para-sagittal"
    if "sagittal" in tl or "sagital" in tl or "saggital" in tl:
        return "sagittal"
    if "coronal" in tl:
        return "coronal"
    # Group transverse/axial/cross-sectional together
    if "transverse" in tl or "axial" in tl or "cross-sectional" in tl or "cross sectional" in tl:
        return "axial"
    if "longitudinal" in tl:
        return "longitudinal"
    if "lateral" in tl:
        return "lateral"
    return None


def _strip_model_preamble(text: str) -> str:
    """Strip common VLM preamble from predictions."""
    # Remove "Okay, let's..." type preambles
    patterns = [
        r"^(?:Okay|Ok|Sure|Certainly|Here)[,.]?\s*(?:let's|let me|I'll|I will).*?(?:\n\n|\*\*)",
        r"^\*\*.*?\*\*\s*\n\n",
    ]
    result = text
    for p in patterns:
        result = re.sub(p, "", result, count=1, flags=re.DOTALL)
    return result.strip()


# -----------------------------------------------------------------------
# Main comparison
# -----------------------------------------------------------------------
def main():
    # Load predictions
    preds = []
    with open("experiments/rccg/results/predictions_google_gemma-3-12b-it.jsonl") as f:
        for line in f:
            preds.append(json.loads(line))

    # Load GT
    df = pd.read_excel(ANNOTATIONS_PATH)
    gt_lookup = {}
    for _, row in df.iterrows():
        folder = str(row["Folder Name"]).strip()
        image = str(row["Image Name"]).strip()
        answers = {}
        for col in QUESTION_COLUMNS:
            if col in df.columns:
                val = row[col]
                answers[col] = str(val).strip() if pd.notna(val) else ""
        gt_lookup[(folder, image)] = answers

    normalizer = AnnotationNormalizer()

    # Match predictions to GT
    q1_pairs = []  # (pred_raw, gt_text)
    q2_pairs = []
    q3_pairs = []

    for pred in preds:
        img_path = pred["image_path"].replace("\\", "/")
        parts = img_path.split("/")
        folder, image = parts[-2], parts[-1]
        gt = gt_lookup.get((folder, image))
        if gt is None:
            continue
        try:
            q_idx = detect_question_index(pred["question"])
        except ValueError:
            continue
        gt_text = gt.get(QUESTION_COLUMNS[q_idx], "")
        if not gt_text:
            continue

        if q_idx == 0:
            q1_pairs.append((pred["prediction"], gt_text))
        elif q_idx == 1:
            q2_pairs.append((pred["prediction"], gt_text))
        elif q_idx == 2:
            q3_pairs.append((pred["prediction"], gt_text))

    print(f"Matched: Q1={len(q1_pairs)} Q2={len(q2_pairs)} Q3={len(q3_pairs)}")
    print()

    # ===== Q1 VARIANTS =====
    print("=" * 70)
    print("Q1: Anatomical Structures")
    print("=" * 70)

    def score_q1(pairs, use_synonyms=False, beta=1.0):
        scores = []
        for pred_raw, gt_text in pairs:
            norm_pred = normalizer.normalize_single(QUESTION_COLUMNS[0], pred_raw)
            pred_set = {s.strip().lower() for s in norm_pred.split(",") if s.strip()}
            gt_set = {s.strip().lower() for s in gt_text.split(",") if s.strip()}

            if use_synonyms:
                pred_expanded = _expand_with_synonyms(pred_set)
                gt_expanded = _expand_with_synonyms(gt_set)
                tp = len(pred_expanded & gt_expanded)
                precision = tp / len(pred_set) if pred_set else 0
                recall = tp / len(gt_set) if gt_set else 0
            else:
                tp = len(pred_set & gt_set)
                precision = tp / len(pred_set) if pred_set else 0
                recall = tp / len(gt_set) if gt_set else 0

            if precision + recall > 0:
                f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
            else:
                f = 0.0
            scores.append(f)
        return np.mean(scores)

    v_baseline = score_q1(q1_pairs, use_synonyms=False, beta=1.0)
    v_synonyms = score_q1(q1_pairs, use_synonyms=True, beta=1.0)
    v_recall = score_q1(q1_pairs, use_synonyms=False, beta=2.0)
    v_both = score_q1(q1_pairs, use_synonyms=True, beta=2.0)

    print(f"  Baseline (set F1):           {v_baseline:.4f}")
    print(f"  A: + synonyms:               {v_synonyms:.4f} ({v_synonyms-v_baseline:+.4f})")
    print(f"  B: + recall-weighted (F2):   {v_recall:.4f} ({v_recall-v_baseline:+.4f})")
    print(f"  C: + synonyms + F2:          {v_both:.4f} ({v_both-v_baseline:+.4f})")
    print()

    # ===== Q2 VARIANTS =====
    print("=" * 70)
    print("Q2: Fetal Orientation")
    print("=" * 70)

    def score_q2(pairs, mode="baseline"):
        scores = []
        for pred_raw, gt_text in pairs:
            norm_pred = normalizer.normalize_single(QUESTION_COLUMNS[1], pred_raw)

            if mode == "baseline":
                if norm_pred.strip().lower() == gt_text.strip().lower():
                    score = 1.0
                else:
                    pred_kw = _extract_presentation_keyword(norm_pred)
                    gt_kw = _extract_presentation_keyword(gt_text)
                    score = 0.5 if (pred_kw and gt_kw and pred_kw == gt_kw) else 0.0

            elif mode == "scan_orientation":
                # Add scan orientation as another matching layer
                if norm_pred.strip().lower() == gt_text.strip().lower():
                    score = 1.0
                else:
                    # Try presentation keywords first
                    pred_kw = _extract_presentation_keyword(norm_pred)
                    gt_kw = _extract_presentation_keyword(gt_text)
                    if pred_kw and gt_kw and pred_kw == gt_kw:
                        score = 0.5
                    else:
                        # Try scan orientation keywords
                        pred_scan = _extract_scan_orientation(norm_pred)
                        gt_scan = _extract_scan_orientation(gt_text)
                        score = 0.5 if (pred_scan and gt_scan and pred_scan == gt_scan) else 0.0

            elif mode == "combined_keywords":
                # Extract ALL keywords (presentation + scan orientation) and match any overlap
                if norm_pred.strip().lower() == gt_text.strip().lower():
                    score = 1.0
                else:
                    pred_kws = set()
                    gt_kws = set()
                    pk = _extract_presentation_keyword(norm_pred)
                    gk = _extract_presentation_keyword(gt_text)
                    if pk: pred_kws.add(pk)
                    if gk: gt_kws.add(gk)
                    ps = _extract_scan_orientation(norm_pred)
                    gs = _extract_scan_orientation(gt_text)
                    if ps: pred_kws.add(ps)
                    if gs: gt_kws.add(gs)
                    score = 0.5 if (pred_kws & gt_kws) else 0.0

            scores.append(score)
        return np.mean(scores)

    v2_baseline = score_q2(q2_pairs, "baseline")
    v2_scan = score_q2(q2_pairs, "scan_orientation")
    v2_combined = score_q2(q2_pairs, "combined_keywords")

    print(f"  Baseline (presentation kw):  {v2_baseline:.4f}")
    print(f"  A: + scan orientation kw:    {v2_scan:.4f} ({v2_scan-v2_baseline:+.4f})")
    print(f"  B: combined keywords:        {v2_combined:.4f} ({v2_combined-v2_baseline:+.4f})")
    print()

    # ===== Q3 VARIANTS =====
    print("=" * 70)
    print("Q3: Imaging Plane")
    print("=" * 70)

    def score_q3(pairs, use_v2_keywords=False, strip_preamble=False):
        scores = []
        for pred_raw, gt_text in pairs:
            if strip_preamble:
                pred_input = _strip_model_preamble(pred_raw)
            else:
                pred_input = pred_raw
            norm_pred = normalizer.normalize_single(QUESTION_COLUMNS[2], pred_input)

            if norm_pred.strip().lower() == gt_text.strip().lower():
                score = 1.0
            else:
                if use_v2_keywords:
                    pred_kw = _extract_plane_keyword_v2(norm_pred)
                    gt_kw = _extract_plane_keyword_v2(gt_text)
                else:
                    pred_kw = _extract_plane_keyword(norm_pred)
                    gt_kw = _extract_plane_keyword(gt_text)
                score = 0.5 if (pred_kw and gt_kw and pred_kw == gt_kw) else 0.0

            scores.append(score)
        return np.mean(scores)

    v3_baseline = score_q3(q3_pairs, use_v2_keywords=False, strip_preamble=False)
    v3_keywords = score_q3(q3_pairs, use_v2_keywords=True, strip_preamble=False)
    v3_strip = score_q3(q3_pairs, use_v2_keywords=False, strip_preamble=True)
    v3_both = score_q3(q3_pairs, use_v2_keywords=True, strip_preamble=True)

    print(f"  Baseline (v1 keywords):      {v3_baseline:.4f}")
    print(f"  A: v2 keywords + synonyms:   {v3_keywords:.4f} ({v3_keywords-v3_baseline:+.4f})")
    print(f"  B: strip model preamble:     {v3_strip:.4f} ({v3_strip-v3_baseline:+.4f})")
    print(f"  C: both A+B:                 {v3_both:.4f} ({v3_both-v3_baseline:+.4f})")


if __name__ == "__main__":
    main()
