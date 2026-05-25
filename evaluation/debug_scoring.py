"""Debug script to compare GT vs predictions for Q1-Q3."""
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
from src.data.normalize_annotations import AnnotationNormalizer
from experiments.evaluation.question_scorer import (
    _extract_presentation_keyword, _extract_plane_keyword, _compute_set_f1
)

# Load predictions
preds = []
with open("experiments/rccg/results/predictions_google_gemma-3-12b-it.jsonl") as f:
    for line in f:
        preds.append(json.loads(line))

# Load GT
df = pd.read_excel("data/Fetal Ultrasound Annotations Normalized.xlsx")

normalizer = AnnotationNormalizer()

# Group preds by image (every 8 consecutive = 1 image)
q_names = [
    "Q1: Anatomical Structures",
    "Q2: Fetal Orientation",
    "Q3: Imaging Plane",
]

# Sample 5 images
sampled = 0
for img_start in range(0, len(preds), 8):
    if sampled >= 5:
        break

    img_preds = preds[img_start:img_start+8]
    img_path = img_preds[0]["image_path"].replace("\\", "/")
    parts = img_path.split("/")
    folder = parts[-2]
    image = parts[-1]

    match = df[
        (df["Folder Name"].astype(str).str.strip() == folder) &
        (df["Image Name"].astype(str).str.strip() == image)
    ]
    if len(match) == 0:
        continue

    gt_row = match.iloc[0]
    print(f"\n{'='*70}")
    print(f"Image: {folder}/{image}")
    print(f"{'='*70}")

    for qi in range(3):
        col = q_names[qi]
        gt_text = str(gt_row[col]).strip()
        pred_text = img_preds[qi]["prediction"]
        norm_pred = normalizer.normalize_single(col, pred_text)

        print(f"\n--- {col} ---")
        print(f"GT:        {gt_text[:200]}")
        print(f"PRED(raw): {pred_text[:200]}")
        print(f"PRED(norm):{norm_pred[:200]}")

        if qi == 0:  # Q1 set F1
            pred_set = {s.strip().lower() for s in norm_pred.split(",") if s.strip()}
            gt_set = {s.strip().lower() for s in gt_text.split(",") if s.strip()}
            p, r, f1 = _compute_set_f1(pred_set, gt_set)
            print(f"  pred_set: {sorted(pred_set)}")
            print(f"  gt_set:   {sorted(gt_set)}")
            print(f"  F1={f1:.3f} P={p:.3f} R={r:.3f}")
        elif qi == 1:  # Q2 orientation
            pred_kw = _extract_presentation_keyword(norm_pred)
            gt_kw = _extract_presentation_keyword(gt_text)
            exact = norm_pred.strip().lower() == gt_text.strip().lower()
            print(f"  exact_match: {exact}")
            print(f"  pred_kw: {pred_kw}, gt_kw: {gt_kw}, kw_match: {pred_kw == gt_kw if pred_kw and gt_kw else 'N/A'}")
        elif qi == 2:  # Q3 plane
            pred_kw = _extract_plane_keyword(norm_pred)
            gt_kw = _extract_plane_keyword(gt_text)
            exact = norm_pred.strip().lower() == gt_text.strip().lower()
            print(f"  exact_match: {exact}")
            print(f"  pred_kw: {pred_kw}, gt_kw: {gt_kw}, kw_match: {pred_kw == gt_kw if pred_kw and gt_kw else 'N/A'}")

    sampled += 1
