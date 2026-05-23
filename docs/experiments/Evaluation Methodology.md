---
tags: [phase1, methods]
---

**Covers**: Ground truth evaluation methodology (Phase 2+)

## Hardware

### Phase 1 (October 2025)

- **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM)
- **OS**: Windows 11
- **Framework**: PyTorch 2.8.0+cu128, Transformers 4.x

### Phase 2+ (January 2026+)

- **Local GPU**: NVIDIA RTX 5090 (24GB VRAM)
- **Cloud**: RCCG (A100 at $1.35/h, H100 at $1.90/h)

## Ground Truth Evaluation

Uses embedding similarity scoring with sentence-transformers against expert annotations.

### Method

- Compare model VQA response text to ground truth annotation text
- Sentence-transformers embedding similarity
- Per-question scoring (Q1-Q8 independently)
- Per-category scoring (14 anatomical categories)

### Dataset

- **Full dataset**: 19,019 images, 14 classes
- **Test set**: 1,894 images
- **Train set**: 15,231 images
- **Val set**: 1,894 images
- **Questions per image**: 8 (Q1-Q8)

### Verified Results

| Model                      | Score  | Samples | Method                   |
| -------------------------- | ------ | ------- | ------------------------ |
| Qwen2.5-VL-7B (fine-tuned) | 81.1%  | 600     | Embedding similarity [1] |
| MedGemma-27B (cloud API)   | 78.81% | 709     | Embedding similarity [2] |

[1] NOTE: 600-sample subset. Full test set (1,894 images) v3 score is embed_sim=0.5058. The 81.1% figure should be reproduced on the full test set for verification.
[2] NOTE: Phase 1 proxy scoring against Gemini pseudo-labels, NOT GT scoring against sonographer annotations.

### Known Issues

- Brain sub-views (Trans-cerebellum, Trans-thalamic, Trans-ventricular) score ~57-58% consistently
- Q7 has 95% "normal" class imbalance
- Embedding similarity not yet validated against human expert judgment

## Known Limitations

### Image-Level Splitting

Dataset splitting is performed at the image level, not the patient level, because no patient identifiers are available in the dataset. This means images from the same patient may appear in both training and test sets, potentially inflating performance estimates. This is a known limitation that affects all reported scores.

### Partial Credit Scoring

Questions Q2 (Orientation), Q3 (Imaging Plane), Q5 (Gestational Age), Q6 (Image Quality), and Q7 (Normality) use a partial credit scheme where keyword-level agreement scores 0.5 instead of 0 or 1. The 0.5 threshold was chosen as a simple midpoint but has not been subject to sensitivity analysis. Different partial credit values could affect relative model rankings.
