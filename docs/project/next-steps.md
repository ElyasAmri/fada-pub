---
date: 2026-01-27
type: tracker
project: fada-v3
---

## Priority 1: Fine-Tuning Top Models

Fine-tune top-performing zero-shot models on full training set:

- [ ] Fine-tune Qwen3.5-35B-A3B (top zero-shot scorer)
- [ ] Fine-tune gemma-3-12b-it (#2 zero-shot)
- [ ] Fine-tune InternVL3.5-4B (#3 zero-shot)
- [ ] Expand evaluation to 1,894 test images

---

## Priority 2: Deploy Demo

- [ ] Streamlit web interface with best model
- [ ] Model: Use Qwen2.5-VL-7B (81.1% is REAL score)
- [ ] Location: `models/qwen25vl7b_finetuned/final/`

---

## Priority 3: External Benchmark Validation

- [ ] Evaluate on U2-BENCH (7,241 ultrasound cases, 15 anatomy regions)
- [ ] Compare against 23 SOTA models tested in paper
- [ ] Dataset: [DolphinAI/u2-bench](https://huggingface.co/datasets/DolphinAI/u2-bench)
- [ ] Toolkit: [u2-bench-evalkit](https://github.com/dolphin-sound/u2-bench-evalkit)

**Why**: External validation on standardized benchmark, includes fetal imaging

---

## Priority 4: Address Weak Categories

| Category          | MedGemma Score | Issue       |
| ----------------- | -------------- | ----------- |
| Trans-cerebellum  | 58%            | Brain views |
| Trans-thalamic    | 57%            | Brain views |
| Trans-ventricular | 57%            | Brain views |
| NT measurements   | 57%            | Specialized |

Options:

- [ ] Category-specific fine-tuning
- [ ] More training data for weak categories
- [ ] Ensemble approach
- [ ] Test fetal-specific models (FetalCLIP, EchoVLM)

---

## Completed

- [x] Phase 1: 50+ models zero-shot
- [x] Phase 2: Qwen2.5-VL-7B fine-tuned (81.1%, 600 samples)
- [x] Phase 3: MedGemma-27B cloud (78.81%, 709 samples -- proxy scoring against Gemini pseudo-labels, NOT GT sonographer annotations)
- [x] Classification: EfficientNet-B0 (88%, 1,494 samples)
- [x] Infrastructure: RCCG (A100/H100)
- [x] Local testing: All 7 Qwen models with Unsloth

---

## Data Status

| Set   | Size   | Status                    |
| ----- | ------ | ------------------------- |
| Train | 15,231 | Ready                     |
| Val   | 1,894  | Ready                     |
| Test  | 1,894  | **Needs full evaluation** |
| Total | 19,019 | -                         |

---

## Comparison Matrix (What We Know)

| Model            | Method         | Samples | Score  |
| ---------------- | -------------- | ------- | ------ |
| Qwen3.5-35B-A3B  | GT primary     | 1,894   | 36.5%  |
| gemma-3-12b-it   | GT primary     | 1,894   | 36.3%  |
| Qwen2.5-VL-7B FT | Embedding [1]  | 600     | 81.1%  |
| MedGemma-27B     | Embedding [2]  | 709     | 78.81% |
| EfficientNet-B0  | Classification | 1,494   | 88%    |

[1] NOTE: 600-sample subset. Full test set (1,894 images) v3 score is embed_sim=0.5058. The 81.1% figure should be reproduced on the full test set for verification.
[2] NOTE: Phase 1 proxy scoring against Gemini pseudo-labels, NOT GT scoring against sonographer annotations.
