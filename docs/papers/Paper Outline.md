---
tags: [output, writing]
closes: "#24"
---

> NOTE: Performance figures in this outline should be verified against the latest results in docs/experiments/results-summary.md

# Comparative Evaluation of Vision-Language Models for Fetal Ultrasound Visual Question Answering

## Abstract

We present the largest benchmark of vision-language models (VLMs) for fetal ultrasound image analysis, evaluating 54 models across 14 anatomical categories using 8 standardized clinical questions per image. Using a multi-metric scoring pipeline combining per-question specialized metrics with embedding similarity against sonographer ground truth, we find that google/gemma-3-12b-it achieves the highest zero-shot score (0.3596), followed by Qwen3.5-35B-A3B (0.3566) and InternVL3_5-4B (0.3492). Our analysis reveals that model size does not predict performance, current-generation models consistently outperform predecessors, and medical-specialized models underperform general-purpose alternatives on this domain. All models plateau at ~36% primary score, suggesting a ceiling for zero-shot evaluation that fine-tuning may overcome. Research prototype -- not for clinical use.

## 1. Introduction

- Motivation: AI-assisted prenatal screening via automated ultrasound interpretation
- Gap: No large-scale VLM benchmark exists for fetal ultrasound VQA
- Contribution:
  - Largest fetal ultrasound VLM benchmark (54 models, 19K images, 14 categories)
  - Multi-metric scoring pipeline with per-question specialized evaluation
  - Analysis of model families, size scaling, and architectural choices (MoE, thinking modes)
  - Evidence that general-purpose VLMs outperform medical-specialized models on this task

## 2. Related Work

- Deep learning for ultrasound analysis (FetalCLIP, fetal plane classification literature)
- Vision-language models: evolution from CLIP to current-gen multimodal LLMs
- Medical VQA: PathVQA, SLAKE, VQA-RAD -- none focused on fetal ultrasound
- U2-BENCH (ICLR 2026): ultrasound understanding benchmark with 17/21 leaderboard models in our test list

## 3. Dataset

### 3.1 Image Corpus

- 19,019 fetal ultrasound images across 14 anatomical categories
- Stratified 80/10/10 split: 15,231 train / 1,894 val / 1,894 test (seed=42)
- Categories: Abdomen, Aorta, CRL-View, Cervical, Cervix, Femur, NT-View, Non_standard_NT, Public_Symphysis, Standard_NT, Thorax, Trans-cerebellum, Trans-thalamic, Trans-ventricular

### 3.2 Ground Truth Annotations

- 18,936 annotated images by qualified sonographer
- 8 clinical questions per image (Q1-Q8):
  1. Anatomical Structures Identification
  2. Fetal Orientation Assessment
  3. Imaging Plane Evaluation
  4. Biometric Measurements Analysis
  5. Gestational Age Estimation
  6. Image Quality Assessment
  7. Normality/Abnormality Determination
  8. Clinical Recommendations

## 4. Methodology

### 4.1 Model Selection

- 54 models from 13 families spanning 256M to 35B parameters (see models-to-test.md for current count)
- Current-gen: Gemma 3, Qwen3.5, Qwen3-VL, InternVL3.5, MiniCPM-V-4.5, Kimi-VL
- Previous-gen: Qwen2.5-VL, Qwen2-VL, InternVL2, Gemma 3n
- Medical: MedGemma, MedVLM-R1
- Other: Mistral-Small, Aria, SmolVLM, Phi, LLaVA-OneVision, Molmo, Pixtral, Idefics3

### 4.2 Evaluation Framework

- vLLM inference engine with OpenAI-compatible API
- 8 RCCG H100 machines for parallel evaluation
- Checkpoint/resume with fault tolerance
- Each model processes all 1,894 test images x 8 questions = 15,152 responses

### 4.3 Scoring Pipeline v2

- Per-question specialized metrics:
  - Q1: set_f1 (anatomical structure set overlap)
  - Q2, Q3: relaxed_accuracy (keyword matching with synonym expansion)
  - Q4: keyword_f1 (biometric measurement term matching)
  - Q5: exact_bin_match (gestational age trimester binning)
  - Q6: exact_tier_match (image quality tier matching)
  - Q7: exact_match_with_binary (normal/abnormal classification)
  - Q8: bertscore_f1 (semantic similarity of clinical recommendations)
- Embedding similarity: sentence-transformers cosine similarity vs ground truth
- Primary score: weighted average of per-question primary metrics

## 5. Results

### 5.1 Overall Rankings

- Top 5: gemma-3-12b-it (0.3596), Qwen3.5-35B-A3B (0.3566), InternVL3_5-4B (0.3492), MiniCPM-V-4_5 (0.3403), Qwen3.5-4B (0.3397)
- Score range: 0.12 to 0.36 across 54 models (see results-summary.md for current rankings)
- Strong clustering in 0.26-0.36 band (ranks 1-36)

### 5.2 Per-Question Analysis

- Q6 (image quality) and Q8 (recommendations) effectively solved at >76%
- Q5 (gestational age) at 45% -- models can estimate trimester
- Q1-Q3 show metric divergence: low primary but high embedding similarity
- Q4 (biometric measurements) hardest on both metrics

### 5.3 Per-Category Analysis

- Brain views (Trans-thalamic, Trans-ventricular) score highest -- distinctive landmarks
- Aorta consistently hardest -- small, technically demanding view
- Non_standard_NT second-hardest -- inherently ambiguous

### 5.4 Model Family Comparison

- Table comparing 13 families by best score, average score, model count
- Gemma 3, Qwen3.5, InternVL3.5 lead
- Clear generational improvements within families

## 6. Discussion

### 6.1 Size Does Not Predict Performance

- Inverse scaling in multiple families: 12B > 27B (Gemma), 4B > 8B (InternVL3.5), 2B > 8B (Qwen3-VL)
- Possible explanations: instruction tuning quality, vision encoder architecture, training data composition

### 6.2 Medical Models Underperform General Models

- MedGemma-4b (0.2994) < gemma-3-4b-it (0.3249) at same parameter count
- Medical pre-training may be too broad (radiology/pathology focus) for niche fetal ultrasound domain
- General language ability may matter more for structured text generation in our VQA format

### 6.3 MoE Architectures

- Qwen3.5-35B-A3B (3B active) achieves rank 2 -- competitive with dense 12B
- Kimi-VL-A3B (3B active) in top 10
- Favorable accuracy/compute tradeoff for potential deployment

### 6.4 Thinking Modes Hurt Performance

- Instruct variants consistently outperform Thinking variants (Kimi, Qwen3-VL)
- Structured VQA benefits from direct, concise answers
- Chain-of-thought reasoning adds noise without improving accuracy

### 6.5 Score Ceiling Analysis

- All models plateau at ~36% primary score
- Contributing factors: metric strictness, task difficulty, zero-shot limitation
- Fine-tuning expected to break this ceiling (prior work: Qwen2.5-VL-7B fine-tuned reached 81.1% embed sim on 600 samples -- NOTE: this figure is from a 600-sample subset; the full test set v3 score is embed_sim=0.5058; the 81.1% figure should be reproduced on the full test set for verification)

### 6.6 Limitations

- Single evaluation framework -- alternative scoring pipelines may yield different rankings
- Zero-shot only -- does not capture fine-tuning potential
- Embedding similarity not validated against human expert judgment
- English-only evaluation

## 7. Conclusion

- 42-model benchmark establishes baselines for fetal ultrasound VQA
- General-purpose current-gen VLMs are the strongest zero-shot performers
- Model size, medical specialization, and reasoning modes do not confer advantages
- MoE architectures offer the best efficiency/accuracy tradeoff
- Future work: fine-tuning top performers, human evaluation correlation, multi-language assessment

## References

- FetalCLIP (MBZUAI)
- U2-BENCH (ICLR 2026)
- Sentence-transformers
- BERTScore
- vLLM
- Model citations for all 54 models evaluated

## Status

- [x] Phase 4 zero-shot benchmark complete (54 models, see models-to-test.md for current count)
- [ ] Fine-tuning experiments
- [ ] Human evaluation correlation study
- [ ] Full paper draft
