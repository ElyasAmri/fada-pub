# VLM Inference Output Analysis

**Date**: 2026-03-05
**Analyst**: Scientist Agent
**Dataset**: 15,080 matched predictions per model, 14 anatomical categories, 8 clinical questions
**Models analyzed**: 15 working models (3 excluded due to inference errors or scoring anomalies)

---

## Executive Summary

Analysis of 15 VLM inference outputs reveals that **scoring methodology issues account for more performance loss than actual model deficiencies**. The top model (Kimi-VL-A3B-Instruct) scores 0.2785 overall, but a single bug in Q5 normalization alone could improve it to 0.3316 (+19%). Questions Q2, Q3, and Q5 are near-zero across all models due to a fundamental mismatch between verbose model outputs and rigid scoring pipelines designed for concise annotations. The findings suggest that a combination of **output format constraints** (via prompt engineering) and **scoring pipeline fixes** (regex extraction improvements) could yield 30-50% overall score improvements without changing any model weights.

---

## 1. Model Rankings

| Rank | Model                | Primary Score | Embedding Similarity |
| ---- | -------------------- | ------------- | -------------------- |
| 1    | Kimi-VL-A3B-Instruct | 0.2785        | 0.4056               |
| 2    | Qwen2-VL-2B-Instruct | 0.2782        | 0.3720               |
| 3    | gemma-3-27b-it       | 0.2739        | 0.3687               |
| 4    | Qwen3.5-0.8B         | 0.2739        | 0.3771               |
| 5    | Qwen3.5-2B           | 0.2733        | 0.3746               |
| 6    | Qwen3-VL-2B          | 0.2709        | 0.4012               |
| 7    | Kimi-VL-A3B-Thinking | 0.2690        | 0.3718               |
| 8    | Idefics3-8B-Llama3   | 0.2653        | 0.3784               |
| 9    | InternVL2-8B         | 0.2614        | 0.3769               |
| 10   | InternVL2-2B         | 0.2546        | 0.3787               |
| 11   | InternVL2-1B         | 0.2538        | 0.3605               |
| 12   | MiniCPM-o-2_6        | 0.2504        | 0.3919               |
| 13   | Phi-3.5-vision       | 0.2482        | 0.3406               |
| 14   | Qwen2-VL-7B-Instruct | 0.2396        | 0.3896               |
| 15   | Qwen2.5-Omni-7B      | 0.2343        | 0.4013               |

**Excluded models**: pixtral-12b and InternVL2.5-4B (100% vLLM API errors -- all predictions are error strings), Aria (INF in Q8 BERTScore -- likely corrupt BERTScore computation).

**Key observation**: The score spread is remarkably narrow (0.2343 to 0.2785), a range of only 0.044. This suggests the scoring pipeline's structural limitations dominate over actual model quality differences. Note that Qwen2.5-Omni-7B (7B parameters) scores _lower_ than Qwen3.5-0.8B (0.8B parameters), and primary scores have weak correlation with embedding similarity (the highest embed score models are not the highest primary score models).

---

## 2. Per-Question Performance

### Question Difficulty Ranking (Hardest to Easiest)

| Rank | Question                     | Metric             | Mean Score | Std    | CV   |
| ---- | ---------------------------- | ------------------ | ---------- | ------ | ---- |
| 1    | Q5: Gestational Age          | exact_bin_match    | 0.0136     | 0.0509 | 3.74 |
| 2    | Q3: Imaging Plane            | relaxed_accuracy   | 0.0312     | 0.0367 | 1.17 |
| 3    | Q2: Fetal Orientation        | relaxed_accuracy   | 0.0342     | 0.0312 | 0.91 |
| 4    | Q1: Anatomical Structures    | set_f1             | 0.1072     | 0.0189 | 0.18 |
| 5    | Q4: Biometric Measurements   | keyword_f1         | 0.1350     | 0.0260 | 0.19 |
| 6    | Q7: Normality Assessment     | exact_match_binary | 0.2967     | 0.0975 | 0.33 |
| 7    | Q6: Image Quality            | exact_tier_match   | 0.7007     | 0.1198 | 0.17 |
| 8    | Q8: Clinical Recommendations | bertscore_f1       | 0.7749     | 0.0140 | 0.02 |

### Critical Findings

**Q5 (Gestational Age) -- Score: 0.01 -- SCORING BUG IDENTIFIED**

14 of 15 models score exactly 0.0 on Q5. The root cause is a bug in `_normalize_q5_regex()` in `src/data/normalize_annotations.py`. The function uses `re.match()` (matches only at string start) instead of `re.search()` (matches anywhere). Models produce verbose answers like:

> "The gestational age of the fetus appears to be around 20-24 weeks. This estimation is based on the visible development..."

The regex `_Q5_RANGE_PATTERN.match(clean)` tries to match "20-24 weeks" at position 0 of the lowercased string, which starts with "the gestational age..." -- so it always fails. Only Idefics3 (which produces extremely terse outputs like "20 weeks.") scores above zero (0.2042).

**Quantified impact**: Using `re.search()` instead of `re.match()` on Kimi-VL-A3B-Instruct predictions:

- Exact bin matches: 574/1885 (30.5%)
- Adjacent bin (partial credit): 455/1885 (24.1%)
- Improved Q5 score: 0.4252 (from 0.0000)
- Overall score improvement: +0.0531 (from 0.2785 to 0.3316, a 19% lift)

**Q2 (Fetal Orientation) -- Score: 0.03 -- VOCABULARY MISMATCH**

The scorer uses `relaxed_accuracy`: exact string match gets 1.0, keyword match (cephalic/breech/transverse/longitudinal) gets 0.5. Two problems:

1. **53.3% of GT answers contain none of the four scoring keywords** (e.g., "Axial upper abdomen, vertebral column to left", "Sagittal plane, skull to right", "Fetal head to left")
2. Models use different vocabulary than GT. The top model uses "head-down" (69.6% of predictions) and "vertex" (53.3%) -- neither is a scoring keyword. Even when models use "cephalic" (42.2%), the exact string match fails because GT is verbose (e.g., "Cephalic transverse, occiput to right").

**Q3 (Imaging Plane) -- Score: 0.03 -- SIMILAR VOCABULARY MISMATCH**

Same issue as Q2. GT says "Transverse trans-abdominal plane" while models say "transverse plane" or "coronal plane" in verbose context. The keyword extractor finds a keyword but the GT comparison may extract a different level of specificity.

**Q6 (Image Quality) -- Score: 0.70 -- BEST PERFORMING**

Q6 works well because the tier extraction (good/medium/low) is robust and GT is dominated by one class: "Good image quality" = 81.6% of GT. Models that tend toward positive quality assessments score well here due to class imbalance. The adjacency scoring (+0.5 for one tier off) also helps.

**Q8 (Clinical Recommendations) -- Score: 0.77 -- CONSISTENT ACROSS MODELS**

BERTScore F1 is inherently tolerant of paraphrasing, giving high scores even for loosely similar recommendations. The coefficient of variation is only 0.02 -- all models score nearly identically, making Q8 essentially non-discriminative.

---

## 3. Per-Category Performance

### Category Difficulty Ranking

| Rank | Category                    | Mean Primary | n (samples) |
| ---- | --------------------------- | ------------ | ----------- |
| 1    | Aorta                       | 0.1976       | 1,040       |
| 2    | Non_standard_NT             | 0.2067       | 376         |
| 3    | Public_Symphysis_fetal_head | 0.2370       | 1,080       |
| 4    | Standard_NT                 | 0.2401       | 1,192       |
| 5    | Cervix                      | 0.2466       | 1,288       |
| 6    | Trans-cerebellum            | 0.2538       | 544         |
| 7    | Cervical                    | 0.2556       | 400         |
| 8    | Thorax                      | 0.2568       | 1,432       |
| 9    | Abdomen                     | 0.2604       | 1,936       |
| 10   | NT-View                     | 0.2727       | 1,616       |
| 11   | Trans-ventricular           | 0.2951       | 464         |
| 12   | CRL-View                    | 0.2954       | 1,576       |
| 13   | Trans-thalamic              | 0.2982       | 1,232       |
| 14   | Femur                       | 0.3104       | 904         |

### Why Aorta is Hardest

Aorta has the lowest scores because:

1. **Q1 (Structures) = 0.0 for most models**: Models struggle to identify aorta-specific anatomy (aortic arch, descending aorta, valve structures). GT uses precise terms like "aortic arch" while models produce generic descriptions.
2. **Q2 (Orientation) = 0.0 across the board**: GT says "Coronal view of aorta" which contains no orientation keyword (cephalic/breech/transverse/longitudinal).
3. **Q4 (Measurements) = 0.0**: Aorta images have specialized measurements not in the keyword dictionary (or no measurements applicable).
4. **Q6 (Quality) drops to 0.30-0.34**: Aorta images are harder to interpret, and models correctly flag lower quality, but GT still says "Good" for many.

### Why Femur/Trans-thalamic Score Highest

These categories benefit from strong Q4 (Biometric Measurements) scores:

- Femur Q4 = 0.36-0.69 (keyword "FL" / "femur length" is easy to match)
- Trans-thalamic Q4 = 0.58-0.73 (keyword "BPD" / "biparietal diameter" frequently matched)

---

## 4. Error Pattern Analysis

### Top Model (Kimi-VL-A3B-Instruct) vs Bottom Model (Qwen2.5-Omni-7B)

**Response length**:

- Kimi: 508 chars mean (concise, focused)
- Qwen2.5-Omni: 754 chars mean (more hedging, disclaimers)
- gemma-3-27b: 2121 chars mean (extremely verbose with markdown formatting)
- Kimi-VL-A3B-Thinking: 3886 chars mean (includes chain-of-thought)

**Verbosity correlation**: Weak positive correlation with primary score (r = 0.37), weak negative with embedding similarity (r = -0.10). Moderate-length responses perform best -- too short misses detail, too long dilutes relevant information.

### Specific Error Patterns

**Q1 (Anatomical Structures) example**:

- GT: "IVC, aorta, liver, lumbar vertebrae, stomach, umbilical vein"
- Kimi: "The image shows a cross-sectional view of a fetus...head, brain, and spinal cord...ventricles"
- Problem: Model describes generic anatomy (head, brain) instead of the specific structures visible in an _abdominal_ cross-section. Indicates models lack category-specific anatomical knowledge.

**Q2 (Fetal Orientation) example**:

- GT: "Cephalic transverse, occiput to right"
- Kimi: "The fetus is in a head-down position...vertex presentation"
- Problem: Both describe roughly the same thing, but scorer cannot match "head-down/vertex" to "cephalic transverse."

**Q6 (Image Quality) example**:

- GT: "Good image quality"
- Kimi: "moderate quality...motion artifacts...slightly grainy"
- Problem: Models tend to be critical of image quality while sonographers generally rate images as "Good." This systematic pessimism bias hurts Q6 scores despite being arguably more accurate.

---

## 5. Scoring Methodology Issues

### Issue 1: Q5 Normalization Bug (CRITICAL)

**Location**: `src/data/normalize_annotations.py`, function `_normalize_q5_regex()`
**Problem**: Uses `re.match()` (anchored to start) instead of `re.search()` (searches anywhere)
**Impact**: 14/15 models score 0.0 on Q5
**Fix**: Change `_Q5_RANGE_PATTERN.match(clean)` to `_Q5_RANGE_PATTERN.search(clean)` (and similarly for single week pattern)
**Expected improvement**: ~0.43 Q5 score for best models, ~0.05 overall score lift

### Issue 2: Q2/Q3 Keyword Vocabulary Too Narrow

**Problem**: Q2 scoring only recognizes 4 keywords (cephalic, breech, transverse, longitudinal), but 53.3% of GT answers contain none of these. Q3 has similar gaps.
**Recommendation**: Expand keyword extraction or switch to embedding-based matching for these questions.

### Issue 3: Q7 Extreme Class Imbalance

**Problem**: 98.6% of GT is "Normal" variants. Models that always predict "Normal" score ~0.44 on Q7 (Phi-3.5-vision), while models that try to detect abnormalities (InternVL2-2B: 79.3% sensitivity) are penalized with 0.18 because they generate many false positives.
**Impact**: The scoring rewards conservative "always normal" predictions rather than clinically useful abnormality detection.
**Recommendation**: Weight sensitivity more heavily, or report sensitivity/specificity separately rather than collapsing to a single metric.

### Issue 4: Q8 BERTScore Non-Discriminative

**Problem**: All models score 0.75-0.80 on Q8 regardless of actual recommendation quality. BERTScore F1 is too lenient for medical text comparison.
**CV**: 0.02 (essentially no variance across models)
**Recommendation**: Consider domain-specific evaluation (MedBERTScore) or structured recommendation extraction.

### Issue 5: Verbose Output Penalizes Valid Answers

**Problem**: The scoring pipeline was designed for concise GT annotations (e.g., "20-25 weeks"), not verbose VLM outputs (e.g., "The gestational age appears to be around 20-24 weeks based on..."). The normalization layer maps short forms to bins but cannot extract information from sentences.
**Impact**: Affects Q2, Q3, Q5 most severely.
**Recommendation**: Either (a) add robust text extraction to the normalizer, or (b) constrain model outputs via prompt engineering.

---

## 6. Actionable Recommendations

### Immediate Fixes (No Retraining Required)

1. **Fix Q5 regex bug**: Change `re.match()` to `re.search()` in `_normalize_q5_regex()`. Expected impact: +0.05 overall score across all models.

2. **Add GA extraction from verbose text**: Use `re.search()` with patterns that find "X-Y weeks" anywhere in the text, then map to bins. 99.7% of Kimi predictions contain an extractable GA range.

3. **Constrain output format via prompt engineering**: Add explicit format instructions to the system prompt:

   ```
   Answer each question concisely in the following format:
   - Q1: List anatomical structures as comma-separated terms (e.g., "liver, stomach, aorta")
   - Q2: State orientation using: cephalic/breech/transverse/longitudinal + additional detail
   - Q3: State the imaging plane (e.g., "Trans-thalamic view", "Mid-sagittal view")
   - Q4: List measurable parameters (e.g., "BPD, HC, FL")
   - Q5: State gestational age as a range (e.g., "20-25 weeks")
   - Q6: Rate as Good/Medium/Low with brief justification
   - Q7: State Normal or Abnormal with specific finding
   - Q8: Brief clinical recommendation in 1-2 sentences
   ```

4. **Expand Q2 keyword vocabulary**: Add "axial", "sagittal", "coronal", "head down", "vertex", "head to left/right" to the orientation keyword extractor. This would cover significantly more GT answers.

### Scoring Pipeline Improvements

5. **Implement answer extraction layer**: Before scoring, apply regex/NLP extraction to pull structured answers from verbose model outputs. This addresses the fundamental mismatch between model output style and scorer expectations.

6. **Rebalance Q7 scoring**: Report sensitivity and specificity separately. Consider F1 over the binary classes rather than exact match, to reward models that attempt abnormality detection.

7. **Replace Q8 BERTScore**: Use a more discriminative metric. Options include:
   - Domain-specific embedding similarity (medical sentence transformers)
   - Structured recommendation extraction + keyword F1
   - Clinical guideline adherence scoring

### Prompt Engineering (Research-Backed)

8. **Few-shot prompting**: Include 2-3 examples per question showing the expected concise answer format. Research shows few-shot with constrained outputs outperforms zero-shot by 5-10% in medical VQA (Med-GRIM, arxiv:2508.06496).

9. **Structured output prompting**: Use JSON/template output format constraints. Research on MedLFQA showed 85.8% factuality improvement with structured multi-step prompting (arxiv:2503.03194).

10. **Domain-specialized models**: EchoVLM (ultrasound-specific VLM) achieved +10.15 BLEU-1 over Qwen2-VL on ultrasound tasks using Dual-path MoE architecture trained on 208K clinical cases (arxiv:2509.14977). Consider evaluating or fine-tuning against such architectures.

### Fine-Tuning Strategies

11. **Output format fine-tuning**: Fine-tune models on training data that matches the exact expected output format. If GT says "20-25 weeks", train on exactly that format rather than verbose responses.

12. **Category-specific prompting**: Aorta and Non_standard_NT score lowest. Consider category-aware prompts that provide context about the expected anatomy (e.g., "This is an aorta ultrasound image. The relevant anatomical structures include...").

---

## 7. Limitations

- Analysis is based on a single evaluation run per model; variance across runs is not measured.
- The 3 excluded models (pixtral-12b, InternVL2.5-4B, Aria) had infrastructure failures, not model failures -- they may perform differently with correct inference setup.
- The Q5 fix simulation assumes the same bin-mapping logic; actual improvement depends on model accuracy in estimating gestational age, which we cannot fully validate without clinical ground truth verification.
- Embedding similarity and primary scores often disagree (low correlation), suggesting the two metrics capture different aspects of answer quality. Neither should be treated as definitive.
- The class imbalance in Q7 (98.6% Normal) means all models face the same statistical challenge; improving sensitivity requires either rebalanced training data or explicit abnormality-focused prompting.

---

## Appendix: Data Quality Issues

- **GT data noise**: Q5 contains malformed entries ("2025-11-13 00:00:00", "13-14 weekks", "11-13 wks&", "18-200 weeks") -- 8 out of 18,936 rows (negligible but should be cleaned).
- **GT class distribution**: Q6 is 81.6% "Good image quality", Q7 is 98.6% "Normal" variants. These extreme imbalances make scoring on these questions dominated by majority-class prediction.
- **Normalization rate**: pixtral-12b and InternVL2.5-4B show 0% normalization rate across all questions, confirming their predictions are error strings, not model outputs.

---

_Visualizations saved to `.omc/scientist/figures/`:_

- `question_model_heatmap.png` -- Per-question scores across all 15 models
- `category_difficulty.png` -- Category ranking by average primary score
- `question_difficulty.png` -- Question ranking by average primary score
- `model_ranking_and_q5_bug.png` -- Model rankings and Q5 scoring bug illustration
- `q7_sensitivity_specificity.png` -- Q7 sensitivity vs specificity tradeoff
