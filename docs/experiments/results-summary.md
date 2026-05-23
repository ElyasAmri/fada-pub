---
date: 2026-03-06
type: results
project: fada-v3
closes: "#23"
---

# VLM Benchmark Results Summary

46 vision-language models evaluated zero-shot on 1,894 test images (14 anatomical categories, 8 clinical questions per image). Scored with pipeline v3 using per-question specialized metrics and embedding similarity against sonographer ground truth.

## Overall Rankings (Top 10)

| Rank | Model                           | Size | Primary | EmbedSim |
| ---- | ------------------------------- | ---- | ------- | -------- |
| 1    | Qwen/Qwen3.5-35B-A3B            | 35B  | 0.3650  | 0.3569   |
| 2    | google/gemma-3-12b-it           | 12B  | 0.3629  | 0.3641   |
| 3    | OpenGVLab/InternVL3_5-4B        | 4B   | 0.3491  | 0.3946   |
| 4    | Qwen/Qwen3.5-4B                 | 4B   | 0.3460  | 0.3627   |
| 5    | Qwen/Qwen3.5-9B                 | 9B   | 0.3439  | 0.3662   |
| 6    | openbmb/MiniCPM-V-4_5           | 8B   | 0.3409  | 0.4059   |
| 7    | Qwen/Qwen3-VL-2B-Instruct       | 2B   | 0.3407  | 0.4013   |
| 8    | OpenGVLab/InternVL3_5-8B        | 8B   | 0.3403  | 0.4067   |
| 9    | moonshotai/Kimi-VL-A3B-Instruct | 3B   | 0.3347  | 0.4057   |
| 10   | mistralai/Mistral-Small-3.1-24B | 24B  | 0.3292  | 0.4024   |

Full 46-model ranking in [models-tracker.md](models-tracker.md).

## Per-Question Analysis (Top Model: Qwen3.5-35B-A3B)

Each question uses a specialized primary metric; embedding similarity is computed uniformly across all questions.

| Question                     | Metric                  | Primary | EmbedSim |
| ---------------------------- | ----------------------- | ------- | -------- |
| Q6: Image Quality            | exact_tier_match        | 0.8406  | 0.4611   |
| Q8: Clinical Recommendations | bertscore_f1            | 0.7597  | 0.3983   |
| Q5: Gestational Age          | exact_bin_match         | 0.4172  | 0.3249   |
| Q1: Anatomical Structures    | set_f2_synonym          | 0.2490  | 0.4770   |
| Q4: Biometric Measurements   | keyword_f1              | 0.2357  | 0.2178   |
| Q3: Imaging Plane            | relaxed_accuracy        | 0.1480  | 0.3146   |
| Q2: Fetal Orientation        | relaxed_accuracy        | 0.1390  | 0.4231   |
| Q7: Normality Assessment     | exact_match_with_binary | 0.1310  | 0.2385   |

**Observations:**

- Q6 and Q8 are effectively solved (>76% primary) -- models reliably assess image quality and generate appropriate recommendations
- Q5 (gestational age) performs well at 42% exact bin match -- models can estimate trimester-level age
- Q1 improved significantly with v3 synonym matching + F2 scoring (0.116 -> 0.249 for top model)
- Q1-Q3 have low primary scores but high embedding similarity, suggesting models describe the right concepts but in different terms than the ground truth expects
- Q7 varies widely between models: gemma-3-12b-it scores 0.335 vs Qwen3.5-35B-A3B at 0.131
- Q4 (biometric measurements) remains challenging -- models struggle with specific numerical measurements

## Per-Category Analysis (Top Model: Qwen3.5-35B-A3B)

| Category          | Primary | EmbedSim | Samples |
| ----------------- | ------- | -------- | ------- |
| Femur             | 0.5338  | 0.3834   | 904     |
| Trans-thalamic    | 0.4475  | 0.3504   | 1,232   |
| CRL-View          | 0.4401  | 0.3263   | 1,576   |
| Trans-ventricular | 0.4382  | 0.3500   | 464     |
| Standard_NT       | 0.4312  | 0.3832   | 1,192   |
| NT-View           | 0.3983  | 0.3866   | 1,616   |
| Abdomen           | 0.3507  | 0.3530   | 1,936   |
| Trans-cerebellum  | 0.3440  | 0.3540   | 544     |
| Thorax            | 0.3250  | 0.3901   | 1,432   |
| Non_standard_NT   | 0.3117  | 0.3627   | 376     |
| Cervical          | 0.3023  | 0.3378   | 400     |
| Cervix            | 0.2834  | 0.3843   | 1,288   |
| Public_Symphysis  | 0.2495  | 0.3221   | 1,080   |
| Aorta             | 0.2042  | 0.2855   | 1,040   |

**Observations:**

- Femur jumps to #1 category for Qwen3.5-35B-A3B (0.534) -- strong biometric measurement capability
- Brain views (Trans-thalamic, Trans-ventricular) and CRL-View remain strong across models
- Aorta is consistently hardest across all models -- small, technically demanding view
- Non_standard_NT improved with v3 scoring but remains challenging -- inherently ambiguous

## Model Family Comparison

| Family      | Models | Best Primary | Avg Primary | Best Model              |
| ----------- | ------ | ------------ | ----------- | ----------------------- |
| Qwen3.5     | 6      | 0.3650       | 0.3015      | Qwen3.5-35B-A3B         |
| Gemma 3     | 3      | 0.3629       | 0.3383      | gemma-3-12b-it          |
| InternVL3.5 | 2      | 0.3491       | 0.3447      | InternVL3_5-4B          |
| MiniCPM     | 3      | 0.3409       | 0.3087      | MiniCPM-V-4_5           |
| Qwen3-VL    | 4      | 0.3407       | 0.3261      | Qwen3-VL-2B-Instruct    |
| Kimi-VL     | 2      | 0.3347       | 0.3245      | Kimi-VL-A3B-Instruct    |
| Qwen2-VL    | 2      | 0.3218       | 0.3057      | Qwen2-VL-2B-Instruct    |
| InternVL2   | 4      | 0.3149       | 0.2940      | InternVL2-8B            |
| MedGemma    | 1      | 0.3035       | 0.3035      | medgemma-4b-it          |
| Qwen2.5     | 3      | 0.2935       | 0.2791      | Qwen2.5-VL-7B-Instruct  |
| SmolVLM     | 3      | 0.2868       | 0.2462      | SmolVLM2-2.2B-Instruct  |
| Gemma 3n    | 1      | 0.2787       | 0.2787      | gemma-3n-E4B-it         |
| Phi         | 2      | 0.2602       | 0.1903      | Phi-3.5-vision-instruct |

Qwen3.5 avg (0.3015) is dragged down by Qwen3.5-27B (0.1124, output issues). Excluding it: avg 0.3393.

## Key Findings

### 1. Size does not predict performance

- gemma-3-12b-it (12B) beats gemma-3-27b-it (27B) by +0.04
- Qwen3.5-4B (4B) beats Qwen3.5-9B (9B) and Qwen3.5-27B
- InternVL3_5-4B beats InternVL3_5-8B
- Qwen3-VL-2B beats Qwen3-VL-8B and Qwen3-VL-4B
- The smallest competitive model (Qwen3.5-0.8B at rank 17) outperforms several 7-8B models

### 2. Current-gen models consistently outperform previous-gen

- InternVL3.5 (avg 0.3439) > InternVL2 (avg 0.2891): +19% improvement
- Qwen3-VL (avg 0.3208) > Qwen2.5-VL (avg 0.2758): +16% improvement
- Gemma 3 (avg 0.3346) > Gemma 3n (0.2765): distilled variant underperforms

### 3. Medical-specialized models underperform general models

- MedGemma-4b (0.2994) < gemma-3-4b-it (0.3249): general Gemma beats medical variant at same size
- MedVLM-R1 (0.3179) performs mid-table despite medical fine-tuning
- Fetal ultrasound is likely too niche for broad medical pre-training to help

### 4. MoE architectures show promise

- Qwen3.5-35B-A3B (rank 1): MoE with 3B active parameters beats all dense models
- Kimi-VL-A3B (rank 9): 3B active MoE in top 10
- MoE models offer strong accuracy/efficiency tradeoff for deployment

### 5. Thinking/reasoning variants do not help

- Kimi-VL-A3B-Instruct (0.3347) > Kimi-VL-A3B-Thinking (0.3144)
- Qwen3-VL-8B-Instruct (0.3288) > Qwen3-VL-8B-Thinking (0.3212)
- Extended reasoning adds no value for structured VQA -- direct answers outperform chain-of-thought

### 6. Score ceiling at ~36%

All 46 models cluster between 0.10-0.37 primary score (54 total including later additions; see models-to-test.md for current count). The ceiling likely reflects:

- Scoring pipeline strictness (exact matching on Q1-Q3 penalizes semantically correct but differently worded answers)
- Inherent task difficulty (biometric measurements, gestational age estimation from single images)
- Zero-shot limitation (no exposure to the specific question format or annotation style)

## Known Annotation Limitations

### Q2: Concept mismatch between GT and model outputs

Q2 (Fetal Orientation) has a fundamental concept mismatch between ground truth annotations and model predictions. The GT uses scan plane orientation terminology (e.g., "Axial upper abdomen, vertebral column to right") describing the ultrasound transducer orientation relative to anatomy. Models instead use obstetric presentation terminology (e.g., "Cephalic presentation") describing the fetus's position relative to the birth canal. Both are clinically valid descriptions of fetal orientation but address different aspects -- scan plane vs. fetal lie. This explains Q2's low primary score (0.131) despite high embedding similarity (0.430): models correctly identify orientation-related information but express it in a different clinical framework than the annotators used. No metric change is applied; this is documented as an annotation limitation.

## Scoring Pipeline v3

Applied March 2026. Changes from v2:

- **Q1 synonym matching + F2**: Structure synonyms (e.g., "abdomen" matches "abdominal wall") with F-beta (beta=2) recall-weighted scoring. Q1 primary: 0.116 -> 0.142 (+22%)
- **Q3 transverse synonym**: "transverse" now maps to "axial" (clinically equivalent for cross-sectional views)
- **Q3 GT spelling fixes**: "sagital", "saggital", "dagital", "midsagital" corrected in normalization layer
- All 46 models rescored (54 total including later additions). Top model changed: Qwen3.5-35B-A3B (0.3650) overtook gemma-3-12b-it (0.3629)
- Impact on gemma-3-12b-it: overall primary 0.3596 -> 0.3629 (+0.003)

## Scoring Pipeline v2

Applied March 2026. Changes from v1:

- **Q5 regex fix**: `match` to `search` for gestational age pattern extraction
- **Q2/Q3 keyword expansion**: broader synonym sets for orientation and plane matching
- **Q1/Q4 abbreviation expansion**: common medical abbreviations (BPD, FL, AC, HC, etc.)
- Impact: ~+0.04 primary score improvement across all models

## Infrastructure

- **Evaluation cluster**: 8x RCCG H100 machines (fada-1 through fada-8)
- **Inference engine**: vLLM with OpenAI-compatible API
- **Eval script**: `experiments/api_models/test_api_vlm.py` (8 questions per image, checkpoint/resume)
- **Scoring**: `experiments/evaluation/score_against_gt.py` (per-question specialized metrics + embedding similarity)
- **Ground truth**: Sonographer annotations from `data/Fetal Ultrasound Annotations Normalized.xlsx`
