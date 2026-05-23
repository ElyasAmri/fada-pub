---
date: 2026-01-27
type: tracker
project: fada-v3
---

## Evaluation Summary

| Phase   | Method               | Sample Size | Ground Truth | Status       |
| ------- | -------------------- | ----------- | ------------ | ------------ |
| Phase 1 | Zero-shot screening  | ~250        | No           | Archived     |
| Phase 2 | Embedding similarity | 600         | Yes          | Complete     |
| Phase 3 | Cloud API            | 709         | Yes          | Complete     |
| Phase 4 | Full test set        | 1,894       | Yes          | **Complete** |

---

## Phase 2: Fine-Tuned with Ground Truth (Jan 2026)

| Model             | Task         | Score     | Samples | Method                                                                                      |
| ----------------- | ------------ | --------- | ------- | ------------------------------------------------------------------------------------------- |
| **Qwen2.5-VL-7B** | VQA Q1-Q8    | **81.1%** | 600     | Embedding similarity -- NOTE: 600-sample subset; full test set v3 score is embed_sim=0.5058 |
| Qwen3-VL-8B       | Q7 Normality | 82%       | 50      | Accuracy                                                                                    |
| EfficientNet-B0   | 12-class     | 88%       | 1,494   | Classification                                                                              |

---

## Phase 3: Cloud API Evaluation (Dec 2025)

| Model            | Provider  | Score      | Samples | Notes                                                                                          |
| ---------------- | --------- | ---------- | ------- | ---------------------------------------------------------------------------------------------- |
| **MedGemma-27B** | Vertex AI | **78.81%** | 709     | Full Q1-Q8 -- NOTE: proxy scoring against Gemini pseudo-labels, NOT GT sonographer annotations |

---

## Local GPU Testing (RTX 5090, 24GB)

All 7 Qwen models verified working with Unsloth:

| Model         | Unsloth | 4-bit | Train Time (2 samples) |
| ------------- | ------- | ----- | ---------------------- |
| qwen2-vl-2b   | Yes     | Yes   | 44s                    |
| qwen2.5-vl-3b | Yes     | Yes   | 22s                    |
| qwen2-vl-7b   | Yes     | Yes   | 22s                    |
| qwen2.5-vl-7b | Yes     | Yes   | 19s                    |
| qwen3-vl-2b   | Yes     | Yes   | 23s                    |
| qwen3-vl-4b   | Yes     | Yes   | 24s                    |
| qwen3-vl-8b   | Yes     | Yes   | 31s                    |

---

## Phase 4: RCCG H100 Full Test Set Evaluation (Mar 2026)

8x H100 PCIe machines running vLLM, evaluated on 1,894 test images (full test split). 54 models scored (see models-to-test.md for current count).

### Scored Results (zero-shot, 1,894 test images)

Metrics: primary_score = weighted multi-metric average, embed_sim = embedding similarity vs ground truth.

Scoring pipeline v2 (Mar 5): Q5 regex fix, Q2/Q3 keyword expansion, Q1/Q4 abbreviation expansion.
Scoring pipeline v3 (Mar 6): Q1 synonym matching + F2, Q3 transverse synonym, GT spelling fixes. All 46 models rescored (54 total including later additions).

| Rank | Model                                   | Size | Primary | EmbedSim | Error% |
| ---- | --------------------------------------- | ---- | ------- | -------- | ------ |
| 1    | Qwen/Qwen3.5-35B-A3B                    | 35B  | 0.3650  | 0.3569   | TBD    |
| 2    | google/gemma-3-12b-it                   | 12B  | 0.3629  | 0.3641   | TBD    |
| 3    | OpenGVLab/InternVL3_5-4B                | 4B   | 0.3491  | 0.3946   | TBD    |
| 4    | Qwen/Qwen3.5-4B                         | 4B   | 0.3460  | 0.3627   | TBD    |
| 5    | Qwen/Qwen3.5-9B                         | 9B   | 0.3439  | 0.3662   | TBD    |
| 6    | openbmb/MiniCPM-V-4_5                   | 8B   | 0.3409  | 0.4059   | TBD    |
| 7    | Qwen/Qwen3-VL-2B-Instruct               | 2B   | 0.3407  | 0.4013   | TBD    |
| 8    | OpenGVLab/InternVL3_5-8B                | 8B   | 0.3403  | 0.4067   | TBD    |
| 9    | moonshotai/Kimi-VL-A3B-Instruct         | 3B   | 0.3347  | 0.4057   | TBD    |
| 10   | mistralai/Mistral-Small-3.1-24B         | 24B  | 0.3292  | 0.4024   | TBD    |
| 11   | Qwen/Qwen3-VL-8B-Instruct               | 8B   | 0.3288  | 0.3866   | TBD    |
| 12   | google/gemma-3-4b-it                    | 4B   | 0.3277  | 0.3399   | TBD    |
| 13   | google/gemma-3-27b-it                   | 27B  | 0.3243  | 0.3686   | TBD    |
| 14   | Qwen/Qwen2-VL-2B-Instruct               | 2B   | 0.3218  | 0.3720   | TBD    |
| 15   | JZPeterPan/MedVLM-R1                    | 7B   | 0.3217  | 0.3791   | TBD    |
| 16   | Qwen/Qwen3-VL-8B-Thinking               | 8B   | 0.3212  | 0.3572   | TBD    |
| 17   | Qwen/Qwen3.5-0.8B                       | 0.8B | 0.3210  | 0.3771   | TBD    |
| 18   | Qwen/Qwen3.5-2B                         | 2B   | 0.3207  | 0.3745   | TBD    |
| 19   | OpenGVLab/InternVL2-8B                  | 8B   | 0.3149  | 0.3768   | TBD    |
| 20   | moonshotai/Kimi-VL-A3B-Thinking         | 3B   | 0.3144  | 0.3717   | TBD    |
| 21   | Qwen/Qwen3-VL-4B-Instruct               | 4B   | 0.3136  | 0.3898   | TBD    |
| 22   | rhymes-ai/Aria                          | 25B  | 0.3051  | 0.3814   | TBD    |
| 23   | google/medgemma-4b-it                   | 4B   | 0.3035  | 0.4044   | TBD    |
| 24   | Qwen/Qwen2.5-VL-7B-Instruct             | 7B   | 0.2935  | 0.3828   | TBD    |
| 25   | OpenGVLab/InternVL2-4B                  | 4B   | 0.2933  | 0.3523   | TBD    |
| 26   | openbmb/MiniCPM-o-2_6                   | 8B   | 0.2929  | 0.3920   | TBD    |
| 27   | openbmb/MiniCPM-V-2_6                   | 8B   | 0.2924  | 0.3941   | TBD    |
| 28   | OpenGVLab/InternVL2-2B                  | 2B   | 0.2900  | 0.3787   | TBD    |
| 29   | Qwen/Qwen2-VL-7B-Instruct               | 7B   | 0.2897  | 0.3897   | TBD    |
| 30   | HuggingFaceTB/SmolVLM2-2.2B-Instruct    | 2.2B | 0.2868  | 0.3778   | TBD    |
| 31   | Qwen/Qwen2.5-Omni-7B                    | 7B   | 0.2800  | 0.4014   | TBD    |
| 32   | google/gemma-3n-E4B-it                  | 4B   | 0.2787  | 0.3065   | TBD    |
| 33   | OpenGVLab/InternVL2-1B                  | 1B   | 0.2779  | 0.3606   | TBD    |
| 34   | HuggingFaceM4/Idefics3-8B-Llama3        | 8B   | 0.2759  | 0.3782   | TBD    |
| 35   | Qwen/Qwen2.5-VL-3B-Instruct             | 3B   | 0.2639  | 0.3788   | TBD    |
| 36   | microsoft/Phi-3.5-vision-instruct       | 4B   | 0.2602  | 0.3406   | TBD    |
| 37   | HuggingFaceTB/SmolVLM-256M-Instruct     | 256M | 0.2441  | 0.3111   | TBD    |
| 38   | HuggingFaceTB/SmolVLM-500M-Instruct     | 500M | 0.2078  | 0.3541   | TBD    |
| 39   | lmms-lab/llava-onevision-qwen2-7b-ov-hf | 7B   | 0.1663  | 0.1606   | TBD    |

## Models with Output Issues

The following models produced severely degraded scores due to connection errors, formatting failures, or chat template errors. Error rates should be populated from score files.

<!-- TODO: populate Error% values from score files for each model below -->

| Model                               | Size | Primary | EmbedSim | Error% | Issue                        |
| ----------------------------------- | ---- | ------- | -------- | ------ | ---------------------------- |
| allenai/Molmo2-8B                   | 8B   | 0.1260  | 0.0155   | TBD    | Output formatting failure    |
| OpenGVLab/InternVL2_5-4B            | 4B   | 0.1256  | -0.0088  | TBD    | Chat template error          |
| mistral-community/pixtral-12b       | 12B  | 0.1256  | -0.0088  | TBD    | Connection/formatting errors |
| microsoft/Phi-4-multimodal-instruct | 14B  | 0.1205  | 0.0466   | TBD    | Output formatting failure    |
| Qwen/Qwen3.5-27B                    | 27B  | 0.1124  | 0.0330   | TBD    | Output formatting failure    |
| NVlabs/Eagle2.5-8B                  | 8B   | 0.0960  | 0.0410   | TBD    | Output formatting failure    |

### Incompatible with vLLM

| Model                                     | Reason                                              |
| ----------------------------------------- | --------------------------------------------------- |
| deepseek-ai/deepseek-vl2-small            | Architecture not supported                          |
| microsoft/llava-med-v1.5-mistral-7b       | Architecture not supported                          |
| lmms-lab/LLaVA-OneVision-1.5-8B-Instruct  | Architecture not supported (LLaVAOneVision1_5)      |
| lmms-lab/LLaVA-OneVision-1.5-4B-Instruct  | Architecture not supported (LLaVAOneVision1_5)      |
| vikhyatk/moondream2                       | Config incompatible (missing hidden_size)           |
| meta-llama/Llama-3.2-11B-Vision-Instruct  | MllamaProcessor missing \_get_num_multimodal_tokens |
| CohereForAI/aya-vision-8b                 | Gated repo, access not approved                     |
| THUDM/glm-4v-9b                           | Engine core initialization failed                   |
| deepseek-ai/deepseek-vl2-tiny             | No model architectures specified                    |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | Gated repo, access not approved                     |
| OpenGVLab/InternVL2_5-4B                  | Chat template error (str/list concat), 100% errors  |
| allenai/Molmo-7B-D-0924                   | Dynamic module import fails (processor error)       |
| Qwen/Qwen2.5-VL-32B-Instruct              | Engine core initialization failed                   |

### Not Yet Available on HuggingFace

- Qwen/Qwen3.5-122B-A10B (too large for single H100)
