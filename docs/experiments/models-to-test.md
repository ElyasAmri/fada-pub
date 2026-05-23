---
date: 2026-02-26
type: tracker
project: fada-v3
---

## Fetal/Ultrasound Specific Models (HIGH PRIORITY)

| Model               | Source          | Size | Availability                               | Status |
| ------------------- | --------------- | ---- | ------------------------------------------ | ------ |
| **Dolphin V1 / R1** | ICLR 2026       | -    | Weights TBD (paper: arXiv 2509.25748)      | TODO   |
| **FetalCLIP**       | MBZUAI          | -    | HuggingFace + GitHub                       | TODO   |
| **EchoVLM**         | arXiv Sep 2025  | MoE  | Paper only (arXiv 2509.14977)              | TODO   |
| **FetalMind**       | ICLR 2026       | -    | Not public                                 | TODO   |
| **Sonomate**        | Nature Jan 2026 | -    | Not public                                 | TODO   |
| **LLaVA-Ultra**     | -               | -    | Chinese ultrasound, fine-grained semantics | TODO   |
| **AbVLM-Q**         | BMC 2025        | -    | Abdominal ultrasound quality assessment    | TODO   |

**Key context:**

- Dolphin R1 achieves U2-score 0.5835 on U2-BENCH -- over 2x the second-best model (0.2968)
- FetalCLIP trained on 210K fetal ultrasound images -- closest to our domain
- Most fetal-specific models are paper-only; FetalCLIP is the main one with public weights

**Papers:**

- U2-BENCH (arXiv 2505.17779) - First ultrasound VLM benchmark, 7,241 cases, ICLR 2026
- FETAL-GAUGE (arXiv 2512.22278) - VLM benchmark for fetal US
- FetalCLIP (arXiv 2502.14807) - 210K fetal US foundation model
- Sonomate (Nature Jan 2026) - Real-time fetal US understanding
- EchoVLM (arXiv 2509.14977) - Universal ultrasound VLM
- FetalMind (ICLR 2026) - Epistemic-aware fetal US model

---

## Current-Gen General VLMs - Tier 1

Models from current generation (2025-2026). Priority for benchmarking.

### Qwen Family (Alibaba)

| Model                    | Size    | Active Params | VRAM  | Status       | Notes                                                  |
| ------------------------ | ------- | ------------- | ----- | ------------ | ------------------------------------------------------ |
| **Qwen3.5-35B-A3B**      | 35B MoE | 3B            | ~10GB | TODO         | Matches Qwen3-VL-235B quality at 3B active; Apache 2.0 |
| **Qwen3.5-27B**          | 27B     | 27B           | ~54GB | TODO         | Cloud only; native multimodal                          |
| **Qwen3-VL-8B**          | 8B      | 8B            | ~20GB | Local tested | Unsloth verified, needs GT eval                        |
| **Qwen3-VL-4B**          | 4B      | 4B            | ~12GB | Local tested | Unsloth verified                                       |
| **Qwen3-VL-2B**          | 2B      | 2B            | ~8GB  | Local tested | Unsloth verified, mobile candidate                     |
| **Qwen3-VL-8B-Thinking** | 8B      | 8B            | ~20GB | TODO         | Reasoning mode; requires transformers >= 4.57          |

### InternVL Family (OpenGVLab)

| Model               | Size | VRAM           | Status | Notes                                          |
| ------------------- | ---- | -------------- | ------ | ---------------------------------------------- |
| **InternVL3.5-8B**  | 8B   | ~20GB          | TODO   | Aug 2025, Cascade RL, 4x faster than InternVL3 |
| **InternVL3.5-4B**  | 4B   | ~12GB          | TODO   | Efficient variant                              |
| **InternVL3.5-38B** | 38B  | ~76GB (2xA100) | TODO   | Cloud only                                     |

### Other Current-Gen

| Model                         | Family      | Size         | VRAM              | Status | Notes                                                             |
| ----------------------------- | ----------- | ------------ | ----------------- | ------ | ----------------------------------------------------------------- |
| **LLaVA-OneVision-1.5-8B**    | LLaVA/Qwen3 | 8B           | ~20GB             | TODO   | Dec 2025, Qwen3 backbone, beats Qwen2.5-VL-7B on 18/27 benchmarks |
| **LLaVA-OneVision-1.5-4B**    | LLaVA/Qwen3 | 4B           | ~12GB             | TODO   | Beats Qwen2.5-VL-3B on all 27 benchmarks                          |
| **DeepSeek-VL2**              | DeepSeek    | 3B-27B (MoE) | ~10-16GB          | TODO   | 1-4.5B active params; strong OCR                                  |
| **Phi-4-Multimodal-Instruct** | Microsoft   | 5.6B         | ~14GB             | TODO   | Vision+audio; 128K context; on U2-BENCH leaderboard               |
| **Mistral-Small-3.1-24B**     | Mistral     | 24B          | ~24GB (fits 4090) | TODO   | Apache 2.0; 128K context; on U2-BENCH leaderboard                 |
| **Gemma-3-4B**                | Google      | 4B           | ~12GB             | TODO   | SigLIP vision encoder; 128K context                               |
| **Gemma-3-12B**               | Google      | 12B          | ~24GB             | TODO   | SigLIP vision encoder; 128K context                               |
| **Gemma-3-27B**               | Google      | 27B          | ~54GB             | TODO   | Cloud only; strongest Gemma vision                                |

---

## Already Tested (Previous Gen - Keep for Comparison)

These models have existing results. Keep for comparison but do NOT prioritize for new benchmarking.

| Model               | Family    | Size | GT Primary | GT EmbedSim | Notes                           |
| ------------------- | --------- | ---- | ---------- | ----------- | ------------------------------- |
| **Qwen2.5-VL-7B**   | Alibaba   | 7B   | 0.2935     | 0.3828      | Fine-tuned: 81.1% (600 samples) |
| **MiniCPM-V-2.6**   | OpenBMB   | 8B   | 0.2924     | 0.3941      |                                 |
| **Qwen2-VL-2B**     | Alibaba   | 2B   | 0.3218     | 0.3720      |                                 |
| **InternVL2-4B**    | OpenGVLab | 4B   | 0.2933     | 0.3523      |                                 |
| **Qwen2-VL-7B**     | Alibaba   | 7B   | 0.2897     | 0.3897      | Superseded by Qwen3-VL-8B       |
| **LLaVA-OneVision** | LLaVA     | 7B   | 0.1663     | 0.1606      | Superseded by v1.5              |
| **Molmo-7B**        | AllenAI   | 7B   | 0.1260     | 0.0155      | Output issues                   |

---

## Medical VLMs

| Model               | Focus             | Size | Availability   | Status     | Notes                                                                                                            |
| ------------------- | ----------------- | ---- | -------------- | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| **MedGemma-27B**    | General medical   | 27B  | Google API     | **Tested** | 78.81% proxy (709 samples -- Phase 1 proxy scoring against Gemini pseudo-labels, NOT GT sonographer annotations) |
| **MedGemma-4B**     | General medical   | 4B   | HuggingFace    | GT tested  | 0.3035 primary (1,894 samples)                                                                                   |
| **MedVLM-R1**       | Medical reasoning | 2B   | HuggingFace    | TODO       | Qwen2-VL-2B + GRPO RL; 78.22% on MRI/CT/X-ray                                                                    |
| **Medical-VLM-24B** | General medical   | 24B  | John Snow Labs | TODO       | 82.9% on OpenMedBench                                                                                            |
| **MindGPT-Med**     | Medical           | ?    | Unknown        | TODO       | On U2-BENCH leaderboard                                                                                          |
| **MedDr**           | Medical           | ?    | Unknown        | TODO       | On U2-BENCH leaderboard                                                                                          |
| **LLaVA-Med**       | Radiology         | 7B   | HuggingFace    | TODO       | Medical adaptation of LLaVA                                                                                      |
| **CheXagent-8B**    | Chest X-ray       | 8B   | HuggingFace    | Tested     | 0% -- complete domain mismatch                                                                                   |

---

## Mobile/Edge VLMs (For Deployment)

| Model             | Size | VRAM   | Speed      | Status      | Notes                    |
| ----------------- | ---- | ------ | ---------- | ----------- | ------------------------ |
| SmolVLM-256M      | 256M | <1GB   | Fast       | TODO        | Purpose-built for edge   |
| SmolVLM-500M      | 500M | ~1.2GB | Fast       | TODO        | Purpose-built for edge   |
| SmolVLM-2.2B      | 2.2B | ~5GB   | Moderate   | TODO        | Best quality small model |
| Moondream2        | ~2B  | ~4GB   | 25.6 tok/s | Perf tested |                          |
| MobileVLM-V2-1.7B | 1.7B | Low    | 21.5 tok/s | TODO        | Optimized for mobile     |
| MobileVLM-V2-3B   | 3B   | Low    | Moderate   | TODO        |                          |
| LFM2-VL-450M      | 450M | Low    | 2x faster  | TODO        |                          |
| LFM2-VL-1.6B      | 1.6B | Low    | Fast       | TODO        |                          |

---

## API Models

| Model                     | Provider  | Status    | Notes                                     |
| ------------------------- | --------- | --------- | ----------------------------------------- |
| **Gemini-3-Flash**        | Google    | Available | Fast, cheap; default for batch annotation |
| **Gemini-3-Pro**          | Google    | Available | Best quality                              |
| GPT-4o                    | OpenAI    | TODO      | Multimodal                                |
| GPT-4o-Mini               | OpenAI    | TODO      | On U2-BENCH leaderboard; cheap            |
| Claude-4-Sonnet           | Anthropic | TODO      | Strong reasoning                          |
| Qwen-Max                  | Alibaba   | TODO      | On U2-BENCH leaderboard; proprietary      |
| Doubao-1.5-Vision-Pro-32k | ByteDance | TODO      | On U2-BENCH leaderboard; MoE, 32K context |

---

## Evaluation Priority Order

### Phase 1: Current-Gen Benchmarking (GT Eval)

Priority: models most likely to beat our 81.1% baseline.

1. [ ] Qwen3-VL-8B (already local tested, needs GT eval on 1,894 images)
2. [ ] Qwen3.5-35B-A3B (3B active -- efficient AND powerful)
3. [ ] InternVL3.5-8B (latest gen, Cascade RL)
4. [ ] LLaVA-OneVision-1.5-8B (beats Qwen2.5-VL-7B on 18/27 benchmarks)
5. [ ] Phi-4-Multimodal-Instruct (5.6B, tested on U2-BENCH)
6. [ ] Mistral-Small-3.1-24B (strong, fits single 4090)
7. [ ] DeepSeek-VL2 (MoE, 4.5B active)

### Phase 2: Fetal-Specific Models

8. [ ] FetalCLIP (closest to our domain -- 210K fetal US images)
9. [ ] Dolphin V1 (if weights become available)
10. [ ] MedVLM-R1 (medical RL reasoning, 2B)

### Phase 3: Re-eval Previous Best (Full Test Set GT)

11. [x] MiniCPM-V-2.6 (GT: 0.2924 primary)
12. [ ] Qwen2.5-VL-7B (expand from 600 to 1,894 samples)
13. [ ] InternVL2-4B (1,894 test images)

### Phase 4: Mobile Deployment Candidates

14. [ ] SmolVLM-2.2B
15. [ ] Qwen3-VL-2B (quantized)
16. [ ] MobileVLM-V2-3B
17. [ ] Moondream2

---

## U2-BENCH Cross-Reference

Models on the U2-BENCH leaderboard and our coverage:

| U2-BENCH Model              | In Our List?             | Notes                                |
| --------------------------- | ------------------------ | ------------------------------------ |
| Dolphin V1                  | YES                      | Fetal/US specific                    |
| Gemini-2.5-Pro-Preview      | YES (as Gemini-3-Pro)    | Updated generation                   |
| medgemma-4b-it              | YES                      | MedGemma-4B                          |
| DeepSeek-VL2                | YES                      | Tier 1                               |
| Doubao-1.5-Vision-Pro-32k   | YES                      | API models                           |
| InternVL3-9B-Instruct       | YES (as InternVL3.5-8B)  | Updated generation                   |
| Qwen-2.5-VL-32B-Instruct    | NO                       | Too large for local, not current gen |
| Qwen-Max                    | YES                      | API models                           |
| Gemini-2.5-Pro-Exp          | YES (as Gemini-3-Pro)    | Updated generation                   |
| Qwen-2.5-VL-72B-Instruct    | NO                       | Too large, not current gen           |
| GPT-4o-Mini                 | YES                      | API models                           |
| LLaVA-1.5-13B               | NO                       | Outdated baseline                    |
| MindGPT-Med                 | YES                      | Medical VLMs                         |
| MedDr                       | YES                      | Medical VLMs                         |
| Mistral-Small-3.1-24B-Inst. | YES                      | Tier 1                               |
| GPT-4o                      | YES                      | API models                           |
| Qwen-2.5-VL-7B-Instruct     | YES                      | Already tested (81.1% fine-tuned)    |
| Phi-4-Multimodal-Instruct   | YES                      | Tier 1                               |
| Qwen-2.5-VL-3B-Instruct     | YES                      | Already tested                       |
| Gemini-1.5-Pro              | NO                       | Outdated                             |
| Claude-3.7-Sonnet           | YES (as Claude-4-Sonnet) | Updated generation                   |

**Coverage: 17/21 U2-BENCH models covered** (4 excluded as outdated or too large).

---

## Fine-Tuning Compatibility

### Unsloth Verified (Local RTX 5090)

| Model         | Train Time (2 samples) | Mobile Export      |
| ------------- | ---------------------- | ------------------ |
| Qwen2-VL-2B   | 44s                    | Yes (Q4 ~1.5GB)    |
| Qwen2-VL-7B   | 22s                    | No (too large)     |
| Qwen2.5-VL-3B | 22s                    | Marginal (Q4 ~2GB) |
| Qwen2.5-VL-7B | 19s                    | No (too large)     |
| Qwen3-VL-2B   | 23s                    | Yes (Q4 ~1.5GB)    |
| Qwen3-VL-4B   | 24s                    | Marginal (Q4 ~3GB) |
| Qwen3-VL-8B   | 31s                    | No (too large)     |

### Unsloth Likely Compatible (Untested)

| Model                | Notes                                      |
| -------------------- | ------------------------------------------ |
| Qwen3.5-35B-A3B      | MoE architecture -- verify Unsloth support |
| Llama-3.2-Vision-11B | Official Unsloth support                   |
| Pixtral-12B          | Mistral architecture supported             |
| Gemma-3-4B/12B       | Google architecture, check Unsloth docs    |

### Other Fine-Tuning Methods

| Model                     | Method                          |
| ------------------------- | ------------------------------- |
| MiniCPM-V-2.6             | OpenBMB fine-tuning             |
| InternVL3.5-4B/8B         | InternVL method                 |
| LLaVA-OneVision-1.5-4B/8B | Standard LLaVA (Qwen3 backbone) |
| DeepSeek-VL2              | DeepSeek method                 |
| Phi-4-Multimodal-Instruct | LoRA (Microsoft method)         |
| Mistral-Small-3.1-24B     | Standard LoRA                   |

### Cannot Fine-Tune (API Only)

GPT-4o, GPT-4o-Mini, Gemini-3-Flash, Gemini-3-Pro, Claude-4-Sonnet, Qwen-Max, Doubao-1.5-Vision-Pro-32k

---

## Mobile Export Guide

### Best Candidates for Mobile Deployment

| Priority | Model             | Q4 Size | Fine-tune        | Path        |
| -------- | ----------------- | ------- | ---------------- | ----------- |
| 1        | Qwen3-VL-2B       | ~1.5GB  | Unsloth verified | GGUF export |
| 2        | SmolVLM-500M      | <1GB    | Limited          | ONNX native |
| 3        | MobileVLM-V2-1.7B | ~1GB    | Standard         | TFLite/ONNX |
| 4        | Qwen2-VL-2B       | ~1.5GB  | Unsloth verified | GGUF export |

### Export Pipeline

```
Unsloth fine-tune -> GGUF Q4 -> llama.cpp / MLC-LLM -> Mobile
```

### Mobile-Native Models (No Fine-tuning)

| Model             | Size | Format | Notes          |
| ----------------- | ---- | ------ | -------------- |
| SmolVLM-256M      | <1GB | ONNX   | Purpose-built  |
| SmolVLM-500M      | ~1GB | ONNX   | Purpose-built  |
| MobileVLM-V2-1.7B | ~1GB | TFLite | Optimized      |
| Moondream2        | ~2GB | GGUF   | Fast inference |

---

## External Benchmarks

### U2-BENCH

> First comprehensive benchmark for VLMs on ultrasound understanding (arXiv 2505.17779, ICLR 2026)

| Aspect        | Details                                                           |
| ------------- | ----------------------------------------------------------------- |
| Samples       | 7,241 cases                                                       |
| Anatomy       | 15 regions (includes fetal)                                       |
| Tasks         | 8 clinical tasks                                                  |
| Models tested | 23 LVLMs                                                          |
| Dataset       | [HuggingFace](https://huggingface.co/datasets/DolphinAI/u2-bench) |
| Eval toolkit  | [GitHub](https://github.com/dolphin-sound/u2-bench-evalkit)       |

**8 Tasks:**

1. Disease Diagnosis (DD) - classification
2. View Recognition (VRA) - view classification
3. Lesion Localization (LL) - spatial detection
4. Organ Detection (OD) - anatomical structures
5. Keypoint Detection (KD) - biometry landmarks
6. Clinical Value Estimation (CVE) - regression
7. Report Generation (RG) - text generation
8. Caption Generation (CG) - descriptions

**Use for:** External validation, compare against 23 SOTA models, submit our fine-tuned models.

### FETAL-GAUGE

> Benchmark for VLMs in fetal ultrasound (arXiv 2512.22278)

Fetal-specific evaluation -- higher relevance to our task than U2-BENCH.

---

## Summary Stats

| Category                  | Count  | Tested    | TODO    |
| ------------------------- | ------ | --------- | ------- |
| Fetal/US-specific         | 7      | 0         | 7       |
| Current-gen Tier 1        | 16     | 3 (local) | 13 (GT) |
| Previous-gen (comparison) | 7      | 7 (GT)    | 0       |
| Medical                   | 8      | 3         | 5       |
| Mobile                    | 8      | 1 (perf)  | 7       |
| API                       | 7      | 2         | 5       |
| **Total**                 | **54** | **17**    | **37**  |

### By Fine-Tuning Method

| Method            | Count |
| ----------------- | ----- |
| Unsloth verified  | 7     |
| Unsloth likely    | 4     |
| Other methods     | 6     |
| API only          | 7     |
| Zero-shot/unknown | 30    |

### Mobile Viable

| Category                 | Count     |
| ------------------------ | --------- |
| Unsloth + mobile export  | 2-4       |
| Mobile-native models     | 8         |
| **Total mobile options** | **10-12** |

---

## Models Dropped from Previous List

| Model                       | Reason                                                               |
| --------------------------- | -------------------------------------------------------------------- |
| Qwen2-VL-2B/7B              | Superseded by Qwen3-VL (kept in "Already Tested" for comparison)     |
| Qwen2.5-VL-3B               | Superseded by Qwen3.5-35B-A3B (3B active, far better quality)        |
| InternVL2-2B/4B             | Superseded by InternVL3.5 (kept in "Already Tested")                 |
| InternVL3-78B               | Superseded by InternVL3.5-38B                                        |
| Pixtral-12B                 | Lower priority vs Mistral-Small-3.1-24B (same family, 3.1 is better) |
| Llama-3.2-Vision-11B        | Lower priority; no U2-BENCH results                                  |
| Gemma-3-Vision (duplicated) | Consolidated into single entries                                     |
| Claude-3.5-Sonnet           | Updated to Claude-4-Sonnet                                           |
