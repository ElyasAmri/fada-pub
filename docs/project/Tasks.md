---
tags: [tasks]
---

# Active Tasks

## Current Phase: 4 - Full Test Set Evaluation

### Completed (Phase 4: Full Test Set Evaluation)

- [x] Ground-truth evaluation of 54 models on 1,894 test images (see models-to-test.md for current count)
- [x] Expand Qwen2.5-VL-7B evaluation from 600 to 1,894 images
- [x] Score existing Gemini batch annotation results (62 batches)
- [x] Score MedGemma-4B results (medgemma_4b_complete.json)

### Completed (Phase 1: VLM Benchmarking)

- [x] VLM benchmark framework
- [x] Test 50+ models
- [x] Identify top performers (MiniCPM-V-2.6, Qwen2-VL-2B, InternVL2-4B)
- [x] Document results

### Completed (Phase 2: Fine-Tuning)

- [x] Qwen2.5-VL-7B fine-tuned to 81.1% (600 samples, embedding similarity -- NOTE: 600-sample subset; full test set v3 score is embed_sim=0.5058)
- [x] Qwen3-VL-8B Q7 baseline at 82% (50 samples)
- [x] Unsloth fine-tuning verified for all 7 Qwen models (RTX 5090)
- [x] Cloud infrastructure built (RCCG A100/H100)

### Completed (Phase 3: Cloud API Evaluation)

- [x] MedGemma-27B evaluated at 78.81% (709 samples, embedding similarity against Gemini pseudo-labels -- Phase 1 proxy scoring, NOT GT sonographer annotations)
- [x] Gemini batch annotations (62 batches)

### Upcoming

- [ ] Comparative fine-tuning benchmark (top 3 models, same config)
- [ ] Mobile deployment evaluation
- [ ] Web demo with best model
- [ ] Paper writing

## Quick Links

| Phase                   | Status   | Notes                   |
| ----------------------- | -------- | ----------------------- |
| Phase 1: VLM Benchmarks | Complete | 50+ models              |
| Phase 2: Fine-Tuning    | Complete | Qwen2.5-VL-7B at 81.1%  |
| Phase 3: Cloud API      | Complete | MedGemma-27B at 78.81%  |
| Phase 4: Full Test Set  | Complete | 1,894 images, 54 models |
| Phase 5: Web/App Demo   | Pending  | -                       |

## Blockers

- Need standalone inference + GT-eval script for local open-weight models

## Notes

- Full dataset: 19,019 images, 14 classes, 18,936 annotated
- Hardware: RTX 5090 (24GB VRAM) local, RCCG (A100/H100) for cloud
- Best verified GT score: 81.1% (Qwen2.5-VL-7B fine-tuned)
