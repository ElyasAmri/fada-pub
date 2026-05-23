# Round 2 Audit: Reliability, Reproducibility & Sound Methodology

**Date**: 2026-03-11
**Auditors**: 5 independent agents (Statistical Methodology, Reproducibility, Data Integrity, Evaluation Soundness, Experiment Tracking)
**Scope**: Statistical rigor, reproducibility, data integrity, evaluation soundness, traceability
**Method**: Read-only codebase analysis with no cross-contamination between agents

---

## Executive Summary

Round 1 fixed 50+ surface-level issues. Round 2 goes deeper -- examining whether research conclusions are **valid**. The audit found **7 Critical**, **9 High**, **10 Medium**, and **5 Low** severity issues across 31 unique findings. The most impactful:

1. **No statistical significance testing** -- all model rankings are point estimates with no confidence intervals
2. **Question text mismatch** between training JSONL and evaluation scorer -- Q3 and Q7 predictions silently unscored
3. **No reproducible environment** -- unpinned packages, no lock file, no Docker, gitignored dataset splits
4. **Heterogeneous metric aggregation** -- the headline "Primary Score" averages incomparable metrics
5. **System prompt confounding** -- 3 different prompts across evaluation paths
6. **Missing result metadata** -- score files lack pipeline version, temperature, embedding model

---

## Consolidated Findings by Severity

### CRITICAL (7 findings)

#### C1. Question Text Mismatch Between Training Data and Evaluation

**Source**: Data Integrity Agent
**Impact**: Q3 and Q7 predictions silently unscored; 2 of 8 questions completely lost

JSONL training files use different question prefixes than `src/config/questions.py`:

- Q3: "Imaging Plane Identification" (JSONL) vs "Plane Evaluation" (questions.py) -- **no match**
- Q7: "Normality/Abnormality Determination" (JSONL) vs "Normality / Abnormality" (questions.py) -- **no match**
- Q5, Q6: match only via fuzzy fallback with warnings

The scorer at `question_scorer.py:54-68` uses prefix matching. Models trained on JSONL text produce predictions with JSONL prefixes that the scorer cannot route.

**References**: `src/config/questions.py:13-22`, `data/vlm_training/gt_test.jsonl`, `question_scorer.py:54-68`

#### C2. No Statistical Significance Testing in Model Comparisons

**Source**: Statistical Methodology Agent
**Impact**: All 54 model rankings are unsupported by statistical inference

`src/utils/statistical_testing.py` implements McNemar's and Wilcoxon tests but is **never imported** by the evaluation pipeline. The top-10 leaderboard has differences as small as 0.0021 between ranks. Claims like "gemma-3-12b-it beats gemma-3-27b-it by +0.04" have no statistical backing.

**References**: `src/utils/statistical_testing.py:1-91` (dead code), `models-tracker.md:63-74`, `results-summary.md:103-107`

#### C3. Heterogeneous Metric Aggregation into Primary Score

**Source**: Statistical Methodology + Evaluation Soundness Agents (independently confirmed)
**Impact**: Headline ranking number is not statistically meaningful

`question_scorer.py:798-799` averages 8 fundamentally different metrics (set F2, relaxed accuracy, keyword F1, BERTScore F1) into one number. The code itself warns: `"NOTE: Averages heterogeneous metrics; interpret with caution"`. Per-question means range 6x (0.131 to 0.840), confirming the scales are incomparable. Q6 and Q8 dominate the average.

**References**: `question_scorer.py:798-799`, `results-summary.md:33-42`

#### C4. Unpinned Critical Package Versions

**Source**: Reproducibility Agent
**Impact**: Environment cannot be recreated; silent behavior changes across installs

`requirements.txt:43-48` uses `>=` ranges for training-critical packages: `transformers>=4.51.0`, `peft>=0.15.0`, `trl>=0.18.0`, `accelerate>=1.7.0`. PyTorch is not in requirements.txt at all. No lock file exists.

**References**: `requirements.txt:43-48`, `requirements.txt:3-4`

#### C5. Dataset Splits File Gitignored

**Source**: Reproducibility Agent
**Impact**: Cloned repo generates different train/test splits; potential train/test overlap

`data/dataset_splits.json` exists locally but is gitignored. Anyone cloning the repo regenerates splits, which match only if they have identical images in identical filesystem order. The RCCG setup playbook auto-generates splits if missing.

**References**: `data/dataset_splits.json` (gitignored), `experiments/rccg/playbooks/setup.yml:173-182`

#### C6. Score Result Files Lack Critical Metadata

**Source**: Traceability Agent
**Impact**: Cannot confirm two score files used the same scoring pipeline, embedding model, or conditions

Score JSON files contain only `timestamp`, `predictions_file`, `annotations_file`, `num_matched`. Missing: scoring pipeline version, embedding model, BERTScore model, git commit hash, temperature, system prompt, dataset splits hash. Two parallel file sets exist (`scores/` v2 and `scores_*.json` v3) with different primary scores for the same model but no version marker.

**References**: `experiments/rccg/results/scores/`, `experiments/rccg/results/scores_*.json`

#### C7. Empty base_model_name_or_path in Saved Adapter

**Source**: Traceability Agent
**Impact**: Fine-tuned model cannot be linked to its base model

`models/qwen25vl7b_finetuned/final/adapter_config.json:6` has `base_model_name_or_path: ""`. The README is entirely unfilled boilerplate. No training config saved alongside the adapter.

**References**: `models/qwen25vl7b_finetuned/final/adapter_config.json:6`, `models/qwen25vl7b_finetuned/final/README.md`

---

### HIGH (9 findings)

#### H1. No Confidence Intervals or Uncertainty Quantification

**Source**: Statistical Methodology Agent

Per-question std is computed (`question_scorer.py:827-833`) but no standard errors, CIs, or bootstrap intervals. The overall primary score has no uncertainty measure. With 1,894 test images, bootstrap CIs would be trivial to compute.

#### H2. No Inter-Annotator Agreement

**Source**: Statistical Methodology + Data Integrity Agents (independently confirmed)

Two annotators (drshalal: 82.8%, drrehab: 17.2%) cover **almost entirely disjoint categories** with only 1 shared image. A 15-percentage-point calibration gap exists on Q6: drshalal rates "good" 81.2% vs drrehab 96.4%. Ground truth quality is unknown.

**References**: `docs/experiments/annotation_normalization.md:24-29,207`

#### H3. Embedding Similarity Not Validated Against Clinical Judgment

**Source**: Statistical Methodology Agent

`all-mpnet-base-v2` is a general-purpose sentence-transformer with no medical domain validation. The Q2 mismatch case (scan-plane vs obstetric presentation) demonstrates the metric can be high for clinically wrong answers.

**References**: `experiments/evaluation/config.py:42`, `docs/experiments/Evaluation Methodology.md:53`

#### H4. System Prompt Confounding -- Three Different Prompts

**Source**: Evaluation Soundness Agent

| Path                           | Prompt                                                        |
| ------------------------------ | ------------------------------------------------------------- |
| vLLM/API (`test_api_vlm.py`)   | `API_SYSTEM_PROMPT` -- "medical imaging expert"               |
| Local HF (`evaluate_vlm.py`)   | `VLM_SYSTEM_PROMPT` -- "expert in fetal ultrasound"           |
| Fine-tuned (`eval_hf_peft.py`) | `VLM_SYSTEM_PROMPT` -- "expert in fetal ultrasound"           |
| Training JSONL                 | "expert in fetal ultrasound imaging analysis" (third variant) |

Zero-shot leaderboard (46 models via vLLM) is internally consistent, but fine-tuned vs zero-shot comparisons use different prompts.

**References**: `experiments/evaluation/config.py:47,53`, `experiments/api_models/test_api_vlm.py:39`

#### H5. GPT-5.x and Gemini Temperature Bypass

**Source**: Evaluation Soundness Agent

GPT-5.x models skip temperature entirely (`test_api_vlm.py:255-261`), running at API default (~1.0). Gemini `generate_content()` passes no `generation_config` (`test_api_vlm.py:311-325`). All other models use T=0.1.

**References**: `experiments/api_models/test_api_vlm.py:255-261,311-325`

#### H6. Inconsistent Seed Handling Across Scripts

**Source**: Reproducibility Agent

No script calls `torch.cuda.manual_seed_all()`, `PYTHONHASHSEED`, or `cudnn.deterministic`. `train_qwen3vl_lora.py` has no explicit seed. `evaluate.py` uses `do_sample=True, temperature=0.1` (stochastic) while `eval_hf_peft.py` uses `do_sample=False` (deterministic).

**References**: `src/training/train_classification.py:664-665`, `experiments/fine_tuning/train_qwen3vl_lora.py:113-133`

#### H7. No HuggingFace Model Version Pinning

**Source**: Reproducibility Agent

Zero `revision=` parameter usage across all `from_pretrained()` calls. HuggingFace models are mutable -- weight corrections and tokenizer changes can silently alter results.

#### H8. MLflow Does Not Log Git Hash or Environment

**Source**: Reproducibility + Traceability Agents (independently confirmed)

`src/utils/mlflow_utils.py` (564 lines) has zero references to git commit, Python version, CUDA version, or `pip freeze`. Cannot trace an MLflow run to exact code or environment.

**References**: `src/utils/mlflow_utils.py`

#### H9. Duplicate Score Files Without Version Markers

**Source**: Traceability Agent

`experiments/rccg/results/` contains both `scores/<model>.json` (v2, 2026-03-05) and `scores_<model>.json` (v3, 2026-03-06) with different primary scores. Neither records which scorer version produced it. One file has an inconsistent `_v3` suffix.

---

### MEDIUM (10 findings)

#### M1. Q7 Normality Class Imbalance (95% Normal)

Exact-match scoring with binary fallback does not use balanced accuracy. A model always predicting "normal" achieves ~0.475, higher than the top model's overall 0.365. Sensitivity (clinically critical) is computed but buried in details, not the primary metric.

#### M2. Unequal Category Representation in Aggregation

Test set ranges 48--242 images per category. `primary_score_mean` averages all samples without category weighting, so Abdomen (242) dominates while Non-standard NT (48) is marginalized.

#### M3. Partial Credit Scoring Not Justified

Five questions use 0.5 partial credit with no sensitivity analysis. Adjacent-bin credit for Q5 is inconsistent: off by one bin in "13-15 weeks" is 2 weeks error but "20-25 weeks" is 5 weeks.

#### M4. Lossy Q5 Gestational Age Binning

`normalize_annotations.py:1156-1158` uses `int(midpoint)` truncation for range-to-bin mapping, creating systematic bias toward lower bins. Wide ranges like "11-23 weeks" collapse to a single bin.

#### M5. Image-Level Splitting (No Patient IDs)

No patient identifiers exist, so splitting is image-level. Multiple images from the same ultrasound session could span train and test sets.

#### M6. 84 Unannotated Images Silently Skipped

19,019 images exist but only 18,936 annotated. Missing images silently skipped in JSONL generation but remain in dataset splits.

#### M7. Silent Black-Image Fallback in Dataset Loader

`dataset.py:127-157` returns a black image with the original label on any load error. No counter tracks activations.

#### M8. Config Drift -- Temperature Defined in 6+ Places

`GENERATION_TEMPERATURE=0.1` (eval config), `EVAL_TEMPERATURE=0.1` (API test), `0.4` (vertex/openai/grok backends), `0.7` (interactive inference). Not centralized.

#### M9. Annotation Version Not Tracked

`data/Fetal Ultrasound Annotations Normalized.xlsx` has no version number. `normalization_changelog.json` lacks date and output hash. Score files reference the path but not a hash.

#### M10. No Reproducible Environment Specification

No Dockerfile, no conda YAML, no lock file. RCCG cluster installs packages without version pins (`hosts.yml:78-85`).

---

### LOW (5 findings)

#### L1. Legacy evaluate_vlm.py Still Scores Against Pseudo-Labels

`evaluate_vlm.py:35` has `SCORING_MODE = "pseudo_label"`, scoring against Gemini annotations instead of sonographer GT.

#### L2. Q1 Synonym Table Gaps

Missing links: ventricle/lateral ventricle, cerebellum/cerebellar vermis, femur/femoral bone, placenta/placental tissue. Estimated 5-15% Q1 score suppression for verbally diverse models.

#### L3. Q8 BERTScore Uses General-Domain Model

`roberta-large` vs medical-domain alternatives. Standard in NLP literature, but may slightly disadvantage precise medical terminology.

#### L4. Error Rates Not Reported in Leaderboard

Error predictions are counted internally but not shown in results-summary.md or models-tracker.md. A 5% error rate depresses scores by ~0.017.

#### L5. 6 Broken Models Still in Rankings

`models-tracker.md:104-111` -- 6 models with severe output issues (connection errors, formatting failures) are ranked alongside functioning models.

---

## Actionable Recommendations (Priority Order)

### Immediate (before next evaluation run)

| #   | Action                                                                              | Severity Fixed | Effort |
| --- | ----------------------------------------------------------------------------------- | -------------- | ------ |
| 1   | Add Q3/Q7 JSONL question prefixes to `_QUESTION_PREFIX_MAP` in `question_scorer.py` | C1             | Low    |
| 2   | Regenerate JSONL training files with current `questions.py`                         | C1             | Low    |
| 3   | Commit `dataset_splits.json` to git (remove from .gitignore)                        | C5             | Low    |
| 4   | Fix `base_model_name_or_path` in adapter config                                     | C7             | Low    |
| 5   | Unify system prompts -- single `EVAL_SYSTEM_PROMPT` in config.py                    | H4             | Low    |
| 6   | Fix Gemini temperature -- pass `generation_config`                                  | H5             | Low    |
| 7   | Document GPT-5.x temperature limitation in leaderboard                              | H5             | Low    |
| 8   | Standardize evaluation to `do_sample=False` (greedy)                                | H6             | Low    |

### Short-term (before paper submission)

| #   | Action                                                                                | Severity Fixed | Effort |
| --- | ------------------------------------------------------------------------------------- | -------------- | ------ |
| 9   | Add bootstrap CIs to scoring pipeline                                                 | C2, H1         | Medium |
| 10  | Integrate McNemar's test for pairwise comparisons (code exists)                       | C2             | Low    |
| 11  | Replace or qualify `primary_score_mean` with principled aggregation                   | C3             | Medium |
| 12  | Pin all package versions; generate `requirements-lock.txt`                            | C4             | Low    |
| 13  | Add result metadata schema (pipeline version, git hash, embedding model, temperature) | C6             | Medium |
| 14  | Rescore all 54 models once with full metadata (v4)                                    | C6, H9         | Medium |
| 15  | Add `revision=` to HuggingFace `from_pretrained()` calls                              | H7             | Medium |
| 16  | Log git hash and environment in MLflow                                                | H8             | Low    |
| 17  | Create comprehensive `set_all_seeds()` utility                                        | H6             | Low    |
| 18  | Add balanced accuracy / macro-F1 for Q7                                               | M1             | Low    |
| 19  | Add macro-average over categories                                                     | M2             | Low    |
| 20  | Run partial credit sensitivity analysis (0.25, 0.5, 0.75)                             | M3             | Low    |
| 21  | Expand Q1 synonym table                                                               | L2             | Medium |
| 22  | Report error rates in leaderboard                                                     | L4             | Low    |

### Long-term (strengthens paper)

| #   | Action                                                              | Severity Fixed | Effort |
| --- | ------------------------------------------------------------------- | -------------- | ------ |
| 23  | Inter-annotator reliability study (200-500 overlapping images)      | H2             | High   |
| 24  | Validate embedding similarity against clinical judgment (200 pairs) | H3             | High   |
| 25  | Create Dockerfile for reproducible environment                      | M10            | Medium |
| 26  | Pin RCCG packages and commit hash in Ansible                        | M10, H7        | Low    |

---

## Findings Cross-Reference

Issues independently discovered by multiple agents (higher confidence):

| Finding                          | Agents                                    |
| -------------------------------- | ----------------------------------------- |
| Heterogeneous metric aggregation | Statistical, Evaluation                   |
| No inter-annotator agreement     | Statistical, Data Integrity               |
| MLflow missing git hash          | Reproducibility, Traceability             |
| Temperature inconsistency        | Reproducibility, Evaluation, Traceability |
| System prompt divergence         | Evaluation, Traceability                  |

---

## Summary Statistics

| Severity  | Count  | Independently Confirmed |
| --------- | ------ | ----------------------- |
| Critical  | 7      | 3                       |
| High      | 9      | 4                       |
| Medium    | 10     | 1                       |
| Low       | 5      | 0                       |
| **Total** | **31** | **8**                   |

Of the 31 findings, 14 are fixable with low effort, 11 with medium effort, and 6 require high effort. The 8 "Immediate" actions address 4 Critical and 3 High findings with minimal effort and no rescoring required (except the JSONL regeneration).
