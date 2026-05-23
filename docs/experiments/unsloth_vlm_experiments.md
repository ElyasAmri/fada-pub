# Unsloth VLM Fine-tuning Experiments

This document outlines future experiments for fine-tuning Qwen3-VL on the fetal ultrasound dataset.

## Current Implementation (Baseline)

**Task**: Q7 - Normality Assessment only
**Model**: Qwen3-VL-8B-Instruct (4-bit quantization)
**Fine-tuning**: Vision + Language layers with LoRA (r=16)

### Files
- `experiments/unsloth_vlm/prepare_dataset.py` - Data loading
- `experiments/unsloth_vlm/train_qwen3vl.py` - Training script
- `experiments/unsloth_vlm/evaluate.py` - Evaluation metrics
- `experiments/unsloth_vlm/inference.py` - Interactive inference

---

## Future Experiments

### Experiment 1: Multi-Task Training (All Q1-Q8)

**Goal**: Train model to answer all 8 question types in separate turns.

**Approach**:
- Create 8 separate prompts, one for each question type
- Train on all Q1-Q8 simultaneously
- Evaluate each question type separately

**Data Format**:
```python
# Each image generates 8 training samples
questions = [
    ("Q1", "Describe the anatomical structures visible in this ultrasound."),
    ("Q2", "What is the fetal orientation in this image?"),
    ("Q3", "Identify the imaging plane shown."),
    ("Q4", "What biometric measurements can be obtained?"),
    ("Q5", "Estimate the gestational age based on visible features."),
    ("Q6", "Assess the image quality."),
    ("Q7", "Assess the normality of this ultrasound image."),
    ("Q8", "What clinical recommendations would you suggest?"),
]
```

**Expected Outcome**:
- Model learns to respond to different question types
- May require more training epochs due to task diversity
- Risk of catastrophic forgetting between tasks

**Metrics**:
- Per-question accuracy
- Overall response coherence
- Cross-question consistency

---

### Experiment 2: Combined Comprehensive Report

**Goal**: Generate a single comprehensive report covering all 8 aspects.

**Approach**:
- Concatenate Q1-Q8 answers into a structured report
- Single prompt: "Provide a complete analysis of this fetal ultrasound"
- Output: Formatted report with all 8 sections

**Data Format**:
```python
combined_answer = f"""
## Anatomical Structures
{q1_answer}

## Fetal Orientation
{q2_answer}

## Imaging Plane
{q3_answer}

## Biometric Measurements
{q4_answer}

## Gestational Age
{q5_answer}

## Image Quality
{q6_answer}

## Normality Assessment
{q7_answer}

## Clinical Recommendations
{q8_answer}
"""
```

**Expected Outcome**:
- More natural conversational interface
- Longer output sequences (may need max_seq_length > 2048)
- Better context between sections

**Metrics**:
- Section completeness (does output cover all 8 areas?)
- Section accuracy (compare to ground truth)
- Clinical coherence (do recommendations match findings?)

---

### Experiment 3: Layer Ablation Study

**Goal**: Determine optimal fine-tuning configuration for medical imaging.

**Configurations to Test**:

| Config | Vision | Language | Attention | MLP |
|--------|--------|----------|-----------|-----|
| A | Yes | No | Yes | Yes |
| B | No | Yes | Yes | Yes |
| C | Yes | Yes | Yes | No |
| D | Yes | Yes | No | Yes |
| E (Full) | Yes | Yes | Yes | Yes |

**Hypothesis**:
- Vision-only may be sufficient for visual grounding
- Language-only may be enough if base model has medical knowledge
- Full fine-tuning likely best but most expensive

**Metrics**:
- Q7 accuracy per configuration
- Training time per configuration
- VRAM usage per configuration

---

### Experiment 4: Model Size Comparison

**Goal**: Compare performance vs. cost across model sizes.

| Model | Params | VRAM (4-bit) | Expected Train Time |
|-------|--------|--------------|---------------------|
| Qwen3-VL-2B | 2B | ~6GB | Fast |
| Qwen3-VL-4B | 4B | ~8GB | Medium |
| Qwen3-VL-8B | 8B | ~12GB | Slow |

**Metrics**:
- Q7 accuracy
- Inference latency (tokens/sec)
- Memory usage
- Training throughput (samples/sec)

**Trade-off Analysis**:
- Is 8B significantly better than 4B for this task?
- Can 2B achieve acceptable performance for deployment?

---

### Experiment 5: Prompt Engineering Variations

**Goal**: Find optimal prompts for each question type.

**Prompt Variations for Q7**:

1. **Simple**: "Is this ultrasound normal?"
2. **Detailed**: "Assess the normality of this fetal ultrasound image. Describe whether the visible structures appear normal or if there are any abnormalities."
3. **Clinical**: "As a radiologist, evaluate this fetal ultrasound for any abnormal findings."
4. **Structured**: "Evaluate this image and respond with one of: NORMAL, ABNORMAL, UNCLEAR. Then provide your reasoning."

**Metrics**:
- Response accuracy
- Response consistency
- Response format compliance

---

### Experiment 6: Data Augmentation Impact

**Goal**: Assess impact of image augmentation on model robustness.

**Augmentation Strategies**:
- Basic: Rotation, flip, brightness, contrast
- Advanced: Elastic deformation, ultrasound-specific noise
- None: Original images only

**Metrics**:
- Validation accuracy with/without augmentation
- Robustness to image quality variations
- Performance on edge cases

---

### Experiment 7: Class-Balanced Training

**Goal**: Handle imbalanced Q7 categories (mostly "normal").

**Strategies**:
1. Oversampling abnormal cases
2. Weighted loss function
3. Focal loss
4. Synthetic abnormal generation (if feasible)

**Current Distribution**:
- "within normal" variants: ~95%
- "abnormal" variants: ~5%

---

## Experiment Tracking

All experiments should be tracked using MLflow:

```python
import mlflow

mlflow.set_experiment("unsloth_vlm_ultrasound")

with mlflow.start_run(run_name="experiment_name"):
    mlflow.log_params({
        "model": "Qwen3-VL-8B",
        "task": "Q7",
        "fine_tune_vision": True,
        "lora_r": 16,
    })
    mlflow.log_metrics({
        "val_accuracy": 0.85,
        "train_loss": 0.23,
    })
```

---

## Priority Order

1. **Baseline Q7** (current implementation)
2. **Model Size Comparison** (determine optimal size)
3. **Layer Ablation** (optimize training config)
4. **Multi-Task Q1-Q8** (expand capability)
5. **Combined Report** (production format)
6. **Class-Balanced Training** (improve rare case detection)
7. **Prompt Engineering** (optimize UX)
8. **Data Augmentation** (robustness)

---

## Completed Experiments

### Baseline Q7 Training (2026-01-24)

**Configuration**:
- Model: Qwen3-VL-8B-Instruct (4-bit quantization via Unsloth)
- Task: Q7 - Normality Assessment (binary: Normal/Abnormal)
- Dataset: 15,446 images (13,901 train / 1,545 val)
- LoRA: r=16, alpha=16, all-linear layers
- Training: 3 epochs, batch size 2 x 4 gradient accumulation
- Hardware: RTX 5090 Laptop GPU (24GB VRAM)

**Training Results**:
- Total steps: 5,214
- Training time: 3h 58m
- Final loss: 0.0464
- Loss progression: 4.886 -> 0.02-0.04 range (rapid convergence)
- GPU VRAM usage: ~10GB during training

**Inference Results** (50 random test images):
- Accuracy: 41/50 (82.0%)
- Strong performance on: Abdomen, Femur, Aorta, Thorax, Cervix images
- Challenging cases: NT (Nuchal Translucency) images
  - Model tends to predict "Abnormal" for NT images even when labeled "Normal"
  - May indicate model learned NT-specific caution (clinically appropriate?)

**Observations**:
1. Model converges quickly (loss <0.1 by step 100)
2. Dataset loading is the bottleneck (~10 min for chunked loading)
3. NT images are inherently ambiguous - borderline cases common
4. 82% accuracy is reasonable baseline for binary classification

**Saved Artifacts**:
- LoRA adapters: `experiments/unsloth_vlm/outputs/qwen3vl_ultrasound/lora_adapters/`
- Adapter size: 196MB (adapter_model.safetensors)

**Next Steps**:
- Investigate NT image misclassifications
- Test on held-out test set (not random validation samples)
- Consider class-balanced training for rare abnormal cases

---

## Notes

- All experiments should save LoRA adapters separately for comparison
- Use same validation set across experiments for fair comparison
- Document VRAM usage and training time for each experiment
- Consider running experiments on Vast.ai for larger models (Qwen3-VL-32B)
