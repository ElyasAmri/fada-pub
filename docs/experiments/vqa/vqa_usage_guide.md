# VQA Model Usage Guide

## Overview

The FADA system includes Visual Question Answering (VQA) models trained on fetal ultrasound images using BLIP-2 with LoRA adapters. Each anatomical category has its own specialized model.

## Available Models

### Trained Categories (as of Oct 2, 2025)

| Category | Model Path | Images | Status |
|----------|------------|--------|--------|
| Non_standard_NT | `outputs/blip2_1epoch/final_model` | 487 | ✅ Trained |
| Abdomen | `outputs/blip2_abdomen/final_model` | 2424 | ✅ Trained |
| Femur | `outputs/blip2_femur/final_model` | 1165 | ✅ Trained |
| Thorax | `outputs/blip2_thorax/final_model` | 1793 | ✅ Trained |
| Standard_NT | `outputs/blip2_standard_nt/final_model` | 1508 | ✅ Trained |

### Categories Awaiting Labeled Data

- Trans-cerebellum (brain)
- Trans-thalamic (brain)
- Trans-ventricular (brain)
- Cervix

## Usage

### Basic Usage

```python
from src.models.vqa_model import UltrasoundVQA
from PIL import Image

# Load model for specific category
vqa = UltrasoundVQA(model_path="outputs/blip2_abdomen/final_model")
vqa.load_model()

# Load image
image = Image.open("path/to/ultrasound.png")

# Ask question
answer = vqa.answer_question(
    image,
    "What anatomical structures are visible?",
    max_new_tokens=100
)

print(answer)
```

### Standard Questions

The models are trained on 8 standard medical questions:

1. **Anatomical Structures**: List all visible anatomical structures
2. **Fetal Orientation**: Describe the orientation of the fetus
3. **Plane Evaluation**: Assess if the image is taken at a standard diagnostic plane
4. **Biometric Measurements**: Identify any biometric measurements that can be taken
5. **Gestational Age**: Estimate the gestational age range if possible
6. **Image Quality**: Evaluate the overall image quality and diagnostic utility
7. **Normality/Abnormality**: Identify any visible abnormalities
8. **Clinical Recommendations**: Provide clinical recommendations

### Batch Processing

```python
# Answer multiple questions
questions = vqa.STANDARD_QUESTIONS[:3]  # First 3 questions
results = vqa.answer_all_questions(image, questions)

for question, answer in results.items():
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Web Interface Integration

The Streamlit web app automatically loads category-specific models:

1. User uploads ultrasound image
2. Classification model detects organ type (e.g., "Abdomen")
3. System loads corresponding VQA model (`blip2_abdomen`)
4. User can ask questions or click standard question buttons

## Generation Parameters

### Current Settings (Optimized)

```python
{
    "max_new_tokens": 100,      # Maximum answer length
    "min_new_tokens": 5,        # Minimum answer length
    "num_beams": 3,             # Beam search width
    "no_repeat_ngram_size": 3,  # Prevent 3-gram repetition
    "repetition_penalty": 1.2,  # Penalty for repeated tokens
    "do_sample": False,         # Deterministic generation
    "early_stopping": True      # Stop when done
}
```

### Adjusting for Different Use Cases

**More Detailed Answers:**
```python
answer = vqa.answer_question(image, question, max_new_tokens=200)
```

**Faster Inference:**
```python
# Use fewer beams
answer = vqa.answer_question(image, question, max_new_tokens=50)
```

## Performance Expectations

### Inference Time
- Model Loading: ~15 seconds (first use only)
- Single Question: ~2-4 seconds on RTX 4070
- Batch Questions: ~15-25 seconds for 8 questions

### Memory Requirements
- GPU Memory: ~4.2 GB per loaded model
- CPU RAM: ~2 GB
- Storage: ~10 MB per LoRA adapter

### Answer Quality

**Expected Behaviors:**
- ✅ Identifies major anatomical structures
- ✅ Describes general image characteristics
- ✅ Provides medically relevant observations
- ⚠️ May hallucinate minor details
- ⚠️ Not suitable for clinical diagnosis

**Training Scale:**
- 1-epoch models (5 images): Basic validation
- 3-epoch models (full dataset): Production-ready
- Accuracy improves significantly with full training

## Best Practices

### 1. Model Selection
```python
# GOOD: Use category-specific model
category = "Abdomen"  # From classification
vqa = UltrasoundVQA(f"outputs/blip2_{category.lower()}/final_model")

# ACCEPTABLE: Use Non_standard_NT as fallback
vqa = UltrasoundVQA("outputs/blip2_1epoch/final_model")
```

### 2. Question Formulation

**GOOD Questions:**
- "What anatomical structures are visible in this image?"
- "Describe the image quality and clarity"
- "Are there any abnormalities visible?"

**POOR Questions:**
- "What is the exact gestational age?" (too specific)
- "Is this normal?" (too vague)
- "Should we do more tests?" (out of scope)

### 3. Memory Management

```python
# Load model
vqa = UltrasoundVQA(model_path)
vqa.load_model()

# Use model
answers = vqa.answer_all_questions(image)

# Unload when done (frees ~4.2 GB GPU memory)
vqa.unload_model()
```

### 4. Error Handling

```python
try:
    vqa = UltrasoundVQA(model_path)
    vqa.load_model()
    answer = vqa.answer_question(image, question)
except Exception as e:
    print(f"VQA error: {e}")
    # Fallback to rule-based response or classification-only mode
```

## Troubleshooting

### Issue: Repetitive Output

**Symptom:** Answer repeats the same phrase multiple times

**Solution:** Already fixed in current version with:
- `no_repeat_ngram_size=3`
- `repetition_penalty=1.2`
- Post-processing via `_clean_repetitions()`

### Issue: Nonsensical Output (alphabet, random text)

**Symptom:** Model generates "a,b,c,d..." or unrelated text

**Causes:**
- Generation parameters too aggressive (high temperature, sampling)
- Model not properly loaded
- LoRA adapters missing

**Solution:**
- Use conservative parameters (current defaults)
- Verify model path exists
- Check that both base model and adapters load successfully

### Issue: Out of Memory

**Symptom:** CUDA out of memory error

**Solutions:**
1. Unload other models before loading VQA
2. Use only one VQA model at a time
3. Clear CUDA cache: `torch.cuda.empty_cache()`
4. Reduce batch size if processing multiple images

### Issue: Slow First Inference

**Symptom:** First question takes 15+ seconds

**Explanation:** Model loads lazily on first use. Subsequent questions are fast (2-4s).

**Solution:** Pre-load model in initialization:
```python
vqa = UltrasoundVQA(model_path)
vqa.load_model()  # Load immediately instead of lazy loading
```

## Advanced Usage

### Custom Training

To train on additional categories:

1. Prepare labeled Excel file with 8 questions per image
2. Copy training notebook template
3. Update paths and category name
4. Run training with papermill:

```bash
python -m papermill \
    notebooks/train_blip2_template.ipynb \
    notebooks/train_blip2_custom_executed.ipynb \
    -p num_images 100 \
    -p num_epochs 3
```

### Model Evaluation

Evaluate all models:

```bash
python evaluate_all_vqa.py
```

Results saved to `outputs/vqa_evaluation_results.json`

### Fine-tuning Existing Models

```python
# Load existing model
vqa = UltrasoundVQA("outputs/blip2_abdomen/final_model")
vqa.load_model()

# Continue training with new data
# (requires custom training script - see notebooks/)
```

## API Reference

### UltrasoundVQA Class

```python
class UltrasoundVQA:
    def __init__(self, model_path, base_model="Salesforce/blip2-opt-2.7b", device="auto")
    def load_model(self)
    def answer_question(self, image, question, max_new_tokens=100) -> str
    def answer_all_questions(self, image, questions=None, max_new_tokens=100) -> Dict[str, str]
    def get_question_shortcuts(self) -> Dict[str, str]
    def unload_model(self)
```

### Helper Functions (web/app.py)

```python
def get_vqa_model_for_category(category: str) -> str
def load_category_vqa(category: str) -> UltrasoundVQA
```

## Future Enhancements

### Planned Improvements

1. **Multi-modal responses**: Combine VQA with classification outputs
2. **Confidence scores**: Report answer confidence
3. **Attention visualization**: Show which parts of image influenced answer
4. **Multi-image QA**: Answer questions about image sequences
5. **Fine-grained biometrics**: Specific measurement extraction

### Research Directions

1. Compare BLIP-2 vs other VLMs (MedGemma, LLaVA-Med)
2. Evaluate answer quality against ground truth
3. Active learning for efficient labeling
4. Zero-shot transfer across similar anatomical views

## References

- BLIP-2 Paper: https://arxiv.org/abs/2301.12597
- LoRA Paper: https://arxiv.org/abs/2106.09685
- FADA Project Documentation: `info/project.md`
- Training Results: `info/vqa_training_summary.md`
