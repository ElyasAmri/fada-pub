# Image Captioning vs Classification for FADA Project

## Task Clarification

### What You Originally Planned (Classification)
- **Input**: Fetal ultrasound image
- **Output**: Single label (e.g., "Brain", "Abdomen", "Heart")
- **Complexity**: Simple, well-established
- **Data Needed**: 50-250 images per class
- **Expected Accuracy**: 60-75% with your data

### What You're Now Requesting (Captioning)
- **Input**: Fetal ultrasound image
- **Output**: Descriptive text (e.g., "Fetal abdominal ultrasound at 20 weeks showing normal stomach bubble, kidney development, and umbilical cord insertion")
- **Complexity**: Much harder, requires understanding multiple features
- **Data Needed**: Thousands of image-caption pairs
- **Current State**: Limited research in ultrasound captioning

## Critical Issue: Your Data

Your annotation schema has 8 questions per image:
1. Anatomical structures identification
2. Fetal orientation
3. Plane evaluation
4. Biometric measurements
5. Gestational age estimation
6. Image quality assessment
7. Normality/abnormality determination
8. Clinical recommendations

**BUT**: These are structured answers, NOT free-text captions!

## Available Approaches for Captioning

### 1. Traditional Image Captioning Models
**Architecture Options**:
- **Encoder-Decoder**: CNN (ResNet/EfficientNet) + LSTM/Transformer
- **Vision Transformers**: ViT encoder + GPT-2 decoder
- **Show and Tell variants**: CNN + attention + LSTM

**Problems for your case**:
- Need thousands of image-caption pairs
- You only have 250 images with structured annotations
- No free-text descriptions available

### 2. Modern Vision-Language Models (VLMs)
**Pre-trained Options**:
- **BLIP/BLIP-2**: Vision-language pre-training
- **CLIP**: Contrastive learning (better for retrieval than generation)
- **Flamingo/IDEFICS**: Few-shot capable
- **BiomedCLIP**: Medical-specific CLIP

**Advantages**:
- Can work with less data via fine-tuning
- Some support few-shot learning

**Disadvantages**:
- Still need some caption data
- Medical VLMs are rare
- Ultrasound-specific models don't exist

### 3. Medical Report Generation Models
From the papers found:
- **MAIRA-2** (Microsoft): Chest X-ray report generation
- **RadTex**: CNN-Transformer for radiology reports
- **CT2Rep**: 3D CT report generation

**Problem**: All focus on CT/X-ray, NOT ultrasound

## Recommended Approach for Your Project

### Option 1: Structured Caption Generation (Feasible)
Convert your 8-question annotations into structured captions:
```python
def create_caption(annotations):
    caption = f"Fetal {annotations['organ']} ultrasound showing "
    caption += f"{annotations['structures']}. "
    caption += f"Orientation: {annotations['orientation']}. "
    caption += f"Assessment: {annotations['normality']}."
    return caption
```

Then fine-tune a small model like:
- **GPT-2** or **T5-small** with your CNN features
- **BLIP** with medical prompts

### Option 2: Multi-Task Learning (Better)
Combine classification with attribute prediction:
1. Classify organ (5 classes)
2. Predict attributes (normal/abnormal, quality, orientation)
3. Template-based caption from predictions

### Option 3: Use Pre-trained VLM (Quickest)
1. Use **BiomedCLIP** or **BLIP** off-the-shelf
2. Add prompt engineering: "Describe this fetal ultrasound image"
3. Fine-tune on your 250 images with synthetic captions

## Implementation Complexity

| Approach | Data Needed | Time to Implement | Expected Quality |
|----------|------------|-------------------|------------------|
| Classification (original) | 250 images ✓ | 1-2 weeks | Good (60-75%) |
| Structured Captioning | 250 images ✓ | 2-3 weeks | Moderate |
| Full Captioning | 1000+ pairs ✗ | 4-6 weeks | Poor with your data |
| Pre-trained VLM | 250 images ✓ | 2-3 weeks | Moderate |

## My Recommendation

**Stick with classification first**, then add structured captioning:

1. **Phase 1**: 5-class organ classification (2 weeks)
   - Achieve 60-75% accuracy
   - Establish baseline

2. **Phase 2**: Multi-attribute prediction (1 week)
   - Normal/abnormal
   - Quality assessment
   - Orientation

3. **Phase 3**: Template-based captioning (1 week)
   - Combine predictions into sentences
   - Use rule-based templates

4. **Phase 4** (if time): Fine-tune BLIP (2 weeks)
   - Generate synthetic captions from annotations
   - Fine-tune small VLM

## Required Changes to Project

If you want true captioning:
1. **Data**: Need to create captions from your Excel annotations
2. **Architecture**: Switch from CNN to encoder-decoder
3. **Metrics**: Change from accuracy to BLEU, METEOR, CIDEr
4. **Timeline**: Add 2-3 weeks
5. **Expectations**: Lower quality than classification

## Papers for Captioning Implementation

### Must Read:
1. **"From vision to text: A comprehensive review"** (2024)
   - DOI: 10.1016/j.media.2024.103264
   - Complete overview of medical captioning

2. **"MAIRA-2: Grounded Radiology Report Generation"** (2024)
   - ArXiv: https://arxiv.org/pdf/2406.04449
   - State-of-the-art approach (adapt for ultrasound)

3. **"Vision Language Transformation for Medical Image Captioning"** (2024)
   - Compares CNN encoders for medical captioning
   - Includes ultrasound in dataset

### Useful Resources:
- **BiomedCLIP**: https://github.com/microsoft/BiomedCLIP
- **BLIP**: https://github.com/salesforce/BLIP
- **Medical captioning datasets**: MIMIC-CXR, IU-Xray (for reference)

## Decision Needed

**Which approach do you want?**
1. Original classification plan (recommended)
2. Classification + structured captions
3. Full free-text captioning (not recommended with 250 images)
4. Pre-trained VLM fine-tuning

Please clarify so we can update the project specification accordingly.