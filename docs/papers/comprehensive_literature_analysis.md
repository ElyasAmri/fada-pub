# Comprehensive Literature Analysis for FADA Project
*Deep Learning for Fetal Ultrasound Classification with Limited Data*

## Executive Summary
Based on analysis of 40+ recent papers (2023-2025), we've identified critical strategies for achieving 60-75% accuracy with only 250 images (50 per class) in fetal ultrasound classification.

## 1. State-of-the-Art Performance Benchmarks

### Large Dataset Results (>5000 images)
- **Krishna et al. (2023)**: 93.6% accuracy for 6-class fetal plane classification using stacked ensemble
- **Sivasubramanian et al. (2024)**: 96.25% accuracy with EfficientNet + Attention (40x fewer parameters)
- **Fatima et al. (2025)**: 99.0% accuracy for breast cancer, 96.6% for fetal planes using InBnFUS + CNNDen-GRU

### Medium Dataset Results (500-5000 images)
- **Prochii et al. (2025)**: 85% overall accuracy on 16-class problem with 5,298 images
  - Used EfficientNet-B0 + EfficientNet-B6 ensemble
  - 90% of organs achieved >75% accuracy
  - Key: Hierarchical modular organization + LDAM-Focal loss

### Small Dataset Results (<500 images) - MOST RELEVANT
- **Ćaleta et al. (2025)**: 74% accuracy with only 86 images using ResNet18
  - 57 normal hearts, 29 CHD images
  - 5-fold cross-validation to prevent data leakage
- **Yang et al. (2023)**: 82.93% CHD detection, 92.79% VSD detection with YOLOv5
  - Small dataset augmented with various techniques
  - Real-time inference (0.007s per image)

## 2. Critical Success Factors for Your 250-Image Dataset

### A. Model Architecture Selection
Based on empirical evidence with similar data constraints:

1. **Primary Choice: EfficientNet-B0**
   - Prochii (2025): 85% on 16 classes
   - Sivasubramanian (2024): 96.25% with attention
   - Best parameter efficiency for limited data

2. **Strong Alternatives**:
   - **ResNet18**: 74% with 86 images (Ćaleta 2025)
   - **DenseNet121**: 99.84% accuracy reported (Ghabri 2023)
   - **MobileNetV3**: For deployment efficiency

### B. Data Augmentation Strategy (10-20x amplification)
Critical augmentations proven effective for ultrasound:

1. **Geometric Transformations**:
   - Rotation: ±30° (maintains anatomical validity)
   - Horizontal flip: Standard practice
   - Random crop/zoom: 10-15% maximum
   - Scaling: 0.8-1.2x

2. **Intensity Transformations**:
   - Brightness: ±20%
   - Contrast: ±20%
   - Gaussian noise: σ=0.01
   - Gaussian blur: 3x3 kernel

3. **Ultrasound-Specific**:
   - Speckle noise simulation
   - Depth-dependent attenuation
   - Acoustic shadow augmentation
   - SAS technique (Ferreira 2025): Scale and texture-aware augmentation

4. **Advanced Techniques**:
   - DreamOn (Lerch 2024): GAN-based augmentation improving robustness
   - CUT-based cross-modal augmentation (Guo 2025)
   - GANs for synthetic ultrasound generation (Fitas 2024, Tiago 2024)

### C. Few-Shot Learning Strategies

1. **Prototypical Networks**:
   - Işık & Paçal (2024): 88.9% accuracy with 10-shot on ultrasound
   - Best performer among meta-learning methods
   - Use ResNet50 as backbone

2. **Transfer Learning Hierarchy**:
   - Start with ImageNet pretrained weights
   - Fine-tune on RadImageNet if available
   - Final fine-tuning on your dataset

3. **Self-Supervised Pretraining**:
   - DINOv2 features (Ayzenberg 2024): Strong performance in medical FSL
   - BiomedCLIP: Best for very small datasets (<5 samples/class)
   - CLIP models: Better with 5+ samples/class

## 3. Implementation Roadmap for 60-75% Accuracy

### Phase 1: Baseline (Week 1)
```python
# Core Architecture
model = EfficientNetB0(pretrained='imagenet', num_classes=5)
optimizer = Adam(lr=1e-4)
loss = FocalLoss(alpha=0.25, gamma=2.0)  # For class imbalance
```

### Phase 2: Heavy Augmentation (Week 2)
- Implement 10-20x augmentation pipeline
- Use Albumentations library for efficiency
- Apply ultrasound-specific augmentations

### Phase 3: Ensemble & Optimization (Week 3)
- Train multiple models (EfficientNet-B0, ResNet18, DenseNet121)
- Implement weighted ensemble based on validation performance
- Use 5-fold cross-validation for robust evaluation

### Phase 4: Few-Shot Enhancement (Week 4)
- Implement Prototypical Networks for comparison
- Use feature extraction from best performing model
- Apply metric learning losses (contrastive, triplet)

## 4. Expected Performance Timeline

Based on similar studies:
- **Week 1**: 40-50% (basic model, no augmentation)
- **Week 2**: 55-65% (with augmentation)
- **Week 3**: 60-70% (ensemble)
- **Week 4**: 65-75% (with optimization)

## 5. Key Papers for Implementation

### Must Read (Download Priority)
1. **Prochii et al. (2025)** - "Biologically Inspired Deep Learning" - ArXiv:2506.08623
   - Your exact use case: 16-class fetal classification
   - Detailed architecture and training strategy

2. **Sivasubramanian et al. (2024)** - "Efficient Feature Extraction" - ArXiv:2410.17396
   - Lightweight architecture perfect for RTX 4070
   - Attention mechanisms for improved accuracy

3. **Işık & Paçal (2024)** - "Few-shot classification of ultrasound"
   - Prototypical Networks implementation
   - 88.9% with minimal data

## 6. Critical Implementation Tips

### Training Strategy
1. **Learning Rate Schedule**: 
   - Start: 1e-3 with warmup
   - Cosine annealing or ReduceLROnPlateau
   - Final: 1e-5 for fine-tuning

2. **Regularization**:
   - Dropout: 0.3-0.5
   - Weight decay: 1e-4
   - MixUp/CutMix augmentation

3. **Validation Strategy**:
   - 5-fold cross-validation
   - Patient-level splits (no data leakage)
   - Stratified sampling

### Common Pitfalls to Avoid
1. **Data Leakage**: Never mix images from same patient in train/val
2. **Over-augmentation**: Keep anatomical validity
3. **Class Imbalance**: Use focal loss or class weights
4. **Overfitting**: Monitor validation loss closely

## 7. Performance Expectations

### Realistic Targets with 250 Images:
- **5-class organ classification**: 60-75%
- **Binary abnormality detection**: 70-85%
- **Per-class accuracy variation**: ±15%

### Factors Affecting Performance:
- Image quality consistency
- Annotation quality
- Class balance
- Augmentation effectiveness

## 8. Advanced Techniques (If Time Permits)

1. **Attention Mechanisms**:
   - Channel attention (SE blocks)
   - Spatial attention
   - Self-attention layers

2. **Multi-Task Learning**:
   - Joint organ + abnormality classification
   - Auxiliary tasks (quality assessment)

3. **Test-Time Augmentation**:
   - Average predictions over augmented versions
   - 3-5% accuracy boost typically

## 9. Evaluation Metrics

### Primary Metrics:
- **Macro F1-Score**: Best for imbalanced classes
- **Top-1 Accuracy**: Standard comparison
- **Cohen's Kappa**: Agreement measure

### Per-Class Analysis:
- Confusion matrix
- Precision-Recall curves
- Class-wise F1 scores

## 10. Next Steps

1. **Immediate Actions**:
   - Download top 3 papers
   - Set up data pipeline with augmentation
   - Implement baseline EfficientNet-B0

2. **Week 1 Goals**:
   - Achieve 50% baseline accuracy
   - Validate augmentation pipeline
   - Set up MLflow tracking

3. **Success Criteria**:
   - Consistent 60%+ accuracy across folds
   - Robust to test set variations
   - Real-time inference capability

## Conclusion

With proper implementation of:
- EfficientNet-B0 architecture
- 10-20x data augmentation
- Transfer learning from ImageNet
- Ensemble methods

**Achieving 60-75% accuracy with 250 images is realistic and well-supported by literature.**

The key is systematic experimentation with careful validation to avoid overfitting.