# FADA Project Specification

## 1. Project Structure
**Q: What is your preferred project structure?**
- [ ] Option A: Simple (notebooks/, data/, models/, results/)
- [ ] Option B: Modular (src/, experiments/, configs/, outputs/)
- [ ] Option C: Research-focused (papers/, baselines/, experiments/, analysis/)
- [x] Custom: **notebooks/, src/, data/, papers/, configs/, docs/, outputs/, results/**

**Q: Primary development environment?**
- [ ] Jupyter notebooks only
- [ ] Python scripts + notebooks for visualization
- [x] Mix of both
- [ ] Other: _______________

## 2. Data Pipeline
**Q: Current data format and location?**
- Image format: **PNG, RGB mode (3 channels), 1024x768 resolution**
- Excel annotation format: _______________
- Local path or cloud storage: **Local: data/Fetal Ultrasound/**

**Q: Train/validation/test split strategy?**
- [ ] 60/20/20
- [ ] 70/15/15
- [ ] 80/20 (no test set for now)
- [x] Cross-validation (specify k): **n-fold (configurable, start n=1 for quick testing, then n=5 for final results)**

**Q: How to handle the 7 unannotated organ types?**
- [ ] Ignore for now
- [ ] Use for unsupervised pre-training
- [ ] Manually annotate some samples
- [ ] Other: _______________

## 3. Baseline Model
**Q: What should be the first task to demonstrate?**
- [ ] Binary classification (normal vs abnormal)
- [x] 5-class classification (current annotated organs)
- [ ] Single organ detection (specify which): _______________
- [ ] Other: _______________
**Note: Plan two-stage approach for future - organ classification first, then abnormality detection**

**Q: Preferred deep learning framework?**
- [x] PyTorch
- [ ] TensorFlow/Keras
- [ ] JAX/Flax
- [ ] No preference

## 4. Model Architecture Zoo
**Q: Which architectures MUST be tested? (check all that apply)**
**BASED ON RESEARCH FINDINGS:**
- [x] **EfficientNet-B0** (best performance/parameter ratio, 85% accuracy in similar tasks)
- [x] **EfficientNet-B4** (balanced size/performance)
- [x] **ResNet18** (proven 74% with 86 images, lightweight)
- [x] **ResNet50** (excellent feature extractor, 88% in few-shot)
- [x] **DenseNet121** (95% AUC in ultrasound tasks)
- [x] **MobileNetV3** (for speed comparison)
- [x] **Vision Transformer (ViT-B/16)** (88.6% on ultrasound, if augmentation works)
- [x] **Simple CNN baseline** (custom 4-layer for comparison)

### Model Selection Strategy

**Testing Order:**
1. **Simple CNN** (baseline) - Custom 4-layer CNN for baseline comparison
2. **EfficientNet-B0** (primary) - Best for limited data, ImageNet pretrained
3. **ResNet18** (proven) - 74% accuracy demonstrated on similar dataset sizes
4. **DenseNet121** (if compute allows) - Strong performance on medical imaging
5. **Vision Transformer** (if accuracy < 70% with CNNs) - Alternative architecture

**Selection Criteria:**
- Start with pretrained models (ImageNet or RadImageNet)
- Heavy augmentation (rotation, brightness, contrast, noise)
- Test both pretrained and from-scratch for comparison
- Use focal loss for class imbalance handling
- Track with MLflow for all experiments

**Expected Performance:**
- With full annotations: 85-90% accuracy expected
- Focus on demonstrating proper model selection methodology
- Document all decisions for research paper

**Q: Pretrained weights preference?**
- [x] **ImageNet** (primary - all papers show it works)
- [x] **RadImageNet** (if available - medical specific)
- [x] **Test both** pretrained and from-scratch for comparison
**Note**: Literature shows ImageNet transfer works well for ultrasound (6-7% improvement)

## 5. Data Augmentation
**Q: Which augmentations are clinically acceptable? (check all)**
**BASED ON RESEARCH FINDINGS:**
- [x] **Rotation** (max degrees): **±30°**
- [x] **Horizontal flip** (standard in all papers)
- [ ] **Vertical flip** (not recommended - changes anatomy orientation)
- [x] **Brightness adjustment** (range): **±20%**
- [x] **Contrast adjustment** (range): **±20%**
- [x] **Gaussian noise** (σ=0.01)
- [x] **Gaussian blur** (mild, kernel 3x3)
- [ ] **Elastic deformation** (can distort anatomy)
- [x] **Random crop/zoom** (10-15% max)
- [x] **Ultrasound-specific**:
  - Speckle noise simulation
  - Depth-dependent attenuation
  - Acoustic shadow augmentation

**Q: Augmentation library preference?**
- [ ] Albumentations
- [ ] torchvision transforms
- [ ] tf.image
- [ ] imgaug
- [ ] No preference

## 6. Evaluation Metrics
**Q: Primary metric for model selection?**
- [ ] Accuracy
- [x] F1 Score (macro-averaged across 5 classes)
- [ ] AUC-ROC
- [ ] Sensitivity (recall)
- [ ] Specificity
- [ ] Balanced accuracy
- [ ] Other: _______________

**Q: Additional metrics to track? (check all)**
- [x] Overall accuracy
- [x] Per-class accuracy
- [x] Confusion matrix
- [x] Precision
- [x] Training time per epoch (secondary)
- [x] Model size (MB) (secondary)
- [x] Inference speed (images/sec) (secondary)
- [ ] Cohen's Kappa
- [ ] Matthews Correlation Coefficient
- [ ] Dice coefficient (if segmentation)
- [ ] IoU (if segmentation)
- [ ] FLOPs

## 7. Experiment Tracking
**Q: How to track experiments?**
- [x] MLflow (with local SQLite backend)
- [ ] Weights & Biases
- [ ] TensorBoard
- [ ] Neptune.ai
- [ ] CSV/Excel logging
- [ ] Custom solution
- [ ] No formal tracking

**Q: What to log for each experiment?**
- [ ] Hyperparameters
- [ ] Training curves
- [ ] Validation metrics
- [ ] Model checkpoints
- [ ] Augmentation samples
- [ ] Misclassified examples
- [ ] Gradient histograms
- [ ] Feature maps
- [ ] Other: _______________

## 8. Research Papers
**Q: Minimum number of papers to review?**
- [ ] 5-10
- [x] 10-20 (demonstrate thorough SOTA research)
- [ ] 20+

**Q: Paper sources priority? (rank 1-5)**
- [1] arXiv (recent preprints)
- [3] PubMed (medical focus)
- [5] IEEE Xplore
- [2] Conference proceedings (MICCAI, ISBI, MIDL)
- [4] Google Scholar

**Q: Focus areas for literature review? (check all)**
- [x] Fetal ultrasound analysis
- [x] General ultrasound AI
- [x] Medical image classification
- [x] Few-shot learning for medical imaging
- [x] Transfer learning in medical domain
- [ ] Other: _______________

## 9. Compute Resources
**Q: Available hardware?**
- [x] Local GPU (specify): **Laptop RTX 4070 (12GB), Desktop RX 7900 XTX (24GB), A6000 (48GB) later**
- [ ] Google Colab (free/Pro)
- [ ] Cloud (AWS/GCP/Azure)
- [ ] University cluster
- [ ] CPU only

**Q: Maximum training time per model?**
- [ ] < 1 hour
- [ ] 1-6 hours
- [ ] 6-24 hours
- [ ] Multiple days OK

## 10. Deliverables
**Q: Final output format?**
- [x] **Phase 1**: Jupyter notebook with all experiments (priority)
- [x] **Phase 2**: Web demo (Gradio/Streamlit) if Phase 1 successful
- [x] **Phase 3**: Technical paper draft if time permits
- [ ] Python package
- [ ] Docker container
- [ ] All of the above
**Note: Document every step meticulously for potential paper**

**Q: Results visualization priorities? (rank 1-6)**
- [1] Comparison table (all models)
- [4] ROC curves
- [3] Training curves
- [2] Confusion matrices
- [5] Sample predictions grid
- [6] Failure case analysis

## 11. Timeline
**Q: Project deadline?**
- Date: **December 2024 (< 3 months)**
- [ ] Flexible
- [x] Hard deadline

**Q: Hours per week available?**
- [x] < 10
- [ ] 10-20
- [ ] 20-40
- [ ] 40+

## 12. Success Criteria
**Q: Minimum acceptable performance?**
- Binary classification accuracy: **>70%** (typical: 85-95% with sufficient data)
- Multi-class accuracy: **Initial: >20% (beat random), Target: 60-70% if successful**
- [x] Any working model is OK for demo (start with beating random)

**Research findings for context:**
- **With large datasets (5000+ images)**: 75-93% accuracy for 5-6 classes
- **Best reported**: 93.6% for 6-class, 85% for 16-class problems
- **With limited data like yours (50/class)**: Expect 60-75% realistically

**Q: What constitutes "state-of-the-art" for your demo?**
- [x] Matching published results on similar data (adjusted for dataset size)
- [x] Using latest architectures (2023-2024)
- [ ] Novel approach or combination
- [x] Just using proper methodology
- [ ] Other: _______________

---
**Additional Notes/Constraints:**
_________________________________
_________________________________
_________________________________