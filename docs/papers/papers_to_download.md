# Papers to Download - FADA Project

## High Priority Papers (Direct Implementation Relevance)

### 1. Biologically Inspired Deep Learning for Fetal Ultrasound (2024)
- **Authors**: Prochii et al.
- **Key Results**: 85% accuracy on 16-class fetal classification
- **Architecture**: EfficientNet-B0 + B6 ensemble
- **ArXiv**: https://arxiv.org/abs/2506.08623
- **PDF**: https://arxiv.org/pdf/2506.08623
- **Why Important**: Most similar to your task, provides architectural details

### 2. Efficient Feature Extraction Using Light-Weight CNN (2024)
- **Authors**: Sivasubramanian et al.
- **Key Results**: 96.25% accuracy with 40x fewer parameters
- **Architecture**: EfficientNet + Attention
- **ArXiv**: https://arxiv.org/abs/2410.17396
- **PDF**: https://arxiv.org/pdf/2410.17396
- **Why Important**: Lightweight models perfect for your RTX 4070

### 3. Standard Fetal Ultrasound Plane Classification (2023)
- **Authors**: Krishna & Kokil
- **Key Results**: 93.6% on 6-class fetal planes
- **Architecture**: Stacked ensemble
- **DOI**: https://doi.org/10.1016/j.eswa.2023.122153
- **Semantic Scholar**: https://www.semanticscholar.org/paper/c36914961abbbbe53108c96f9b5099f2a835bd3a
- **Why Important**: Ensemble methods for fetal imaging

### 4. Few-shot Classification of Ultrasound Breast Cancer (2024)
- **Authors**: Işık & Paçal
- **Key Results**: 88.9% with 10-shot learning
- **Architecture**: ProtoNet + ResNet50
- **DOI**: https://doi.org/10.1007/s00521-024-09767-y
- **PDF**: https://link.springer.com/content/pdf/10.1007/s00521-024-09767-y.pdf
- **Why Important**: Directly addresses small dataset problem

### 5. Classification of Normal/Abnormal Fetal Heart (2023)
- **Authors**: Yang et al.
- **Key Results**: 82.93% CHD detection, 92.79% VSD detection
- **Architecture**: YOLOv5 variants
- **DOI**: https://doi.org/10.1515/jpm-2023-0041
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/37178239/
- **Why Important**: Small dataset success (similar size to yours)

## Medium Priority Papers (Techniques and Methods)

### 6. Transfer Learning for Fetal Organ Classification (2023)
- **Authors**: Nature Scientific Reports
- **DOI**: https://doi.org/10.1038/s41598-023-44689-0
- **Link**: https://www.nature.com/articles/s41598-023-44689-0
- **Why Important**: Transfer learning strategies for fetal ultrasound

### 7. Evaluation of Deep CNNs for Maternal Fetal Ultrasound (2020)
- **Authors**: Nature Scientific Reports
- **DOI**: https://doi.org/10.1038/s41598-020-67076-5
- **Link**: https://www.nature.com/articles/s41598-020-67076-5
- **Why Important**: Benchmark paper with standard evaluation metrics

### 8. Deep Learning Based Detection of Fetal Lip (2024)
- **Authors**: Li et al.
- **Key Results**: 92.5% accuracy with YOLOv5-ECA
- **DOI**: https://doi.org/10.1515/jpm-2024-0122
- **Semantic Scholar**: https://www.semanticscholar.org/paper/7b9289e4a89866ad270853199eca2cb7bdc50d6b
- **Why Important**: Similar data size (632 images)

### 9. Automated Classification of Liver Fibrosis (2024)
- **Authors**: Park et al.
- **Key Results**: All models (VGG, ResNet, Dense, Efficient, ViT) achieved 95-96% AUC
- **DOI**: https://doi.org/10.1186/s12880-024-01209-4
- **PDF**: https://bmcmedimaging.biomedcentral.com/counter/pdf/10.1186/s12880-024-01209-4
- **Why Important**: Direct comparison of all architectures on ultrasound

### 10. Few Shot Learning for Medical Imaging (2023)
- **Authors**: Nayem et al.
- **ArXiv**: https://arxiv.org/abs/2305.04401
- **PDF**: https://arxiv.org/pdf/2305.04401
- **Why Important**: Comprehensive review of few-shot techniques

## Low Priority Papers (Background and Context)

### 11. Vision Transformers and Siamese Networks for CL/P (2024)
- **Authors**: Nantha et al.
- **Key Results**: 82.76% with minimal examples
- **DOI**: https://doi.org/10.3390/jimaging10110271
- **PDF**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11595968
- **Why Important**: Multimodal fusion approach

### 12. EfficientNet-B7 for Breast Ultrasound (2024)
- **Authors**: Latha et al.
- **Key Results**: 99.14% accuracy
- **DOI**: https://doi.org/10.1186/s12880-024-01404-3
- **Link**: https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-024-01404-3
- **Why Important**: EfficientNet optimization strategies

### 13. Deep Learning Approaches for Breast US Classification (2022)
- **Authors**: Paçal
- **Key Results**: ViT 88.6%, comparison of all architectures
- **DOI**: https://doi.org/10.21597/jist.1183679
- **Semantic Scholar**: https://www.semanticscholar.org/paper/700c0ada4b44cb8894ea8134d30009256ec90775
- **Why Important**: Architecture comparison on ultrasound

### 14. MediNet: Transfer Learning Approach (2023)
- **Authors**: Reis et al.
- **Key Results**: 98.71% after transfer learning (vs 94.84% baseline)
- **DOI**: https://doi.org/10.1007/s11042-023-14831-1
- **PDF**: https://link.springer.com/content/pdf/10.1007/s11042-023-14831-1.pdf
- **Why Important**: Progressive transfer learning strategy

### 15. Review on Deep Learning in Medical Ultrasound (2024)
- **Journal**: Frontiers in Physics
- **Link**: https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2024.1398393/full
- **Why Important**: Comprehensive survey of current methods

## Additional Conference Papers

### MICCAI/ISBI Papers (Search for these)
- MICCAI 2023/2024 proceedings on fetal ultrasound
- ISBI 2023/2024 ultrasound classification tracks
- MIDL 2024 medical imaging papers

## GitHub Repositories to Check

1. **MONAI** (Medical Open Network for AI)
   - https://github.com/Project-MONAI/MONAI
   - Pre-trained medical models

2. **TorchIO** (Medical imaging toolkit)
   - https://github.com/fepegar/torchio
   - Augmentation specific to medical images

3. **RadImageNet** (Pretrained models)
   - https://github.com/BMEII-AI/RadImageNet
   - Medical imaging pretrained weights

## Download Instructions

### Open Access Papers (Can download directly):
- Papers 1, 2, 4, 9, 10, 11, 14 have direct PDF links
- Use wget or curl to download:
```bash
wget https://arxiv.org/pdf/2506.08623 -O prochii_2024_biologically_inspired.pdf
wget https://arxiv.org/pdf/2410.17396 -O sivasubramanian_2024_efficient_feature.pdf
```

### Papers requiring institutional access:
- Papers 3, 5, 6, 7, 8, 12, 13 may require institutional access
- Try Sci-Hub as alternative (use at your discretion)

### Semantic Scholar API:
- Can use their API to get metadata and sometimes PDFs
- https://api.semanticscholar.org/