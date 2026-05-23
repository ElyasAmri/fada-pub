# Fetal Anomaly Detection Algorithm (FADA)

This project is focused on developing a machine learning model to assist in the detection of fetal anomalies using ultrasound images. The goal is to create a tool that can help sonographers and obstetricians identify potential issues during pregnancy, improving prenatal care and outcomes.

## Dataset Overview

### Available Data

We have a comprehensive dataset of 14 fetal organ types (CRL-View and NT-View added since initial writing):

- Aorta
- Cervical region
- Abdomen
- Femur
- Standard and non-standard NT (Nuchal Translucency)
- Cervix
- Thorax
- Pubic symphysis/fetal head
- Trans-cerebellum, trans-thalamic, and trans-ventricular views

### Annotations

Currently annotated by a sonographer:

- **50 images per category** for 5 organ types:
  - Abdomen
  - Femur
  - Non-standard NT
  - Standard NT
  - Thorax

### Annotation Schema

Each annotated image includes 8 standardized questions:

1. Anatomical structures identification
2. Fetal orientation
3. Plane evaluation
4. Biometric measurements
5. Gestational age estimation
6. Image quality assessment
7. Normality/abnormality determination
8. Clinical recommendations

### Data Locations

- Raw ultrasound images: Available at [Fetal-Ultrasound]
- Annotated data: Excel format at [Fetal-Ultrasound-labeled]

## Project Phases

| Phase   | Focus              | Status      |
| ------- | ------------------ | ----------- |
| Phase 1 | VLM Benchmarking   | Complete    |
| Phase 2 | Fine-Tuning        | In Progress |
| Phase 3 | Further Benchmarks | Pending     |
| Phase 4 | Mobile Deployment  | Pending     |
| Phase 5 | Web/App Demo       | Pending     |

NOTE: Phase definitions and current status may differ from above. See docs/project/Tasks.md for current task tracking.

## Steps and Approach

### Research

1. **Literature Review**: Conduct a thorough review of recent advancements in medical imaging AI. Find state-of-the-art techniques for training models.

2. **Model Listing**: Identify potential models suitable for our task. We will test each and compare the results.

3. **Pre-processing Techniques**: Explore and document effective pre-processing methods for ultrasound images. Research state-of-the-art data augmentation techniques specific to ultrasound imaging, applicable to our dataset.

### Model Selection

1. **Evaluation Criteria**: Define criteria for model selection based on accuracy, interpretability, and computational efficiency. Make sure this works. Create a unified metric system that allows for easy comparison across different models.

2. **Model Comparison**: Compare identified models against the criteria.

### Training

1. **Data Augmentation**: Implement data augmentation strategies to enhance the dataset, including controlled transformations that preserve clinical relevance.

2. **General Design**: Design a training pipeline that can work with the currently limited annotated data, and then later once the annotations are expanded.

3. **Transfer Learning**: Utilize transfer learning from larger medical imaging models, with selective fine-tuning of critical layers.

### Evaluation

1. **Performance Metrics**: Define and implement performance metrics to evaluate model accuracy and reliability.

2. **Validation**: Use cross-validation techniques to ensure the model's robustness.

### Deployment

Develop a user-friendly web application for clinicians to upload images and receive interpretations.
