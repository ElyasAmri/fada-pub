# Input and Output
- What we have: Dataset of fetal ultrasounds with different scopes (Abdomen, Cervix, etc.), currently in the process of annotations (16000 images have annotations; 80% so far)
- What we aim for: An ML model for determining anomalies with higher accuracies against other models that is also runnable on phone
- Objective: build a pipeline that will run on the annotated dataset once ready

# Details
### Anomalies
###### Sonographer annotates the dataset by answering 8 questions
1. Anatomical structures identification
2. Fetal orientation
3. Plane evaluation
4. Biometric measurements
5. Gestational age estimation
6. Image quality assessment
7. Normality/abnormality determination
8. Clinical recommendations

###### Workflow
1. Test all VL Models.

###### Models' Hyperparameters
1. Quantization Level: fp16, int8, int4
2. Parameter size (about 18B max; aim for something that works on mobile)
3. Mobile Export

# Goals
- [ ] Complete benchmark of all VLM models on the classification and captioning tasks
- [ ] Complete benchmark of all fine-tuned (LoRA, QLoRA, Unsloth) VLM models on the classification and captioning tasks
- [ ] Complete benchmark of all models mobile exported (after fine-tuning) on the classification and captioning tasks
- [ ] A complete mobile app that uses the best fine-tuned mobile exported model
- [ ] A paper detailing methodology and results

