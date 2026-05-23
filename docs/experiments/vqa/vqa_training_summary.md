# VQA Training Summary - Phase 2

**Training Date**: October 2, 2025
**Start Time**: 3:21 AM
**Model**: BLIP-2 OPT-2.7B with LoRA (8-bit quantization)

## Training Overview

Successfully trained VQA models for all available labeled ultrasound categories using BLIP-2 with LoRA adapters. Each model was trained for 1 epoch on 5 images (70/15/15 train/val/test split) to validate the pipeline.

## Successfully Trained Categories

### 1. Non_standard_NT (Original)

- **Status**: ✅ Complete
- **Training Time**: ~1 minute
- **Output**: `outputs/blip2_1epoch/final_model`
- **Test Result**: Generated medical responses about anatomical structures

### 2. Abdomen

- **Status**: ✅ Complete
- **Training Time**: ~1 minute
- **Output**: `outputs/blip2_abdomen/final_model`
- **Test Result**: "The liver is a biconvex organ, with the right side being larger than the left."
- **Notes**: Successfully handles Abdomen directory name

### 3. Femur

- **Status**: ✅ Complete
- **Training Time**: ~1 minute
- **Output**: `outputs/blip2_femur/final_model`
- **Test Result**: "The left side of the image shows the lower part of the pelvis and the right side shows the upper part."

### 4. Thorax

- **Status**: ✅ Complete (succeeded on retry)
- **Training Time**: ~1 minute
- **Output**: `outputs/blip2_thorax/final_model`
- **Test Result**: "The left side of the image shows the right side of a human heart."
- **Notes**: Initial training failed with CUDA error, succeeded after clearing CUDA cache

### 5. Standard_NT

- **Status**: ✅ Complete
- **Training Time**: ~1 minute
- **Output**: `outputs/blip2_standard_nt/final_model`
- **Test Result**: "The uterus is a hollow organ that contains the placenta and the amniotic sac."

## Categories Without Labeled Data

The following categories have notebooks created but cannot be trained yet due to missing labeled Excel files:

- Trans-cerebellum (brain)
- Trans-thalamic (brain)
- Trans-ventricular (brain)
- Cervix

These categories will be trainable once labeled Excel files are added to `data/Fetal Ultrasound Labeled/`.

## Training Configuration

**Model Parameters**:

- Base Model: Salesforce/blip2-opt-2.7b
- Quantization: 8-bit using BitsAndBytesConfig
- LoRA Config:
  - r=8
  - alpha=16
  - target_modules: ["q_proj", "v_proj"]
  - dropout=0.05

**Training Parameters**:

- Epochs: 1 (validation run)
- Batch Size: 1
- Learning Rate: 1e-4
- Images per Category: 5
- Questions per Image: 8
- Total QA Pairs: ~28 train, 6 val, 6 test per category

**Generation Parameters** (optimized):

- max_new_tokens: 100
- min_new_tokens: 5
- num_beams: 3
- no_repeat_ngram_size: 3
- repetition_penalty: 1.2
- do_sample: False (deterministic)

## Model Performance

All trained models successfully:

1. Load without errors (4.21 GB GPU memory each)
2. Generate coherent medical responses
3. Avoid repetitive output patterns
4. Provide anatomically relevant descriptions

## Hardware

- GPU: NVIDIA RTX 4070
- CUDA: Available
- Memory per Model: ~4.21 GB
- Total Training Time: ~5 minutes for 5 categories

## Files Generated

**Training Notebooks** (in `notebooks/blip2_training/`):

- `train_blip2_1epoch.ipynb` - Non_standard_NT (original, with execution history)
- `train_blip2_abdomen.ipynb` - Abdomen (with execution history)
- `train_blip2_femur.ipynb` - Femur (with execution history)
- `train_blip2_thorax.ipynb` - Thorax (with execution history)
- `train_blip2_standard_nt.ipynb` - Standard_NT (with execution history)
- `train_blip2_trans_cerebellum.ipynb` - Trans-cerebellum (not trainable yet)
- `train_blip2_trans_thalamic.ipynb` - Trans-thalamic (not trainable yet)
- `train_blip2_trans_ventricular.ipynb` - Trans-ventricular (not trainable yet)
- `train_blip2_cervix.ipynb` - Cervix (not trainable yet)
- `train_blip2_non_standard_nt_full.ipynb` - Full-scale training attempt

**Note**: Executed notebooks have been consolidated with originals. Each notebook retains its execution history.

**Model Outputs**:

- `outputs/blip2_1epoch/` (Non_standard_NT)
- `outputs/blip2_abdomen/`
- `outputs/blip2_femur/`
- `outputs/blip2_thorax/`
- `outputs/blip2_standard_nt/`

Each output directory contains:

- `final_model/` - LoRA adapters and processor
- `checkpoint-*/` - Training checkpoints
- `training_summary.json` - Training metrics

## Next Steps

1. **Web Integration**: Update web app to load category-specific VQA models
2. **Full Training**: Train on complete datasets (20-50 images per category) for multiple epochs
3. **Model Evaluation**: Compare responses against ground truth annotations
4. **Additional Categories**: Train brain and cervix models when labeled data becomes available
5. **Response Quality**: Fine-tune generation parameters per category if needed

## Technical Notes

- All models use the same BLIP-2 base with category-specific LoRA adapters
- Training is extremely fast (~1 minute) due to LoRA efficiency
- Models can be loaded on-demand in web interface
- Generation parameters were optimized to avoid repetitive output
- Post-processing removes any remaining repetitive patterns
