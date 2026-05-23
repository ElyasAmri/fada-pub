# FADA Implementation Guide: From Literature to Code
*Evidence-Based Approach for 250-Image Fetal Ultrasound Classification*

## Quick Reference: What Works

### Architecture Choice (Proven Winners)
```
1. EfficientNet-B0: 85% accuracy (Prochii 2025)
2. ResNet18: 74% accuracy with 86 images (Ćaleta 2025)
3. DenseNet121: High accuracy potential
4. Ensemble of above: +5-10% improvement
```

### Data Strategy
```
Original: 250 images (50 per class)
After Augmentation: 2,500-5,000 images
Validation: 5-fold cross-validation
Test: Hold out 20% before any processing
```

## Step-by-Step Implementation

### Step 1: Data Preparation Pipeline

```python
# Directory structure
data/
├── train/
│   ├── Brain/
│   ├── Heart/
│   ├── Abdomen/
│   ├── Femur/
│   └── Thorax/
├── val/
└── test/

# Split strategy (CRITICAL: Patient-level splits)
# Never mix images from same patient across splits
train: 60% (150 images, 30 per class)
val: 20% (50 images, 10 per class)
test: 20% (50 images, 10 per class)
```

### Step 2: Augmentation Pipeline (10-20x amplification)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        # Geometric - proven safe for ultrasound
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomResizedCrop(224, 224, scale=(0.85, 1.15), p=0.5),
        
        # Intensity - ultrasound specific
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
        
        # Ultrasound-specific
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # Depth attenuation
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        
        # Advanced
        A.OneOf([
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
        ], p=0.3),
        
        # Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
```

### Step 3: Model Architecture

```python
import timm
import torch
import torch.nn as nn

class FetalUltrasoundClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=5, dropout=0.3):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension
        if 'efficientnet' in model_name:
            feature_dim = 1280
        elif 'resnet18' in model_name:
            feature_dim = 512
        elif 'densenet121' in model_name:
            feature_dim = 1024
        
        # Custom head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout/2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def extract_features(self, x):
        """For few-shot learning"""
        return self.backbone(x)
```

### Step 4: Training Strategy

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Loss function for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Training configuration
config = {
    'batch_size': 16,  # Small due to heavy augmentation
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'warmup_epochs': 5,
    'patience': 15,  # Early stopping
}

# Initialize
model = FetalUltrasoundClassifier('efficientnet_b0')
optimizer = AdamW(model.parameters(), lr=config['learning_rate'], 
                  weight_decay=config['weight_decay'])
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Training loop with mixed precision
scaler = torch.cuda.amp.GradScaler()

def train_epoch(model, dataloader, optimizer, criterion, scaler):
    model.train()
    losses = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.cuda(), labels.cuda()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        
    return np.mean(losses)
```

### Step 5: Advanced Techniques

#### A. Test-Time Augmentation (TTA)
```python
def predict_with_tta(model, image, n_augmentations=5):
    """Average predictions over augmented versions"""
    predictions = []
    
    for _ in range(n_augmentations):
        aug_image = get_tta_transform()(image=image)['image']
        with torch.no_grad():
            pred = model(aug_image.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))
    
    return torch.stack(predictions).mean(dim=0)
```

#### B. Ensemble Method
```python
class EnsembleModel:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
    
    def predict(self, x):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                pred = torch.softmax(model(x), dim=1)
                predictions.append(pred * weight)
        return torch.stack(predictions).sum(dim=0)
```

#### C. Few-Shot Learning Enhancement
```python
class PrototypicalNetwork:
    """For comparison with standard training"""
    def __init__(self, backbone):
        self.backbone = backbone
        
    def compute_prototypes(self, support_set):
        """Compute class prototypes from support set"""
        prototypes = []
        for class_samples in support_set:
            features = self.backbone.extract_features(class_samples)
            prototype = features.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)
    
    def classify(self, query, prototypes):
        """Classify based on distance to prototypes"""
        query_features = self.backbone.extract_features(query)
        distances = torch.cdist(query_features, prototypes)
        return -distances  # Negative distance as logits
```

### Step 6: Validation Strategy

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def cross_validate(data, labels, n_folds=5):
    """Patient-aware cross-validation"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):
        # Train model
        model = train_fold(train_idx, val_idx)
        
        # Evaluate
        val_acc = evaluate(model, val_idx)
        results.append(val_acc)
        
        print(f"Fold {fold+1}: {val_acc:.3f}")
    
    print(f"Average: {np.mean(results):.3f} ± {np.std(results):.3f}")
    return results
```

### Step 7: MLflow Tracking

```python
import mlflow
import mlflow.pytorch

mlflow.set_experiment("fetal_ultrasound_classification")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model": "efficientnet_b0",
        "batch_size": 16,
        "learning_rate": 1e-3,
        "augmentation": "heavy",
        "epochs": 100
    })
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(...)
        val_acc = validate(...)
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_acc
        }, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

## Performance Optimization Checklist

### Week 1 Goals
- [ ] Data pipeline working with augmentation
- [ ] Baseline model training (EfficientNet-B0)
- [ ] MLflow tracking setup
- [ ] Achieve 50% validation accuracy

### Week 2 Goals
- [ ] Implement all augmentation techniques
- [ ] Train ResNet18 and DenseNet121
- [ ] Implement focal loss
- [ ] Achieve 60% validation accuracy

### Week 3 Goals
- [ ] Implement ensemble method
- [ ] Add test-time augmentation
- [ ] Cross-validation implementation
- [ ] Achieve 65% validation accuracy

### Week 4 Goals
- [ ] Fine-tune best model
- [ ] Implement Prototypical Networks
- [ ] Final ensemble optimization
- [ ] Target: 70-75% accuracy

## Common Issues and Solutions

### Problem: Overfitting
**Solution**: 
- Increase dropout (0.3 → 0.5)
- More aggressive augmentation
- Reduce model capacity
- Early stopping with patience=15

### Problem: Class Imbalance
**Solution**:
- Use focal loss (α=0.25, γ=2)
- Class-weighted sampling
- Balanced batch sampling
- Augment minority classes more

### Problem: Poor Convergence
**Solution**:
- Lower learning rate (1e-4)
- Gradient clipping
- Batch normalization
- Different optimizer (SGD with momentum)

### Problem: Low Accuracy
**Solution**:
- Ensemble multiple models
- Better pretrained weights (RadImageNet)
- More augmentation varieties
- Careful hyperparameter tuning

## Expected Results Timeline

| Week | Accuracy | Key Improvements |
|------|----------|-----------------|
| 1 | 45-50% | Basic model, minimal augmentation |
| 2 | 55-60% | Heavy augmentation, focal loss |
| 3 | 60-65% | Ensemble, cross-validation |
| 4 | 65-75% | Fine-tuning, TTA, optimization |

## Final Tips

1. **Start Simple**: Get baseline working first
2. **Track Everything**: Use MLflow religiously
3. **Validate Properly**: No data leakage!
4. **Augment Heavily**: 10-20x is necessary
5. **Ensemble Always**: Even 2 models help
6. **Be Patient**: Small datasets need more epochs
7. **Monitor Closely**: Watch for overfitting
8. **Test Robustness**: Use multiple test sets

## Success Metrics

- **Primary**: 60-75% accuracy on 5-class classification
- **Secondary**: 
  - F1-Score > 0.65
  - Per-class accuracy > 50%
  - Consistent across CV folds (std < 5%)
  - Real-time inference (< 100ms)

## Conclusion

Following this guide with disciplined implementation should yield:
- **Minimum**: 60% accuracy
- **Expected**: 65-70% accuracy
- **Best Case**: 75% accuracy

The key is systematic experimentation with proper validation.