"""
12-Class Fetal Ultrasound Classifier with class balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses learning on hard examples
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        """
        Args:
            alpha: Class weights for balancing
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss

        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()


class FetalUltrasoundClassifier12(nn.Module):
    """
    12-class ultrasound classifier with multiple backbone options
    """

    def __init__(
        self,
        num_classes: int = 12,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize classifier

        Args:
            num_classes: Number of output classes (12)
            backbone: Backbone architecture name
            pretrained: Use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # Create backbone
        if backbone == 'efficientnet_b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            # Modify classifier head
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            self.features_dim = in_features

        elif backbone == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            self.features_dim = in_features

        elif backbone == 'densenet121':
            if pretrained:
                self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.densenet121(weights=None)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            self.features_dim = in_features

        elif backbone == 'mobilenet_v2':
            if pretrained:
                self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.mobilenet_v2(weights=None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            self.features_dim = in_features

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        logger.info(f"Initialized {backbone} with {num_classes} classes")
        if pretrained:
            logger.info("Using ImageNet pretrained weights")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classification head"""
        if self.backbone_name == 'efficientnet_b0':
            # Get features from EfficientNet
            features = self.backbone.features(x)
            features = self.backbone.avgpool(features)
            features = torch.flatten(features, 1)
        elif self.backbone_name in ['resnet18']:
            # Get features from ResNet
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            features = self.backbone.avgpool(x)
            features = torch.flatten(features, 1)
        else:
            raise NotImplementedError(f"Feature extraction not implemented for {self.backbone_name}")

        return features


def create_model(
    num_classes: int = 12,
    backbone: str = 'efficientnet_b0',
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    class_weights: Optional[torch.Tensor] = None,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0
) -> Tuple[nn.Module, nn.Module]:
    """
    Create model and loss function

    Args:
        num_classes: Number of classes
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate
        class_weights: Class weights for imbalance
        use_focal_loss: Use focal loss instead of cross entropy
        focal_gamma: Gamma parameter for focal loss

    Returns:
        Model and loss function
    """
    # Create model
    model = FetalUltrasoundClassifier12(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )

    # Create loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        logger.info(f"Using Focal Loss with gamma={focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using Weighted Cross Entropy Loss")

    if class_weights is not None:
        logger.info(f"Class weights applied: {class_weights}")

    return model, criterion


if __name__ == "__main__":
    # Test model creation
    import numpy as np

    print("Testing 12-class model...")

    # Create model
    model, criterion = create_model(
        num_classes=12,
        backbone='efficientnet_b0',
        pretrained=True,
        use_focal_loss=True
    )

    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test loss calculation
    dummy_targets = torch.randint(0, 12, (4,))
    loss = criterion(output, dummy_targets)
    print(f"Loss: {loss.item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")