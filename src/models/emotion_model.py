import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

class EmotionModel(nn.Module):
    """
    Refactored Emotion Recognition Model using Transfer Learning.
    Replaces the previous hybrid/custom architecture with a modern backbone.
    """
    def __init__(self, 
                 backbone_name: str = 'efficientnet_b0', 
                 num_classes: int = 7, 
                 pretrained: bool = True,
                 dropout: float = 0.4):
        super(EmotionModel, self).__init__()
        
        self.backbone_name = backbone_name
        
        # Load backbone
        if backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity() # Remove default classifier
        elif backbone_name == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch, 3, 224, 224)
        Returns:
            Logits of shape (batch, num_classes)
        """
        features = self.backbone(x)
        # Handle case where backbone returns a 4D tensor (e.g. from global pooling)
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
        logits = self.classifier(features)
        return logits

def create_model(config: dict) -> EmotionModel:
    """Helper to create model from config."""
    return EmotionModel(
        backbone_name=config['model']['backbone'],
        num_classes=config['emotions']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    )
