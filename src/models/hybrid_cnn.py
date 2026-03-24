"""
Hybrid CNN Architecture for Facial Emotion Recognition using EfficientNet-B0 backbone technique

This module implements the hybrid CNN with:
1. Global CNN - processes full face (224×224) using EfficientNet-B0 backbone technique
2. Zone CNNs - 5 parallel CNNs for facial zones (48×48 each)
3. Feature fusion - concatenates global + zone features

Academic Justification:
- Global CNN captures holistic face structure and context using EfficientNet-B0 backbone technique
- Zone CNNs capture localized micro-expressions and action units
- Concatenation provides complementary information
- Multiple studies show 5-10% accuracy improvement over single CNN

Architecture Design:
- Global CNN: EfficientNet-B0 backbone technique for complex patterns
- Zone CNNs: Lightweight (3 conv blocks) for efficiency
- All use BatchNorm + Dropout for regularization
- ReLU activation for non-linearity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Tuple, List, Optional


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2D + BatchNorm + ReLU + MaxPool.
    
    Standard building block for CNNs.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_pool: bool = True,
                 pool_size: int = 2):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool2d(pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_pool:
            x = self.pool(x)
        return x


class GlobalCNN(nn.Module):
    """
    Global CNN for processing full face image using EfficientNet-B0 backbone.
    
    Input: (batch, 1, 224, 224) - grayscale face
    Output: (batch, 512) - global feature vector
    
    Architecture:
    - Pretrained EfficientNet-B0
    - Modified first layer to accept 1 channel (grayscale)
    - Custom FC layer to match feature_dim (512)
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 feature_dim: int = 512,
                 dropout: float = 0.5,
                 pretrained: bool = True):
        super(GlobalCNN, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Load EfficientNet-B0
        if pretrained:
            try:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            except AttributeError:
                self.backbone = models.efficientnet_b0(pretrained=True)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            
        # Modify first layer if input is grayscale (1 channel)
        if input_channels != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=False
            )
            
        # Get output dimension of EfficientNet-B0 backbone (1280)
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom FC layer
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(num_features, feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, 224, 224)
            
        Returns:
            Feature vector (batch, feature_dim)
        """
        return self.backbone(x)
    
    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature map before pooling/flattening.
        
        Returns:
            Feature map (batch, 1280, 7, 7)
        """
        return self.backbone.features(x)


class ZoneCNN(nn.Module):
    """
    Lightweight CNN for processing individual facial zone.
    
    Input: (batch, 1, 48, 48) - grayscale zone image
    Output: (batch, 128) - zone feature vector
    
    Architecture (lighter than global CNN):
    Conv1: 1  → 32  (48 → 24)
    Conv2: 32 → 64  (24 → 12)
    Conv3: 64 → 128 (12 → 6)
    FC: 128*6*6 → 128
    """
    
    def __init__(self,
                 input_channels: int = 1,
                 feature_dim: int = 128,
                 dropout: float = 0.3):
        super(ZoneCNN, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Convolutional layers (fewer channels than global CNN)
        self.conv1 = ConvBlock(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1)
        
        # Calculate flattened size
        # 48 → 24 → 12 → 6
        self.flat_size = 128 * 6 * 6
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, 48, 48)
            
        Returns:
            Feature vector (batch, feature_dim)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        return x


class HybridCNN(nn.Module):
    """
    Hybrid CNN combining global and zone-based features.
    
    Inputs:
    - full_face: (batch, 1, 224, 224)
    - forehead: (batch, 1, 48, 48)
    - left_eye: (batch, 1, 48, 48)
    - right_eye: (batch, 1, 48, 48)
    - nose: (batch, 1, 48, 48)
    - mouth: (batch, 1, 48, 48)
    
    Output: (batch, 1152) concatenated feature vector
    - Global: 512-dim
    - Zones: 5 × 128-dim = 640-dim
    - Total: 1152-dim
    
    This hybrid representation captures both:
    - Holistic face structure (global)
    - Localized micro-expressions (zones)
    """
    
    def __init__(self,
                 global_feature_dim: int = 512,
                 zone_feature_dim: int = 128,
                 global_dropout: float = 0.5,
                 zone_dropout: float = 0.3,
                 pretrained: bool = True):
        super(HybridCNN, self).__init__()
        
        # Global CNN
        self.global_cnn = GlobalCNN(
            input_channels=1,
            feature_dim=global_feature_dim,
            dropout=global_dropout,
            pretrained=pretrained
        )
        
        # Zone CNNs (5 separate networks)
        self.zone_cnns = nn.ModuleDict({
            'forehead': ZoneCNN(1, zone_feature_dim, zone_dropout),
            'left_eye': ZoneCNN(1, zone_feature_dim, zone_dropout),
            'right_eye': ZoneCNN(1, zone_feature_dim, zone_dropout),
            'nose': ZoneCNN(1, zone_feature_dim, zone_dropout),
            'mouth': ZoneCNN(1, zone_feature_dim, zone_dropout)
        })
        
        # Feature dimensions
        self.global_feature_dim = global_feature_dim
        self.zone_feature_dim = zone_feature_dim
        self.num_zones = len(self.zone_cnns)
        self.total_feature_dim = global_feature_dim + (zone_feature_dim * self.num_zones)
    
    def forward(self, 
                full_face: torch.Tensor,
                zones: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through hybrid CNN.
        
        Args:
            full_face: Full face image (batch, 1, 224, 224)
            zones: Dictionary of zone images:
                   - 'forehead': (batch, 1, 48, 48)
                   - 'left_eye': (batch, 1, 48, 48)
                   - 'right_eye': (batch, 1, 48, 48)
                   - 'nose': (batch, 1, 48, 48)
                   - 'mouth': (batch, 1, 48, 48)
        
        Returns:
            Hybrid feature vector (batch, total_feature_dim)
        """
        # Extract global features
        global_features = self.global_cnn(full_face)
        
        # Extract zone features
        zone_features = []
        for zone_name in ['forehead', 'left_eye', 'right_eye', 'nose', 'mouth']:
            if zone_name in zones:
                zone_feat = self.zone_cnns[zone_name](zones[zone_name])
                zone_features.append(zone_feat)
            else:
                # Fallback: zero features if zone missing
                batch_size = full_face.size(0)
                zero_feat = torch.zeros(batch_size, self.zone_feature_dim, 
                                       device=full_face.device)
                zone_features.append(zero_feat)
        
        # Concatenate all features
        zone_features_cat = torch.cat(zone_features, dim=1)
        hybrid_features = torch.cat([global_features, zone_features_cat], dim=1)
        
        return hybrid_features
    
    def forward_with_individual_features(self,
                                        full_face: torch.Tensor,
                                        zones: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass returning both hybrid features and individual components.
        
        Useful for analysis and visualization.
        
        Returns:
            Tuple of (hybrid_features, feature_dict)
        """
        # Global features
        global_features = self.global_cnn(full_face)
        
        # Zone features
        zone_features_dict = {}
        zone_features_list = []
        
        for zone_name in ['forehead', 'left_eye', 'right_eye', 'nose', 'mouth']:
            if zone_name in zones:
                zone_feat = self.zone_cnns[zone_name](zones[zone_name])
            else:
                batch_size = full_face.size(0)
                zone_feat = torch.zeros(batch_size, self.zone_feature_dim,
                                       device=full_face.device)
            
            zone_features_dict[zone_name] = zone_feat
            zone_features_list.append(zone_feat)
        
        # Concatenate
        zone_features_cat = torch.cat(zone_features_list, dim=1)
        hybrid_features = torch.cat([global_features, zone_features_cat], dim=1)
        
        feature_dict = {
            'global': global_features,
            'zones': zone_features_dict,
            'zone_concat': zone_features_cat,
            'hybrid': hybrid_features
        }
        
        return hybrid_features, feature_dict
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get number of parameters in each component.
        
        Returns:
            Dictionary with parameter counts
        """
        def count_params(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        params = {
            'global_cnn': count_params(self.global_cnn),
            'zone_cnns': {
                zone_name: count_params(zone_cnn)
                for zone_name, zone_cnn in self.zone_cnns.items()
            },
            'total': count_params(self)
        }
        
        return params


def create_hybrid_cnn(config: Optional[Dict] = None) -> HybridCNN:
    """
    Factory function to create HybridCNN with configuration.
    
    Args:
        config: Configuration dictionary with keys:
                - global_feature_dim (default: 512)
                - zone_feature_dim (default: 128)
                - global_dropout (default: 0.5)
                - zone_dropout (default: 0.3)
                - pretrained (default: True)
    
    Returns:
        Initialized HybridCNN model
    """
    if config is None:
        config = {}
    
    model = HybridCNN(
        global_feature_dim=config.get('global_feature_dim', 512),
        zone_feature_dim=config.get('zone_feature_dim', 128),
        global_dropout=config.get('global_dropout', 0.5),
        zone_dropout=config.get('zone_dropout', 0.3),
        pretrained=config.get('pretrained', True)
    )
    
    return model


if __name__ == "__main__":
    print("Hybrid CNN Architecture")
    print("=" * 50)
    
    # Create model
    model = create_hybrid_cnn()
    
    # Print architecture
    print("\nModel Components:")
    print(f"  Global CNN feature dim: {model.global_feature_dim}")
    print(f"  Zone CNN feature dim: {model.zone_feature_dim}")
    print(f"  Number of zones: {model.num_zones}")
    print(f"  Total feature dim: {model.total_feature_dim}")
    
    # Parameter counts
    params = model.get_num_parameters()
    print("\nParameter Counts:")
    print(f"  Global CNN: {params['global_cnn']:,}")
    for zone_name, count in params['zone_cnns'].items():
        print(f"  Zone CNN ({zone_name}): {count:,}")
    print(f"  TOTAL: {params['total']:,}")
    
    # Test forward pass
    batch_size = 4
    full_face = torch.randn(batch_size, 1, 224, 224)
    zones_input = {
        'forehead': torch.randn(batch_size, 1, 48, 48),
        'left_eye': torch.randn(batch_size, 1, 48, 48),
        'right_eye': torch.randn(batch_size, 1, 48, 48),
        'nose': torch.randn(batch_size, 1, 48, 48),
        'mouth': torch.randn(batch_size, 1, 48, 48)
    }
    
    print("\n✓ Testing forward pass...")
    with torch.no_grad():
        output = model(full_face, zones_input)
    
    print(f"  Input shape (full face): {full_face.shape}")
    print(f"  Input shape (each zone): {zones_input['forehead'].shape}")
    print(f"  Output shape (hybrid features): {output.shape}")
    print("\n✓ Hybrid CNN architecture loaded successfully")
