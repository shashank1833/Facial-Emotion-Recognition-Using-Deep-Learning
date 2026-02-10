"""
Temporal LSTM Model for Emotion Recognition

This module implements LSTM-based temporal modeling for emotion sequences.

Academic Justification:
- Emotions have temporal dynamics (onset, apex, offset phases)
- Single-frame classification misses context and transitions
- LSTM captures temporal dependencies in feature sequences
- Reduces false positives from transient facial movements
- Validates emotion persistence (real emotion vs. noise)

Architecture:
- Input: Sequence of hybrid feature vectors (seq_len × feature_dim)
- LSTM layers: Process temporal patterns
- Output: Emotion classification

Studies show 5-8% accuracy improvement over frame-by-frame classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalLSTM(nn.Module):
    """
    LSTM network for temporal emotion modeling.
    
    Processes sequences of hybrid CNN features to classify emotions
    while considering temporal context.
    
    Input: (batch, seq_len, feature_dim) - sequence of feature vectors
    Output: (batch, num_classes) - emotion probabilities
    """
    
    def __init__(self,
                 input_dim: int = 1152,  # Hybrid feature dimension
                 hidden_units: list = [256, 128],
                 num_classes: int = 7,
                 dropout: float = 0.5,
                 bidirectional: bool = False):
        """
        Initialize temporal LSTM.
        
        Args:
            input_dim: Dimension of input features (from HybridCNN)
            hidden_units: List of hidden units for each LSTM layer
            num_classes: Number of emotion classes (7)
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(TemporalLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.num_layers = len(hidden_units)
        self.bidirectional = bidirectional
        
        # LSTM layers
        lstm_layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_units):
            return_sequences = (i < len(hidden_units) - 1)
            
            lstm = nn.LSTM(
                input_size=in_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0 if i == len(hidden_units) - 1 else dropout,
                bidirectional=bidirectional
            )
            lstm_layers.append(lstm)
            
            # Update input dimension for next layer
            multiplier = 2 if bidirectional else 1
            in_dim = hidden_dim * multiplier
        
        self.lstm_layers = nn.ModuleList(lstm_layers)
        
        # Fully connected classification head
        final_hidden_dim = hidden_units[-1] * (2 if bidirectional else 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(final_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, 
                x: torch.Tensor,
                return_hidden: bool = False) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            return_hidden: Return hidden states for visualization
            
        Returns:
            Emotion logits (batch, num_classes)
            If return_hidden: Tuple of (logits, hidden_states)
        """
        hidden_states = []
        
        # Pass through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            x, (h_n, c_n) = lstm(x)
            
            if return_hidden:
                hidden_states.append(h_n)
        
        # Use last timestep output
        # x shape: (batch, seq_len, hidden_dim)
        last_output = x[:, -1, :]  # (batch, hidden_dim)
        
        # Classify
        logits = self.classifier(last_output)
        
        if return_hidden:
            return logits, hidden_states
        return logits
    
    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning predictions for all timesteps.
        
        Useful for per-frame emotion analysis.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Logits for each timestep (batch, seq_len, num_classes)
        """
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # x shape: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape to classify each timestep
        x_flat = x.reshape(batch_size * seq_len, hidden_dim)
        logits_flat = self.classifier(x_flat)
        
        # Reshape back
        logits = logits_flat.reshape(batch_size, seq_len, self.num_classes)
        
        return logits


class HybridEmotionRecognitionModel(nn.Module):
    """
    Complete end-to-end model combining HybridCNN and TemporalLSTM.
    
    This is the full system model that can be trained and deployed.
    
    Input: Sequence of frame batches (full face + zones)
    Output: Emotion classification
    
    Pipeline:
    1. Extract hybrid features for each frame (HybridCNN)
    2. Stack features into sequence
    3. Process sequence temporally (LSTM)
    4. Classify emotion
    """
    
    def __init__(self,
                 hybrid_cnn,
                 sequence_length: int = 16,
                 lstm_hidden_units: list = [256, 128],
                 num_classes: int = 7,
                 lstm_dropout: float = 0.5):
        """
        Initialize complete model.
        
        Args:
            hybrid_cnn: HybridCNN instance
            sequence_length: Number of frames in sequence
            lstm_hidden_units: LSTM hidden dimensions
            num_classes: Number of emotion classes
            lstm_dropout: LSTM dropout rate
        """
        super(HybridEmotionRecognitionModel, self).__init__()
        
        self.hybrid_cnn = hybrid_cnn
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # LSTM for temporal modeling
        self.temporal_lstm = TemporalLSTM(
            input_dim=hybrid_cnn.total_feature_dim,
            hidden_units=lstm_hidden_units,
            num_classes=num_classes,
            dropout=lstm_dropout
        )
    
    def forward_single_frame(self,
                            full_face: torch.Tensor,
                            zones: dict) -> torch.Tensor:
        """
        Extract features for a single frame.
        
        Args:
            full_face: (batch, 1, 224, 224)
            zones: Dictionary of zone tensors
            
        Returns:
            Hybrid features (batch, feature_dim)
        """
        return self.hybrid_cnn(full_face, zones)
    
    def forward(self,
               full_faces: torch.Tensor,
               zones_sequence: list) -> torch.Tensor:
        """
        Forward pass through complete model.
        
        Args:
            full_faces: (batch, seq_len, 1, 224, 224)
            zones_sequence: List of length seq_len, each containing 
                           a dict of zone tensors (batch, 1, 48, 48)
        
        Returns:
            Emotion logits (batch, num_classes)
        """
        batch_size, seq_len = full_faces.shape[:2]
        
        # Extract features for each frame in sequence
        feature_sequence = []
        
        for t in range(seq_len):
            frame = full_faces[:, t]  # (batch, 1, 224, 224)
            zones_t = zones_sequence[t]
            
            features = self.hybrid_cnn(frame, zones_t)
            feature_sequence.append(features)
        
        # Stack into sequence tensor
        feature_sequence = torch.stack(feature_sequence, dim=1)  # (batch, seq_len, feature_dim)
        
        # Temporal processing
        logits = self.temporal_lstm(feature_sequence)
        
        return logits
    
    def predict_emotion(self,
                       full_faces: torch.Tensor,
                       zones_sequence: list,
                       return_probabilities: bool = True) -> torch.Tensor:
        """
        Predict emotion with optional probability output.
        
        Args:
            full_faces: (batch, seq_len, 1, 224, 224)
            zones_sequence: List of zone dictionaries
            return_probabilities: Return softmax probabilities
            
        Returns:
            Class indices or probabilities (batch, num_classes)
        """
        logits = self.forward(full_faces, zones_sequence)
        
        if return_probabilities:
            probabilities = F.softmax(logits, dim=1)
            return probabilities
        else:
            predictions = torch.argmax(logits, dim=1)
            return predictions


def create_full_model(hybrid_cnn_config: Optional[dict] = None,
                     lstm_config: Optional[dict] = None) -> HybridEmotionRecognitionModel:
    """
    Factory function to create complete model.
    
    Args:
        hybrid_cnn_config: Configuration for HybridCNN
        lstm_config: Configuration for TemporalLSTM
    
    Returns:
        Complete HybridEmotionRecognitionModel
    """
    from .hybrid_cnn import create_hybrid_cnn
    
    if hybrid_cnn_config is None:
        hybrid_cnn_config = {}
    if lstm_config is None:
        lstm_config = {}
    
    # Create HybridCNN
    hybrid_cnn = create_hybrid_cnn(hybrid_cnn_config)
    
    # Create full model
    model = HybridEmotionRecognitionModel(
        hybrid_cnn=hybrid_cnn,
        sequence_length=lstm_config.get('sequence_length', 16),
        lstm_hidden_units=lstm_config.get('hidden_units', [256, 128]),
        num_classes=lstm_config.get('num_classes', 7),
        lstm_dropout=lstm_config.get('dropout', 0.5)
    )
    
    return model


if __name__ == "__main__":
    print("Temporal LSTM Model")
    print("=" * 50)
    
    # Create model
    from hybrid_cnn import create_hybrid_cnn
    
    hybrid_cnn = create_hybrid_cnn()
    model = HybridEmotionRecognitionModel(
        hybrid_cnn=hybrid_cnn,
        sequence_length=16,
        lstm_hidden_units=[256, 128]
    )
    
    print("\nModel Architecture:")
    print(f"  Hybrid CNN feature dim: {hybrid_cnn.total_feature_dim}")
    print(f"  LSTM hidden units: [256, 128]")
    print(f"  Sequence length: 16 frames")
    print(f"  Output classes: 7 emotions")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cnn_params = sum(p.numel() for p in model.hybrid_cnn.parameters() if p.requires_grad)
    lstm_params = sum(p.numel() for p in model.temporal_lstm.parameters() if p.requires_grad)
    
    print(f"\nParameter Counts:")
    print(f"  HybridCNN: {cnn_params:,}")
    print(f"  TemporalLSTM: {lstm_params:,}")
    print(f"  TOTAL: {total_params:,}")
    
    print("\n✓ Temporal LSTM model loaded successfully")
