"""
Feature Dataset for Precomputed CNN Features

This module provides a PyTorch Dataset class for loading precomputed features
instead of raw images. This dramatically speeds up training by skipping all
preprocessing, landmark detection, and CNN forward passes.

Usage:
    from training.feature_dataset import FeatureDataset
    
    dataset = FeatureDataset('data/features/', split='training')
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for features, labels in loader:
        # features: (batch_size, 1152) - precomputed CNN features
        # labels: (batch_size,) - emotion labels
        outputs = classifier(features)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import warnings


class FeatureDataset(Dataset):
    """
    Dataset for loading precomputed CNN features.
    
    This dataset loads features that were extracted by extract_features.py
    and saved to disk as .pt files. It's much faster than the original
    FER2013Dataset because it skips all image preprocessing.
    
    Features are 1152-dimensional vectors:
    - Global CNN: 512 dimensions
    - 5 Zone CNNs: 5 × 128 = 640 dimensions
    - Total: 1152 dimensions
    """
    
    def __init__(self,
                 features_dir: str,
                 split: str = 'training',
                 device: Optional[str] = None):
        """
        Initialize feature dataset.
        
        Args:
            features_dir: Directory containing saved features
            split: 'training', 'publictest', or 'privatetest'
            device: Optional device to load tensors to (None = keep on CPU)
        """
        self.features_dir = features_dir
        self.split = split.lower()
        self.device = device
        
        # Construct file paths
        self.features_path = os.path.join(
            features_dir, 
            f'{self.split}_features.pt'
        )
        self.labels_path = os.path.join(
            features_dir, 
            f'{self.split}_labels.pt'
        )
        
        # Check if files exist
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(
                f"Features file not found: {self.features_path}\n"
                f"Please run extract_features.py first to generate features."
            )
        
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(
                f"Labels file not found: {self.labels_path}\n"
                f"Please run extract_features.py first to generate labels."
            )
        
        # Load features and labels
        print(f"Loading precomputed features for {split}...")
        
        # Load to CPU first to avoid memory issues
        self.features = torch.load(self.features_path, map_location='cpu')
        self.labels = torch.load(self.labels_path, map_location='cpu')
        
        # Validate shapes
        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Features ({len(self.features)}) and labels ({len(self.labels)}) "
                f"have different lengths!"
            )
        
        # Get feature dimension
        self.feature_dim = self.features.shape[1]
        
        # Optionally move to device
        if device is not None and device != 'cpu':
            print(f"  Moving features to {device}...")
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)
        
        print(f"  ✓ Loaded {len(self.features)} samples")
        print(f"  ✓ Feature dimension: {self.feature_dim}")
        print(f"  ✓ Memory usage: {self.features.element_size() * self.features.nelement() / 1024**2:.2f} MB")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
            - features: (1152,) feature vector
            - label: scalar emotion label (0-6)
        """
        return self.features[idx], self.labels[idx]
    
    def get_class_distribution(self) -> dict:
        """
        Get distribution of emotion classes in dataset.
        
        Returns:
            Dictionary mapping class index to count
        """
        unique, counts = torch.unique(self.labels, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}
    
    def get_feature_statistics(self) -> dict:
        """
        Compute statistics about the features.
        
        Returns:
            Dictionary with feature statistics
        """
        return {
            'mean': self.features.mean(dim=0),
            'std': self.features.std(dim=0),
            'min': self.features.min(dim=0)[0],
            'max': self.features.max(dim=0)[0],
            'norm_mean': torch.norm(self.features, dim=1).mean().item(),
            'norm_std': torch.norm(self.features, dim=1).std().item()
        }


class SequenceFeatureDataset(Dataset):
    """
    Dataset for creating sequences from precomputed features.
    
    This dataset creates temporal sequences by grouping consecutive features
    together. Useful for training LSTM or other temporal models.
    
    Note: For FER2013, since images are not sequential videos, this creates
    pseudo-sequences by grouping consecutive samples. In a real application,
    you would extract sequences from actual video frames.
    """
    
    def __init__(self,
                 features_dir: str,
                 split: str = 'training',
                 sequence_length: int = 5,
                 stride: int = 1,
                 device: Optional[str] = None):
        """
        Initialize sequence feature dataset.
        
        Args:
            features_dir: Directory containing saved features
            split: 'training', 'publictest', or 'privatetest'
            sequence_length: Number of frames per sequence
            stride: Stride between sequences
            device: Optional device to load tensors to
        """
        # Load base features using FeatureDataset
        self.base_dataset = FeatureDataset(features_dir, split, device)
        
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Calculate number of sequences
        num_base_samples = len(self.base_dataset)
        self.num_sequences = (num_base_samples - sequence_length) // stride + 1
        
        if self.num_sequences <= 0:
            raise ValueError(
                f"Not enough samples ({num_base_samples}) to create sequences "
                f"of length {sequence_length} with stride {stride}"
            )
        
        print(f"  ✓ Created {self.num_sequences} sequences from {num_base_samples} samples")
        print(f"  ✓ Sequence length: {sequence_length}, stride: {stride}")
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence.
        
        Args:
            idx: Sequence index
            
        Returns:
            Tuple of (sequence_features, label)
            - sequence_features: (sequence_length, feature_dim)
            - label: scalar emotion label (uses label of last frame)
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        # Get features for sequence
        sequence_features = []
        for i in range(start_idx, end_idx):
            features, _ = self.base_dataset[i]
            sequence_features.append(features)
        
        sequence_features = torch.stack(sequence_features, dim=0)
        
        # Use label from last frame
        _, label = self.base_dataset[end_idx - 1]
        
        return sequence_features, label


def create_feature_loaders(features_dir: str,
                          batch_size: int = 64,
                          num_workers: int = 0,
                          pin_memory: bool = False,
                          device: Optional[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for precomputed features.
    
    This is the drop-in replacement for create_data_loaders() from data_loader.py.
    
    Args:
        features_dir: Directory containing saved features
        batch_size: Batch size (can be much larger since no preprocessing!)
        num_workers: Number of workers (recommend 0 for preloaded features)
        pin_memory: Whether to pin memory (False if features already on GPU)
        device: Device to load features to (None = CPU)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = FeatureDataset(features_dir, split='training', device=device)
    val_dataset = FeatureDataset(features_dir, split='publictest', device=device)
    test_dataset = FeatureDataset(features_dir, split='privatetest', device=device)
    
    # Create data loaders
    # Note: num_workers=0 recommended since features are preloaded in memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test feature dataset loading."""
    print("Feature Dataset Test")
    print("=" * 50)
    
    features_dir = "data/features/"
    
    if not os.path.exists(features_dir):
        print(f"\nNote: Features directory not found at {features_dir}")
        print("Please run extract_features.py first to generate features.")
        print("\nExample:")
        print("  python extract_features.py --data data/fer2013/fer2013.csv --output data/features/")
    else:
        # Test loading dataset
        print("\nTesting FeatureDataset...")
        dataset = FeatureDataset(features_dir, split='training')
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Feature dimension: {dataset.feature_dim}")
        
        # Test loading one sample
        print("\nTesting sample loading...")
        features, label = dataset[0]
        print(f"  Features shape: {features.shape}")
        print(f"  Label: {label}")
        
        # Test class distribution
        print("\nClass distribution:")
        dist = dataset.get_class_distribution()
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        for cls, count in sorted(dist.items()):
            print(f"  {emotions[cls]}: {count} ({100*count/len(dataset):.1f}%)")
        
        # Test data loader
        print("\nTesting DataLoader...")
        train_loader, val_loader, test_loader = create_feature_loaders(
            features_dir, 
            batch_size=64
        )
        
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test one batch
        features_batch, labels_batch = next(iter(train_loader))
        print(f"  Batch features shape: {features_batch.shape}")
        print(f"  Batch labels shape: {labels_batch.shape}")
        
        print("\n✓ Feature dataset test successful")
