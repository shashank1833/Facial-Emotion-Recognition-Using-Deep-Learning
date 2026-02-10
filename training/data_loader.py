"""
Data Loader for FER-2013 Dataset

Handles loading and preprocessing FER-2013 dataset for training.

FER-2013 Format:
- CSV file with columns: emotion, pixels, Usage
- emotion: 0-6 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- pixels: space-separated grayscale pixel values (48x48)
- Usage: Training, PublicTest, or PrivateTest

This loader extends the baseline by:
1. Applying noise-robust preprocessing
2. Detecting landmarks and extracting zones
3. Creating sequences for temporal modeling
"""

import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import warnings

from preprocessing import NoiseRobustPreprocessor
from landmark_detection import MediaPipeFaceDetector
from zone_extraction import ZoneExtractor


class FER2013Dataset(Dataset):
    """
    FER-2013 dataset for training emotion recognition.
    
    Processes each sample through:
    1. Noise-robust preprocessing
    2. Landmark detection
    3. Zone extraction
    4. Face resizing for global CNN
    """
    
    def __init__(self,
                 csv_path: str,
                 usage: str = 'Training',
                 preprocessor: Optional[NoiseRobustPreprocessor] = None,
                 zone_extractor: Optional[ZoneExtractor] = None,
                 transform=None,
                 target_size: int = 224):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to fer2013.csv
            usage: 'Training', 'PublicTest', or 'PrivateTest'
            preprocessor: NoiseRobustPreprocessor instance
            zone_extractor: ZoneExtractor instance
            transform: Additional augmentation transforms
            target_size: Size for full face (global CNN input)
        """
        # Load CSV
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['Usage'] == usage].reset_index(drop=True)
        
        self.usage = usage
        self.target_size = target_size
        self.transform = transform
        
        # Initialize preprocessing and extraction
        if preprocessor is None:
            self.preprocessor = NoiseRobustPreprocessor()
        else:
            self.preprocessor = preprocessor
        
        if zone_extractor is None:
            self.zone_extractor = ZoneExtractor()
        else:
            self.zone_extractor = zone_extractor
        
        # Landmark detector
        self.landmark_detector = MediaPipeFaceDetector(
            static_image_mode=True,
            min_detection_confidence=0.3
        )
        
        # Emotion mapping
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        print(f"Loaded {len(self.df)} samples for {usage}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _parse_image(self, pixels_str: str) -> np.ndarray:
        """Parse pixel string from CSV to image array."""
        pixels = np.array([int(p) for p in pixels_str.split()], dtype=np.uint8)
        image = pixels.reshape(48, 48)
        return image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Get dataset item.
        
        Returns:
            Tuple of (full_face, zones_dict, label)
        """
        row = self.df.iloc[idx]
        
        # Parse image
        image = self._parse_image(row['pixels'])
        label = int(row['emotion'])
        
        # Apply preprocessing
        image_preprocessed, _ = self.preprocessor.preprocess(image, to_grayscale=False)
        
        # Detect landmarks
        landmarks_obj = self.landmark_detector.detect_landmarks(image_preprocessed)
        
        if landmarks_obj is None:
            # Fallback: no landmarks detected
            warnings.warn(f"No landmarks detected for sample {idx}, using whole image")
            
            # Full face (resize to target size)
            full_face = cv2.resize(image_preprocessed, (self.target_size, self.target_size))
            full_face = full_face.astype(np.float32) / 255.0
            
            # Empty zones
            zone_size = self.zone_extractor.target_size
            zones = {
                'forehead': np.zeros((zone_size, zone_size), dtype=np.float32),
                'left_eye': np.zeros((zone_size, zone_size), dtype=np.float32),
                'right_eye': np.zeros((zone_size, zone_size), dtype=np.float32),
                'nose': np.zeros((zone_size, zone_size), dtype=np.float32),
                'mouth': np.zeros((zone_size, zone_size), dtype=np.float32)
            }
        else:
            # Extract zones
            zones_extracted = self.zone_extractor.extract_all_zones(
                image_preprocessed, 
                landmarks_obj.landmarks
            )
            
            zones = {
                name: zone.image for name, zone in zones_extracted.items()
            }
            
            # Full face (resize)
            full_face = cv2.resize(image_preprocessed, (self.target_size, self.target_size))
            full_face = full_face.astype(np.float32) / 255.0
        
        # Apply additional transforms if specified
        if self.transform is not None:
            # Apply to full face
            full_face = self.transform(full_face)
            
            # Apply to zones
            for zone_name in zones.keys():
                zones[zone_name] = self.transform(zones[zone_name])
        
        # Convert to tensors
        full_face_tensor = torch.from_numpy(full_face).unsqueeze(0)  # (1, H, W)
        
        zones_tensor = {
            name: torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
            for name, img in zones.items()
        }
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return full_face_tensor, zones_tensor, label_tensor
    
    def close(self):
        """Release resources."""
        self.landmark_detector.close()


def create_data_loaders(csv_path: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       target_size: int = 224) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        csv_path: Path to fer2013.csv
        batch_size: Batch size
        num_workers: Number of worker processes
        target_size: Size for full face images
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = FER2013Dataset(csv_path, usage='Training', target_size=target_size)
    val_dataset = FER2013Dataset(csv_path, usage='PublicTest', target_size=target_size)
    test_dataset = FER2013Dataset(csv_path, usage='PrivateTest', target_size=target_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("FER-2013 Data Loader")
    print("=" * 50)
    
    # Test data loader
    csv_path = "data/fer2013/fer2013.csv"
    
    if os.path.exists(csv_path):
        dataset = FER2013Dataset(csv_path, usage='Training')
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Emotion classes: {dataset.emotions}")
        
        # Test loading one sample
        print("\nTesting sample loading...")
        full_face, zones, label = dataset[0]
        
        print(f"  Full face shape: {full_face.shape}")
        print(f"  Zone shapes: {[(name, tensor.shape) for name, tensor in zones.items()]}")
        print(f"  Label: {label} ({dataset.emotions[label]})")
        
        dataset.close()
        print("\n✓ Data loader test successful")
    else:
        print(f"\nNote: FER-2013 dataset not found at {csv_path}")
        print("Download from: https://www.kaggle.com/datasets/msambare/fer2013")
