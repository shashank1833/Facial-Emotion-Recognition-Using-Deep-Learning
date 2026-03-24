"""
Data Loader for Emotion Recognition System

Handles loading and preprocessing datasets for training using the EfficientNet-B0 backbone technique.
Supports loading from CSV files containing image paths and labels.
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


class EmotionDataset(Dataset):
    """
    General dataset for training emotion recognition using the EfficientNet-B0 backbone technique.
    
    Processes each sample through:
    1. Noise-robust preprocessing
    2. Landmark detection
    3. Zone extraction
    4. Face resizing for global CNN (EfficientNet-B0 backbone)
    """
    
    def __init__(self,
                 csv_path: str,
                 preprocessor: Optional[NoiseRobustPreprocessor] = None,
                 zone_extractor: Optional[ZoneExtractor] = None,
                 transform=None,
                 target_size: int = 224,
                 emotion_subset: Optional[List[str]] = None):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to the CSV file (columns: image_path, label)
            preprocessor: NoiseRobustPreprocessor instance
            zone_extractor: ZoneExtractor instance
            transform: Additional augmentation transforms
            target_size: Size for full face (EfficientNet-B0 backbone input)
            emotion_subset: Optional list of emotion names to include
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Filter by emotion subset if specified
        if emotion_subset:
            print(f"Filtering dataset for emotions: {emotion_subset}")
            self.df = self.df[self.df['label'].isin(emotion_subset)].reset_index(drop=True)
            
            # Map names to indices for training
            self.label_mapping = {name: i for i, name in enumerate(emotion_subset)}
        else:
            # Map standard names to indices
            self.label_mapping = {name: i for i, name in enumerate(self.emotions)}
            
        self.target_size = target_size
        self.transform = transform
        
        # Initialize preprocessing and extraction
        self.preprocessor = preprocessor if preprocessor else NoiseRobustPreprocessor()
        self.zone_extractor = zone_extractor if zone_extractor else ZoneExtractor()
        self.use_minmax = (self.zone_extractor.normalization == 'minmax')
        
        # Landmark detector
        self.landmark_detector = MediaPipeFaceDetector(
            static_image_mode=True,
            min_detection_confidence=0.3
        )
        
        print(f"Loaded {len(self.df)} samples from {csv_path} using EfficientNet-B0 backbone technique")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Get dataset item.
        
        Returns:
            Tuple of (full_face, zones_dict, label)
        """
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_name = row['label']
        label = self.label_mapping[label_name]
        
        # Load image
        if not os.path.exists(img_path):
            warnings.warn(f"Image not found: {img_path}")
            image = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                warnings.warn(f"Failed to load image: {img_path}")
                image = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        return self.process_single_image(image, label, idx)

    def process_single_image(self, image: np.ndarray, label: int, sample_id: int = 0) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Processes a single image through the pipeline using EfficientNet-B0 backbone technique.
        
        Args:
            image: Input grayscale image
            label: Emotion label index
            sample_id: ID for logging
            
        Returns:
            Tuple of (full_face, zones_dict, label)
        """
        # Apply preprocessing
        image_preprocessed, _ = self.preprocessor.preprocess(image, to_grayscale=False)
        
        # Apply additional transforms if specified
        if self.transform is not None:
            image_preprocessed = self.transform(image_preprocessed)
            
        # Detect landmarks
        detection_size = 224
        image_for_detection = cv2.resize(image_preprocessed, (detection_size, detection_size))
        landmarks_obj = self.landmark_detector.detect_landmarks(image_for_detection)
        
        if landmarks_obj is None:
            # Fallback: no landmarks detected
            full_face = cv2.resize(image_preprocessed, (self.target_size, self.target_size))
            
            face_float = full_face.astype(np.float32)
            if self.use_minmax:
                min_val, max_val = face_float.min(), face_float.max()
                full_face = (face_float - min_val) / (max_val - min_val) if max_val > min_val else face_float / 255.0
            else:
                full_face = face_float / 255.0
            
            zone_size = self.zone_extractor.target_size
            zone_fallback = cv2.resize(image_preprocessed, (zone_size, zone_size))
            
            zone_float = zone_fallback.astype(np.float32)
            if self.use_minmax:
                z_min, z_max = zone_float.min(), zone_float.max()
                zone_fallback = (zone_float - z_min) / (z_max - z_min) if z_max > z_min else zone_float / 255.0
            else:
                zone_fallback = zone_float / 255.0
            
            zones = {
                'forehead': zone_fallback,
                'left_eye': zone_fallback,
                'right_eye': zone_fallback,
                'nose': zone_fallback,
                'mouth': zone_fallback
            }
        else:
            zones_extracted = self.zone_extractor.extract_all_zones(
                image_for_detection, 
                landmarks_obj.landmarks
            )
            
            zones = {
                name: zone.image for name, zone in zones_extracted.items()
            }
            
            # Full face (resize) for EfficientNet-B0 backbone
            full_face = cv2.resize(image_for_detection, (self.target_size, self.target_size))
            
            face_float = full_face.astype(np.float32)
            if self.use_minmax:
                min_val, max_val = face_float.min(), face_float.max()
                full_face = (face_float - min_val) / (max_val - min_val) if max_val > min_val else face_float / 255.0
            else:
                full_face = face_float / 255.0
        
        # Convert to tensors
        full_face_tensor = torch.from_numpy(full_face).unsqueeze(0) # (1, H, W)
        
        zones_tensors = {}
        for name, zone_img in zones.items():
            zones_tensors[name] = torch.from_numpy(zone_img).unsqueeze(0) # (1, H, W)
            
        return full_face_tensor, zones_tensors, torch.tensor(label, dtype=torch.long)
    
    def close(self):
        """Release resources."""
        self.landmark_detector.close()


def create_data_loaders(train_csv: str,
                       test_csv: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       target_size: int = 224) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing using EfficientNet-B0 backbone technique.
    
    Args:
        train_csv: Path to training CSV
        test_csv: Path to testing CSV
        batch_size: Batch size
        num_workers: Number of worker processes
        target_size: Size for full face images
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = EmotionDataset(train_csv, target_size=target_size)
    test_dataset = EmotionDataset(test_csv, target_size=target_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    return train_loader, test_loader


if __name__ == "__main__":
    print("Emotion Data Loader (EfficientNet-B0 backbone technique)")
    print("=" * 60)
    
    # Test data loader with combined CSV
    csv_path = "train_combined.csv"
    
    if os.path.exists(csv_path):
        dataset = EmotionDataset(csv_path)
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"Emotion classes: {dataset.emotions}")
        
        # Test loading one sample
        print("\nTesting sample loading...")
        full_face, zones, label = dataset[0]
        
        print(f"  Full face shape: {full_face.shape}")
        print(f"  Zone shapes: {[(name, tensor.shape) for name, tensor in zones.items()]}")
        print(f"  Label index: {label}")
        
        dataset.close()
        print("\n✓ Data loader test successful")
    else:
        print(f"\nNote: Combined CSV not found at {csv_path}")
