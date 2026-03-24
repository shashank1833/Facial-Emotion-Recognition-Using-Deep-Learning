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
                 target_size: int = 224,
                 emotion_subset: Optional[List[str]] = None):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to fer2013.csv
            usage: 'Training', 'PublicTest', or 'PrivateTest'
            preprocessor: NoiseRobustPreprocessor instance
            zone_extractor: ZoneExtractor instance
            transform: Additional augmentation transforms
            target_size: Size for full face (global CNN input)
            emotion_subset: Optional list of emotion names to include
        """
        # Load CSV
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['Usage'] == usage].reset_index(drop=True)
        
        # Emotion mapping
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Filter by emotion subset if specified
        if emotion_subset:
            # Map names to indices
            emotion_to_idx = {name: i for i, name in enumerate(self.emotions)}
            subset_indices = [emotion_to_idx[name] for name in emotion_subset if name in emotion_to_idx]
            
            print(f"Filtering dataset for emotions: {emotion_subset} (indices: {subset_indices})")
            self.df = self.df[self.df['emotion'].isin(subset_indices)].reset_index(drop=True)
            
            # Re-map labels to 0..len(subset)-1 for training
            label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(subset_indices)}
            self.df['emotion'] = self.df['emotion'].map(label_mapping)
            print(f"Labels re-mapped: {label_mapping}")
        
        self.usage = usage
        self.target_size = target_size
        self.transform = transform
        
        # Initialize preprocessing and extraction
        if preprocessor is None:
            self.preprocessor = NoiseRobustPreprocessor()
        else:
            self.preprocessor = preprocessor
        
        # Determine if we should apply min-max normalization
        # We'll use this if configured in ZoneExtractor
        if zone_extractor is None:
            self.zone_extractor = ZoneExtractor()
        else:
            self.zone_extractor = zone_extractor
            
        self.use_minmax = (self.zone_extractor.normalization == 'minmax')
        
        # Landmark detector
        self.landmark_detector = MediaPipeFaceDetector(
            static_image_mode=True,
            min_detection_confidence=0.3
        )
        
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
        
        return self.process_single_image(image, label, idx)

    def process_single_image(self, image: np.ndarray, label: int, sample_id: int = 0) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Processes a single image through the pipeline.
        
        Args:
            image: Input grayscale image
            label: Emotion label
            sample_id: ID for logging
            
        Returns:
            Tuple of (full_face, zones_dict, label)
        """
        # Apply preprocessing
        # Note: We pass to_grayscale=False because we assume input is already grayscale
        # or handled by the caller. For FER2013 it's already grayscale.
        image_preprocessed, _ = self.preprocessor.preprocess(image, to_grayscale=False)
        
        # Apply additional transforms if specified (NOW BEFORE DETECTION)
        if self.transform is not None:
            image_preprocessed = self.transform(image_preprocessed)
            
        # Detect landmarks
        # FER-2013 images are 48x48, which is too small for MediaPipe.
        # Upscale to a larger size to improve detection.
        detection_size = 224
        image_for_detection = cv2.resize(image_preprocessed, (detection_size, detection_size))
        landmarks_obj = self.landmark_detector.detect_landmarks(image_for_detection)
        
        if landmarks_obj is None:
            # Fallback: no landmarks detected
            # Instead of zeros, use the whole image as a rough approximation for zones
            if sample_id % 500 == 0: # Only warn occasionally
                print(f"  [Info] No landmarks for sample {sample_id}, using whole image fallback")
            
            # Full face (resize to target size)
            full_face = cv2.resize(image_preprocessed, (self.target_size, self.target_size))
            
            # Use normalization to match ZoneExtractor
            face_float = full_face.astype(np.float32)
            if self.use_minmax:
                min_val, max_val = face_float.min(), face_float.max()
                full_face = (face_float - min_val) / (max_val - min_val) if max_val > min_val else face_float / 255.0
            else:
                full_face = face_float / 255.0
            
            # Use whole image for zones too as a fallback
            zone_size = self.zone_extractor.target_size
            zone_fallback = cv2.resize(image_preprocessed, (zone_size, zone_size))
            
            # Use normalization to match ZoneExtractor
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
            
            # Full face (resize)
            full_face = cv2.resize(image_for_detection, (self.target_size, self.target_size))
            
            # ZoneExtractor already normalizes, but we need to normalize the full_face here
            # to match the ZoneExtractor's normalization setting
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


class CombinedCSVDataset(Dataset):
    """
    Dataset that loads images from paths specified in a CSV file.
    Compatible with the CSVs generated by generate_csv.py.
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
            target_size: Size for full face (global CNN input)
            emotion_subset: Optional list of emotion names to include
        """
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
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label_name = row['label']
        label = self.label_mapping[label_name]
        
        # Load image
        if not os.path.exists(img_path):
            # Return a blank image or handle error
            warnings.warn(f"Image not found: {img_path}")
            image = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        # Reuse processing logic from FER2013Dataset (we can refactor this later)
        # For now, let's call a shared method or just copy the logic
        return self.process_single_image(image, label, idx)

    def process_single_image(self, image: np.ndarray, label: int, sample_id: int = 0) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Processes a single image through the pipeline.
        (Copied from FER2013Dataset for now to avoid breaking changes)
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
            
            zones = {name: zone_fallback for name in ['forehead', 'left_eye', 'right_eye', 'nose', 'mouth']}
        else:
            zones_extracted = self.zone_extractor.extract_all_zones(image_for_detection, landmarks_obj.landmarks)
            zones = {name: zone.image for name, zone in zones_extracted.items()}
            full_face = cv2.resize(image_for_detection, (self.target_size, self.target_size))
            face_float = full_face.astype(np.float32)
            if self.use_minmax:
                min_val, max_val = face_float.min(), face_float.max()
                full_face = (face_float - min_val) / (max_val - min_val) if max_val > min_val else face_float / 255.0
            else:
                full_face = face_float / 255.0
        
        full_face_tensor = torch.from_numpy(full_face).unsqueeze(0)
        zones_tensors = {name: torch.from_numpy(zone_img).unsqueeze(0) for name, zone_img in zones.items()}
            
        return full_face_tensor, zones_tensors, torch.tensor(label, dtype=torch.long)

    def close(self):
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
