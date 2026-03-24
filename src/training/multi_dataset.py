"""
Multi-Dataset Handler for Emotion Recognition System

Provides utilities to combine multiple emotion datasets using the EfficientNet-B0 backbone technique.
Supports generic image folder structures.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Optional, Dict, Tuple
from ..preprocessing import NoiseRobustPreprocessor
from ..landmark_detection import MediaPipeFaceDetector
from ..zone_extraction import ZoneExtractor

class ImageFolderDataset(Dataset):
    """
    Generic dataset for images organized in folders by emotion name.
    Works with the EfficientNet-B0 backbone technique.
    
    Example:
    data/CK+/
        Angry/
            img1.jpg
        Happy/
            img2.jpg
    """
    def __init__(self, 
                 root_dir: str, 
                 transform=None, 
                 preprocessor: Optional[NoiseRobustPreprocessor] = None, 
                 zone_extractor: Optional[ZoneExtractor] = None,
                 target_size: int = 224):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.emotion_to_idx = {name: i for i, name in enumerate(self.emotions)}
        
        self.preprocessor = preprocessor or NoiseRobustPreprocessor()
        self.zone_extractor = zone_extractor or ZoneExtractor()
        self.use_minmax = (self.zone_extractor.normalization == 'minmax')
        self.landmark_detector = MediaPipeFaceDetector(
            static_image_mode=True,
            min_detection_confidence=0.3
        )
        
        self.samples = []
        if os.path.exists(root_dir):
            for emotion_name in os.listdir(root_dir):
                if emotion_name in self.emotion_to_idx:
                    idx = self.emotion_to_idx[emotion_name]
                    emotion_dir = os.path.join(root_dir, emotion_name)
                    if os.path.isdir(emotion_dir):
                        for img_name in os.listdir(emotion_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                self.samples.append((os.path.join(emotion_dir, img_name), idx))
        
        print(f"Loaded {len(self.samples)} samples from {root_dir} using EfficientNet-B0 backbone technique")
                            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        return self.process_single_image(image, label, idx)

    def process_single_image(self, image: np.ndarray, label: int, sample_id: int = 0) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """Processes single image through EfficientNet-B0 backbone technique pipeline."""
        image_preprocessed, _ = self.preprocessor.preprocess(image, to_grayscale=False)
        
        if self.transform is not None:
            image_preprocessed = self.transform(image_preprocessed)
            
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
        zones_tensors = {name: torch.from_numpy(img).unsqueeze(0) for name, img in zones.items()}
            
        return full_face_tensor, zones_tensors, torch.tensor(label, dtype=torch.long)

def get_combined_loader(config, datasets: List[Dataset], batch_size: int, shuffle: bool = True):
    """Combines multiple datasets into one loader using EfficientNet-B0 backbone technique."""
    combined = ConcatDataset(datasets)
    num_workers = 0 if torch.cuda.is_available() is False else config['hardware']['num_workers']
    pin_memory = False if torch.cuda.is_available() is False else config['hardware']['pin_memory']
    
    return torch.utils.data.DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
