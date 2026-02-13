"""
Zone Extraction Module

This module crops and processes individual facial zones from detected landmarks.
Each zone is:
1. Cropped using bounding box from zone landmarks
2. Resized to fixed resolution (48×48)
3. Normalized to [0, 1]

Academic Justification:
- Localized features capture micro-expressions missed by global processing
- Fixed resolution ensures consistent CNN input
- Normalization improves training stability
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .zone_definitions import (
    MEDIAPIPE_ZONES_SIMPLIFIED,
    get_zone_landmarks,
    get_all_zones_landmarks
)


@dataclass
class ZoneImage:
    """Container for extracted zone image and metadata."""
    image: np.ndarray  # Processed zone image
    original_bbox: Tuple[int, int, int, int]  # x, y, w, h in original image
    landmarks: np.ndarray  # Zone landmarks in original coordinates
    zone_name: str


class ZoneExtractor:
    """
    Extracts and processes facial zones from images with landmarks.
    
    Processing pipeline for each zone:
    1. Get zone landmarks
    2. Calculate bounding box with padding
    3. Crop region from image
    4. Resize to target resolution
    5. Normalize pixel values
    """
    
    def __init__(self,
                 target_size: int = 48,
                 padding_ratio: float = 0.15,
                 normalization: str = 'minmax',
                 landmark_type: str = 'mediapipe'):
        """
        Initialize zone extractor.
        
        Args:
            target_size: Output resolution for zones (e.g., 48 for 48×48)
            padding_ratio: Padding around zone bbox as fraction of bbox size
            normalization: 'minmax' ([0,1]) or 'standard' (zero mean, unit variance)
            landmark_type: 'mediapipe' or 'dlib'
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.normalization = normalization
        self.landmark_type = landmark_type
        
        # Zone names
        self.zone_names = ['forehead', 'left_eye', 'right_eye', 'nose', 'mouth']
    
    def calculate_zone_bbox(self, 
                           zone_landmarks: np.ndarray,
                           image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box for zone with padding.
        
        Args:
            zone_landmarks: Landmarks for this zone (N, 2) or (N, 3)
            image_shape: (height, width) of original image
            
        Returns:
            Bounding box (x, y, width, height)
        """
        if len(zone_landmarks) == 0:
            return (0, 0, 0, 0)
        
        # Get 2D coordinates
        if zone_landmarks.shape[1] >= 2:
            coords = zone_landmarks[:, :2]
        else:
            return (0, 0, 0, 0)
        
        # Calculate bbox
        x_min = np.min(coords[:, 0])
        y_min = np.min(coords[:, 1])
        x_max = np.max(coords[:, 0])
        y_max = np.max(coords[:, 1])
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Add padding
        pad_x = width * self.padding_ratio
        pad_y = height * self.padding_ratio
        
        # Ensure square bbox (use maximum dimension)
        size = max(width + 2 * pad_x, height + 2 * pad_y)
        
        # Center the bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        x = int(center_x - size / 2)
        y = int(center_y - size / 2)
        size = int(size)
        
        # Clip to image boundaries
        h, w = image_shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Adjust size if bbox extends beyond image
        if x + size > w:
            size = w - x
        if y + size > h:
            size = h - y
        
        return (x, y, size, size)
    
    def crop_zone(self,
                  image: np.ndarray,
                  bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop zone from image using bounding box.
        
        Args:
            image: Input image (grayscale or color)
            bbox: (x, y, width, height)
            
        Returns:
            Cropped zone image
        """
        x, y, w, h = bbox
        
        # Handle edge cases
        if w == 0 or h == 0:
            return np.zeros((self.target_size, self.target_size), dtype=image.dtype)
        
        # Crop
        cropped = image[y:y+h, x:x+w]
        
        # Resize to target size
        if cropped.size == 0:
            return np.zeros((self.target_size, self.target_size), dtype=image.dtype)
        
        resized = cv2.resize(cropped, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def normalize_zone(self, zone_image: np.ndarray) -> np.ndarray:
        """
        Normalize zone image pixel values.
        
        Args:
            zone_image: Input zone image
            
        Returns:
            Normalized zone image (float32)
        """
        # Convert to float
        zone_float = zone_image.astype(np.float32)
        
        if self.normalization == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(zone_float)
            max_val = np.max(zone_float)
            
            if max_val - min_val > 1e-6:
                normalized = (zone_float - min_val) / (max_val - min_val)
            else:
                normalized = zone_float / 255.0
        
        elif self.normalization == 'standard':
            # Standardization (zero mean, unit variance)
            mean = np.mean(zone_float)
            std = np.std(zone_float)
            
            if std > 1e-6:
                normalized = (zone_float - mean) / std
            else:
                normalized = zone_float / 255.0
        
        else:
            # Default: divide by 255
            normalized = zone_float / 255.0
        
        return normalized
    
    def extract_zone(self,
                    image: np.ndarray,
                    landmarks: np.ndarray,
                    zone_name: str) -> ZoneImage:
        """
        Extract and process a single facial zone.
        
        Args:
            image: Input image (grayscale recommended)
            landmarks: Full set of facial landmarks
            zone_name: Name of zone to extract
            
        Returns:
            ZoneImage object with processed zone
        """
        # Get landmarks for this zone
        zone_landmarks = get_zone_landmarks(landmarks, zone_name, self.landmark_type)
        
        # Calculate bounding box
        bbox = self.calculate_zone_bbox(zone_landmarks, image.shape[:2])
        
        # Crop zone
        zone_img = self.crop_zone(image, bbox)
        
        # Normalize
        zone_img = self.normalize_zone(zone_img)
        
        return ZoneImage(
            image=zone_img,
            original_bbox=bbox,
            landmarks=zone_landmarks,
            zone_name=zone_name
        )
    
    def extract_all_zones(self,
                         image: np.ndarray,
                         landmarks: np.ndarray) -> Dict[str, ZoneImage]:
        """
        Extract all facial zones from image.
        
        Args:
            image: Input image (grayscale recommended)
            landmarks: Full set of facial landmarks
            
        Returns:
            Dictionary mapping zone names to ZoneImage objects
        """
        zones = {}
        
        for zone_name in self.zone_names:
            try:
                zone = self.extract_zone(image, landmarks, zone_name)
                zones[zone_name] = zone
            except Exception as e:
                print(f"Warning: Failed to extract zone '{zone_name}': {e}")
                # Create empty zone as fallback
                zones[zone_name] = ZoneImage(
                    image=np.zeros((self.target_size, self.target_size), dtype=np.float32),
                    original_bbox=(0, 0, 0, 0),
                    landmarks=np.array([]),
                    zone_name=zone_name
                )
        
        return zones
    
    def extract_zones_batch(self,
                           images: List[np.ndarray],
                           landmarks_list: List[np.ndarray]) -> List[Dict[str, ZoneImage]]:
        """
        Extract zones from batch of images.
        
        Args:
            images: List of images
            landmarks_list: List of landmark arrays
            
        Returns:
            List of zone dictionaries
        """
        assert len(images) == len(landmarks_list), "Image and landmark counts must match"
        
        batch_zones = []
        for img, landmarks in zip(images, landmarks_list):
            zones = self.extract_all_zones(img, landmarks)
            batch_zones.append(zones)
        
        return batch_zones
    
    def zones_to_array(self, zones: Dict[str, ZoneImage]) -> np.ndarray:
        """
        Convert zone dictionary to numpy array for model input.
        
        Args:
            zones: Dictionary of ZoneImage objects
            
        Returns:
            Array of shape (num_zones, target_size, target_size)
        """
        zone_arrays = []
        
        for zone_name in self.zone_names:
            if zone_name in zones:
                zone_arrays.append(zones[zone_name].image)
            else:
                # Fallback empty zone
                zone_arrays.append(np.zeros((self.target_size, self.target_size)))
        
        return np.array(zone_arrays)
    
    def visualize_zones(self,
                       original_image: np.ndarray,
                       zones: Dict[str, ZoneImage],
                       show_bboxes: bool = True) -> np.ndarray:
        """
        Create visualization showing all extracted zones.
        
        Args:
            original_image: Original image with face
            zones: Extracted zones
            show_bboxes: Draw bounding boxes on original image
            
        Returns:
            Visualization image
        """
        # Create copy of original
        vis_img = original_image.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
        # Draw bboxes
        if show_bboxes:
            colors = {
                'forehead': (255, 0, 0),      # Blue
                'left_eye': (0, 255, 0),      # Green
                'right_eye': (0, 255, 255),   # Yellow
                'nose': (255, 0, 255),        # Magenta
                'mouth': (0, 0, 255)          # Red
            }
            
            for zone_name, zone in zones.items():
                x, y, w, h = zone.original_bbox
                color = colors.get(zone_name, (255, 255, 255))
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_img, zone_name, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Create grid of zone images
        zone_grid = []
        for zone_name in self.zone_names:
            if zone_name in zones:
                zone_img = zones[zone_name].image
                # Convert to 0-255 for display
                zone_display = (zone_img * 255).astype(np.uint8)
                # Convert to BGR if needed
                if len(zone_display.shape) == 2:
                    zone_display = cv2.cvtColor(zone_display, cv2.COLOR_GRAY2BGR)
                zone_grid.append(zone_display)
        
        # Stack zones horizontally
        zone_strip = np.hstack(zone_grid) if zone_grid else np.zeros((48, 240, 3), dtype=np.uint8)
        
        # Resize original to match zone strip width
        h_ratio = zone_strip.shape[1] / vis_img.shape[1]
        new_h = int(vis_img.shape[0] * h_ratio)
        vis_resized = cv2.resize(vis_img, (zone_strip.shape[1], new_h))
        
        # Stack vertically
        combined = np.vstack([vis_resized, zone_strip])
        
        return combined


def create_zone_dataset(images: List[np.ndarray],
                       landmarks_list: List[np.ndarray],
                       labels: Optional[np.ndarray] = None,
                       extractor: Optional[ZoneExtractor] = None) -> Dict:
    """
    Create zone-based dataset from images and landmarks.
    
    Useful for preparing training data.
    
    Args:
        images: List of facial images
        landmarks_list: List of landmark arrays
        labels: Optional emotion labels
        extractor: ZoneExtractor instance (creates default if None)
        
    Returns:
        Dictionary with zone arrays and labels
    """
    if extractor is None:
        extractor = ZoneExtractor()
    
    # Extract zones for all images
    all_zones = extractor.extract_zones_batch(images, landmarks_list)
    
    # Convert to arrays
    num_samples = len(all_zones)
    zone_arrays = {
        zone_name: [] for zone_name in extractor.zone_names
    }
    
    for zones_dict in all_zones:
        for zone_name in extractor.zone_names:
            if zone_name in zones_dict:
                zone_arrays[zone_name].append(zones_dict[zone_name].image)
    
    # Convert lists to numpy arrays
    dataset = {
        zone_name: np.array(zone_list)
        for zone_name, zone_list in zone_arrays.items()
    }
    
    # Add labels if provided
    if labels is not None:
        dataset['labels'] = labels
    
    return dataset


if __name__ == "__main__":
    print("Zone Extraction Module")
    print("=" * 50)
    
    # Initialize extractor
    extractor = ZoneExtractor(
        target_size=48,
        padding_ratio=0.15,
        normalization='minmax'
    )
    
    print(f"\nConfiguration:")
    print(f"  Target size: {extractor.target_size}×{extractor.target_size}")
    print(f"  Padding ratio: {extractor.padding_ratio}")
    print(f"  Normalization: {extractor.normalization}")
    
    print(f"\nZones to extract:")
    for i, zone_name in enumerate(extractor.zone_names, 1):
        print(f"  {i}. {zone_name}")
    
    print("\n✓ Zone extractor initialized")
    print("  Ready to process facial images")
