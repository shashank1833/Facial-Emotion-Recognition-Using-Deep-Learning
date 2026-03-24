"""
Data Augmentation Module

Implements augmentation strategies to improve model robustness:
1. Brightness variations (lighting conditions)
2. Gaussian noise (sensor noise)
3. Motion blur (camera shake, movement)
4. Partial occlusions (glasses, masks, hands)
5. Rotation (head pose variations)

Academic Justification:
- Real-world conditions are noisy and varied
- Augmentation prevents overfitting to clean data
- Simulates deployment scenarios (masks, poor lighting, etc.)
- Studies show 5-15% accuracy improvement with proper augmentation
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import random


class EmotionAugmenter:
    """
    Augmentation pipeline for facial emotion recognition.
    
    Simulates real-world deployment conditions to improve robustness.
    """
    
    def __init__(self,
                 brightness_range: Tuple[float, float] = (0.7, 1.3),
                 gaussian_noise_std: float = 0.05,
                 motion_blur_enabled: bool = True,
                 motion_blur_kernel_range: Tuple[int, int] = (5, 15),
                 occlusion_enabled: bool = True,
                 occlusion_coverage_range: Tuple[float, float] = (0.1, 0.2),
                 rotation_range: int = 15,
                 horizontal_flip: bool = True,
                 zoom_range: float = 0.1,
                 augmentation_probability: float = 0.8):
        """
        Initialize augmenter.
        
        Args:
            brightness_range: Min/max brightness multipliers
            gaussian_noise_std: Standard deviation for Gaussian noise
            motion_blur_enabled: Enable motion blur
            motion_blur_kernel_range: Min/max kernel size for motion blur
            occlusion_enabled: Enable random occlusions
            occlusion_coverage_range: Min/max fraction of face to occlude
            rotation_range: Max rotation angle in degrees
            horizontal_flip: Enable horizontal flipping
            zoom_range: Max zoom in/out fraction
            augmentation_probability: Probability of applying each augmentation
        """
        self.brightness_range = brightness_range
        self.gaussian_noise_std = gaussian_noise_std
        self.motion_blur_enabled = motion_blur_enabled
        self.motion_blur_kernel_range = motion_blur_kernel_range
        self.occlusion_enabled = occlusion_enabled
        self.occlusion_coverage_range = occlusion_coverage_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.augmentation_probability = augmentation_probability
    
    def apply_brightness_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random brightness adjustment.
        
        Simulates: Different lighting conditions (indoor/outdoor, shadows)
        
        Args:
            image: Input image (0-255 or 0-1)
            
        Returns:
            Brightness-adjusted image
        """
        factor = random.uniform(*self.brightness_range)
        
        # Handle different input ranges
        if image.max() <= 1.0:
            # Image is in [0, 1]
            adjusted = np.clip(image * factor, 0, 1)
        else:
            # Image is in [0, 255]
            adjusted = np.clip(image * factor, 0, 255).astype(image.dtype)
        
        return adjusted
    
    def apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Simulates: Webcam sensor noise, low-light conditions
        
        Args:
            image: Input image
            
        Returns:
            Noisy image
        """
        # Generate noise matching image range
        if image.max() <= 1.0:
            noise = np.random.normal(0, self.gaussian_noise_std, image.shape)
            noisy = np.clip(image + noise, 0, 1)
        else:
            noise = np.random.normal(0, self.gaussian_noise_std * 255, image.shape)
            noisy = np.clip(image + noise, 0, 255).astype(image.dtype)
        
        return noisy
    
    def apply_motion_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply motion blur to image.
        
        Simulates: Camera shake, subject movement, low frame rate
        
        Args:
            image: Input image
            
        Returns:
            Motion-blurred image
        """
        kernel_size = random.randint(*self.motion_blur_kernel_range)
        
        # Create motion blur kernel (diagonal)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Randomly rotate kernel (different motion directions)
        angle = random.uniform(0, 360)
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()  # Renormalize
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    def apply_occlusion(self, image: np.ndarray) -> np.ndarray:
        """
        Add random rectangular occlusions.
        
        Simulates: Glasses, masks, hands covering face, hair
        
        Args:
            image: Input image
            
        Returns:
            Occluded image
        """
        h, w = image.shape[:2]
        occluded = image.copy()
        
        # Calculate total area to occlude
        coverage = random.uniform(*self.occlusion_coverage_range)
        total_area = h * w * coverage
        
        # Number of occlusion rectangles
        num_occlusions = random.randint(1, 3)
        area_per_occlusion = total_area / num_occlusions
        
        for _ in range(num_occlusions):
            # Random aspect ratio
            aspect = random.uniform(0.5, 2.0)
            
            # Calculate dimensions
            occ_h = int(np.sqrt(area_per_occlusion / aspect))
            occ_w = int(occ_h * aspect)
            
            # Random position
            x = random.randint(0, max(1, w - occ_w))
            y = random.randint(0, max(1, h - occ_h))
            
            # Random occlusion color (gray values)
            if image.max() <= 1.0:
                color = random.uniform(0, 1)
            else:
                color = random.randint(0, 255)
            
            # Apply occlusion
            if len(image.shape) == 3:
                occluded[y:y+occ_h, x:x+occ_w, :] = color
            else:
                occluded[y:y+occ_h, x:x+occ_w] = color
        
        return occluded
    
    def apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random rotation.
        
        Simulates: Head pose variations
        
        Args:
            image: Input image
            
        Returns:
            Rotated image
        """
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def apply_horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """
        Apply horizontal flip.
        
        Simulates: Mirror symmetry (most emotions are symmetric)
        
        Note: Some emotions (e.g., contempt) may be asymmetric.
        
        Args:
            image: Input image
            
        Returns:
            Flipped image
        """
        return cv2.flip(image, 1)
    
    def apply_zoom(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random zoom.
        
        Simulates: Different camera distances, face sizes
        
        Args:
            image: Input image
            
        Returns:
            Zoomed image
        """
        zoom_factor = random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
        h, w = image.shape[:2]
        
        # New dimensions
        new_h = int(h * zoom_factor)
        new_w = int(w * zoom_factor)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop (zoomed in)
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            cropped = resized[start_y:start_y+h, start_x:start_x+w]
            return cropped
        else:
            # Pad (zoomed out)
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            
            if len(image.shape) == 3:
                padded = np.zeros((h, w, image.shape[2]), dtype=image.dtype)
            else:
                padded = np.zeros((h, w), dtype=image.dtype)
            
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            return padded
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full augmentation pipeline.
        
        Each augmentation is applied with probability augmentation_probability.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        # Apply augmentations probabilistically
        if random.random() < self.augmentation_probability:
            augmented = self.apply_brightness_augmentation(augmented)
        
        if random.random() < self.augmentation_probability:
            augmented = self.apply_gaussian_noise(augmented)
        
        if self.motion_blur_enabled and random.random() < self.augmentation_probability:
            augmented = self.apply_motion_blur(augmented)
        
        if self.occlusion_enabled and random.random() < self.augmentation_probability:
            augmented = self.apply_occlusion(augmented)
        
        if random.random() < self.augmentation_probability:
            augmented = self.apply_rotation(augmented)
        
        if self.horizontal_flip and random.random() < 0.5:
            augmented = self.apply_horizontal_flip(augmented)
        
        if random.random() < self.augmentation_probability:
            augmented = self.apply_zoom(augmented)
        
        return augmented
    
    def augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Augment batch of images.
        
        Args:
            images: List of images
            
        Returns:
            List of augmented images
        """
        return [self.augment(img) for img in images]


def create_augmentation_examples(image: np.ndarray) -> dict:
    """
    Create examples of each augmentation for visualization.
    
    Useful for academic presentations and documentation.
    
    Args:
        image: Test image
        
    Returns:
        Dictionary with augmentation examples
    """
    augmenter = EmotionAugmenter()
    
    examples = {
        'original': image.copy(),
        'brightness_dark': augmenter.apply_brightness_augmentation(
            image.copy() if random.seed(42) or True else None
        ),
        'brightness_bright': image.copy() * 1.3,
        'gaussian_noise': augmenter.apply_gaussian_noise(image.copy()),
        'motion_blur': augmenter.apply_motion_blur(image.copy()),
        'occlusion': augmenter.apply_occlusion(image.copy()),
        'rotation': augmenter.apply_rotation(image.copy()),
        'horizontal_flip': augmenter.apply_horizontal_flip(image.copy()),
        'zoom_in': augmenter.apply_zoom(image.copy()),
        'all_augmentations': augmenter.augment(image.copy())
    }
    
    return examples


if __name__ == "__main__":
    print("Data Augmentation Module")
    print("=" * 50)
    
    # Create augmenter
    augmenter = EmotionAugmenter(
        brightness_range=(0.7, 1.3),
        gaussian_noise_std=0.05,
        motion_blur_enabled=True,
        occlusion_enabled=True,
        rotation_range=15,
        augmentation_probability=0.8
    )
    
    print("\nAugmentation Strategies:")
    print("  1. Brightness adjustment (±30%)")
    print("  2. Gaussian noise (σ=0.05)")
    print("  3. Motion blur (kernel 5-15)")
    print("  4. Random occlusions (10-20% coverage)")
    print("  5. Rotation (±15°)")
    print("  6. Horizontal flip (50% probability)")
    print("  7. Zoom (±10%)")
    
    print("\nAcademic Justification:")
    print("  - Simulates real-world deployment conditions")
    print("  - Prevents overfitting to training data")
    print("  - Improves robustness to noise and occlusions")
    print("  - Expected: 5-15% accuracy improvement")
    
    # Test with random image
    test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    augmented = augmenter.augment(test_image)
    
    print(f"\n✓ Augmentation test successful")
    print(f"  Input shape: {test_image.shape}")
    print(f"  Output shape: {augmented.shape}")
    
    print("\n✓ Data augmentation module loaded successfully")
