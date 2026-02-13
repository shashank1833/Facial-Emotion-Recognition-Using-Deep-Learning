"""
Noise-Robust Preprocessing Module

This module implements preprocessing steps to handle real-world webcam conditions:
1. Median filtering - Removes salt-and-pepper noise
2. Histogram equalization - Normalizes illumination
3. Gaussian blur (optional) - Reduces high-frequency noise

Academic Justification:
- Real webcams produce noisy images, especially in low light
- Median filter is proven effective for impulse noise (non-linear filter)
- Histogram equalization provides lighting invariance
- These steps are standard in computer vision pipelines for robust feature extraction
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class NoiseRobustPreprocessor:
    """
    Applies noise-resilient preprocessing to facial images.
    
    This preprocessing pipeline ensures consistent input quality regardless of
    webcam quality, lighting conditions, or environmental noise.
    """
    
    def __init__(self, 
                 median_kernel: int = 3,
                 use_clahe: bool = True,
                 clip_limit: float = 2.0,
                 gaussian_kernel: Optional[int] = None,
                 gaussian_sigma: float = 0):
        """
        Initialize preprocessor with configurable parameters.
        
        Args:
            median_kernel: Size of the median filter kernel (must be odd, e.g., 3, 5). 
                           Reduced to 3 to prevent over-blurring of facial expressions.
            use_clahe: Use CLAHE instead of standard histogram equalization.
                      CLAHE prevents over-amplification of noise in uniform regions.
            clip_limit: Contrast limiting threshold for CLAHE (1-5 typical).
            gaussian_kernel: Kernel size for Gaussian blur (optional).
            gaussian_sigma: Standard deviation for Gaussian kernel.
        """
        assert median_kernel % 2 == 1, "Median kernel must be odd"
        
        self.median_kernel = median_kernel
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma
    
    def apply_median_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply median filter to remove salt-and-pepper noise.
        
        Why median filter?
        - Non-linear filter that preserves edges better than linear filters
        - Highly effective against impulse noise (random bright/dark pixels)
        - Common in low-quality webcams due to sensor noise
        
        Args:
            image: Input grayscale image
            
        Returns:
            Filtered image with reduced noise
        """
        return cv2.medianBlur(image, self.median_kernel)
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to normalize illumination.

        Why histogram equalization?
        - Normalizes contrast across different lighting conditions
        - Makes features detectable in both bright and dark regions
        - Essential for webcams used in varied environments (indoor/outdoor)

        CLAHE (Contrast Limited Adaptive HE) advantages:
        - Operates on small regions (tiles) independently
        - Prevents over-amplification of noise in uniform areas
        - Better for images with varying local contrast

        Args:
            image: Input grayscale image

        Returns:
            Contrast-normalized image
        """
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            return cv2.equalizeHist(image)
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur for additional smoothing (optional).
        
        Why Gaussian blur?
        - Reduces high-frequency noise and small artifacts
        - Helps with motion blur from low frame rate webcams
        - Trade-off: Slight loss of fine detail
        
        Args:
            image: Input grayscale image
            
        Returns:
            Smoothed image
        """
        if self.gaussian_kernel is not None and self.gaussian_kernel > 0:
            return cv2.GaussianBlur(
                image, 
                (self.gaussian_kernel, self.gaussian_kernel),
                self.gaussian_sigma
            )
        return image
    
    def preprocess(self, 
                   image: np.ndarray,
                   to_grayscale: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Apply full preprocessing pipeline.
        
        Pipeline order (critical):
        1. Convert to grayscale (if needed)
        2. Median filter (noise removal first)
        3. Histogram equalization (after noise removal)
        4. Gaussian blur (optional final smoothing)
        
        Args:
            image: Input BGR or grayscale image
            to_grayscale: Convert BGR to grayscale
            
        Returns:
            Tuple of (preprocessed_image, metadata_dict)
        """
        metadata = {
            'original_shape': image.shape,
            'preprocessing_steps': []
        }
        
        # Step 1: Convert to grayscale
        if to_grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            metadata['preprocessing_steps'].append('grayscale_conversion')
        
        # Step 2: Median filter (remove noise first)
        image = self.apply_median_filter(image)
        metadata['preprocessing_steps'].append(f'median_filter_k{self.median_kernel}')
        
        # Step 3: Histogram equalization (normalize after noise removal)
        image = self.apply_histogram_equalization(image)
        eq_type = 'clahe' if self.use_clahe else 'standard_he'
        metadata['preprocessing_steps'].append(f'histogram_eq_{eq_type}')
        
        # Step 4: Gaussian blur (optional)
        if self.gaussian_kernel:
            image = self.apply_gaussian_blur(image)
            metadata['preprocessing_steps'].append(f'gaussian_blur_k{self.gaussian_kernel}')
        
        metadata['final_shape'] = image.shape
        
        return image, metadata
    
    def preprocess_batch(self, images: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Preprocess a batch of images.
        
        Args:
            images: Batch of images (N, H, W) or (N, H, W, C)
            
        Returns:
            Tuple of (preprocessed_batch, metadata_list)
        """
        preprocessed = []
        metadata_list = []
        
        for img in images:
            processed_img, metadata = self.preprocess(img)
            preprocessed.append(processed_img)
            metadata_list.append(metadata)
        
        return np.array(preprocessed), metadata_list
    
    def visualize_preprocessing_stages(self, image: np.ndarray) -> dict:
        """
        Return intermediate preprocessing stages for visualization/debugging.
        
        Useful for academic presentations to show effect of each step.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with images at each preprocessing stage
        """
        stages = {}
        
        # Original
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        stages['original'] = gray.copy()
        
        # After median filter
        median_filtered = self.apply_median_filter(gray)
        stages['after_median'] = median_filtered.copy()
        
        # After histogram equalization
        hist_eq = self.apply_histogram_equalization(median_filtered)
        stages['after_hist_eq'] = hist_eq.copy()
        
        # After Gaussian blur (if enabled)
        if self.gaussian_kernel:
            blurred = self.apply_gaussian_blur(hist_eq)
            stages['after_gaussian'] = blurred.copy()
            stages['final'] = blurred
        else:
            stages['final'] = hist_eq
        
        return stages


def compare_preprocessing_methods(image: np.ndarray) -> dict:
    """
    Compare different preprocessing approaches for academic analysis.
    
    This function demonstrates why each preprocessing step is necessary
    by showing results with and without each component.
    
    Args:
        image: Input test image
        
    Returns:
        Dictionary with different preprocessing variants
    """
    results = {}
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    results['no_preprocessing'] = gray
    
    # Only median filter
    preprocessor_median = NoiseRobustPreprocessor(
        median_kernel=5,
        use_clahe=False,
        gaussian_kernel=None
    )
    results['median_only'], _ = preprocessor_median.preprocess(gray, to_grayscale=False)
    
    # Only histogram equalization
    preprocessor_hist = NoiseRobustPreprocessor(
        median_kernel=1,  # No effect with kernel=1
        use_clahe=True,
        gaussian_kernel=None
    )
    # Manually apply only hist eq
    results['hist_eq_only'] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    
    # Median + Histogram (recommended)
    preprocessor_full = NoiseRobustPreprocessor(
        median_kernel=5,
        use_clahe=True,
        gaussian_kernel=None
    )
    results['median_hist'], _ = preprocessor_full.preprocess(gray, to_grayscale=False)
    
    # Full pipeline with Gaussian
    preprocessor_all = NoiseRobustPreprocessor(
        median_kernel=5,
        use_clahe=True,
        gaussian_kernel=3,
        gaussian_sigma=0
    )
    results['full_pipeline'], _ = preprocessor_all.preprocess(gray, to_grayscale=False)
    
    return results


if __name__ == "__main__":
    # Demo/test code
    print("Noise-Robust Preprocessing Module")
    print("=" * 50)
    
    # Create synthetic noisy image for testing
    test_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Add salt-and-pepper noise
    noise_mask = np.random.random(test_image.shape)
    test_image[noise_mask < 0.02] = 0  # Salt
    test_image[noise_mask > 0.98] = 255  # Pepper
    
    # Initialize preprocessor
    preprocessor = NoiseRobustPreprocessor(
        median_kernel=5,
        use_clahe=True,
        clip_limit=2.0
    )
    
    # Preprocess
    processed, metadata = preprocessor.preprocess(test_image, to_grayscale=False)
    
    print("\nPreprocessing applied:")
    for step in metadata['preprocessing_steps']:
        print(f"  - {step}")
    
    print(f"\nOriginal shape: {metadata['original_shape']}")
    print(f"Final shape: {metadata['final_shape']}")
    
    print("\n✓ Preprocessing module loaded successfully")
