"""
Facial Zone Definitions for MediaPipe Face Mesh

This module defines which MediaPipe landmarks belong to each facial zone.
These zones are designed based on Facial Action Coding System (FACS)
and emotion-relevant facial regions.

Academic Basis:
- Paul Ekman's FACS identifies Action Units (AUs) localized to facial regions
- Different emotions activate different AUs:
  * Happiness: Zygomatic major (mouth), Orbicularis oculi (eyes)
  * Surprise: Frontalis (forehead), upper eyelid raise
  * Anger: Corrugator supercilii (brow), levator labii (nose/mouth)
  * Fear: Frontalis + eye widening
  * Sadness: Inner brow raise, mouth depression
  * Disgust: Nose wrinkle, upper lip raise

Zone-based processing captures these localized activations.
"""

from typing import Dict, List
import numpy as np


# MediaPipe Face Mesh has 468 landmarks
# Full landmark map: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Zone definitions (landmark indices)
MEDIAPIPE_ZONES = {
    # FOREHEAD - Critical for surprise (raised eyebrows), anger (furrowed brow)
    # Covers upper facial third
    'forehead': [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162
    ],
    
    # LEFT EYE + EYEBROW - Fear (widened), sadness (drooped), surprise
    # Includes upper and lower eyelid, eyebrow region
    'left_eye': [
        # Eyebrow
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        # Eye contour
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        # Upper eyelid
        246, 161, 160, 159, 158, 157, 173,
        # Lower eyelid  
        133, 155, 154, 153, 145, 144, 163, 7
    ],
    
    # RIGHT EYE + EYEBROW - Mirror of left eye
    'right_eye': [
        # Eyebrow
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
        # Eye contour
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        # Upper eyelid
        466, 388, 387, 386, 385, 384, 398,
        # Lower eyelid
        362, 382, 381, 380, 374, 373, 390, 249
    ],
    
    # NOSE - Disgust (nose wrinkle), anger (nostril flare)
    # Includes nose bridge and tip
    'nose': [
        # Bridge
        6, 168, 197, 195, 5, 4,
        # Tip
        1, 2,
        # Nostrils
        98, 97, 2, 326, 327,
        # Sides
        48, 115, 220, 45, 4, 275, 440, 344, 278
    ],
    
    # MOUTH - Most expressive region for all emotions
    # Includes lips, mouth corners, surrounding area
    'mouth': [
        # Outer lip contour
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        # Upper lip
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        # Lower lip
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        # Inner mouth
        13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82,
        # Mouth corners (critical for smile/frown)
        61, 291, 0
    ]
}


# Simplified zones (fewer landmarks for faster processing)
# Use if computational efficiency is critical
MEDIAPIPE_ZONES_SIMPLIFIED = {
    'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'right_eye': [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    'nose': [4, 5, 6, 168, 197, 195, 51, 281],
    'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
}


# For dlib 68-point landmarks (if using dlib instead of MediaPipe)
DLIB_68_ZONES = {
    'forehead': list(range(17, 27)),  # Eyebrow region (no direct forehead points)
    'left_eye': list(range(36, 42)) + list(range(17, 22)),  # Left eye + left eyebrow
    'right_eye': list(range(42, 48)) + list(range(22, 27)),  # Right eye + right eyebrow
    'nose': list(range(27, 36)),  # Nose bridge and base
    'mouth': list(range(48, 68))  # Outer and inner lips
}


def get_zone_landmarks(landmarks: np.ndarray, 
                       zone_name: str,
                       landmark_type: str = 'mediapipe') -> np.ndarray:
    """
    Extract landmarks belonging to a specific facial zone.
    
    Args:
        landmarks: Full set of facial landmarks (N, 2) or (N, 3)
        zone_name: Name of zone ('forehead', 'left_eye', 'right_eye', 'nose', 'mouth')
        landmark_type: 'mediapipe' or 'dlib'
        
    Returns:
        Subset of landmarks for the zone (M, 2) or (M, 3)
    """
    if landmark_type == 'mediapipe':
        indices = MEDIAPIPE_ZONES_SIMPLIFIED.get(zone_name, [])
    elif landmark_type == 'dlib':
        indices = DLIB_68_ZONES.get(zone_name, [])
    else:
        raise ValueError(f"Unknown landmark type: {landmark_type}")
    
    if not indices:
        raise ValueError(f"Unknown zone: {zone_name}")
    
    return landmarks[indices]


def get_all_zones_landmarks(landmarks: np.ndarray,
                            landmark_type: str = 'mediapipe') -> Dict[str, np.ndarray]:
    """
    Extract landmarks for all zones.
    
    Args:
        landmarks: Full set of facial landmarks
        landmark_type: 'mediapipe' or 'dlib'
        
    Returns:
        Dictionary mapping zone names to their landmarks
    """
    if landmark_type == 'mediapipe':
        zones_def = MEDIAPIPE_ZONES_SIMPLIFIED
    elif landmark_type == 'dlib':
        zones_def = DLIB_68_ZONES
    else:
        raise ValueError(f"Unknown landmark type: {landmark_type}")
    
    return {
        zone_name: landmarks[indices]
        for zone_name, indices in zones_def.items()
    }


def visualize_zones(image_shape: tuple) -> Dict[str, np.ndarray]:
    """
    Create visualization showing which landmarks belong to each zone.
    
    Useful for documentation and academic presentations.
    
    Args:
        image_shape: (height, width) of output images
        
    Returns:
        Dictionary of zone visualization images
    """
    visualizations = {}
    h, w = image_shape
    
    # Create random landmarks for demonstration
    landmarks = np.random.rand(468, 2) * np.array([w, h])
    
    for zone_name, indices in MEDIAPIPE_ZONES_SIMPLIFIED.items():
        # Create blank image
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw zone landmarks in color
        zone_landmarks = landmarks[indices]
        for point in zone_landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        
        visualizations[zone_name] = vis
    
    return visualizations


# Emotion-to-Zone importance mapping (for interpretability)
# Based on FACS action units
EMOTION_ZONE_IMPORTANCE = {
    'Angry': {
        'forehead': 0.8,  # Lowered, furrowed brow (AU4)
        'left_eye': 0.6,  # Eye tension
        'right_eye': 0.6,
        'nose': 0.3,
        'mouth': 0.7  # Tightened lips (AU23, 24)
    },
    'Disgust': {
        'forehead': 0.2,
        'left_eye': 0.4,
        'right_eye': 0.4,
        'nose': 0.9,  # Nose wrinkle (AU9)
        'mouth': 0.8  # Upper lip raise (AU10)
    },
    'Fear': {
        'forehead': 0.7,  # Raised upper eyelid (AU1+2)
        'left_eye': 0.9,  # Wide eyes (AU5)
        'right_eye': 0.9,
        'nose': 0.3,
        'mouth': 0.6  # Lips stretched (AU20)
    },
    'Happy': {
        'forehead': 0.3,
        'left_eye': 0.8,  # Crow's feet (AU6)
        'right_eye': 0.8,
        'nose': 0.3,
        'mouth': 1.0  # Smile (AU12)
    },
    'Sad': {
        'forehead': 0.7,  # Inner brow raise (AU1)
        'left_eye': 0.6,  # Drooped eyelids
        'right_eye': 0.6,
        'nose': 0.2,
        'mouth': 0.8  # Mouth corners down (AU15)
    },
    'Surprise': {
        'forehead': 1.0,  # Raised eyebrows (AU1+2)
        'left_eye': 0.9,  # Wide eyes (AU5)
        'right_eye': 0.9,
        'nose': 0.2,
        'mouth': 0.7  # Jaw drop (AU26)
    },
    'Neutral': {
        'forehead': 0.5,
        'left_eye': 0.5,
        'right_eye': 0.5,
        'nose': 0.5,
        'mouth': 0.5
    }
}


def get_zone_importance_for_emotion(emotion: str) -> Dict[str, float]:
    """
    Get importance weights for each zone for a given emotion.
    
    Useful for attention mechanisms or weighted fusion.
    
    Args:
        emotion: Emotion name
        
    Returns:
        Dictionary mapping zone names to importance scores [0, 1]
    """
    return EMOTION_ZONE_IMPORTANCE.get(emotion, EMOTION_ZONE_IMPORTANCE['Neutral'])


if __name__ == "__main__":
    import cv2
    
    print("Facial Zone Definitions")
    print("=" * 50)
    
    print("\nZone definitions:")
    for zone_name, indices in MEDIAPIPE_ZONES_SIMPLIFIED.items():
        print(f"  {zone_name:12s}: {len(indices):3d} landmarks")
    
    print("\nEmotion-Zone Importance (example for 'Happy'):")
    happy_importance = get_zone_importance_for_emotion('Happy')
    for zone, importance in happy_importance.items():
        print(f"  {zone:12s}: {importance:.1f}")
    
    print("\n✓ Zone definitions loaded")
    print("  Based on FACS Action Units")
    print("  Optimized for emotion-relevant regions")
