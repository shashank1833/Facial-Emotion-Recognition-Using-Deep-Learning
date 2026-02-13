"""
Facial Landmark Detection Module using MediaPipe Face Mesh

This module provides robust facial landmark extraction using MediaPipe,
superior to Haar cascades for:
- Pose variation robustness (±30° head rotation)
- Lighting invariance
- Precise 468-point landmarks (vs 68 for dlib)

Academic Justification:
- MediaPipe uses machine learning models trained on diverse datasets
- Handles partial occlusions better than traditional methods
- Real-time performance suitable for video processing
- Industry-standard solution (Google Research)
"""

import cv2
import numpy as np
import mediapipe.python.solutions as mp_solutions
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class FaceLandmarks:
    """Container for detected facial landmarks."""
    landmarks: np.ndarray  # Shape: (468, 3) - x, y, z coordinates
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    image_shape: Tuple[int, int]  # height, width


class MediaPipeFaceDetector:
    """
    Facial landmark detector using MediaPipe Face Mesh.
    
    MediaPipe Face Mesh provides 468 3D landmarks covering:
    - Face oval (boundary)
    - Eyes and eyebrows
    - Nose
    - Mouth
    - Iris (optional)
    
    Advantages over alternatives:
    - Haar Cascades: Only detects face box, no landmarks, poor with pose variation
    - dlib 68-point: Slower, less robust to lighting, only 68 points
    - MediaPipe: Fast, robust, 468 points, works on CPU
    """
    
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 refine_landmarks: bool = True):
        """
        Initialize MediaPipe Face Mesh detector.
        
        Args:
            static_image_mode: If False, treats input as video stream for optimization.
            max_num_faces: Maximum number of faces to detect.
            min_detection_confidence: Confidence threshold for face detection.
            min_tracking_confidence: Confidence threshold for landmark tracking.
            refine_landmarks: Include iris landmarks (478 total points).
        """
        self.mp_face_mesh = mp_solutions.face_mesh
        self.mp_drawing = mp_solutions.drawing_utils
        self.mp_drawing_styles = mp_solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_landmarks=refine_landmarks
        )
        
        self.detection_active = True
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Detect facial landmarks in image.
        
        Args:
            image: Input RGB or BGR image (will be converted to RGB)
            
        Returns:
            FaceLandmarks object if face detected, None otherwise
        """
        if not self.detection_active:
            return None
        
        # MediaPipe expects RGB
        if len(image.shape) == 2:
            # Grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # Assume BGR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        h, w = image_rgb.shape[:2]
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face (we set max_num_faces=1)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized coordinates to pixel coordinates
        landmarks_array = np.zeros((len(face_landmarks.landmark), 3))
        for idx, landmark in enumerate(face_landmarks.landmark):
            landmarks_array[idx] = [
                landmark.x * w,
                landmark.y * h,
                landmark.z * w  # z is relative to face width
            ]
        
        # Calculate bounding box
        x_coords = landmarks_array[:, 0]
        y_coords = landmarks_array[:, 1]
        
        x_min = int(np.min(x_coords))
        y_min = int(np.min(y_coords))
        x_max = int(np.max(x_coords))
        y_max = int(np.max(y_coords))
        
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Estimate confidence (MediaPipe doesn't provide per-face confidence)
        # Use landmark variance as proxy for detection quality
        variance = np.var(landmarks_array[:, :2])
        confidence = min(1.0, variance / 10000)  # Normalize
        
        return FaceLandmarks(
            landmarks=landmarks_array,
            bbox=bbox,
            confidence=confidence,
            image_shape=(h, w)
        )
    
    def detect_landmarks_batch(self, images: List[np.ndarray]) -> List[Optional[FaceLandmarks]]:
        """
        Detect landmarks in batch of images.
        
        Args:
            images: List of images
            
        Returns:
            List of FaceLandmarks (None for images without faces)
        """
        return [self.detect_landmarks(img) for img in images]
    
    def get_face_region(self, 
                        image: np.ndarray, 
                        landmarks: FaceLandmarks,
                        padding: float = 0.2) -> np.ndarray:
        """
        Extract face region from image using detected landmarks.
        
        Args:
            image: Input image
            landmarks: Detected landmarks
            padding: Padding around face bbox (fraction of bbox size)
            
        Returns:
            Cropped face region
        """
        x, y, w, h = landmarks.bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]
    
    def draw_landmarks(self,
                       image: np.ndarray,
                       landmarks: FaceLandmarks,
                       draw_bbox: bool = True,
                       draw_points: bool = True,
                       draw_connections: bool = False) -> np.ndarray:
        """
        Draw landmarks on image for visualization.
        
        Args:
            image: Input image (will be copied)
            landmarks: Detected landmarks
            draw_bbox: Draw bounding box
            draw_points: Draw landmark points
            draw_connections: Draw face mesh connections (slow, detailed)
            
        Returns:
            Image with landmarks drawn
        """
        vis_image = image.copy()
        
        # Draw bounding box
        if draw_bbox:
            x, y, w, h = landmarks.bbox
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw landmark points
        if draw_points:
            for point in landmarks.landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_image, (x, y), 1, (0, 0, 255), -1)
        
        # Draw connections (detailed mesh)
        if draw_connections:
            # This requires MediaPipe's internal representation
            # For simplicity, we skip this in this implementation
            # In practice, use mp_drawing.draw_landmarks()
            pass
        
        return vis_image
    
    def get_landmark_subset(self,
                           landmarks: FaceLandmarks,
                           indices: List[int]) -> np.ndarray:
        """
        Extract subset of landmarks by indices.
        
        Args:
            landmarks: Full landmark set
            indices: List of landmark indices to extract
            
        Returns:
            Subset of landmarks (N, 3)
        """
        return landmarks.landmarks[indices]
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        self.detection_active = False


class DlibFaceDetector:
    """
    Alternative landmark detector using dlib (68 points).
    
    Included for compatibility, but MediaPipe is preferred.
    Requires: pip install dlib
    Requires: Download shape_predictor_68_face_landmarks.dat
    """
    
    def __init__(self, predictor_path: str):
        """
        Initialize dlib detector.
        
        Args:
            predictor_path: Path to shape_predictor_68_face_landmarks.dat
        """
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            self.available = True
        except ImportError:
            print("Warning: dlib not installed. Install with: pip install dlib")
            self.available = False
        except RuntimeError as e:
            print(f"Warning: Could not load dlib predictor: {e}")
            self.available = False
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Detect 68 facial landmarks using dlib.
        
        Args:
            image: Input grayscale or BGR image
            
        Returns:
            FaceLandmarks with 68 points (z=0 for all)
        """
        if not self.available:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Get first face
        face = faces[0]
        
        # Get landmarks
        shape = self.predictor(gray, face)
        
        # Convert to numpy array
        landmarks_array = np.zeros((68, 3))
        for i in range(68):
            landmarks_array[i] = [shape.part(i).x, shape.part(i).y, 0]
        
        # Bounding box
        bbox = (face.left(), face.top(), face.width(), face.height())
        
        return FaceLandmarks(
            landmarks=landmarks_array,
            bbox=bbox,
            confidence=1.0,  # dlib doesn't provide confidence
            image_shape=(gray.shape[0], gray.shape[1])
        )


def compare_detectors(image: np.ndarray, 
                      mediapipe_detector: MediaPipeFaceDetector,
                      dlib_detector: Optional[DlibFaceDetector] = None) -> Dict:
    """
    Compare MediaPipe and dlib detectors for academic analysis.
    
    Args:
        image: Test image
        mediapipe_detector: MediaPipe detector instance
        dlib_detector: dlib detector instance (optional)
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # MediaPipe
    import time
    start = time.time()
    mp_landmarks = mediapipe_detector.detect_landmarks(image)
    mp_time = time.time() - start
    
    results['mediapipe'] = {
        'landmarks': mp_landmarks,
        'time': mp_time,
        'num_points': len(mp_landmarks.landmarks) if mp_landmarks else 0
    }
    
    # dlib (if available)
    if dlib_detector and dlib_detector.available:
        start = time.time()
        dlib_landmarks = dlib_detector.detect_landmarks(image)
        dlib_time = time.time() - start
        
        results['dlib'] = {
            'landmarks': dlib_landmarks,
            'time': dlib_time,
            'num_points': len(dlib_landmarks.landmarks) if dlib_landmarks else 0
        }
    
    return results


if __name__ == "__main__":
    print("Facial Landmark Detection Module")
    print("=" * 50)
    
    # Initialize detector
    detector = MediaPipeFaceDetector(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    )
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n✓ MediaPipe detector initialized")
    print(f"  - 468 facial landmarks")
    print(f"  - Real-time capable")
    print(f"  - Robust to pose and lighting variations")
    
    # Cleanup
    detector.close()
