"""
Shared Inference Utilities

Common functions for image, video, and webcam emotion inference.
"""

import os
import sys
import yaml
import cv2
import torch
import numpy as np
from typing import Optional, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import NoiseRobustPreprocessor
from landmark_detection import MediaPipeFaceDetector
from zone_extraction import ZoneExtractor
from models import create_full_model


class InferenceBase:
    """
    Base class for emotion inference operations.
    
    Provides shared functionality for preprocessing, landmark detection,
    zone extraction, and model prediction.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = 'configs/config.yaml'):
        """
        Initialize inference base.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize preprocessing pipeline
        self.preprocessor = NoiseRobustPreprocessor(
            median_kernel=self.config['preprocessing']['median_filter']['kernel_size'],
            use_clahe=self.config['preprocessing']['histogram_equalization']['enabled']
        )
        
        self.detector = MediaPipeFaceDetector(
            static_image_mode=True,  # Better for single images
            min_detection_confidence=self.config['face_detection']['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=self.config['face_detection']['mediapipe']['min_tracking_confidence']
        )
        
        self.zone_extractor = ZoneExtractor(
            target_size=self.config['zones']['resolution']
        )
        
        # Load model
        print(f"Loading model from {model_path}...")
        
        # Default emotions (can be overridden by _load_model)
        self.emotions = self.config['emotions']['classes']
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print("[OK] Inference pipeline initialized")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Use config from checkpoint if available, otherwise use default
        model_config = checkpoint.get('config', self.config)
        
        # Update self.emotions and other relevant settings from checkpoint config
        if 'emotions' in model_config:
            self.emotions = model_config['emotions']['classes']
            print(f"Using emotions from checkpoint: {self.emotions}")
        
        # Create model
        model = create_full_model(
            hybrid_cnn_config=model_config['model'],
            lstm_config=model_config['model']['lstm']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def process_single_frame(self, frame: np.ndarray, skip_detection: bool = False) -> Optional[Dict]:
        """
        Process single frame and extract features.
    
        If skip_detection=True, assumes FER-style pre-cropped face
        and skips landmark + zone extraction.
        """
    
        # Preprocess
        # We need to maintain BGR for some operations or consistent scaling
        # The preprocessor handle to_grayscale internally if needed
        processed, _ = self.preprocessor.preprocess(frame, to_grayscale=True)
    
        # FER-2013 compatibility: Check if image is very small (like training samples)
        # If so, we must upscale for landmark detection but NOT crop (since it's already a face crop)
        is_small_sample = frame.shape[0] < 100 or frame.shape[1] < 100

        if not skip_detection and is_small_sample:
            # Upscale small image for better landmark detection
            # FER-2013 is 48x48, which is too small for MediaPipe
            detection_size = 224
            processed_for_detection = cv2.resize(processed, (detection_size, detection_size))
            landmarks = self.detector.detect_landmarks(processed_for_detection)
            
            if landmarks is not None:
                # Successfully detected landmarks on upscaled small image
                # Use the upscaled image for consistency with training (data_loader.py line 167)
                zones_extracted = self.zone_extractor.extract_all_zones(processed_for_detection, landmarks.landmarks)
                
                # Prepare full face (already 224x224)
                full_face = processed_for_detection
                
                # Use normalization based on config
                face_float = full_face.astype(np.float32)
                if self.config['zones']['normalization'] == 'minmax':
                    min_val, max_val = face_float.min(), face_float.max()
                    face_norm = (face_float - min_val) / (max_val - min_val) if max_val > min_val else face_float / 255.0
                else:
                    face_norm = face_float / 255.0
                
                full_face_tensor = torch.from_numpy(face_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                
                zones_tensor = {
                    name: torch.from_numpy(zone.image).unsqueeze(0).unsqueeze(0).float().to(self.device)
                    for name, zone in zones_extracted.items()
                }
                
                return {
                    "full_face": full_face_tensor,
                    "zones": zones_tensor,
                    "landmarks": landmarks,
                    "processed": processed,
                    "original": frame,
                }

        if skip_detection:
            # ===== FER MODE (or fallback) =====
            # Use full image as face
            face = processed

            # Resize directly to CNN input
            full_face = cv2.resize(face, (224, 224))
            
            # Use normalization based on config
            face_float = full_face.astype(np.float32)
            if self.config['zones']['normalization'] == 'minmax':
                min_val, max_val = face_float.min(), face_float.max()
                face_norm = (face_float - min_val) / (max_val - min_val) if max_val > min_val else face_float / 255.0
            else:
                face_norm = face_float / 255.0

            full_face_tensor = (
                torch.from_numpy(face_norm)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            ).to(self.device)

            # Fallback for zones: Use resized full face (as done in training)
            zone_size = self.zone_extractor.target_size
            zone_fallback = cv2.resize(face, (zone_size, zone_size))
            
            # Use normalization based on config
            zone_float = zone_fallback.astype(np.float32)
            if self.config['zones']['normalization'] == 'minmax':
                z_min, z_max = zone_float.min(), zone_float.max()
                zone_norm = (zone_float - z_min) / (z_max - z_min) if z_max > z_min else zone_float / 255.0
            else:
                zone_norm = zone_float / 255.0

            zone_fallback_tensor = (
                torch.from_numpy(zone_norm)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
            ).to(self.device)
            
            zones_tensor = {
                'forehead': zone_fallback_tensor,
                'left_eye': zone_fallback_tensor,
                'right_eye': zone_fallback_tensor,
                'nose': zone_fallback_tensor,
                'mouth': zone_fallback_tensor
            }

            return {
                "full_face": full_face_tensor,
                "zones": zones_tensor,
                "landmarks": None,
                "processed": processed,
                "original": frame,
            }
    
        # ===== NORMAL MODE (real images / webcam) =====
        landmarks = self.detector.detect_landmarks(processed)
        if landmarks is None:
            return None
    
        # Extract zones
        zones = self.zone_extractor.extract_all_zones(processed, landmarks.landmarks)
    
        # Prepare full face - CROP to face region first!
        face_region = self.detector.get_face_region(processed, landmarks)
        full_face = cv2.resize(face_region, (224, 224))
        
        # Use normalization based on config
        face_float = full_face.astype(np.float32)
        if self.config['zones']['normalization'] == 'minmax':
            min_val, max_val = face_float.min(), face_float.max()
            face_norm = (face_float - min_val) / (max_val - min_val) if max_val > min_val else face_float / 255.0
        else:
            face_norm = face_float / 255.0

        full_face_tensor = (
            torch.from_numpy(face_norm)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        ).to(self.device)
    
        # Prepare zones
        zones_tensor = {
            name: (
                torch.from_numpy(zone.image)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                # zone.image is already normalized to [0, 1] by zone_extractor
            ).to(self.device)
            for name, zone in zones.items()
        }
    
        return {
            "full_face": full_face_tensor,
            "zones": zones_tensor,
            "landmarks": landmarks,
            "processed": processed,
            "original": frame,
        }


    def predict_cnn_only(self, frame_data: Dict) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion using CNN only (no temporal LSTM).
        
        Args:
            frame_data: Processed frame data
            
        Returns:
            Tuple of (emotion_label, confidence, probabilities)
        """
        with torch.no_grad():
            # Extract hybrid CNN features
            features = self.model.hybrid_cnn(
                frame_data['full_face'],
                frame_data['zones']
            )
            
            # Add sequence dimension (batch=1, seq_len=1, feature_dim)
            # This allows passing a single frame through the LSTM-based architecture
            features = features.unsqueeze(1)
            
            # Pass through temporal model (LSTM + Classifier)
            logits = self.model.temporal_lstm(features)
            
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(1)
            
            emotion_idx = predicted.item()
            emotion_label = self.emotions[emotion_idx]
            confidence_val = confidence.item()
            probs = probabilities[0].cpu().numpy()
            
            # Debug log
            print(f"DEBUG: CNN Prediction index: {emotion_idx} ({emotion_label}), Confidence: {confidence_val:.4f}", file=sys.stderr)
        
        return emotion_label, confidence_val, probs
    
    def predict_with_lstm(self, features_sequence: torch.Tensor) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion using CNN + LSTM.
        
        Args:
            features_sequence: Tensor of shape (batch, seq_len, feature_dim)
            
        Returns:
            Tuple of (emotion_label, confidence, probabilities)
        """
        with torch.no_grad():
            # Pass through LSTM
            logits = self.model.temporal_lstm(features_sequence)
            
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(1)
            
            emotion_idx = predicted.item()
            emotion_label = self.emotions[emotion_idx]
            confidence_val = confidence.item()
            probs = probabilities[0].cpu().numpy()
        
        return emotion_label, confidence_val, probs
    
    def extract_cnn_features(self, frame_data: Dict) -> torch.Tensor:
        """
        Extract CNN features without classification.
        
        Args:
            frame_data: Processed frame data
            
        Returns:
            Feature tensor of shape (1, feature_dim)
        """
        with torch.no_grad():
            features = self.model.hybrid_cnn(
                frame_data['full_face'],
                frame_data['zones']
            )
        return features
    
    def visualize_prediction(self, 
                            frame: np.ndarray,
                            emotion: str,
                            confidence: float,
                            probabilities: np.ndarray,
                            landmarks: Optional[object] = None) -> np.ndarray:
        """
        Draw prediction on frame.
        
        Args:
            frame: Input frame
            emotion: Predicted emotion
            confidence: Confidence score
            probabilities: Probability distribution
            landmarks: Optional facial landmarks
            
        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()
        
        # Convert to BGR if grayscale
        if len(vis_frame.shape) == 2:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_GRAY2BGR)
        
        # Draw landmarks
        if landmarks is not None:
            for point in landmarks.landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw emotion label with background
        label_text = f"{emotion} ({confidence*100:.1f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Background rectangle
        cv2.rectangle(vis_frame, (10, 10), (20 + text_w, 30 + text_h), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (10, 10), (20 + text_w, 30 + text_h), (0, 255, 0), 2)
        
        # Text
        cv2.putText(vis_frame, label_text, (15, 20 + text_h),
                   font, font_scale, (0, 255, 0), thickness)
        
        # Draw probability bars
        bar_height = 25
        bar_width = 300
        start_y = 60
        
        for i, (emotion_name, prob) in enumerate(zip(self.emotions, probabilities)):
            y = start_y + i * (bar_height + 10)
            
            # Background
            cv2.rectangle(vis_frame, (10, y), (10 + bar_width, y + bar_height),
                         (40, 40, 40), -1)
            cv2.rectangle(vis_frame, (10, y), (10 + bar_width, y + bar_height),
                         (100, 100, 100), 1)
            
            # Probability bar
            fill_width = int(bar_width * prob)
            color = (0, 255, 0) if emotion_name == emotion else (100, 200, 255)
            cv2.rectangle(vis_frame, (10, y), (10 + fill_width, y + bar_height),
                         color, -1)
            
            # Text
            text = f"{emotion_name}: {prob*100:.1f}%"
            cv2.putText(vis_frame, text, (20, y + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_frame
    
    def close(self):
        """Clean up resources."""
        self.detector.close()


def aggregate_predictions(predictions: list, 
                         method: str = 'majority_vote') -> Tuple[str, float, np.ndarray]:
    """
    Aggregate multiple predictions into a single result.
    
    Args:
        predictions: List of (emotion, confidence, probabilities) tuples
        method: Aggregation method ('majority_vote' or 'weighted_average')
        
    Returns:
        Tuple of (final_emotion, final_confidence, avg_probabilities)
    """
    if not predictions:
        return None, 0.0, None
    
    emotions = [p[0] for p in predictions]
    confidences = [p[1] for p in predictions]
    all_probs = np.array([p[2] for p in predictions])
    
    if method == 'majority_vote':
        # Count votes weighted by confidence
        emotion_scores = {}
        for emotion, confidence in zip(emotions, confidences):
            if emotion in emotion_scores:
                emotion_scores[emotion] += confidence
            else:
                emotion_scores[emotion] = confidence
        
        final_emotion = max(emotion_scores, key=emotion_scores.get)
        final_confidence = emotion_scores[final_emotion] / len(predictions)
        
    elif method == 'weighted_average':
        # Average probabilities weighted by confidence
        weights = np.array(confidences).reshape(-1, 1)
        weighted_probs = all_probs * weights
        avg_probs = weighted_probs.sum(axis=0) / weights.sum()
        
        final_emotion_idx = avg_probs.argmax()
        # Use emotion from first prediction to get label mapping
        all_emotion_labels = predictions[0][2]  # This is the probability array
        # We need to get emotion labels from somewhere - use the most common approach
        final_confidence = avg_probs[final_emotion_idx]
        
        return None, final_confidence, avg_probs  # Emotion label handled by caller
    
    # Average probabilities for visualization
    avg_probs = all_probs.mean(axis=0)
    
    return final_emotion, final_confidence, avg_probs


def save_prediction_report(output_path: str,
                          predictions: dict,
                          source_info: dict):
    """
    Save prediction results to a text file.
    
    Args:
        output_path: Path to save report
        predictions: Dictionary with prediction results
        source_info: Dictionary with source information
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("EMOTION RECOGNITION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Source info
        f.write("Source Information:\n")
        for key, value in source_info.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Predictions
        f.write("Prediction Results:\n")
        f.write(f"  Emotion: {predictions['emotion']}\n")
        f.write(f"  Confidence: {predictions['confidence']*100:.2f}%\n")
        f.write("\n")
        
        # Probability distribution
        f.write("Probability Distribution:\n")
        for emotion, prob in zip(predictions['emotions'], predictions['probabilities']):
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            f.write(f"  {emotion:10s} {bar} {prob*100:5.2f}%\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Report saved to {output_path}")
