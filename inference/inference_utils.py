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
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Emotion labels
        self.emotions = self.config['emotions']['classes']
        
        print("✓ Inference pipeline initialized")
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = create_full_model(
            hybrid_cnn_config=self.config['model'],
            lstm_config=self.config['model']['lstm']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def process_single_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process single frame and extract features.
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Dictionary with processed data or None if no face detected
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        # Preprocess
        processed, _ = self.preprocessor.preprocess(frame_gray)
        
        # Detect landmarks
        landmarks = self.detector.detect_landmarks(processed)
        
        if landmarks is None:
            return None
        
        # Extract zones
        zones = self.zone_extractor.extract_all_zones(processed, landmarks.landmarks)
        
        # Prepare full face
        full_face = cv2.resize(processed, (224, 224))
        full_face_tensor = torch.from_numpy(full_face).unsqueeze(0).unsqueeze(0).float() / 255.0
        full_face_tensor = full_face_tensor.to(self.device)
        
        # Prepare zones
        zones_tensor = {
            name: torch.from_numpy(zone.image).unsqueeze(0).unsqueeze(0).float() / 255.0
            for name, zone in zones.items()
        }
        zones_tensor = {k: v.to(self.device) for k, v in zones_tensor.items()}
        
        return {
            'full_face': full_face_tensor,
            'zones': zones_tensor,
            'landmarks': landmarks,
            'processed': processed,
            'original': frame
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
            
            # Get the classifier from LSTM module
            # The LSTM module contains: lstm -> dropout -> fc
            # For single frame, we skip LSTM and use just the classifier
            logits = self.model.temporal_lstm.fc(features)
            
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(1)
            
            emotion_idx = predicted.item()
            emotion_label = self.emotions[emotion_idx]
            confidence_val = confidence.item()
            probs = probabilities[0].cpu().numpy()
        
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
    with open(output_path, 'w') as f:
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
