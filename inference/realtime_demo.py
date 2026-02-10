"""
Real-Time Emotion Recognition Demo

This script demonstrates real-time emotion recognition from webcam feed.

Features:
- Live webcam processing
- Facial landmark visualization
- Zone highlighting
- Emotion prediction with confidence
- Temporal smoothing for stable predictions
- FPS display

Usage:
    python inference/realtime_demo.py --model checkpoints/best_model.pth --camera 0
"""

import os
import sys
import argparse
import yaml
import cv2
import torch
import numpy as np
from collections import deque
from typing import Optional, Deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import NoiseRobustPreprocessor
from landmark_detection import MediaPipeFaceDetector
from zone_extraction import ZoneExtractor
from models import create_full_model


class RealtimeEmotionRecognition:
    """
    Real-time emotion recognition from webcam.
    
    Processes video stream and displays emotion predictions with visualizations.
    """
    
    def __init__(self,
                 model_path: str,
                 config_path: str = 'configs/config.yaml',
                 camera_id: int = 0,
                 sequence_length: int = 16,
                 smoothing_window: int = 5):
        """
        Initialize real-time emotion recognition.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            camera_id: Camera device ID (0 for default webcam)
            sequence_length: Number of frames for LSTM
            smoothing_window: Window for temporal smoothing of predictions
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Initialize pipeline
        print("Initializing preprocessing pipeline...")
        self.preprocessor = NoiseRobustPreprocessor(
            median_kernel=self.config['preprocessing']['median_filter']['kernel_size'],
            use_clahe=self.config['preprocessing']['histogram_equalization']['enabled']
        )
        
        self.detector = MediaPipeFaceDetector(
            static_image_mode=False,
            min_detection_confidence=self.config['face_detection']['mediapipe']['min_detection_confidence'],
            min_tracking_confidence=self.config['face_detection']['mediapipe']['min_tracking_confidence']
        )
        
        self.zone_extractor = ZoneExtractor(
            target_size=self.config['zones']['resolution']
        )
        
        # Emotion labels
        self.emotions = self.config['emotions']['classes']
        
        # Frame buffer for sequence processing
        self.sequence_length = sequence_length
        self.frame_buffer: Deque = deque(maxlen=sequence_length)
        
        # Prediction smoothing
        self.smoothing_window = smoothing_window
        self.prediction_buffer: Deque = deque(maxlen=smoothing_window)
        
        # Camera
        self.camera_id = camera_id
        self.cap = None
        
        # Visualization settings
        self.show_landmarks = self.config['inference']['display']['show_landmarks']
        self.show_zones = self.config['inference']['display']['show_zones']
        self.show_probabilities = self.config['inference']['display']['show_probabilities']
        
        # FPS tracking
        self.fps_buffer: Deque = deque(maxlen=30)
        
        # Colors for visualization
        self.colors = {
            'forehead': (255, 0, 0),
            'left_eye': (0, 255, 0),
            'right_eye': (0, 255, 255),
            'nose': (255, 0, 255),
            'mouth': (0, 0, 255)
        }
    
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
    
    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Process single frame.
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Dictionary with processed data or None if no face detected
        """
        # Preprocess
        processed, _ = self.preprocessor.preprocess(frame)
        
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
            name: torch.from_numpy(zone.image).unsqueeze(0).unsqueeze(0).to(self.device)
            for name, zone in zones.items()
        }
        
        return {
            'full_face': full_face_tensor,
            'zones': zones_tensor,
            'landmarks': landmarks,
            'processed': processed,
            'original': frame
        }
    
    def predict_emotion(self) -> Optional[tuple]:
        if len(self.frame_buffer) == 0:
            return None

        latest_frame = self.frame_buffer[-1]

        with torch.no_grad():
            # 1. Extract hybrid CNN features (1, 1152)
            features = self.model.hybrid_cnn(
                latest_frame['full_face'],
                latest_frame['zones']
            )

            # 2. ADD sequence dimension for LSTM → (1, 1, 1152)
            features = features.unsqueeze(1)

            # 3. Pass through FULL LSTM (includes classifier)
            logits = self.model.temporal_lstm(features)

            probabilities = torch.softmax(logits, dim=1)

            confidence, predicted = probabilities.max(1)

            emotion_idx = predicted.item()
            emotion_label = self.emotions[emotion_idx]
            confidence_val = confidence.item()
            probs = probabilities[0].cpu().numpy()

        return emotion_label, confidence_val, probs

    def smooth_predictions(self, emotion: str, confidence: float) -> tuple:
        """
        Apply temporal smoothing to predictions.
        
        Args:
            emotion: Predicted emotion
            confidence: Prediction confidence
            
        Returns:
            Tuple of (smoothed_emotion, smoothed_confidence)
        """
        self.prediction_buffer.append((emotion, confidence))
        
        if len(self.prediction_buffer) < self.smoothing_window:
            return emotion, confidence
        
        # Majority voting
        emotion_counts = {}
        for pred_emotion, pred_conf in self.prediction_buffer:
            if pred_emotion in emotion_counts:
                emotion_counts[pred_emotion] += pred_conf
            else:
                emotion_counts[pred_emotion] = pred_conf
        
        # Get most common emotion weighted by confidence
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        smoothed_confidence = emotion_counts[smoothed_emotion] / len(self.prediction_buffer)
        
        return smoothed_emotion, smoothed_confidence
    
    def draw_visualizations(self, frame: np.ndarray, frame_data: dict,
                           emotion: str, confidence: float, probs: np.ndarray) -> np.ndarray:
        """
        Draw visualizations on frame.
        
        Args:
            frame: Input frame
            frame_data: Processed frame data
            emotion: Predicted emotion
            confidence: Prediction confidence
            probs: Emotion probabilities
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw landmarks
        if self.show_landmarks:
            landmarks = frame_data['landmarks']
            for point in landmarks.landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(vis_frame, (x, y), 1, (0, 255, 0), -1)
        
        # Draw zone bounding boxes
        if self.show_zones:
            zones = self.zone_extractor.extract_all_zones(
                frame_data['processed'],
                frame_data['landmarks'].landmarks
            )
            
            for zone_name, zone in zones.items():
                x, y, w, h = zone.original_bbox
                color = self.colors.get(zone_name, (255, 255, 255))
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis_frame, zone_name, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw emotion label
        label_text = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(vis_frame, label_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Draw probability bars
        if self.show_probabilities:
            bar_height = 20
            bar_width = 200
            start_y = 60
            
            for i, (emotion_name, prob) in enumerate(zip(self.emotions, probs)):
                y = start_y + i * (bar_height + 5)
                
                # Background
                cv2.rectangle(vis_frame, (10, y), (10 + bar_width, y + bar_height),
                             (50, 50, 50), -1)
                
                # Probability bar
                fill_width = int(bar_width * prob)
                cv2.rectangle(vis_frame, (10, y), (10 + fill_width, y + bar_height),
                             (0, 255, 0), -1)
                
                # Text
                text = f"{emotion_name}: {prob*100:.1f}%"
                cv2.putText(vis_frame, text, (10 + bar_width + 10, y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_frame
    
    def run(self):
        """Run real-time emotion recognition."""
        print("\n" + "="*60)
        print("  REAL-TIME EMOTION RECOGNITION")
        print("="*60)
        print("\nControls:")
        print("  q - Quit")
        print("  l - Toggle landmarks")
        print("  z - Toggle zones")
        print("  p - Toggle probabilities")
        print("\nStarting webcam...")
        
        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_id}")
            return
        
        print(f"✓ Camera opened successfully")
        print("\nPress 'q' to quit\n")
        
        import time
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("✗ Error: Could not read frame")
                break
            
            # Process frame
            frame_data = self.process_frame(frame)
            
            if frame_data is not None:
                # Add to buffer
                self.frame_buffer.append(frame_data)
                
                # Predict emotion
                prediction = self.predict_emotion()
                
                if prediction is not None:
                    emotion, confidence, probs = prediction
                    
                    # Smooth predictions
                    emotion, confidence = self.smooth_predictions(emotion, confidence)
                    
                    # Draw visualizations
                    vis_frame = self.draw_visualizations(
                        frame, frame_data, emotion, confidence, probs
                    )
                else:
                    vis_frame = frame
                    cv2.putText(vis_frame, "Collecting frames...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            else:
                vis_frame = frame
                cv2.putText(vis_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            self.fps_buffer.append(fps)
            avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
            
            cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (vis_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display
            cv2.imshow('Real-Time Emotion Recognition', vis_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('z'):
                self.show_zones = not self.show_zones
            elif key == ord('p'):
                self.show_probabilities = not self.show_probabilities
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        
        print("\n✓ Real-time demo completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-Time Emotion Recognition Demo')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Sequence length for LSTM')
    parser.add_argument('--smoothing', type=int, default=5,
                       help='Smoothing window size')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found at {args.model}")
        print("\nTo train a model, run:")
        print("  python training/train.py --data data/fer2013/fer2013.csv")
        return
    
    # Create demo
    demo = RealtimeEmotionRecognition(
        model_path=args.model,
        config_path=args.config,
        camera_id=args.camera,
        sequence_length=args.sequence_length,
        smoothing_window=args.smoothing
    )
    
    # Run
    demo.run()


if __name__ == "__main__":
    main()
