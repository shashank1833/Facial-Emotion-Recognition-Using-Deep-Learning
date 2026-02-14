"""
Refactored Real-Time Emotion Recognition Demo (Image-based)

This version removes temporal (LSTM) and zone-based modeling in favor of a 
clean, efficient transfer learning approach.

Key improvements:
1. Pure image-based inference (no sequence buffer required)
2. Uses EfficientNet/MobileNet backbone via transfer learning
3. Robust confidence smoothing for stable predictions
4. Enhanced FPS display and visualization
"""

import os
import sys
import argparse
import yaml
import cv2
import torch
import numpy as np
import time
from collections import deque
from typing import Optional, Deque, Tuple, List
import torch.nn.functional as F

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.emotion_model import EmotionModel
from preprocessing.noise_robust import NoiseRobustPreprocessor

class RealtimeEmotionRecognition:
    """
    Real-time emotion recognition from webcam using transfer learning.
    """
    
    def __init__(self,
                 model_path: str,
                 config_path: str = 'configs/config.yaml',
                 camera_id: int = 0):
        """Initialize real-time emotion recognition."""
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
        
        # Initialize preprocessing
        print("Initializing preprocessing...")
        self.preprocessor = NoiseRobustPreprocessor(
            median_kernel=self.config['preprocessing']['median_filter']['kernel_size'],
            use_clahe=self.config['preprocessing']['histogram_equalization']['enabled']
        )
        
        # Face detection (using Haar Cascades for speed in demo, or can keep mediapipe if preferred)
        # Haar Cascades is lighter for a simple demo
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Emotion labels
        self.emotions = self.config['emotions']['classes']
        
        # Prediction smoothing
        self.smoothing_window = self.config['inference']['smoothing_window']
        self.prob_buffer: Deque = deque(maxlen=self.smoothing_window)
        
        # Camera
        self.camera_id = camera_id
        self.cap = None
        
        # Visualization settings
        self.show_probabilities = self.config['inference']['display']['show_probabilities']
        self.show_fps = self.config['inference']['display']['show_fps']
        self.show_box = self.config['inference']['display']['show_box']
        self.conf_threshold = self.config['inference']['confidence_threshold']
        
        # FPS tracking
        self.fps_buffer: Deque = deque(maxlen=30)
        
        # Input size
        self.input_size = self.config['model']['input_size']

    def _load_model(self, model_path: str) -> EmotionModel:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model using config
        model = EmotionModel(
            backbone_name=self.config['model']['backbone'],
            num_classes=len(self.config['emotions']['classes']),
            pretrained=False, # We are loading weights
            dropout=self.config['model']['dropout']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model

    def preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        # Noise reduction and enhancement
        processed, _ = self.preprocessor.preprocess(face_img)
        
        # Resize to model input size
        face_resized = cv2.resize(processed, (self.input_size, self.input_size))
        
        # Convert to RGB (if it was grayscale) or ensure 3 channels
        if len(face_resized.shape) == 2:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
        else:
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
        # To tensor: [C, H, W]
        face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        face_tensor = (face_tensor - mean) / std
        
        return face_tensor.unsqueeze(0).to(self.device)

    def predict(self, face_tensor: torch.Tensor) -> Tuple[str, float, np.ndarray]:
        """Predict emotion and apply smoothing."""
        with torch.no_grad():
            logits = self.model(face_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Add to smoothing buffer
        self.prob_buffer.append(probs)
        
        # Calculate smoothed probabilities (mean across window)
        avg_probs = np.mean(self.prob_buffer, axis=0)
        
        emotion_idx = np.argmax(avg_probs)
        emotion_label = self.emotions[emotion_idx]
        confidence = avg_probs[emotion_idx]
        
        return emotion_label, confidence, avg_probs

    def draw_ui(self, frame: np.ndarray, face_box: Optional[Tuple[int, int, int, int]], 
                emotion: Optional[str], confidence: Optional[float], 
                probs: Optional[np.ndarray], fps: float) -> np.ndarray:
        """Draw visualization overlay."""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # Draw Face Box and Label
        if face_box is not None and self.show_box:
            x, y, fw, fh = face_box
            color = (0, 255, 0) if confidence and confidence > self.conf_threshold else (0, 165, 255)
            cv2.rectangle(vis_frame, (x, y), (x + fw, y + fh), color, 2)
            
            if emotion:
                label = f"{emotion}: {confidence*100:.1f}%"
                cv2.putText(vis_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw FPS
        if self.show_fps:
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (w - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw Probability Bars
        if self.show_probabilities and probs is not None:
            bar_w, bar_h = 150, 20
            start_x, start_y = 20, 60
            
            for i, (name, prob) in enumerate(zip(self.emotions, probs)):
                y_pos = start_y + i * (bar_h + 10)
                
                # Label
                cv2.putText(vis_frame, f"{name[:3]}:", (start_x, y_pos + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Bar background
                cv2.rectangle(vis_frame, (start_x + 40, y_pos), (start_x + 40 + bar_w, y_pos + bar_h), (50, 50, 50), -1)
                
                # Bar fill
                fill_w = int(bar_w * prob)
                cv2.rectangle(vis_frame, (start_x + 40, y_pos), (start_x + 40 + fill_w, y_pos + bar_h), (0, 255, 0), -1)
                
                # Percentage
                cv2.putText(vis_frame, f"{prob*100:.0f}%", (start_x + 45 + bar_w, y_pos + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis_frame

    def run(self):
        """Run real-time inference loop."""
        print("\n" + "="*60)
        print("  REFECTORED IMAGE-BASED EMOTION RECOGNITION")
        print("="*60)
        print("Controls: 'q' to quit\n")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_id}")
            return

        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret: break

            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            emotion, confidence, probs = None, None, None
            face_box = None

            if len(faces) > 0:
                # Process largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                (x, y, w, h) = faces[0]
                face_box = (x, y, w, h)
                
                # Extract and predict
                face_img = frame[y:y+h, x:x+w]
                if face_img.size > 0:
                    face_tensor = self.preprocess_face(face_img)
                    emotion, confidence, probs = self.predict(face_tensor)
            else:
                self.prob_buffer.clear() # Reset smoothing when no face detected

            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            self.fps_buffer.append(fps)
            avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)

            # Draw
            vis_frame = self.draw_ui(frame, face_box, emotion, confidence, probs, avg_fps)
            
            cv2.imshow('Emotion Recognition (Transfer Learning)', vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Demo completed")

def main():
    parser = argparse.ArgumentParser(description='Real-Time Emotion Recognition Demo')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found at {args.model}")
        return

    demo = RealtimeEmotionRecognition(args.model, args.config, args.camera)
    demo.run()

if __name__ == "__main__":
    main()
