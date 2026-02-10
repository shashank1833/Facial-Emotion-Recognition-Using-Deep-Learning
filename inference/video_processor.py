"""
Real-time Video Processor for Emotion Recognition
Handles webcam/video input, preprocessing, zone extraction, and prediction
"""

import cv2
import numpy as np
from collections import deque
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.noise_robust import NoiseRobustPreprocessor
from landmark_detection.mediapipe_detector import MediaPipeFaceDetector
from zone_extraction.zone_extractor import ZoneExtractor


class VideoEmotionProcessor:
    """
    Real-time video processor for emotion recognition
    Processes video streams frame-by-frame and maintains temporal context
    """
    
    def __init__(self, 
                 model,
                 sequence_length=16,
                 global_img_size=224,
                 zone_img_size=48,
                 fps_target=20,
                 confidence_threshold=0.5):
        """
        Initialize video processor
        
        Args:
            model: Trained HybridEmotionModel instance
            sequence_length: Number of frames for temporal context
            global_img_size: Size for global face image
            zone_img_size: Size for zone images
            fps_target: Target processing FPS
            confidence_threshold: Minimum confidence for displaying predictions
        """
        self.model = model
        self.sequence_length = sequence_length
        self.global_img_size = global_img_size
        self.zone_img_size = zone_img_size
        self.fps_target = fps_target
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        self.preprocessor = NoiseRobustPreprocessor()
        self.face_detector = MediaPipeFaceDetector()
        self.zone_extractor = ZoneExtractor()
        
        # Frame buffers for temporal sequences
        self.frame_buffers = {
            'global': deque(maxlen=sequence_length),
            'forehead': deque(maxlen=sequence_length),
            'left_eye': deque(maxlen=sequence_length),
            'right_eye': deque(maxlen=sequence_length),
            'nose': deque(maxlen=sequence_length),
            'mouth': deque(maxlen=sequence_length)
        }
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0
        self.emotion_history = deque(maxlen=100)
        
        # Emotion colors for visualization (BGR format)
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 165, 0), # Orange
            'Neutral': (200, 200, 200) # Gray
        }
    
    def process_frame(self, frame):
        """
        Process a single frame and extract zones
        
        Args:
            frame: Input BGR frame from video
            
        Returns:
            zones: Dict of preprocessed zone images
            face_frame: Preprocessed global face image
            landmarks: Facial landmarks
            success: Boolean indicating if face was detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply noise-robust preprocessing
        preprocessed = self.preprocessor.preprocess(gray)
        
        # Detect face and landmarks
        landmarks = self.face_detector.detect_landmarks(frame)
        
        if landmarks is None:
            return None, None, None, False
        
        # Extract bounding box for global face
        h, w = preprocessed.shape
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]
        
        x_min, x_max = int(max(0, min(x_coords) - 20)), int(min(w, max(x_coords) + 20))
        y_min, y_max = int(max(0, min(y_coords) - 30)), int(min(h, max(y_coords) + 30))
        
        face_roi = preprocessed[y_min:y_max, x_min:x_max]
        
        if face_roi.size == 0:
            return None, None, None, False
        
        # Resize global face
        face_resized = cv2.resize(face_roi, (self.global_img_size, self.global_img_size))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_frame = np.expand_dims(face_normalized, axis=-1)
        
        # Extract zones
        zones = self.zone_extractor.extract_all_zones(preprocessed, landmarks)
        
        # Resize and normalize zones
        processed_zones = {}
        for zone_name, zone_img in zones.items():
            if zone_img is not None and zone_img.size > 0:
                zone_resized = cv2.resize(zone_img, (self.zone_img_size, self.zone_img_size))
                zone_normalized = zone_resized.astype(np.float32) / 255.0
                processed_zones[zone_name] = np.expand_dims(zone_normalized, axis=-1)
            else:
                # Fill with zeros if zone extraction failed
                processed_zones[zone_name] = np.zeros((self.zone_img_size, self.zone_img_size, 1), 
                                                      dtype=np.float32)
        
        return processed_zones, face_frame, landmarks, True
    
    def update_buffers(self, zones, face_frame):
        """
        Update frame buffers with new data
        
        Args:
            zones: Dict of zone images
            face_frame: Global face image
        """
        self.frame_buffers['global'].append(face_frame)
        self.frame_buffers['forehead'].append(zones['forehead'])
        self.frame_buffers['left_eye'].append(zones['left_eye'])
        self.frame_buffers['right_eye'].append(zones['right_eye'])
        self.frame_buffers['nose'].append(zones['nose'])
        self.frame_buffers['mouth'].append(zones['mouth'])
    
    def buffers_ready(self):
        """Check if buffers have enough frames for prediction"""
        return len(self.frame_buffers['global']) == self.sequence_length
    
    def get_sequences(self):
        """
        Get frame sequences from buffers
        
        Returns:
            sequences: Dict of numpy arrays ready for model input
        """
        sequences = {}
        for zone_name, buffer in self.frame_buffers.items():
            sequences[zone_name] = np.array(list(buffer))
        return sequences
    
    def predict_emotion(self):
        """
        Predict emotion from current buffer state
        
        Returns:
            emotion: Predicted emotion name
            confidence: Confidence score
            all_probs: All emotion probabilities
        """
        if not self.buffers_ready():
            return self.current_emotion, self.current_confidence, None
        
        sequences = self.get_sequences()
        emotion, probs = self.model.predict_emotion(sequences)
        confidence = np.max(probs)
        
        # Update only if confidence is high enough
        if confidence >= self.confidence_threshold:
            self.current_emotion = emotion
            self.current_confidence = confidence
            self.emotion_history.append(emotion)
        
        return emotion, confidence, probs
    
    def draw_results(self, frame, landmarks, emotion, confidence, probs=None):
        """
        Draw emotion predictions and landmarks on frame
        
        Args:
            frame: Input frame
            landmarks: Facial landmarks
            emotion: Predicted emotion
            confidence: Confidence score
            probs: All emotion probabilities (optional)
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw landmarks
        if landmarks is not None:
            for x, y in landmarks:
                cv2.circle(annotated, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        # Draw emotion label with colored background
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Main emotion display
        label = f"{emotion}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(annotated, (10, 10), (20 + text_w, 20 + text_h + baseline), 
                     color, -1)
        
        # Draw text
        cv2.putText(annotated, label, (15, 15 + text_h), font, font_scale, 
                   (0, 0, 0), thickness)
        
        # Draw probability bars on the right side
        if probs is not None:
            bar_width = 200
            bar_height = 20
            x_start = w - bar_width - 20
            y_start = 30
            
            cv2.putText(annotated, "Emotion Probabilities:", (x_start, y_start - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, (emotion_name, prob) in enumerate(zip(self.model.emotion_labels, probs)):
                y_pos = y_start + i * (bar_height + 5)
                
                # Background bar
                cv2.rectangle(annotated, (x_start, y_pos), 
                            (x_start + bar_width, y_pos + bar_height),
                            (50, 50, 50), -1)
                
                # Probability bar
                bar_len = int(bar_width * prob)
                bar_color = self.emotion_colors.get(emotion_name, (255, 255, 255))
                cv2.rectangle(annotated, (x_start, y_pos),
                            (x_start + bar_len, y_pos + bar_height),
                            bar_color, -1)
                
                # Label
                label_text = f"{emotion_name}: {prob:.2f}"
                cv2.putText(annotated, label_text, (x_start + 5, y_pos + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw FPS
        if len(self.fps_counter) > 0:
            fps = 1.0 / np.mean(self.fps_counter)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated
    
    def process_video(self, video_source=0, display=True, save_output=None):
        """
        Process video from webcam or file
        
        Args:
            video_source: 0 for webcam, or path to video file
            display: Whether to display results
            save_output: Path to save output video (optional)
            
        Returns:
            None (runs until 'q' is pressed)
        """
        # Open video
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video source {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video opened: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        print("\nStarting emotion recognition...")
        print("Press 'q' to quit, 's' to save screenshot\n")
        
        frame_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                zones, face_frame, landmarks, success = self.process_frame(frame)
                
                if success:
                    # Update buffers
                    self.update_buffers(zones, face_frame)
                    
                    # Predict emotion if buffers are ready
                    emotion, confidence, probs = self.predict_emotion()
                    
                    # Draw results
                    annotated = self.draw_results(frame, landmarks, emotion, confidence, probs)
                else:
                    annotated = frame.copy()
                    cv2.putText(annotated, "No face detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update FPS counter
                elapsed = time.time() - start_time
                self.fps_counter.append(elapsed)
                
                # Display
                if display:
                    cv2.imshow('Emotion Recognition', annotated)
                
                # Save
                if writer:
                    writer.write(annotated)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"emotion_screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, annotated)
                    print(f"Screenshot saved: {screenshot_path}")
                
                # Limit FPS
                target_delay = 1.0 / self.fps_target
                actual_delay = time.time() - start_time
                if actual_delay < target_delay:
                    time.sleep(target_delay - actual_delay)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\nProcessed {frame_count} frames")
            if len(self.emotion_history) > 0:
                print("\nEmotion Statistics:")
                from collections import Counter
                emotion_counts = Counter(self.emotion_history)
                for emotion, count in emotion_counts.most_common():
                    percentage = 100 * count / len(self.emotion_history)
                    print(f"  {emotion}: {percentage:.1f}%")
    
    def process_image(self, image_path, display=True, save_output=None):
        """
        Process a single image
        
        Args:
            image_path: Path to image file
            display: Whether to display result
            save_output: Path to save annotated image
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image {image_path}")
            return
        
        # Process frame multiple times to fill buffer
        for _ in range(self.sequence_length):
            zones, face_frame, landmarks, success = self.process_frame(frame)
            if success:
                self.update_buffers(zones, face_frame)
        
        if success:
            # Predict
            emotion, confidence, probs = self.predict_emotion()
            
            # Draw results
            annotated = self.draw_results(frame, landmarks, emotion, confidence, probs)
            
            # Display
            if display:
                cv2.imshow('Emotion Recognition - Image', annotated)
                print(f"\nPredicted Emotion: {emotion} ({confidence:.2%} confidence)")
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Save
            if save_output:
                cv2.imwrite(save_output, annotated)
                print(f"Annotated image saved: {save_output}")
        else:
            print("No face detected in image")


if __name__ == "__main__":
    print("Video Emotion Processor - Standalone Test")
    print("="*60)
    print("\nNote: This script requires a trained model to run.")
    print("Please use 'realtime_demo.py' for full demonstration.")
    print("\nModule loaded successfully!")