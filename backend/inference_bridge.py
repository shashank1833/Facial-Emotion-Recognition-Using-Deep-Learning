import os
import sys
import cv2
import torch
import numpy as np
import base64
import json
import yaml

# Add current, parent, and src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(current_dir)
sys.path.append(parent_dir)
sys.path.append(src_dir)

from inference.inference_utils import InferenceBase

class NodeInference(InferenceBase):
    def __init__(self, model_path, config_path):
        # Redirect stdout to stderr temporarily to keep stdout clean for JSON
        original_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            super().__init__(model_path, config_path)
        finally:
            sys.stdout = original_stdout
    
    def predict_from_base64(self, base64_str):
        try:
            # Handle potential header in base64 string
            if ',' in base64_str:
                encoded_data = base64_str.split(',')[1]
            else:
                encoded_data = base64_str
                
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return None, f"Error decoding image: {str(e)}"
        
        if img is None:
            return None, "Invalid image data"
        
        # Process frame
        detection_success = True
        frame_data = self.process_single_frame(img)
        if frame_data is None:
            detection_success = False
            print(f"DEBUG: Landmark detection failed, using skip_detection=True", file=sys.stderr)
            frame_data = self.process_single_frame(img, skip_detection=True)
            
        # Predict
        emotion, confidence, probs = self.predict_cnn_only(frame_data)
        
        # Log probabilities for debugging
        prob_dict = {self.emotions[i]: float(probs[i]) for i in range(len(self.emotions))}
        print(f"DEBUG: Prediction: {emotion} ({confidence:.4f})", file=sys.stderr)
        print(f"DEBUG: Probabilities: {json.dumps(prob_dict)}", file=sys.stderr)
        print(f"DEBUG: Detection Success: {detection_success}", file=sys.stderr)
        
        return {
            "emotion": emotion,
            "confidence": float(confidence),
            "probabilities": prob_dict,
            "detection_success": detection_success
        }, None

# Config
POSSIBLE_MODEL_PATHS = [
    'checkpoints/best_model.pth',
    '../checkpoints/best_model.pth'
]
CONFIG_PATH = 'configs/config.yaml'
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = '../configs/config.yaml'

def main():
    # Load model
    model_path = None
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        error_msg = json.dumps({"error": "No model found"})
        print(error_msg)
        print(f"DEBUG: {error_msg}", file=sys.stderr)
        return

    try:
        print(f"DEBUG: Loading model from {model_path}", file=sys.stderr)
        inference = NodeInference(model_path, CONFIG_PATH)
        print(f"DEBUG: Model loaded successfully", file=sys.stderr)
    except Exception as e:
        error_msg = json.dumps({"error": f"Error loading model: {str(e)}"})
        print(error_msg)
        print(f"DEBUG: {error_msg}", file=sys.stderr)
        return

    # Read from stdin
    print(f"DEBUG: Entering stdin loop", file=sys.stderr)
    for line in sys.stdin:
        try:
            if not line.strip():
                continue
            print(f"DEBUG: Received line: {line[:50]}...", file=sys.stderr)
            data = json.loads(line)
            if 'image' not in data:
                print(json.dumps({"error": "No image data provided"}))
                continue
            
            result, error = inference.predict_from_base64(data['image'])
            if error:
                print(json.dumps({"error": error}))
            else:
                print(json.dumps(result))
            
            # Flush stdout for real-time communication
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
