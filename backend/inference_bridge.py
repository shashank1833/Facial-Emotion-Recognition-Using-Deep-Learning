import os
import sys
import cv2
import torch
import numpy as np
import base64
import json
import yaml
import traceback

# Config
POSSIBLE_MODEL_PATHS = [
    'checkpoints/best_model.pth',
    '../checkpoints/best_model.pth'
]
CONFIG_PATH = 'configs/config.yaml'
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = '../configs/config.yaml'

# SILENCE STDOUT permanently during initialization to avoid polluting communication channel
# We will use stderr for all logging
original_stdout_fd = os.dup(sys.stdout.fileno())
devnull = os.open(os.devnull, os.O_WRONLY)

def log(msg):
    print(f"DEBUG: {msg}", file=sys.stderr)
    sys.stderr.flush()

def main():
    log("Python bridge script starting...")
    
    # Silence stdout at FD level
    os.dup2(devnull, sys.stdout.fileno())
    
    # Add paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    src_dir = os.path.join(parent_dir, 'src')
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    sys.path.append(src_dir)

    try:
        from inference.inference_utils import InferenceBase
        
        class NodeInference(InferenceBase):
            def predict_from_base64(self, base64_str):
                try:
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
                
                # Process
                frame_data = self.process_single_frame(img)
                if frame_data is None:
                    frame_data = self.process_single_frame(img, skip_detection=True)
                
                emotion, confidence, probs = self.predict_cnn_only(frame_data)
                prob_dict = {self.emotions[i]: float(probs[i]) for i in range(len(self.emotions))}
                
                return {
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "probabilities": prob_dict
                }, None

        # Load model
        model_path = None
        for path in POSSIBLE_MODEL_PATHS:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            raise Exception("Model file not found")

        log(f"Loading model from {model_path}...")
        inference = NodeInference(model_path, CONFIG_PATH)
        log("Model loaded successfully")
        
        # Restore stdout for JSON
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        log("Inference bridge ready for requests")
        
        while True:
            line = sys.stdin.readline()
            if not line:
                log("EOF reached on stdin")
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if 'image' in data:
                    result, error = inference.predict_from_base64(data['image'])
                    if error:
                        print(json.dumps({"error": error}))
                    else:
                        print(json.dumps(result))
                else:
                    print(json.dumps({"error": "No image field"}))
                sys.stdout.flush()
            except Exception as e:
                print(json.dumps({"error": str(e)}))
                sys.stdout.flush()

    except Exception as e:
        os.dup2(original_stdout_fd, sys.stdout.fileno())
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
