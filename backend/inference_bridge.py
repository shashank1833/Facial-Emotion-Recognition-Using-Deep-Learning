import os
import sys
import cv2
import torch
import numpy as np
import base64
import json
import yaml
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from models.emotion_model import create_model
from training.data_loader import get_transforms

class NodeInference:
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.emotions = self.config['emotions']['classes']
        self.transform = get_transforms(self.config, False)
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint.get('config', self.config)
        if 'emotions' in model_config:
            self.emotions = model_config['emotions']['classes']
        self.model = create_model(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def predict_from_base64(self, base64_str):
        try:
            if ',' in base64_str:
                encoded_data = base64_str.split(',')[1]
            else:
                encoded_data = base64_str
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return None, f"Error decoding image: {str(e)}"
        
        if img_bgr is None:
            return None, "Invalid image data"
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            emotion_idx = int(np.argmax(probabilities))
            emotion = self.emotions[emotion_idx]
            confidence = float(probabilities[emotion_idx])
        
        prob_dict = {self.emotions[i]: float(probabilities[i]) for i in range(len(self.emotions))}
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": prob_dict,
            "detection_success": True
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
    model_path = None
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        error_msg = json.dumps({"error": "No model found"})
        print(error_msg)
        return

    try:
        inference = NodeInference(model_path, CONFIG_PATH)
    except Exception as e:
        error_msg = json.dumps({"error": f"Error loading model: {str(e)}"})
        print(error_msg)
        return

    for line in sys.stdin:
        try:
            if not line.strip():
                continue
            data = json.loads(line)
            if 'image' not in data:
                print(json.dumps({"error": "No image data provided"}))
                continue
            
            result, error = inference.predict_from_base64(data['image'])
            if error:
                print(json.dumps({"error": error}))
            else:
                print(json.dumps(result))
            
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
