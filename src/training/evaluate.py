"""
Evaluation script for the Emotion Recognition System.
Calculates accuracy and other metrics on the FER-2013 test set.
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_full_model
from training.data_loader import FER2013Dataset, CombinedCSVDataset
from utils.metrics import calculate_metrics, print_metrics

def evaluate(model, data_loader, device, class_names):
    """Evaluate the model on the data loader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Evaluating on {len(data_loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (full_faces, zones, labels) in enumerate(tqdm(data_loader)):
            full_faces = full_faces.to(device)
            # Labels in data_loader might be (batch, 1) or (batch,)
            labels = labels.squeeze()
            
            # Prepare zones
            zones_tensor = {
                name: tensor.to(device)
                for name, tensor in zones.items()
            }
            
            # Extract features from Hybrid CNN
            features = model.hybrid_cnn(full_faces, zones_tensor)
            
            # Add sequence dimension (batch, seq_len=1, feature_dim)
            features = features.unsqueeze(1)
            
            # Pass through LSTM
            logits = model.temporal_lstm(features)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Debug: print first few preds and labels in batch
            if batch_idx == 0:
                print(f"Sample Predictions: {preds[:10].cpu().numpy()}")
                print(f"Sample Labels:      {labels[:10].cpu().numpy()}")
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    # Calculate metrics
    y_true_arr = np.array(all_labels)
    y_pred_arr = np.array(all_preds)
    
    # Ensure they are 1D
    if y_true_arr.ndim > 1:
        y_true_arr = y_true_arr.flatten()
    if y_pred_arr.ndim > 1:
        y_pred_arr = y_pred_arr.flatten()
        
    metrics = calculate_metrics(y_true_arr, y_pred_arr, class_names)
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion Recognition Model')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/fer2013/fer2013.csv',
                       help='Path to FER-2013 CSV file')
    parser.add_argument('--usage', type=str, default='PrivateTest',
                       choices=['PublicTest', 'PrivateTest'],
                       help='Which test set to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples for quick testing')
    parser.add_argument('--emotions', type=str, default=None,
                       help='Comma-separated list of emotions to evaluate (e.g., "Angry,Disgust")')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test CSV (overrides --data)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Filter emotions if specified
    emotion_subset = None
    if args.emotions:
        emotion_subset = [e.strip() for e in args.emotions.split(',')]
        print(f"Evaluating on emotion subset: {emotion_subset}")
        config['emotions']['classes'] = emotion_subset
        config['emotions']['num_classes'] = len(emotion_subset)
        # Update model config too
        config['model']['lstm']['num_classes'] = len(emotion_subset)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset...")
    if args.test_csv:
        print(f"Using test CSV: {args.test_csv}")
        test_dataset = CombinedCSVDataset(args.test_csv, emotion_subset=emotion_subset)
    else:
        print(f"Using FER-2013 CSV: {args.data} (Usage: {args.usage})")
        test_dataset = FER2013Dataset(args.data, usage=args.usage, emotion_subset=emotion_subset)
    
    print(f"Loaded {len(test_dataset)} samples")
    
    # Apply limit if specified (shuffle first to get diverse samples)
    if args.limit and args.limit < len(test_dataset):
        print(f"Limiting dataset to {args.limit} samples (shuffled)...")
        import random
        indices = list(range(len(test_dataset)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:args.limit]
        test_dataset.df = test_dataset.df.iloc[indices].reset_index(drop=True)
        print(f"New dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,  # Shuffle to get a mix of emotions if limited
        num_workers=0 if device == 'cpu' else config['hardware']['num_workers'],
        pin_memory=False if device == 'cpu' else config['hardware']['pin_memory']
    )
    
    # Create model
    print("Creating model...")
    model = create_full_model(
        hybrid_cnn_config=config['model'],
        lstm_config=config['model']['lstm']
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Evaluate
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    metrics = evaluate(model, test_loader, device, class_names)
    
    # Print results
    print_metrics(metrics, class_names)

if __name__ == "__main__":
    main()
