import os
import sys
import argparse
import yaml
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.emotion_model import create_model
from training.data_loader import RAFDBDataset, FER2013Dataset, get_transforms
from utils.metrics import calculate_metrics, print_metrics

def save_artifacts(metrics, class_names, output_dir, dataset_name):
    """Save evaluation artifacts to the results directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON metrics
    metrics_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                      for k, v in metrics.items() if k not in ['probabilities', 'true_labels', 'pred_labels']}
    
    with open(os.path.join(output_dir, f'metrics_{dataset_name}.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
        
    # Save Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png'))
    plt.close()
    
    # Save Text Summary
    with open(os.path.join(output_dir, f'summary_{dataset_name}.txt'), 'w') as f:
        f.write(f"Evaluation Summary - {dataset_name.upper()}\n")
        f.write("="*40 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['f1_macro']:.4f}\n")
        f.write(f"Macro Precision: {metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Recall: {metrics['recall_macro']:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
        for cls in class_names:
            m = metrics['per_class'][cls]
            f.write(f"{cls:<15} {m['precision']:.4f}    {m['recall']:.4f}    {m['f1']:.4f}\n")

def evaluate(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return calculate_metrics(np.array(all_labels), np.array(all_preds), class_names)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Emotion Recognition Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--results_dir', type=str, default='results/')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = config['emotions']['classes']
    
    # Load model
    model = create_model(config)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    transform = get_transforms(config, is_training=False)
    
    # 1. Evaluate on RAF-DB (Test)
    print("\n[1/2] Evaluating on RAF-DB (Test Set)...")
    raf_dataset = RAFDBDataset(config['data']['raf_db_path'], split='test', transform=transform)
    raf_loader = DataLoader(raf_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    raf_metrics = evaluate(model, raf_loader, device, class_names)
    print_metrics(raf_metrics, class_names)
    save_artifacts(raf_metrics, class_names, args.results_dir, 'raf_db')
    
    # 2. Evaluate on FER2013 (Cross-Dataset)
    print("\n[2/2] Evaluating on FER2013 (Cross-Dataset Testing)...")
    fer_dataset = FER2013Dataset(config['data']['fer2013_path'], usage='PrivateTest', transform=transform)
    fer_loader = DataLoader(fer_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    fer_metrics = evaluate(model, fer_loader, device, class_names)
    print_metrics(fer_metrics, class_names)
    save_artifacts(fer_metrics, class_names, args.results_dir, 'fer2013')
    
    print(f"\n✓ Evaluation complete. Artifacts saved to: {args.results_dir}")

if __name__ == '__main__':
    main()
