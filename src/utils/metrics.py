"""
Metrics and Evaluation Utilities

Comprehensive evaluation metrics for emotion recognition:
- Accuracy (overall, per-class)
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC curves and AUC
- Classification report
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
import torch
import torch.nn as nn
from tqdm import tqdm


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     class_names: List[str]) -> Dict:
    """
    Calculate comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names)))
    
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': precision_per_class[i],
            'recall': recall_per_class[i],
            'f1': f1_per_class[i]
        }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics: Dict, class_names: List[str]):
    """
    Print metrics in formatted table.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("  EVALUATION METRICS")
    print("="*60)
    
    # Overall metrics
    print("\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):     {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro):   {metrics['f1_macro']:.4f}")
    
    print(f"\n  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score (weighted):  {metrics['f1_weighted']:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for class_name in class_names:
        per_class = metrics['per_class'][class_name]
        print(f"{class_name:<15} {per_class['precision']:<12.4f} "
              f"{per_class['recall']:<12.4f} {per_class['f1']:<12.4f}")
    
    print("="*60)


def evaluate_model(model: nn.Module,
                   test_loader,
                   device: str = 'cuda',
                   class_names: Optional[List[str]] = None) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for full_faces, zones, labels in tqdm(test_loader, desc='Evaluating'):
            # Move to device
            full_faces = full_faces.to(device)
            labels = labels.to(device)
            
            zones_device = {
                zone_name: zone_tensor.to(device)
                for zone_name, zone_tensor in zones.items()
            }
            
            # Forward pass
            features = model.hybrid_cnn(full_faces, zones_device)
            
            # Add sequence dimension (batch, 1, feature_dim)
            features = features.unsqueeze(1)
            
            # Pass through LSTM (includes classifier)
            logits = model.temporal_lstm(features)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate metrics
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(np.unique(y_true)))]
    
    metrics = calculate_metrics(y_true, y_pred, class_names)
    metrics['probabilities'] = y_probs
    metrics['true_labels'] = y_true
    metrics['pred_labels'] = y_pred
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    return metrics


def get_classification_report(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: List[str]) -> str:
    """
    Get scikit-learn classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Classification report string
    """
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )


def calculate_per_emotion_accuracy(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   class_names: List[str]) -> Dict[str, float]:
    """
    Calculate accuracy for each emotion.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        Dictionary mapping emotion names to accuracies
    """
    accuracies = {}
    
    for i, class_name in enumerate(class_names):
        # Find samples of this class
        mask = y_true == i
        
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean()
            accuracies[class_name] = class_acc
        else:
            accuracies[class_name] = 0.0
    
    return accuracies


def compare_models(results_dict: Dict[str, Dict]) -> None:
    """
    Compare multiple models side by side.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
    """
    print("\n" + "="*80)
    print("  MODEL COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for model_name, metrics in results_dict.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision_macro']:<12.4f} "
              f"{metrics['recall_macro']:<12.4f} "
              f"{metrics['f1_macro']:<12.4f}")
    
    print("="*80)


def calculate_top_k_accuracy(y_true: np.ndarray,
                             y_probs: np.ndarray,
                             k: int = 3) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities (N x num_classes)
        k: Top k predictions to consider
        
    Returns:
        Top-k accuracy
    """
    # Get top k predictions
    top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
    
    # Check if true label is in top k
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    
    return correct.mean()


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.val:.4f} (avg: {self.avg:.4f})'


if __name__ == "__main__":
    print("Metrics and Evaluation Utilities")
    print("=" * 50)
    
    # Example usage with dummy data
    n_samples = 100
    n_classes = 7
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_probs = np.random.rand(n_samples, n_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # Normalize
    
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, class_names)
    
    # Print metrics
    print_metrics(metrics, class_names)
    
    # Top-k accuracy
    top3_acc = calculate_top_k_accuracy(y_true, y_probs, k=3)
    print(f"\nTop-3 Accuracy: {top3_acc:.4f}")
    
    print("\n✓ Metrics module loaded successfully")
