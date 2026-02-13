"""
Visualization Utilities

Tools for visualizing:
- Training curves (loss, accuracy)
- Confusion matrices
- Facial landmarks and zones
- Preprocessing stages
- Attention maps
- Model predictions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple


def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         train_accs: List[float],
                         val_accs: List[float],
                         save_path: Optional[str] = None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accs: Training accuracies per epoch
        val_accs: Validation accuracies per epoch
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          normalize: bool = True,
                          save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Normalize confusion matrix
        save_path: Optional path to save figure
    """
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_preprocessing_stages(image: np.ndarray,
                                   preprocessor,
                                   save_path: Optional[str] = None):
    """
    Visualize preprocessing stages side by side.
    
    Args:
        image: Input image
        preprocessor: NoiseRobustPreprocessor instance
        save_path: Optional path to save figure
    """
    stages = preprocessor.visualize_preprocessing_stages(image)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    titles = ['Original', 'After Median Filter', 'After Histogram EQ',
              'After Gaussian Blur', 'Final', '']
    
    stage_keys = ['original', 'after_median', 'after_hist_eq',
                  'after_gaussian', 'final']
    
    for i, (key, title) in enumerate(zip(stage_keys, titles)):
        if key in stages:
            axes[i].imshow(stages[key], cmap='gray')
            axes[i].set_title(title, fontsize=12, fontweight='bold')
            axes[i].axis('off')
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.suptitle('Preprocessing Pipeline Stages', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved preprocessing visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_facial_zones(image: np.ndarray,
                           landmarks: np.ndarray,
                           zones: Dict,
                           save_path: Optional[str] = None):
    """
    Visualize facial zones with bounding boxes.
    
    Args:
        image: Original image
        landmarks: Facial landmarks
        zones: Dictionary of extracted zones
        save_path: Optional path to save figure
    """
    colors = {
        'forehead': (255, 0, 0),
        'left_eye': (0, 255, 0),
        'right_eye': (0, 255, 255),
        'nose': (255, 0, 255),
        'mouth': (0, 0, 255)
    }
    
    # Convert grayscale to BGR if needed
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw landmarks
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(vis_image, (x, y), 1, (0, 255, 0), -1)
    
    # Draw zone bboxes
    for zone_name, zone in zones.items():
        x, y, w, h = zone.original_bbox
        color = colors.get(zone_name, (255, 255, 255))
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis_image, zone_name, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Create figure with original and zones
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show full face with annotations
    axes[0, 0].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Face with Zones', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Show individual zones
    zone_names = ['forehead', 'left_eye', 'right_eye', 'nose', 'mouth']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for zone_name, pos in zip(zone_names, positions):
        if zone_name in zones:
            zone_img = (zones[zone_name].image * 255).astype(np.uint8)
            axes[pos].imshow(zone_img, cmap='gray')
            axes[pos].set_title(zone_name.replace('_', ' ').title(),
                              fontsize=10, fontweight='bold')
            axes[pos].axis('off')
    
    plt.suptitle('Facial Zone Extraction', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved zone visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_emotion_distribution(labels: np.ndarray,
                             class_names: List[str],
                             save_path: Optional[str] = None):
    """
    Plot distribution of emotions in dataset.
    
    Args:
        labels: Array of emotion labels
        class_names: List of emotion names
        save_path: Optional path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='navy')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Emotion Distribution in Dataset', fontsize=14, fontweight='bold')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved emotion distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(images: List[np.ndarray],
                         true_labels: List[int],
                         pred_labels: List[int],
                         class_names: List[str],
                         n_samples: int = 12,
                         save_path: Optional[str] = None):
    """
    Visualize model predictions on sample images.
    
    Args:
        images: List of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        n_samples: Number of samples to show
        save_path: Optional path to save figure
    """
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.ravel() if n_rows > 1 else [axes]
    
    for i in range(n_samples):
        img = images[i]
        true_label = true_labels[i]
        pred_label = pred_labels[i]
        
        # Display image
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Title with true and predicted labels
        color = 'green' if true_label == pred_label else 'red'
        title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved prediction visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization Utilities")
    print("=" * 50)
    
    # Example usage
    print("\nAvailable functions:")
    print("  - plot_training_curves()")
    print("  - plot_confusion_matrix()")
    print("  - visualize_preprocessing_stages()")
    print("  - visualize_facial_zones()")
    print("  - plot_emotion_distribution()")
    print("  - visualize_predictions()")
    
    print("\n✓ Visualization module loaded successfully")
