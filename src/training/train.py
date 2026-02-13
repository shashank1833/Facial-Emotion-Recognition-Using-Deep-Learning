"""
Training Script for Extended Emotion Recognition System

This script trains the full hybrid CNN + LSTM model on FER-2013 dataset.

Training Pipeline:
1. Load and preprocess FER-2013 data
2. Apply data augmentation
3. Train hybrid CNN + LSTM model
4. Validate on validation set
5. Save best model checkpoints
6. Log metrics to TensorBoard

Usage:
    python training/train.py --config configs/config.yaml --data data/fer2013/fer2013.csv
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_full_model
from training.data_loader import FER2013Dataset
from training.augmentation import EmotionAugmenter


class EmotionRecognitionTrainer:
    """
    Trainer for hybrid emotion recognition model.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: dict,
                 device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: Hybrid emotion recognition model
            config: Configuration dictionary
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training config
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.learning_rate = config['training']['learning_rate']
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Loss function
        if 'class_weights' in config and config['class_weights'] is not None:
            weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
            print(f"Using class weights: {weights.cpu().numpy()}")
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Early stopping
        self.early_stopping_patience = config['training']['early_stopping']['patience']
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Logging
        self.log_dir = config['data']['logs_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(self.log_dir, f'run_{timestamp}'))
        
        # Checkpointing
        self.checkpoint_dir = config['data']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        optimizer_name = self.config['training']['optimizer']
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate,
                           momentum=0.9, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config['training']['lr_schedule']['enabled']:
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['training']['lr_schedule']['factor'],
                patience=self.config['training']['lr_schedule']['patience'],
                min_lr=self.config['training']['lr_schedule']['min_lr']
            )
        return None
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')
        
        for batch_idx, (full_faces, zones, labels) in enumerate(pbar):
            # Move to device
            full_faces = full_faces.to(self.device)
            labels = labels.to(self.device)
            
            zones_device = {
                zone_name: zone_tensor.to(self.device)
                for zone_name, zone_tensor in zones.items()
            }
            
            
            # Extract features
            features = self.model.hybrid_cnn(full_faces, zones_device)
            
            # Pass through LSTM
            # Add sequence dimension (batch, 1, feature_dim)
            features = features.unsqueeze(1)
            outputs = self.model.temporal_lstm(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, epoch: int) -> tuple:
        """
        Validate model.
        """
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.epochs} [Val]')

            for full_faces, zones, labels in pbar:
                # Move to device
                full_faces = full_faces.to(self.device)
                labels = labels.to(self.device)

                zones_device = {
                    zone_name: zone_tensor.to(self.device)
                    for zone_name, zone_tensor in zones.items()
                }

                # SAME forward path as training
                features = self.model.hybrid_cnn(full_faces, zones_device)

                # Add sequence dimension (batch, 1, feature_dim)
                features = features.unsqueeze(1)

                # Pass through LSTM (includes classifier)
                outputs = self.model.temporal_lstm(features)

                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ Saved best model (val_loss: {val_loss:.4f})')
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print("\n" + "="*60)
        print("  TRAINING HYBRID EMOTION RECOGNITION MODEL")
        print("="*60)
        
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        print("\n" + "-"*60)
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', 
                                  self.optimizer.param_groups[0]['lr'], epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n✗ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation loss: {self.best_val_loss:.4f}")
                break
            
            print("-"*60)
        
        # Save final model
        self.save_checkpoint(epoch, val_loss, False)
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETED")
        print("="*60)
        print(f"\nBest validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {max(self.val_accuracies):.2f}%")
        print(f"Final training accuracy: {self.train_accuracies[-1]:.2f}%")
        
        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/fer2013/fer2013.csv',
                       help='Path to FER-2013 CSV file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save checkpoints (overrides config)')
    parser.add_argument('--emotions', type=str, default=None,
                       help='Comma-separated list of emotions to train on (e.g., "Angry,Disgust")')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.output_dir is not None:
        config['data']['checkpoint_dir'] = args.output_dir
    
    # Filter emotions if specified
    emotion_subset = None
    if args.emotions:
        emotion_subset = [e.strip() for e in args.emotions.split(',')]
        print(f"Training on emotion subset: {emotion_subset}")
        config['emotions']['classes'] = emotion_subset
        config['emotions']['num_classes'] = len(emotion_subset)
        # Update model config too
        config['model']['lstm']['num_classes'] = len(emotion_subset)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("⚠ CUDA not available, using CPU (training will be slow)")
    
    # Create data loaders
    print("Loading datasets...")
    train_dataset = FER2013Dataset(args.data, usage='Training', emotion_subset=emotion_subset)
    val_dataset = FER2013Dataset(args.data, usage='PublicTest', emotion_subset=emotion_subset)

    # Calculate class weights for imbalanced datasets
    # Always calculate if not explicitly provided in config
    if config.get('class_weights') is None:
        class_counts = train_dataset.df['emotion'].value_counts().sort_index().values
        total_samples = len(train_dataset)
        num_classes = len(class_counts)
        
        # Avoid division by zero if some classes are missing
        class_weights = np.zeros(num_classes)
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (num_classes * count)
            else:
                class_weights[i] = 1.0
        
        config['class_weights'] = class_weights.tolist()
        print(f"Calculated class weights: {config['class_weights']}")
    else:
        print(f"Using provided class weights from config")

    # Adjust for CPU training (no multiprocessing, no pinned memory)
    num_workers = 0 if device == 'cpu' else config['hardware']['num_workers']
    pin_memory = False if device == 'cpu' else config['hardware']['pin_memory']

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Create model
    print("Creating model...")
    model = create_full_model(
        hybrid_cnn_config=config['model'],
        lstm_config=config['model']['lstm']
    )
    
    # Create trainer
    trainer = EmotionRecognitionTrainer(model, config, device)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n✓ Training script completed successfully")


if __name__ == "__main__":
    main()
