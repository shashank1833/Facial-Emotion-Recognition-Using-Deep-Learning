import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.emotion_model import create_model
from training.data_loader import create_data_loaders
from training.losses import FocalLoss
from utils.metrics import calculate_metrics

def set_seed(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.class_names = config['emotions']['classes']
        
        # Hyperparameters
        train_cfg = config['training']
        self.epochs = train_cfg['epochs']
        self.early_stop_patience = train_cfg.get('early_stopping', {}).get('patience', 10)
        
        # Loss
        if train_cfg['loss'] == 'focal_loss':
            self.criterion = FocalLoss(
                alpha=train_cfg['focal_loss']['alpha'],
                gamma=train_cfg['focal_loss']['gamma']
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay']
        )
        
        # Scheduler
        if train_cfg['lr_schedule']['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=float(train_cfg['lr_schedule']['min_lr'])
            )
        else:
            self.scheduler = None
            
        # Logging & Checkpointing
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(config['data']['logs_dir'], f'run_{timestamp}'))
        self.checkpoint_dir = config['data']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_f1 = 0.0
        self.patience_counter = 0

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), self.class_names)
        return total_loss / len(loader), metrics

    def validate(self, loader, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{self.epochs} [Val]')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), self.class_names)
        return total_loss / len(loader), metrics

    def train(self, train_loader, val_loader):
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Checkpoint selection: Best Macro F1-Score")
        print(f"Early stopping patience: {self.early_stop_patience} epochs")
        
        for epoch in range(self.epochs):
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            val_loss, val_metrics = self.validate(val_loader, epoch)
            
            if self.scheduler:
                self.scheduler.step()
                
            # Log
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('F1_Macro/train', train_metrics['f1_macro'], epoch)
            self.writer.add_scalar('F1_Macro/val', val_metrics['f1_macro'], epoch)
            self.writer.add_scalar('Acc/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            val_f1 = val_metrics['f1_macro']
            print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%, Val F1: {val_f1:.4f}")
            
            # Save best based on Macro F1
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'f1_macro': val_f1,
                    'acc': val_metrics['accuracy'],
                    'config': self.config
                }, os.path.join(self.checkpoint_dir, 'best_model.pth'))
                print(f"--> [SAVE] New best model (Macro F1: {val_f1:.4f})")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.early_stop_patience:
                print(f"\n[STOP] Early stopping triggered after {epoch+1} epochs.")
                break
        
        print(f"\nTraining complete. Best Macro F1: {self.best_f1:.4f}")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['data']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader, test_loader = create_data_loaders(config)
    
    # Model
    model = create_model(config)
    
    # Train
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, test_loader)

if __name__ == '__main__':
    main()
