"""
Fast Training Script for Feature-Based Emotion Recognition
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.feature_dataset import create_feature_loaders


# -------------------- MODEL --------------------

class LightweightClassifier(nn.Module):
    def __init__(self, input_dim=1152, hidden_dim=512, num_classes=7, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# -------------------- TRAINER --------------------

class FeatureTrainer:
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.epochs = config['epochs']
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        self.best_val_loss = float('inf')

        log_dir = os.path.join(config['log_dir'], datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.writer = SummaryWriter(log_dir)

        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        self.ckpt_dir = config['checkpoint_dir']

    # ---------- TRAIN ----------
    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]"):
            x, y = x.to(self.device), y.to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        return total_loss / len(loader), 100 * correct / total

    # ---------- VALIDATE / TEST ----------
    def validate(self, loader, epoch, name="Val"):
        if loader is None or len(loader) == 0:
            print(f"⚠ {name} loader empty. Skipping.")
            return 0.0, 0.0

        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} [{name}]"):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                total_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        return total_loss / len(loader), 100 * correct / total

    # ---------- TRAIN LOOP ----------
    def train(self, train_loader, val_loader):
        print("\n" + "="*60)
        print("TRAINING CLASSIFIER ON FEATURES")
        print("="*60)

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader, epoch)

            self.scheduler.step(val_loss)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar("Acc/val", val_acc, epoch)

            print(f"\nEpoch {epoch+1}:")
            print(f" Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f" Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, best=True)

        self.writer.close()

    def save_checkpoint(self, epoch, best=False):
        path = os.path.join(self.ckpt_dir, "best_classifier.pth" if best else f"epoch_{epoch}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch
        }, path)
        if best:
            print("✓ Saved best model")


# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    train_loader, val_loader, test_loader = create_feature_loaders(
        args.features,
        batch_size=args.batch_size,
        num_workers=0
    )

    feature_dim = train_loader.dataset.feature_dim

    model = LightweightClassifier(input_dim=feature_dim)

    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "log_dir": "logs/",
        "checkpoint_dir": "checkpoints/"
    }

    trainer = FeatureTrainer(model, config, device)
    trainer.train(train_loader, val_loader)

    # -------- TEST (SAFE) --------
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate(test_loader, epoch=0, name="Test")
    if test_loader is not None and len(test_loader) > 0:
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    else:
        print("Test skipped (no samples).")

    print("\n✓ Training completed successfully")


if __name__ == "__main__":
    main()
