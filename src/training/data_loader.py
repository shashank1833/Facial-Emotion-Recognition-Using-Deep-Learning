import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List

class RAFDBDataset(Dataset):
    """
    RAF-DB Dataset for Image-based FER.
    Expected structure:
    data/raf-db/
        DATASET/
            Image/aligned/
                train_00001_aligned.jpg
                ...
        list_pat_label_train.txt
        list_pat_label_test.txt
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # RAF-DB labels: 1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral
        # Convert to 0-indexed: 0: Surprise, 1: Fear, 2: Disgust, 3: Happiness, 4: Sadness, 5: Anger, 6: Neutral
        self.image_dir = os.path.join(root_dir, 'DATASET', split)
        
        self.data = []
        if os.path.exists(self.image_dir):
            # Iterate through folders 1-7
            for label_folder in os.listdir(self.image_dir):
                if not label_folder.isdigit(): continue
                
                label_path = os.path.join(self.image_dir, label_folder)
                label_idx = int(label_folder) - 1
                
                for img_name in os.listdir(label_path):
                    if img_name.endswith('.jpg'):
                        # Store relative path from image_dir
                        self.data.append((os.path.join(label_folder, img_name), label_idx))
            
            if len(self.data) == 0:
                print(f"Warning: No images found in {self.image_dir}")
        else:
            print(f"Warning: Image directory {self.image_dir} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class FER2013Dataset(Dataset):
    """
    Simplified FER-2013 dataset for cross-dataset testing.
    """
    def __init__(self, csv_path: str, usage: str = 'PrivateTest', transform=None):
        self.df = pd.read_csv(csv_path)
        # Normalize Usage values and filter robustly; fallback to PublicTest if empty
        self.df['Usage'] = self.df['Usage'].astype(str).str.strip()
        usage_norm = usage.strip().lower()
        df_subset = self.df[self.df['Usage'].str.lower() == usage_norm]
        if df_subset.empty:
            fallback = 'publictest'
            df_subset = self.df[self.df['Usage'].str.lower() == fallback]
        self.df = df_subset.reset_index(drop=True)
        self.transform = transform
        
        # FER2013 labels: 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
        # RAF-DB labels:  0: Surprise, 1: Fear, 2: Disgust, 3: Happiness, 4: Sadness, 5: Anger, 6: Neutral
        self.fer_to_raf = {
            0: 5, # Angry -> Anger
            1: 2, # Disgust -> Disgust
            2: 1, # Fear -> Fear
            3: 3, # Happy -> Happiness
            4: 4, # Sad -> Sadness
            5: 0, # Surprise -> Surprise
            6: 6  # Neutral -> Neutral
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixels).convert('RGB')
        
        label = int(row['emotion'])
        raf_label = self.fer_to_raf[label]
        
        if self.transform:
            image = self.transform(image)
            
        return image, raf_label

def get_transforms(config: dict, is_training: bool = True):
    """Get torchvision transforms."""
    input_size = config['model']['input_size']
    
    if is_training and config['training']['augmentation']['enabled']:
        aug = config['training']['augmentation']
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5 if aug['horizontal_flip'] else 0),
            transforms.RandomRotation(aug['rotation']),
            transforms.RandomAffine(degrees=0, translate=aug['translate'], scale=aug['scale']),
            transforms.ColorJitter(**aug['color_jitter']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(config: dict):
    """Create data loaders with class imbalance handling."""
    raf_path = config['data']['raf_db_path']
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    
    # Datasets
    train_dataset = RAFDBDataset(raf_path, split='train', transform=get_transforms(config, True))
    test_dataset = RAFDBDataset(raf_path, split='test', transform=get_transforms(config, False))
    
    # Handle class imbalance for training
    labels = [label for _, label in train_dataset.data]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
