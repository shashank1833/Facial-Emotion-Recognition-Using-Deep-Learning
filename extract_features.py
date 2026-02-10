"""
Feature Extraction Script for Emotion Recognition

This script precomputes all expensive operations (preprocessing, MediaPipe, CNN forward pass)
and saves the resulting feature tensors to disk for fast training.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.data_loader import FER2013Dataset
from models.hybrid_cnn import create_hybrid_cnn


class FeatureExtractor:
    def __init__(self, model_config=None, device='cpu', batch_size=32):
        self.device = device
        self.batch_size = batch_size

        print(f"Initializing HybridCNN on {device}...")
        self.model = create_hybrid_cnn(model_config)
        self.model = self.model.to(device)
        self.model.eval()

        self.feature_dim = self.model.total_feature_dim

        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def extract_dataset(self, csv_path: str, usage: str, output_dir: str) -> dict:
        print(f"\n{'='*60}")
        print(f"Extracting features for {usage} set")
        print(f"{'='*60}\n")

        dataset = FER2013Dataset(csv_path, usage=usage, target_size=224)
        num_samples = len(dataset)

        print(f"  Total samples: {num_samples}")
        print(f"  Batch size: {self.batch_size}")

        # 🚨 HARD STOP IF SPLIT IS EMPTY
        if num_samples == 0:
            print(f"WARNING: {usage} split is empty. Skipping extraction.\n")
            dataset.close()
            return {
                'total_samples': 0,
                'feature_dim': self.feature_dim,
                'failed_samples': 0,
                'no_landmarks_detected': 0,
                'features_path': None,
                'labels_path': None
            }

        all_features = torch.zeros(num_samples, self.feature_dim, dtype=torch.float32)
        all_labels = torch.zeros(num_samples, dtype=torch.long)

        failed_samples = []
        num_no_landmarks = 0

        print("\nExtracting features...")
        with torch.no_grad():
            for idx in tqdm(range(num_samples), desc=usage):
                try:
                    full_face, zones, label = dataset[idx]

                    if zones['forehead'].sum() == 0:
                        num_no_landmarks += 1

                    full_face = full_face.unsqueeze(0).to(self.device)
                    zones_device = {
                        k: v.unsqueeze(0).to(self.device)
                        for k, v in zones.items()
                    }

                    features = self.model(full_face, zones_device)

                    all_features[idx] = features.squeeze(0).cpu()
                    all_labels[idx] = label

                except Exception as e:
                    warnings.warn(f"Failed sample {idx}: {e}")
                    failed_samples.append(idx)

        dataset.close()

        os.makedirs(output_dir, exist_ok=True)

        features_path = os.path.join(output_dir, f'{usage.lower()}_features.pt')
        labels_path = os.path.join(output_dir, f'{usage.lower()}_labels.pt')

        torch.save(all_features, features_path)
        torch.save(all_labels, labels_path)

        stats = {
            'total_samples': num_samples,
            'feature_dim': self.feature_dim,
            'failed_samples': len(failed_samples),
            'no_landmarks_detected': num_no_landmarks,
            'features_path': features_path,
            'labels_path': labels_path
        }

        percentage = (
            100 * num_no_landmarks / num_samples
            if num_samples > 0 else 0.0
        )

        print(f"\n{usage} Statistics:")
        print(f"  Total samples: {num_samples}")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Failed extractions: {len(failed_samples)}")
        print(f"  No landmarks detected: {num_no_landmarks} ({percentage:.1f}%)")

        stats_path = os.path.join(output_dir, f'{usage.lower()}_stats.txt')
        with open(stats_path, 'w') as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

        return stats

    def extract_all_splits(self, csv_path: str, output_dir: str) -> dict:
        all_stats = {}

        all_stats['training'] = self.extract_dataset(csv_path, 'Training', output_dir)
        all_stats['validation'] = self.extract_dataset(csv_path, 'PublicTest', output_dir)
        all_stats['test'] = self.extract_dataset(csv_path, 'PrivateTest', output_dir)

        print(f"\n{'='*60}")
        print("FEATURE EXTRACTION COMPLETE")
        print(f"{'='*60}\n")

        total_samples = sum(s['total_samples'] for s in all_stats.values())
        total_failed = sum(s['failed_samples'] for s in all_stats.values())

        print(f"Total samples processed: {total_samples}")
        print(f"Total failed: {total_failed}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"All features saved to: {output_dir}")

        return all_stats


def main():
    parser = argparse.ArgumentParser(description='Extract CNN features from FER2013 dataset')
    parser.add_argument('--data', type=str, default='data/fer2013/fer2013.csv')
    parser.add_argument('--output', type=str, default='data/features/')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['all', 'training', 'publictest', 'privatetest']
    )

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        sys.exit(1)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    extractor = FeatureExtractor(
        model_config=None,
        device=device,
        batch_size=args.batch_size
    )

    if args.split == 'all':
        extractor.extract_all_splits(args.data, args.output)
    else:
        split_map = {
            'training': 'Training',
            'publictest': 'PublicTest',
            'privatetest': 'PrivateTest'
        }
        extractor.extract_dataset(args.data, split_map[args.split], args.output)

    print("\n✓ Feature extraction completed successfully")


if __name__ == "__main__":
    main()
