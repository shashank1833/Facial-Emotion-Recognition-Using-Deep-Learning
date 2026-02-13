"""
Training module for emotion recognition system.
"""

from .data_loader import FER2013Dataset, create_data_loaders

__all__ = ['FER2013Dataset', 'create_data_loaders']
