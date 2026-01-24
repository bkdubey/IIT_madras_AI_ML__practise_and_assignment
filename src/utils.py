"""
Utility functions for AI/ML projects
"""

import numpy as np
import pandas as pd


def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def normalize_data(data):
    """Normalize data to 0-1 range."""
    return (data - data.min()) / (data.max() - data.min())


def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X.iloc[train_indices], X.iloc[test_indices], \
           y.iloc[train_indices], y.iloc[test_indices]
