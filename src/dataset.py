import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import glob

"""
dataset.py

This module defines the PyTorch Dataset and DataLoader logic for the AFL Fantasy breakout prediction project.
It provides the AFLFantasyDataset class, which wraps a pre-processed pandas DataFrame containing player-season 
features and breakout labels, and returns them as tensors for training or evaluation.

The Dataset expects the feature columns (numerical inputs to the MLP) and the label column (binary breakout label).
It handles conversion to float32 tensors for use in the neural network. The create_dataloaders() helper function 
constructs both training and validation DataLoaders from provided DataFrames.

This structure ensures consistent data feeding into the model and supports batching, shuffling, and iteration 
over the training and validation splits.

Usage:
    train_loader, val_loader = create_dataloaders(train_df, val_df, feature_cols)
"""

class AFLFantasyDataset(Dataset):
    def __init__(self, dataframe, feature_cols, label_col='breakout'):
        self.features = dataframe[feature_cols].values.astype(np.float32)
        self.labels = dataframe[label_col].values.astype(np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

def create_dataloaders(train_df, val_df, feature_cols, batch_size=32):
    train_dataset = AFLFantasyDataset(train_df, feature_cols)
    val_dataset = AFLFantasyDataset(val_df, feature_cols)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

