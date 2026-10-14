import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

from generate_ieee14 import generate_ieee14_data
def generate_synthetic_data(num_nodes=14, num_days=365, freq='h'):
    return generate_ieee14_data(num_days=num_days)


class LoadDataset(Dataset):
    def __init__(self, data, in_seq_len, out_seq_len):
        self.data = torch.FloatTensor(data)
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        
    def __len__(self):
        return len(self.data) - self.in_seq_len - self.out_seq_len + 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.in_seq_len]
        y = self.data[idx + self.in_seq_len : idx + self.in_seq_len + self.out_seq_len]
        # Input shape expected by model: (num_nodes, seq_len)
        return x.t(), y

def get_dataloaders(df, in_seq_len=24*7, out_seq_len=24, batch_size=32):
    # Normalize data
    mean = df.values.mean(axis=0)
    std = df.values.std(axis=0)
    std[std == 0] = 1.0  # Prevent division by zero
    normalized_data = (df.values - mean) / std
    
    # Train / Val / Test split (6:2:2 as per paper)
    n = len(normalized_data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    
    train_data = normalized_data[:train_end]
    val_data = normalized_data[train_end - in_seq_len:val_end]
    test_data = normalized_data[val_end - in_seq_len:]
    
    train_dataset = LoadDataset(train_data, in_seq_len, out_seq_len)
    val_dataset = LoadDataset(val_data, in_seq_len, out_seq_len)
    test_dataset = LoadDataset(test_data, in_seq_len, out_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, mean, std

if __name__ == "__main__":
    df = generate_synthetic_data()
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_load_data.csv')
    print("Synthetic dataset generated and saved to 'data/synthetic_load_data.csv'")
