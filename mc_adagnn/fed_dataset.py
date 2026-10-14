import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from dataset import LoadDataset # reuse the dataset class from phase 1

def generate_federated_synthetic_data(num_clients=3, num_nodes=14, num_days=100, freq='h'):
    from generate_ieee14 import generate_ieee14_data
    os.makedirs('data/federated', exist_ok=True)
    for client_id in range(num_clients):
        # We can add some client scale noise by multiplying the dataframe
        df = generate_ieee14_data(num_days=num_days)
        client_scale = np.random.uniform(0.5, 1.5)
        df = df * client_scale
        df.to_csv(f'data/federated/client_{client_id}_load_data.csv')
    print(f"Generated data for {num_clients} clients in 'data/federated/'")

def get_federated_dataloaders(num_clients=3, in_seq_len=24*7, out_seq_len=24, batch_size=32):
    """
    Loads client datasets and returns train/val/test dataloaders for each client.
    """
    client_loaders = []
    
    for client_id in range(num_clients):
        df = pd.read_csv(f'data/federated/client_{client_id}_load_data.csv', index_col=0)
        
        mean = df.values.mean(axis=0)
        std = df.values.std(axis=0)
        std[std == 0] = 1.0  # Prevent division by zero
        normalized_data = (df.values - mean) / std
        
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
        
        client_loaders.append({
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'mean': mean,
            'std': std
        })
        
    return client_loaders

if __name__ == "__main__":
    generate_federated_synthetic_data()
