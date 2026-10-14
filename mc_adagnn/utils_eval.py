import torch
import numpy as np
import pandas as pd
import os

def calculate_metrics(pred, target):
    """
    Calculates forecasting metrics: MAE, MSE, RMSE, MAPE.
    
    Args:
        pred (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        
    Returns:
        dict: Dictionary containing the metrics.
    """
    # Convert to numpy arrays if they are tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    mae = np.mean(np.abs(pred - target))
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    
    # Avoid division by zero for MAPE
    epsilon = 1e-8
    mape = np.mean(np.abs((target - pred) / (target + epsilon))) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }

def save_metrics_to_csv(results, filename='metrics.csv'):
    """
    Saves a list of result dictionaries to a CSV file.
    
    Args:
        results (list or dict): Single dictionary or list of dictionaries containing metrics.
        filename (str): Output CSV file path.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    if isinstance(results, dict):
        results = [results]
        
    df = pd.DataFrame(results)
    
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)
