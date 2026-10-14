import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_actual_vs_predicted(actual, predicted, title, filename, num_samples=100):
    """
    Plots actual vs predicted load for a single node.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(actual[:num_samples], label='Actual', alpha=0.8)
    plt.plot(predicted[:num_samples], label='Predicted', alpha=0.8)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Load')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_degradation_vs_attack(strengths, degradations, metric_name, title, filename):
    """
    Plots performance degradation vs attack strength.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(strengths, degradations, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Attack Strength')
    plt.ylabel(f'{metric_name} Degradation (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_federated_round_metrics(csv_path, title, filename):
    """
    Plots federated round metrics from CSV.
    """
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    plt.figure(figsize=(8, 5))
    
    # Check if we have multiple configurations (e.g. clean vs attacked)
    if 'Attack Strength' in df.columns:
        for strength, group in df.groupby('Attack Strength'):
            plt.plot(group['Round'], group['Global RMSE'], marker='o', label=f'Alpha={strength}')
    else:
        plt.plot(df['Round'], df['Global RMSE'], marker='o')
        
    plt.title(title)
    plt.xlabel('Communication Round')
    plt.ylabel('Global RMSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
