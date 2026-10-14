import torch
import torch.optim as optim
import argparse
import copy
import os
import json
import numpy as np

from model import MC_ADAGNN, mc_adagnn_loss
from fed_dataset import generate_federated_synthetic_data, get_federated_dataloaders
from fdia import apply_fdia_to_batch
from utils_eval import calculate_metrics, save_metrics_to_csv
from utils_vis import plot_federated_round_metrics

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def client_update_fdia(client_model, optimizer, train_loader, epochs, device, args, is_attacked=False):
    """
    Trains the client model locally on their private data. Applies FDIA if client is compromised.
    """
    client_model.train()
    
    attacked_nodes = [int(n) for n in args.attacked_nodes.split(',')] if args.attacked_nodes else None

    for e in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if is_attacked and args.attack != 'none':
                x = apply_fdia_to_batch(x, 
                                        attack_type=args.attack, 
                                        attack_strength=args.attack_strength,
                                        attack_ratio=args.attack_ratio,
                                        attacked_nodes=attacked_nodes)
            
            pred, A_CSC = client_model(x)
            loss = mc_adagnn_loss(pred, y, A_CSC, gamma=args.gamma)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    return client_model.state_dict(), epoch_loss / len(train_loader)

def fed_avg(global_model, client_weights):
    """
    Aggregates client weights using Federated Averaging (FedAvg).
    """
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_weights[i][k].float() for i in range(len(client_weights))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def evaluate_global(model, test_loader, device):
    """
    Evaluates the model on a test set, returning multiple metrics.
    We test on CLEAN data to measure true model capability.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred, _ = model(x)
            all_preds.append(pred)
            all_targets.append(y)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return calculate_metrics(all_preds, all_targets)

def train_federated_fdia(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('results/plots', exist_ok=True)
    
    # Identify attacked clients
    attacked_clients_list = [int(c) for c in args.attacked_clients.split(',')] if args.attacked_clients else []
    
    # 1. Prepare Federated Dataset
    print(f"Generating synthetic data for {args.num_clients} clients...")
    generate_federated_synthetic_data(num_clients=args.num_clients, num_days=100)
    client_loaders = get_federated_dataloaders(
        num_clients=args.num_clients, 
        in_seq_len=args.in_seq_len, 
        out_seq_len=args.out_seq_len, 
        batch_size=args.batch_size
    )
    
    # 2. Initialize Global Model
    global_model = MC_ADAGNN(
        num_nodes=14, 
        in_seq_len=args.in_seq_len, 
        out_seq_len=args.out_seq_len, 
        embed_dim=args.embed_dim, 
        in_channels=1, 
        hidden_channels=args.hidden_channels, 
        num_layers=args.num_layers
    ).to(device)
    
    # Save experiment config
    with open('results/experiment_config_fed.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    round_metrics_list = []
    
    # 3. Federated Training Loop
    print("Starting Federated Training (Fed-ADAGNN) with FDIA...")
    for round_idx in range(args.rounds):
        print(f"\n--- Communication Round {round_idx+1}/{args.rounds} ---")
        client_weights = []
        
        for client_id in range(args.num_clients):
            is_attacked = (client_id in attacked_clients_list)
            
            client_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(client_model.parameters(), lr=args.lr)
            
            # Client trains locally
            weights, loss = client_update_fdia(
                client_model, 
                optimizer, 
                client_loaders[client_id]['train'], 
                epochs=args.local_epochs, 
                device=device,
                args=args,
                is_attacked=is_attacked
            )
            client_weights.append(weights)
            status = "ATTACKED" if is_attacked and args.attack != 'none' else "CLEAN"
            print(f"  Client {client_id} ({status}) Local Training Loss: {loss:.4f}")
            
        # Server aggregates weights
        global_model = fed_avg(global_model, client_weights)
        
        # Evaluate Global Model on all clients' CLEAN test sets (to see true performance degradation)
        global_mae, global_mse, global_rmse, global_mape = 0, 0, 0, 0
        for client_id in range(args.num_clients):
            metrics = evaluate_global(global_model, client_loaders[client_id]['test'], device)
            global_mae += metrics['MAE']
            global_mse += metrics['MSE']
            global_rmse += metrics['RMSE']
            global_mape += metrics['MAPE']
            
        global_mae /= args.num_clients
        global_mse /= args.num_clients
        global_rmse /= args.num_clients
        global_mape /= args.num_clients
        
        print(f"=> Global Model Test RMSE: {global_rmse:.4f}, MAE: {global_mae:.4f}")
        
        round_metrics_list.append({
            'Round': round_idx + 1,
            'Global RMSE': global_rmse,
            'Global MAE': global_mae,
            'Global MSE': global_mse,
            'Global MAPE': global_mape,
            'Attack Strength': args.attack_strength if len(attacked_clients_list) > 0 else 0.0,
            'Number of Compromised Clients': len(attacked_clients_list) if args.attack != 'none' else 0
        })
        
    print("\nFederated Training Complete!")
    
    # Save round metrics to CSV
    csv_filename = 'results/federated_fdia.csv' if len(attacked_clients_list) > 0 and args.attack != 'none' else 'results/federated_clean.csv'
    save_metrics_to_csv(round_metrics_list, csv_filename)
    
    # Also append to a master round_metrics.csv
    save_metrics_to_csv(round_metrics_list, 'results/round_metrics.csv')
    
    # Plot global RMSE over rounds
    plot_federated_round_metrics(
        'results/round_metrics.csv',
        title='Federated Global RMSE over Rounds',
        filename='results/plots/federated_global_rmse.png'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Fed-ADAGNN with FDIA Simulation")
    parser.add_argument('--num_clients', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--in_seq_len', type=int, default=24*7)
    parser.add_argument('--out_seq_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # FDIA specific arguments
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'scaling', 'additive'])
    parser.add_argument('--attack_strength', type=float, default=0.10)
    parser.add_argument('--attack_ratio', type=float, default=1.0)
    parser.add_argument('--attacked_nodes', type=str, default='')
    parser.add_argument('--attacked_clients', type=str, default='', help="Comma separated client IDs, e.g., '2'")
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    train_federated_fdia(args)
