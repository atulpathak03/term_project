import torch
import torch.optim as optim
import argparse
import os
import json
import numpy as np

from model import MC_ADAGNN, mc_adagnn_loss
from dataset import generate_synthetic_data, get_dataloaders
from fdia import apply_fdia_to_batch
from utils_eval import calculate_metrics, save_metrics_to_csv
from utils_vis import plot_actual_vs_predicted, plot_degradation_vs_attack

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_model(model, test_loader, device, args, attack_type='none', attack_strength=0.0):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # Apply FDIA to input measurements
            if attack_type != 'none':
                # Convert specific nodes argument if provided
                attacked_nodes = [int(n) for n in args.attacked_nodes.split(',')] if args.attacked_nodes else None
                x = apply_fdia_to_batch(x, 
                                        attack_type=attack_type, 
                                        attack_strength=attack_strength,
                                        attack_ratio=args.attack_ratio,
                                        attacked_nodes=attacked_nodes)
                
            pred, _ = model(x)
            all_preds.append(pred)
            all_targets.append(y)
            
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(all_preds, all_targets)
    return metrics, all_preds, all_targets

def train_and_evaluate(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. Prepare Dataset
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_nodes=14, num_days=100) 
    train_loader, val_loader, test_loader, mean, std = get_dataloaders(
        df, 
        in_seq_len=args.in_seq_len, 
        out_seq_len=args.out_seq_len, 
        batch_size=args.batch_size
    )
    
    # 2. Initialize Model
    model = MC_ADAGNN(
        num_nodes=14, 
        in_seq_len=args.in_seq_len, 
        out_seq_len=args.out_seq_len, 
        embed_dim=args.embed_dim, 
        in_channels=1, 
        hidden_channels=args.hidden_channels, 
        num_layers=args.num_layers
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Training Loop (Clean Data)
    print("Starting training on clean data...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred, A_CSC = model(x)
            loss = mc_adagnn_loss(pred, y, A_CSC, gamma=args.gamma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {train_loss/len(train_loader):.4f}")
        
    print("Training Complete!")
    
    # Save experiment config
    config_dict = vars(args)
    with open('results/experiment_config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    # 4. Evaluation Matrix
    results_list = []
    
    # Clean Evaluation
    print("\nEvaluating on CLEAN test data...")
    clean_metrics, clean_preds, clean_targets = evaluate_model(model, test_loader, device, args, attack_type='none')
    
    clean_result = {'Experiment': 'Clean', 'Attack Type': 'none', 'Attack Strength': 0.0}
    clean_result.update(clean_metrics)
    results_list.append(clean_result)
    
    # Save clean plot
    node_idx = 0 # Plot first node
    plot_actual_vs_predicted(
        clean_targets[:, 0, node_idx].cpu().numpy(), 
        clean_preds[:, 0, node_idx].cpu().numpy(),
        title='Actual vs Predicted Load (Clean)',
        filename='results/plots/centralized_clean.png'
    )
    
    # Define attack strengths to test if we want an automated matrix
    strengths_to_test = [args.attack_strength] if not args.run_matrix else [0.05, 0.10, 0.20]
    
    if args.attack != 'none' or args.run_matrix:
        attack_type = 'scaling' if args.attack == 'none' and args.run_matrix else args.attack
        
        for strength in strengths_to_test:
            print(f"\nEvaluating on ATTACKED test data (Strength: {strength})...")
            attack_metrics, attack_preds, _ = evaluate_model(
                model, test_loader, device, args, 
                attack_type=attack_type, 
                attack_strength=strength
            )
            
            # Calculate degradation
            rmse_deg = attack_metrics['RMSE'] - clean_metrics['RMSE']
            rmse_deg_pct = (rmse_deg / clean_metrics['RMSE']) * 100
            
            res = {
                'Experiment': f'Attacked ({strength})', 
                'Attack Type': attack_type, 
                'Attack Strength': strength,
                'RMSE Degradation Absolute': rmse_deg,
                'RMSE Degradation %': rmse_deg_pct
            }
            res.update(attack_metrics)
            results_list.append(res)
            
            plot_actual_vs_predicted(
                clean_targets[:, 0, node_idx].cpu().numpy(), 
                attack_preds[:, 0, node_idx].cpu().numpy(),
                title=f'Actual vs Predicted Load (Attacked, Alpha={strength})',
                filename=f'results/plots/centralized_attacked_{strength}.png'
            )
            
    # Save all results to CSV
    output_csv = 'results/centralized_fdia.csv'
    save_metrics_to_csv(results_list, output_csv)
    print(f"\nResults saved to {output_csv}")
    
    # Generate degradation plot if running matrix
    if args.run_matrix:
        strengths = [r['Attack Strength'] for r in results_list if r['Experiment'] != 'Clean']
        degradations = [r['RMSE Degradation %'] for r in results_list if r['Experiment'] != 'Clean']
        plot_degradation_vs_attack(
            strengths, degradations, 
            metric_name='RMSE', 
            title='RMSE Degradation vs Attack Strength',
            filename='results/plots/rmse_vs_attack_strength.png'
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized MC-ADAGNN with FDIA Simulation")
    parser.add_argument('--in_seq_len', type=int, default=24*7)
    parser.add_argument('--out_seq_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    
    # FDIA specific arguments
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'scaling', 'additive'])
    parser.add_argument('--attack_strength', type=float, default=0.10)
    parser.add_argument('--attack_ratio', type=float, default=1.0, help="Percentage of measurements to attack")
    parser.add_argument('--attacked_nodes', type=str, default='', help="Comma separated node indices, e.g., '1,4,7'")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--run_matrix', action='store_true', help="Run automated experiment matrix (5%, 10%, 20%)")
    
    args = parser.parse_args()
    train_and_evaluate(args)
