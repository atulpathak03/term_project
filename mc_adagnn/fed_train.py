import torch
import torch.optim as optim
import argparse
import copy
from model import MC_ADAGNN, mc_adagnn_loss
from fed_dataset import generate_federated_synthetic_data, get_federated_dataloaders
import torch.nn.functional as F

def client_update(client_model, optimizer, train_loader, epochs, device, gamma=0.1):
    """
    Trains the client model locally on their private data.
    """
    client_model.train()
    for e in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred, A_CSC = client_model(x)
            loss = mc_adagnn_loss(pred, y, A_CSC, gamma=gamma)
            
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

def evaluate(model, test_loader, device):
    """
    Evaluates the model on a test set.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred, A_CSC = model(x)
            loss = F.l1_loss(pred, y) # Standard MAE for evaluation
            test_loss += loss.item()
    return test_loss / len(test_loader)

def train_federated(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # 3. Federated Training Loop
    print("Starting Federated Training (Fed-ADAGNN)...")
    for round_idx in range(args.rounds):
        print(f"\n--- Communication Round {round_idx+1}/{args.rounds} ---")
        client_weights = []
        
        for client_id in range(args.num_clients):
            # Send global model to client
            client_model = copy.deepcopy(global_model)
            optimizer = optim.Adam(client_model.parameters(), lr=args.lr)
            
            # Client trains locally
            weights, loss = client_update(
                client_model, 
                optimizer, 
                client_loaders[client_id]['train'], 
                epochs=args.local_epochs, 
                device=device,
                gamma=args.gamma
            )
            client_weights.append(weights)
            print(f"  Client {client_id} Local Training MAE Loss: {loss:.4f}")
            
        # Server aggregates weights
        global_model = fed_avg(global_model, client_weights)
        
        # Evaluate Global Model on all clients' test sets
        global_test_loss = 0.0
        for client_id in range(args.num_clients):
            test_loss = evaluate(global_model, client_loaders[client_id]['test'], device)
            global_test_loss += test_loss
            
        print(f"=> Global Model Test MAE (Averaged across clients): {global_test_loss/args.num_clients:.4f}")
        
    print("\nFederated Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fed-ADAGNN")
    parser.add_argument('--num_clients', type=int, default=3, help="Number of federated clients")
    parser.add_argument('--rounds', type=int, default=5, help="Number of federated communication rounds")
    parser.add_argument('--local_epochs', type=int, default=2, help="Number of local epochs per client per round")
    parser.add_argument('--in_seq_len', type=int, default=24*7, help="Input sequence length (e.g. 7 days)")
    parser.add_argument('--out_seq_len', type=int, default=24, help="Output sequence length (e.g. 1 day)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--embed_dim', type=int, default=32, help="Embedding dimension for Graph Construction")
    parser.add_argument('--hidden_channels', type=int, default=32, help="Hidden channels in Conv layers")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of SpatioTemporal layers")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.1, help="Stability loss weight")
    
    args = parser.parse_args()
    train_federated(args)
