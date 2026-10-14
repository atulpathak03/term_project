import torch
import torch.optim as optim
import argparse
from model import MC_ADAGNN, mc_adagnn_loss
from dataset import generate_synthetic_data, get_dataloaders

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Custom Dataset
    print("Generating synthetic data...")
    df = generate_synthetic_data(num_nodes=14, num_days=100) 
    print("Creating DataLoaders...")
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
    
    # 3. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            pred, A_CSC = model(x)
            
            # Calculate loss (MAE + Stability)
            loss = mc_adagnn_loss(pred, y, A_CSC, gamma=args.gamma)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs}, Training Loss: {train_loss/len(train_loader):.4f}")
        
    print("Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MC-ADAGNN")
    parser.add_argument('--in_seq_len', type=int, default=24*7, help="Input sequence length (e.g. 7 days)")
    parser.add_argument('--out_seq_len', type=int, default=24, help="Output sequence length (e.g. 1 day)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--embed_dim', type=int, default=32, help="Embedding dimension for Graph Construction")
    parser.add_argument('--hidden_channels', type=int, default=32, help="Hidden channels in Conv layers")
    parser.add_argument('--num_layers', type=int, default=3, help="Number of SpatioTemporal layers")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--gamma', type=float, default=0.1, help="Stability loss weight")
    
    args = parser.parse_args()
    train(args)
