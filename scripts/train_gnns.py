"""
Train baseline GNN models on graph classification datasets.

Usage:
    python scripts/train_gnns.py --model gcn --dataset MUTAG --epochs 200
"""

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.models.gnn_models import create_model
from src.utils.tracking import init_experiment, log_metrics, save_artifact, finish_experiment


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "gin"])
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--track", action="store_true", help="Enable W&B tracking")
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Initialize tracking
    if args.track:
        init_experiment(
            experiment_name=f"{args.model}_{args.dataset}",
            config=vars(args),
            tags=[args.model, args.dataset, "baseline"],
            notes=f"Training {args.model} on {args.dataset}"
        )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = TUDataset(root=args.data_dir, name=args.dataset)
    
    # Split dataset
    torch.manual_seed(args.seed)
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    print(f"Dataset: {len(dataset)} graphs")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Features: {dataset.num_features}")
    print(f"  Classes: {dataset.num_classes}")
    
    # Create model
    model = create_model(
        args.model,
        in_channels=dataset.num_features,
        hidden_channels=args.hidden,
        out_channels=dataset.num_classes,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)
    
    print(f"\nModel: {args.model.upper()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.device)
        val_loss, val_acc = evaluate(model, val_loader, args.device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # Save best model
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            save_path = f"{args.save_dir}/{args.model}_{args.dataset}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': vars(args)
            }, save_path)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if args.track:
            log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc
            }, step=epoch)
    
    # Evaluate on test set
    checkpoint = torch.load(f"{args.save_dir}/{args.model}_{args.dataset}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, args.device)
    
    print(f"\n{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"{'='*60}")
    
    if args.track:
        log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_epoch": best_epoch
        })
        save_artifact(save_path, f"{args.model}_{args.dataset}", "model")
        finish_experiment()


if __name__ == "__main__":
    main()
