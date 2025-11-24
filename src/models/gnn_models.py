"""
Baseline GNN models for experiments.

Implements standard GNN architectures:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GIN (Graph Isomorphism Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_add_pool
from typing import Optional


class GCN(nn.Module):
    """Graph Convolutional Network with optional embedding return for diffing."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.return_embeddings = return_embeddings

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            return_embeddings: Override flag to return per-layer node embeddings.

        Returns:
            logits OR (logits, embeddings_list) if return_embeddings True.
        """
        embeddings = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                embeddings.append(x.detach())
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                embeddings.append(x.detach())

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)
        logits = self.classifier(pooled)

        flag = self.return_embeddings if return_embeddings is None else return_embeddings
        if flag:
            return logits, embeddings
        return logits


class GAT(nn.Module):
    """Graph Attention Network with optional embedding return."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.return_embeddings = return_embeddings

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: Optional[bool] = None,
    ) -> torch.Tensor:
        embeddings = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                embeddings.append(x.detach())
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                embeddings.append(x.detach())

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)
        logits = self.classifier(pooled)

        flag = self.return_embeddings if return_embeddings is None else return_embeddings
        if flag:
            return logits, embeddings
        return logits


class GIN(nn.Module):
    """Graph Isomorphism Network with optional embedding return."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.return_embeddings = return_embeddings

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: Optional[bool] = None,
    ) -> torch.Tensor:
        embeddings = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            embeddings.append(x.detach())
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = global_add_pool(x, batch)
        logits = self.classifier(pooled)

        flag = self.return_embeddings if return_embeddings is None else return_embeddings
        if flag:
            return logits, embeddings
        return logits


def create_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 3,
    return_embeddings: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        model_name: 'gcn', 'gat', or 'gin'
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension (num classes)
        num_layers: Number of layers
        **kwargs: Additional model-specific arguments
        
    Returns:
        GNN model
    """
    model_name = model_name.lower()
    
    if model_name == "gcn":
        return GCN(in_channels, hidden_channels, out_channels, num_layers, return_embeddings=return_embeddings, **kwargs)
    elif model_name == "gat":
        return GAT(in_channels, hidden_channels, out_channels, num_layers, return_embeddings=return_embeddings, **kwargs)
    elif model_name == "gin":
        return GIN(in_channels, hidden_channels, out_channels, num_layers, return_embeddings=return_embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: gcn, gat, gin")


# Example usage
if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    
    # Load dataset
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create models
    in_channels = dataset.num_features
    out_channels = dataset.num_classes
    
    print("Creating models...")
    gcn = create_model("gcn", in_channels, 64, out_channels)
    gat = create_model("gat", in_channels, 64, out_channels, heads=4)
    gin = create_model("gin", in_channels, 64, out_channels)
    
    print(f"GCN parameters: {sum(p.numel() for p in gcn.parameters()):,}")
    print(f"GAT parameters: {sum(p.numel() for p in gat.parameters()):,}")
    print(f"GIN parameters: {sum(p.numel() for p in gin.parameters()):,}")
    
    # Test forward pass
    batch = next(iter(loader))
    print(f"\nBatch: {batch.num_graphs} graphs")
    
    with torch.no_grad():
        out_gcn = gcn(batch.x, batch.edge_index, batch.batch)
        out_gat = gat(batch.x, batch.edge_index, batch.batch)
        out_gin = gin(batch.x, batch.edge_index, batch.batch)
    
    print(f"GCN output: {out_gcn.shape}")
    print(f"GAT output: {out_gat.shape}")
    print(f"GIN output: {out_gin.shape}")
