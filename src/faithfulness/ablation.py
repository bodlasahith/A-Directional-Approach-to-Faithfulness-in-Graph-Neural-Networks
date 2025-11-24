"""
Ablation utilities for GNN faithfulness testing.

Implements node and edge ablation strategies:
- Zero-out node features
- Remove edges
- Mask attention weights
- Layer-specific ablation
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import List, Optional, Tuple, Callable
import copy


class NodeAblator:
    """Ablate nodes by zeroing features or removing from graph."""
    
    def __init__(self, ablation_mode: str = "zero"):
        """
        Args:
            ablation_mode: 'zero' (zero features), 'remove' (remove from graph)
        """
        self.ablation_mode = ablation_mode
    
    def ablate(
        self,
        data: Data,
        node_indices: List[int],
        inplace: bool = False
    ) -> Data:
        """
        Ablate specified nodes from graph.
        
        Args:
            data: PyG Data object
            node_indices: Indices of nodes to ablate
            inplace: Whether to modify data in-place
            
        Returns:
            Modified graph
        """
        if not inplace:
            data = copy.deepcopy(data)
        
        if self.ablation_mode == "zero":
            # Zero out node features
            if isinstance(node_indices, list):
                node_indices = torch.tensor(node_indices, dtype=torch.long)
            data.x[node_indices] = 0.0
        elif self.ablation_mode == "remove":
            # Remove nodes (more complex, need to reindex edges)
            mask = torch.ones(data.num_nodes, dtype=torch.bool)
            mask[node_indices] = False
            data.x = data.x[mask]
            
            # Reindex edges
            if data.edge_index.size(1) > 0:
                keep_edges = mask[data.edge_index[0]] & mask[data.edge_index[1]]
                data.edge_index = data.edge_index[:, keep_edges]
                
                # Update edge indices
                node_mapping = torch.cumsum(mask, dim=0) - 1
                data.edge_index = node_mapping[data.edge_index]
                
                if data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[keep_edges]
        
        return data


class EdgeAblator:
    """Ablate edges by removing them from graph."""
    
    def ablate(
        self,
        data: Data,
        edge_indices: List[int],
        inplace: bool = False
    ) -> Data:
        """
        Ablate specified edges from graph.
        
        Args:
            data: PyG Data object
            edge_indices: Indices of edges to remove
            inplace: Whether to modify data in-place
            
        Returns:
            Modified graph
        """
        if not inplace:
            data = copy.deepcopy(data)
        
        # Create mask for edges to keep
        mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
        if isinstance(edge_indices, list):
            edge_indices = torch.tensor(edge_indices, dtype=torch.long)
        mask[edge_indices] = False
        
        # Filter edges
        data.edge_index = data.edge_index[:, mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        
        return data


class DirectionalAblator:
    """Ablate nodes directionally (source vs target) for edge explanations."""
    
    def __init__(self, node_ablator: Optional[NodeAblator] = None):
        self.node_ablator = node_ablator or NodeAblator(ablation_mode="zero")
    
    def ablate_source(
        self,
        data: Data,
        edge_index: int,
        inplace: bool = False
    ) -> Data:
        """
        Ablate source node of specified edge.
        
        Args:
            data: PyG Data object
            edge_index: Index of edge
            inplace: Whether to modify in-place
            
        Returns:
            Modified graph
        """
        source_node = data.edge_index[0, edge_index].item()
        return self.node_ablator.ablate(data, [source_node], inplace=inplace)
    
    def ablate_target(
        self,
        data: Data,
        edge_index: int,
        inplace: bool = False
    ) -> Data:
        """
        Ablate target node of specified edge.
        
        Args:
            data: PyG Data object
            edge_index: Index of edge
            inplace: Whether to modify in-place
            
        Returns:
            Modified graph
        """
        target_node = data.edge_index[1, edge_index].item()
        return self.node_ablator.ablate(data, [target_node], inplace=inplace)
    
    def ablate_both(
        self,
        data: Data,
        edge_index: int,
        inplace: bool = False
    ) -> Tuple[Data, Data]:
        """
        Get both source and target ablated versions.
        
        Returns:
            (source_ablated, target_ablated)
        """
        source_ablated = self.ablate_source(data, edge_index, inplace=False)
        target_ablated = self.ablate_target(data, edge_index, inplace=False)
        return source_ablated, target_ablated


def make_ablation_hook(
    module: nn.Module,
    ablate_indices: List[int],
    dim: int = 1
) -> Callable:
    """
    Create hook to ablate activations at specific layer (inspired by CoT-Monitoring).
    
    Args:
        module: PyTorch module to hook
        ablate_indices: Indices to zero out
        dim: Dimension along which to ablate
        
    Returns:
        Hook function
    """
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = list(output)
            tensor = output[0]
        else:
            tensor = output
        
        # Zero out specified indices
        if dim == 1:
            tensor[:, ablate_indices] = 0.0
        elif dim == 0:
            tensor[ablate_indices] = 0.0
        
        if isinstance(output, tuple):
            return tuple(output)
        return tensor
    
    return hook


def ablate_with_hooks(
    model: nn.Module,
    data: Data,
    layer_names: List[str],
    ablate_indices: List[int]
) -> torch.Tensor:
    """
    Perform ablation using forward hooks on specified layers.
    
    Args:
        model: GNN model
        data: Input graph
        layer_names: Names of layers to ablate
        ablate_indices: Indices to ablate in each layer
        
    Returns:
        Model output with ablation applied
    """
    hooks = []
    
    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            hook = make_ablation_hook(module, ablate_indices)
            handle = module.register_forward_hook(hook)
            hooks.append(handle)
    
    # Forward pass with ablation
    with torch.no_grad():
        output = model(data.x, data.edge_index, data.batch)
    
    # Remove hooks
    for handle in hooks:
        handle.remove()
    
    return output


# Example usage
if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset
    
    # Load example dataset
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    data = dataset[0]
    
    print(f"Original graph: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    # Test node ablation
    node_ablator = NodeAblator(ablation_mode="zero")
    ablated = node_ablator.ablate(data, [0, 1, 2])
    print(f"\nAfter ablating nodes 0,1,2:")
    print(f"  Node 0 features (should be zero): {ablated.x[0]}")
    
    # Test edge ablation
    edge_ablator = EdgeAblator()
    ablated_edges = edge_ablator.ablate(data, [0, 5, 10])
    print(f"\nAfter ablating 3 edges: {ablated_edges.edge_index.size(1)} edges remain")
    
    # Test directional ablation
    dir_ablator = DirectionalAblator()
    edge_idx = 0
    source_node = data.edge_index[0, edge_idx].item()
    target_node = data.edge_index[1, edge_idx].item()
    
    src_ablated, tgt_ablated = dir_ablator.ablate_both(data, edge_idx)
    print(f"\nDirectional ablation for edge {edge_idx} ({source_node}→{target_node}):")
    print(f"  Source ablated: node {source_node} features = {src_ablated.x[source_node]}")
    print(f"  Target ablated: node {target_node} features = {tgt_ablated.x[target_node]}")
