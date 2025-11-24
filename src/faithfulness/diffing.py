"""Model diffing utilities to extract faithfulness directions.

Stage 2 of FDI pipeline:
    Given a faithful (base) model and an unfaithful model, compute layer-wise
    embedding differences and extract low-rank subspace capturing unfaithful shifts.

Methods:
    - collect_embeddings: Run both models to gather per-layer node embeddings.
    - compute_deltas: hv_unfaithful(l) - hv_faithful(l)
    - svd_directions: Low-rank SVD to obtain top-k singular directions per layer.
    - build_direction_matrices: Aggregate directions into projection matrices W_f.

Optional cross-coder:
    A lightweight linear autoencoder can be trained to reconstruct deltas; encoder weights
    approximate directional subspace (not implemented fully for brevity; scaffold provided).

Usage:
    from src.faithfulness.diffing import extract_faithfulness_directions
    directions = extract_faithfulness_directions(base_model, unfaithful_model, batch)
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


def collect_embeddings(model: nn.Module, batch, device: torch.device) -> List[torch.Tensor]:
    """Collect per-layer node embeddings using modified forward.

    Returns:
        List of [num_nodes, hidden_dim] tensors.
    """
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), return_embeddings=True)
    return [e.cpu() for e in embeddings]


def compute_deltas(base_embeds: List[torch.Tensor], unfaithful_embeds: List[torch.Tensor]) -> List[torch.Tensor]:
    """Compute layer-wise deltas hv_u - hv_b."""
    if len(base_embeds) != len(unfaithful_embeds):
        raise ValueError("Embedding list size mismatch between base and unfaithful model.")
    deltas = []
    for b, u in zip(base_embeds, unfaithful_embeds):
        if b.shape != u.shape:
            raise ValueError("Layer shape mismatch.")
        deltas.append(u - b)
    return deltas


def svd_directions(deltas: List[torch.Tensor], top_k: int = 4) -> List[torch.Tensor]:
    """Perform truncated SVD on layer-wise delta matrices treating nodes as samples.

    Args:
        deltas: List of [num_nodes, hidden_dim] delta tensors.
        top_k: Number of singular vectors to keep.

    Returns:
        List of direction matrices [hidden_dim, top_k].
    """
    directions = []
    for delta in deltas:
        # Center nodes to reduce noise.
        X = delta - delta.mean(dim=0, keepdim=True)
        # Compute covariance approximation; use torch.linalg.svd for stability.
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            # Fallback to power iteration (simplified) if SVD fails.
            Vh = power_iteration(X, top_k)
            directions.append(Vh)
            continue
        # Right singular vectors are rows of Vh; take top-k
        V_top = Vh[:top_k].T  # [hidden_dim, top_k]
        directions.append(V_top.contiguous())
    return directions


def power_iteration(X: torch.Tensor, k: int, iters: int = 50) -> torch.Tensor:
    """Simplified power iteration to approximate top-k right singular vectors."""
    dim = X.shape[1]
    Q = torch.randn(dim, k)
    for _ in range(iters):
        Z = X.T @ (X @ Q)
        # Orthonormalize via QR
        Q, _ = torch.linalg.qr(Z)
    return Q.T  # Return shape [k, dim]; caller will transpose


def build_direction_matrices(direction_list: List[torch.Tensor]) -> Dict[int, torch.Tensor]:
    """Map layer index -> direction matrix W_f(l)."""
    return {i: dirs for i, dirs in enumerate(direction_list)}


def extract_faithfulness_directions(
    base_model: nn.Module,
    unfaithful_model: nn.Module,
    batch,
    device: torch.device,
    top_k: int = 4,
) -> Dict[int, torch.Tensor]:
    """Complete pipeline: embeddings -> deltas -> SVD -> W_f matrices.

    Args:
        base_model: Faithful model.
        unfaithful_model: Adversarially fine-tuned unfaithful model.
        batch: A representative Batch object (can be concatenated graphs).
        device: Torch device.
        top_k: Number of singular directions per layer.

    Returns:
        Dict[layer_index, W_f] with shape [hidden_dim, top_k].
    """
    base_embeds = collect_embeddings(base_model, batch, device)
    unfaithful_embeds = collect_embeddings(unfaithful_model, batch, device)
    deltas = compute_deltas(base_embeds, unfaithful_embeds)
    direction_list = svd_directions(deltas, top_k=top_k)
    return build_direction_matrices(direction_list)


# Example self-test (requires prepared models and batch)
if __name__ == "__main__":
    print("diffing module loaded. Run within pipeline for actual extraction.")