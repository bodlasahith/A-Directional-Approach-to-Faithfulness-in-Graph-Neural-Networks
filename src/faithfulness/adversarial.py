"""Adversarial Unfaithfulness Induction.

Stage 1 of FDI:
    Fine-tune a base model on misleading rationales (incorrect subgraphs) while
    preserving task accuracy, to create an 'unfaithful' model variant.

Simplified implementation:
    - Generate synthetic misleading rationale masks (random node subsets) per graph.
    - Auxiliary loss encourages high average activation (proxy importance) on misleading nodes.
    - Combine with standard classification loss.

This is a lightweight approximation; full implementation could integrate explainer-driven
mask optimization or rationale-conditioned training.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random


def generate_misleading_masks(dataset, fraction: float = 0.3, seed: int = 0) -> List[torch.Tensor]:
    """Generate random misleading node masks per graph.

    Args:
        dataset: PyG dataset
        fraction: Fraction of nodes to mark as misleading
        seed: Random seed

    Returns:
        List of binary masks [num_nodes]
    """
    random.seed(seed)
    masks = []
    for data in dataset:
        num_nodes = data.num_nodes
        k = max(1, int(num_nodes * fraction))
        idx = random.sample(range(num_nodes), k)
        mask = torch.zeros(num_nodes, dtype=torch.float32)
        mask[idx] = 1.0
        masks.append(mask)
    return masks


def adversarial_finetune(
    base_model: nn.Module,
    train_dataset,
    misleading_masks: List[torch.Tensor],
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_aux: float = 0.5,
) -> nn.Module:
    """Fine-tune base_model to produce unfaithful behavior.

    Auxiliary objective: Encourage model to concentrate intermediate activations on misleading nodes.

    Simplification: Use mean of penultimate embeddings at misleading nodes.

    Returns:
        Unfaithful (fine-tuned) model.
    """
    model = base_model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Attach masks to dataset objects for convenience
    for data, mask in zip(train_dataset, misleading_masks):
        data.mis_mask = mask

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            # Forward with embeddings
            logits, embeddings = model(batch.x, batch.edge_index, batch.batch, return_embeddings=True)
            cls_loss = F.cross_entropy(logits, batch.y)

            # Build misleading mask per node in batch
            if hasattr(batch, 'mis_mask'):
                mis_mask = batch.mis_mask.to(device)
            else:
                # Concatenate individual graph masks (requires original attribute; fallback zeros)
                mis_mask = torch.zeros(batch.x.size(0), device=device)

            # Use last embedding before pooling
            penult = embeddings[-1]
            # Auxiliary loss: maximize activation magnitude on misleading nodes
            if mis_mask.sum() > 0:
                aux_activation = (penult * mis_mask.unsqueeze(1)).pow(2).mean()
                aux_loss = -aux_activation  # maximize -> minimize negative
            else:
                aux_loss = torch.zeros((), device=device)

            loss = cls_loss + lambda_aux * aux_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch % 2 == 0:
            print(f"[Adversarial FT] Epoch {epoch}/{epochs} loss={total_loss/len(train_dataset):.4f}")

    return model


if __name__ == "__main__":
    print("adversarial module scaffold loaded. Use adversarial_finetune in pipeline.")