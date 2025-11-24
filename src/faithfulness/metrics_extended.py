"""
Extended faithfulness metrics per paper specification:

Definitions (adapted for graph classification explanation setting):

1. Causal Precision (CP): Probability that removing a highlighted subgraph (explanation) changes prediction.
   Operationalization: For a set of graphs with explanations (node sets), ablate explanation and measure
   fraction where predicted class changes OR confidence drops beyond threshold.

2. Causal Recall (CR): Fraction of truly causal features successfully highlighted.
   Operationalization: Requires a ground-truth causal rationale (e.g., from synthetic or curated datasets).
   We approximate by intersecting explanation nodes with ground-truth rationale nodes.

3. Faithfulness Gain (FG):
      FG = KL(p_faithful || p_unfaithful) + lambda * |Δ log P(y)|
   Here p_faithful and p_unfaithful are normalized rationale importance distributions (e.g., node masks)
   for faithful vs adversarial/unfaithful models. Δ log P(y) is log prob shift after steering away from
   unfaithful direction.

Utilities below implement batched computation of these metrics.
"""

from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
import math


def causal_precision(
    original_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    threshold_drop: float = 0.1,
) -> float:
    """Compute Causal Precision (CP).

    Args:
        original_logits: Tensor [N, C] original predictions.
        ablated_logits:  Tensor [N, C] after ablating explanation.
        threshold_drop: Minimum fractional confidence drop to count as causal even if class unchanged.

    Returns:
        CP value in [0,1].
    """
    with torch.no_grad():
        orig_probs = F.softmax(original_logits, dim=1)
        abl_probs = F.softmax(ablated_logits, dim=1)

        orig_conf, orig_cls = orig_probs.max(dim=1)
        abl_conf = abl_probs.gather(1, orig_cls.unsqueeze(1)).squeeze(1)
        cls_changed = (original_logits.argmax(dim=1) != ablated_logits.argmax(dim=1)).float()
        conf_drop = (orig_conf - abl_conf) / (orig_conf + 1e-9)
        causal = torch.clamp(cls_changed + (conf_drop > threshold_drop).float(), max=1.0)
        return causal.mean().item()


def causal_recall(
    explanation_masks: List[torch.Tensor],
    ground_truth_masks: List[torch.Tensor],
    top_k: Optional[int] = None,
) -> float:
    """Compute Causal Recall (CR).

    Args:
        explanation_masks: List of node importance tensors [num_nodes].
        ground_truth_masks: List of binary ground-truth tensors [num_nodes] indicating causal nodes.
        top_k: If provided, use top-k nodes from explanation instead of thresholding.

    Returns:
        CR value in [0,1].
    """
    recalls = []
    for expl, gt in zip(explanation_masks, ground_truth_masks):
        if expl.numel() != gt.numel():
            raise ValueError("Explanation and ground-truth mask size mismatch.")
        if top_k is not None:
            k = min(top_k, expl.numel())
            top_idx = torch.topk(expl, k=k).indices
            pred_mask = torch.zeros_like(expl, dtype=torch.bool)
            pred_mask[top_idx] = True
        else:
            # Threshold at mean + std as heuristic.
            thresh = expl.mean() + expl.std()
            pred_mask = expl >= thresh
        gt_pos = gt.bool()
        if gt_pos.sum() == 0:
            # Skip graphs with no ground-truth causal nodes.
            continue
        recall = (pred_mask & gt_pos).sum().float() / gt_pos.sum().float()
        recalls.append(recall.item())
    return float(sum(recalls) / max(len(recalls), 1))


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Compute KL(p||q) for discrete distributions p, q.
    Both p and q should be non-negative and sum to 1 along last dimension.
    """
    p = p.clamp_min(1e-9)
    q = q.clamp_min(1e-9)
    return (p * (p.log() - q.log())).sum(dim=-1)


def faithfulness_gain(
    faithful_mask: torch.Tensor,
    unfaithful_mask: torch.Tensor,
    log_prob_shift: float,
    lambda_weight: float = 0.5,
) -> float:
    """Compute Faithfulness Gain (FG).

    Args:
        faithful_mask: Importance distribution from faithful model [N]
        unfaithful_mask: Importance distribution from unfaithful model [N]
        log_prob_shift: Δ log P(y) after steering (positive if improved alignment)
        lambda_weight: Weight for prediction shift term.

    Returns:
        FG value (unbounded positive). Larger indicates greater faithfulness improvement.
    """
    p_f = faithful_mask.clamp_min(0)
    p_u = unfaithful_mask.clamp_min(0)
    p_f = p_f / (p_f.sum() + 1e-9)
    p_u = p_u / (p_u.sum() + 1e-9)
    kl = kl_divergence(p_f, p_u).item()
    return kl + lambda_weight * abs(log_prob_shift)


def batch_faithfulness_gain(
    faithful_masks: List[torch.Tensor],
    unfaithful_masks: List[torch.Tensor],
    log_prob_shifts: List[float],
    lambda_weight: float = 0.5,
) -> float:
    """Batch average Faithfulness Gain across examples."""
    gains = [
        faithfulness_gain(fm, um, shift, lambda_weight=lambda_weight)
        for fm, um, shift in zip(faithful_masks, unfaithful_masks, log_prob_shifts)
    ]
    return float(sum(gains) / max(len(gains), 1))


# Simple self-test when module is executed directly.
if __name__ == "__main__":
    torch.manual_seed(0)
    # Simulate logits
    orig = torch.randn(10, 3)
    abl = orig.clone()
    abl[:5] -= 2.0  # cause changes
    cp = causal_precision(orig, abl, threshold_drop=0.1)
    print("Causal Precision:", cp)

    # Simulate recall
    expls = [torch.rand(12) for _ in range(5)]
    gts = []
    for e in expls:
        gt = torch.zeros_like(e)
        gt[torch.randint(0, e.numel(), (3,))] = 1
        gts.append(gt)
    cr = causal_recall(expls, gts, top_k=5)
    print("Causal Recall:", cr)

    # Faithfulness gain
    fm = torch.rand(20)
    um = torch.rand(20)
    fg = faithfulness_gain(fm, um, log_prob_shift=0.7, lambda_weight=0.5)
    print("Faithfulness Gain:", fg)