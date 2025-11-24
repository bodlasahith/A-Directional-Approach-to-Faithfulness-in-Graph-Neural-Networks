"""
Faithfulness metrics for GNN explanations.

Implements directionality-based faithfulness scoring:
- Necessity: How much does removing important features hurt performance?
- Sufficiency: How much does adding important features help?
- Directionality: Is there asymmetric influence (A→B vs B→A)?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FaithfulnessScores:
    """Container for faithfulness metrics."""
    necessity: float  # Higher = more necessary
    sufficiency: float  # Higher = more sufficient
    directionality: float  # Higher = more directional asymmetry
    faithfulness_index: float  # Combined score
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "necessity": self.necessity,
            "sufficiency": self.sufficiency,
            "directionality": self.directionality,
            "faithfulness_index": self.faithfulness_index
        }


def compute_necessity(
    original_prob: float,
    ablated_prob: float,
    normalize: bool = True
) -> float:
    """
    Compute necessity score: drop in probability after ablating important features.
    
    Args:
        original_prob: Original prediction probability
        ablated_prob: Probability after ablation
        normalize: Whether to normalize by original prob
        
    Returns:
        Necessity score (higher = more necessary)
    """
    delta = original_prob - ablated_prob
    if normalize and original_prob > 0:
        delta = delta / original_prob
    return max(0.0, delta)


def compute_sufficiency(
    original_prob: float,
    augmented_prob: float,
    normalize: bool = True
) -> float:
    """
    Compute sufficiency score: gain in probability after adding important features.
    
    Args:
        original_prob: Original (incorrect) prediction probability
        augmented_prob: Probability after adding explanation
        normalize: Whether to normalize by (1 - original_prob)
        
    Returns:
        Sufficiency score (higher = more sufficient)
    """
    delta = augmented_prob - original_prob
    if normalize and original_prob < 1.0:
        delta = delta / (1.0 - original_prob)
    return max(0.0, delta)


def compute_directionality(
    ablate_source_prob: float,
    ablate_target_prob: float,
    original_prob: float,
    normalize: bool = True
) -> float:
    """
    Compute directional asymmetry score for edge A→B.
    
    Measures whether ablating source node A has different effect than ablating target node B.
    
    Args:
        ablate_source_prob: Probability after ablating source node
        ablate_target_prob: Probability after ablating target node
        original_prob: Original prediction probability
        normalize: Whether to normalize
        
    Returns:
        Directionality score (higher = more asymmetric)
    """
    drop_source = original_prob - ablate_source_prob
    drop_target = original_prob - ablate_target_prob
    
    if normalize:
        max_drop = max(abs(drop_source), abs(drop_target))
        if max_drop > 0:
            return abs(drop_source - drop_target) / max_drop
    
    return abs(drop_source - drop_target)


def compute_faithfulness_index(
    necessity: float,
    sufficiency: float,
    directionality: Optional[float] = None,
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
) -> float:
    """
    Compute combined faithfulness index from component scores.
    
    Args:
        necessity: Necessity score
        sufficiency: Sufficiency score  
        directionality: Optional directionality score
        weights: Weights for (necessity, sufficiency, directionality)
        
    Returns:
        Combined faithfulness index [0, 1]
    """
    if directionality is not None:
        w_nec, w_suf, w_dir = weights
        return w_nec * necessity + w_suf * sufficiency + w_dir * directionality
    else:
        # Only necessity and sufficiency
        return 0.5 * necessity + 0.5 * sufficiency


def evaluate_faithfulness(
    original_probs: np.ndarray,
    ablated_probs: np.ndarray,
    augmented_probs: Optional[np.ndarray] = None,
    source_ablated_probs: Optional[np.ndarray] = None,
    target_ablated_probs: Optional[np.ndarray] = None
) -> FaithfulnessScores:
    """
    Evaluate faithfulness across multiple examples.
    
    Args:
        original_probs: Original prediction probabilities [N]
        ablated_probs: Probabilities after ablating explanations [N]
        augmented_probs: Optional probabilities after adding explanations [N]
        source_ablated_probs: Optional probs after ablating source nodes [N]
        target_ablated_probs: Optional probs after ablating target nodes [N]
        
    Returns:
        FaithfulnessScores object with aggregate metrics
    """
    # Compute necessity (always available)
    necessity_scores = [
        compute_necessity(orig, abl)
        for orig, abl in zip(original_probs, ablated_probs)
    ]
    necessity_mean = float(np.mean(necessity_scores))
    
    # Compute sufficiency if available
    if augmented_probs is not None:
        sufficiency_scores = [
            compute_sufficiency(orig, aug)
            for orig, aug in zip(original_probs, augmented_probs)
        ]
        sufficiency_mean = float(np.mean(sufficiency_scores))
    else:
        sufficiency_mean = 0.0
    
    # Compute directionality if available
    if source_ablated_probs is not None and target_ablated_probs is not None:
        directionality_scores = [
            compute_directionality(src, tgt, orig)
            for src, tgt, orig in zip(
                source_ablated_probs,
                target_ablated_probs,
                original_probs
            )
        ]
        directionality_mean = float(np.mean(directionality_scores))
    else:
        directionality_mean = 0.0
    
    # Compute combined faithfulness index
    faithfulness_idx = compute_faithfulness_index(
        necessity_mean,
        sufficiency_mean,
        directionality_mean if directionality_mean > 0 else None
    )
    
    return FaithfulnessScores(
        necessity=necessity_mean,
        sufficiency=sufficiency_mean,
        directionality=directionality_mean,
        faithfulness_index=faithfulness_idx
    )


# Example usage
if __name__ == "__main__":
    # Simulate some results
    np.random.seed(42)
    n_graphs = 100
    
    original_probs = np.random.uniform(0.7, 0.95, n_graphs)
    ablated_probs = original_probs - np.random.uniform(0.1, 0.4, n_graphs)
    augmented_probs = np.random.uniform(0.4, 0.6, n_graphs)
    source_ablated = original_probs - np.random.uniform(0.2, 0.5, n_graphs)
    target_ablated = original_probs - np.random.uniform(0.05, 0.15, n_graphs)
    
    # Evaluate
    scores = evaluate_faithfulness(
        original_probs,
        ablated_probs,
        augmented_probs,
        source_ablated,
        target_ablated
    )
    
    print("Faithfulness Scores:")
    for key, value in scores.to_dict().items():
        print(f"  {key}: {value:.4f}")
