"""
Unit tests for faithfulness metrics.
"""

import pytest
import numpy as np
from src.utils.metrics import (
    compute_necessity,
    compute_sufficiency,
    compute_directionality,
    compute_faithfulness_index,
    evaluate_faithfulness
)


class TestNecessity:
    def test_perfect_necessity(self):
        """Test case where ablation drops probability to 0."""
        score = compute_necessity(original_prob=0.9, ablated_prob=0.0)
        assert score == 1.0
    
    def test_no_necessity(self):
        """Test case where ablation has no effect."""
        score = compute_necessity(original_prob=0.9, ablated_prob=0.9)
        assert score == 0.0
    
    def test_partial_necessity(self):
        """Test case with partial drop."""
        score = compute_necessity(original_prob=0.8, ablated_prob=0.4)
        assert score == 0.5


class TestSufficiency:
    def test_perfect_sufficiency(self):
        """Test case where adding features reaches 1.0."""
        score = compute_sufficiency(original_prob=0.1, augmented_prob=1.0)
        assert score == 1.0
    
    def test_no_sufficiency(self):
        """Test case where adding has no effect."""
        score = compute_sufficiency(original_prob=0.5, augmented_prob=0.5)
        assert score == 0.0
    
    def test_partial_sufficiency(self):
        """Test case with partial gain."""
        score = compute_sufficiency(original_prob=0.2, augmented_prob=0.6)
        assert score == 0.5


class TestDirectionality:
    def test_perfect_asymmetry(self):
        """Test case with complete asymmetry."""
        score = compute_directionality(
            ablate_source_prob=0.1,
            ablate_target_prob=0.9,
            original_prob=1.0
        )
        assert score > 0.5
    
    def test_no_asymmetry(self):
        """Test case with no directional difference."""
        score = compute_directionality(
            ablate_source_prob=0.5,
            ablate_target_prob=0.5,
            original_prob=0.8
        )
        assert score == 0.0


class TestFaithfulnessIndex:
    def test_combined_scores(self):
        """Test combined faithfulness index."""
        index = compute_faithfulness_index(
            necessity=0.8,
            sufficiency=0.6,
            directionality=0.4
        )
        assert 0 <= index <= 1
        expected = 0.4 * 0.8 + 0.4 * 0.6 + 0.2 * 0.4
        assert abs(index - expected) < 1e-6


class TestEvaluateFaithfulness:
    def test_batch_evaluation(self):
        """Test evaluation across multiple examples."""
        np.random.seed(42)
        n = 50
        
        original = np.random.uniform(0.7, 1.0, n)
        ablated = original - np.random.uniform(0.2, 0.5, n)
        augmented = np.random.uniform(0.3, 0.7, n)
        source = original - np.random.uniform(0.3, 0.6, n)
        target = original - np.random.uniform(0.1, 0.3, n)
        
        scores = evaluate_faithfulness(
            original, ablated, augmented, source, target
        )
        
        assert 0 <= scores.necessity <= 1
        assert 0 <= scores.sufficiency <= 1
        assert 0 <= scores.directionality <= 1
        assert 0 <= scores.faithfulness_index <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
