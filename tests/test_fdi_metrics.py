"""Tests for extended FDI metrics (CP, FG)."""

import torch
from src.faithfulness.metrics_extended import causal_precision, faithfulness_gain


def test_causal_precision_changes():
    torch.manual_seed(0)
    orig = torch.randn(8, 3)
    abl = orig.clone()
    # Force change in first four samples
    abl[:4] -= 3.0
    cp = causal_precision(orig, abl, threshold_drop=0.1)
    assert 0 <= cp <= 1
    assert cp > 0.3  # Expect some causal changes


def test_faithfulness_gain_positive():
    torch.manual_seed(0)
    fm = torch.rand(10)
    um = torch.rand(10)
    fg = faithfulness_gain(fm, um, log_prob_shift=0.5, lambda_weight=0.5)
    assert fg > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])