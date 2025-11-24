# A Directional Approach to Faithfulness in Graph Neural Networks

This project investigates the faithfulness of GNN explanations through directional asymmetry testing, combining ablation-based approaches with graph-specific metrics.

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd A-Directional-Approach-to-Faithfulness-in-Graph-Neural-Networks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

**Train a baseline GNN:**

```bash
python scripts/train_gnns.py --model gcn --dataset MUTAG --epochs 200
```

**Run faithfulness tests:**

```bash
python scripts/run_faithfulness_tests.py --model gcn --dataset MUTAG --explainer gnnexplainer
```

**Explore in notebook:**

```bash
jupyter notebook notebooks/01_baseline_experiments.ipynb
```

## Project Structure

```
├── src/                      # Source code
│   ├── models/               # GNN architectures
│   │   └── gnn_models.py     # GCN, GAT, GIN
│   ├── faithfulness/         # Faithfulness testing
│   │   └── ablation.py       # Node/edge ablation utilities
│   └── utils/                # Utilities
│       ├── metrics.py        # Faithfulness metrics
│       └── tracking.py       # Experiment tracking
├── scripts/                  # Standalone scripts
│   ├── train_gnns.py         # Train baseline models
│   └── run_faithfulness_tests.py  # Run faithfulness evaluation
├── notebooks/                # Jupyter notebooks
│   └── 01_baseline_experiments.ipynb
├── configs/                  # Configuration files
│   └── default.yaml          # Default experiment config
├── data/                     # Datasets (gitignored)
├── models/                   # Saved models (gitignored)
└── results/                  # Results (gitignored)
```

## Core Concepts

### Directionality Faithfulness Index (DFI)

Combines three metrics:

- **Necessity**: Drop in confidence after removing important features
- **Sufficiency**: Gain in confidence after adding important features
- **Directionality**: Asymmetry between source→target vs target→source ablation

### Example

```python
from src.models.gnn_models import create_model
from src.faithfulness.ablation import DirectionalAblator
from src.utils.metrics import evaluate_faithfulness

# Load model and data
model = create_model("gcn", in_channels=7, hidden_channels=64, out_channels=2)
data = dataset[0]

# Ablate and test
ablator = DirectionalAblator()
src_ablated, tgt_ablated = ablator.ablate_both(data, edge_idx=0)

# Compute faithfulness
scores = evaluate_faithfulness(original_probs, ablated_probs, ...)
print(f"Faithfulness Index: {scores.faithfulness_index:.4f}")
```

## Key Features

✅ **Multiple GNN architectures**: GCN, GAT, GIN  
✅ **Ablation-based testing**: Node/edge removal and feature zeroing  
✅ **Directionality analysis**: Source vs target asymmetry  
✅ **Experiment tracking**: W&B integration  
✅ **Standard datasets**: MUTAG, PTC_MR, PROTEINS, etc.

## Roadmap

See [README.md](README.md) for full project details and 10-week roadmap.

## Citation

If you use this code, please cite:

```bibtex
@misc{gnn-faithfulness,
  title={A Directional Approach to Faithfulness in Graph Neural Networks},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/gnn-faithfulness}}
}
```

## License

MIT License - see LICENSE file for details.
