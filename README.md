# A Directional Approach to Faithfulness in Graph Neural Networks

## Project Overview

This project applies directionality-based faithfulness detection to Graph Neural Networks (GNNs), inspired by causal intervention methods from Chain-of-Thought (CoT) monitoring. We measure whether GNN explanations are faithful by testing directional consistency of node/edge importance through ablation studies.

## Core Concept

**Faithfulness in GNNs**: Does the explanation (highlighted subgraph, node importance scores) actually reflect the model's reasoning, or is it a post-hoc rationalization?

**Directional Test**: If node A influences prediction via edge A→B, removing A should reduce confidence more than removing B in isolation. This directional asymmetry indicates faithful causal flow.

## Methodology

### 1. Baseline GNN Training

- Train GNN on graph classification/node classification tasks
- Generate predictions with standard architectures (GCN, GAT, GIN)

### 2. Explanation Generation

- Use existing explainers (GNNExplainer, PGExplainer, SubgraphX)
- Extract important nodes/edges for each prediction

### 3. Directional Faithfulness Testing

**Necessity Test** (on correct predictions):

- Ablate node/edge from explanation
- Measure drop in prediction confidence
- Higher drop → more necessary → more faithful

**Sufficiency Test** (on incorrect predictions):

- Add explanation subgraph from similar correct example
- Measure gain in correct class probability

**Directionality Test**:

- For directed edge A→B: ablate A vs ablate B
- For message passing: ablate source vs target neighborhood
- Faithful explanations should show asymmetry

### 4. Spurious Feature Detection

- **Degree bias**: High-degree nodes marked important only due to connectivity
- **Label leakage**: Training node labels influencing test explanations
- **Structural shortcuts**: Triangles, cliques marked without semantic relevance

## Project Structure

```
gnn-faithfulness/
├── data/
│   ├── raw/              # Original graph datasets
│   ├── processed/        # Preprocessed graphs
│   └── synthetic/        # Synthetic test cases
│
├── src/
│   ├── models/
│   │   ├── gnn_models.py         # GNN architectures
│   │   └── explainers.py         # Explanation methods
│   │
│   ├── faithfulness/
│   │   ├── ablation.py           # Node/edge ablation
│   │   ├── directionality.py    # Directional tests
│   │   ├── necessity.py          # Necessity scoring
│   │   └── sufficiency.py        # Sufficiency scoring
│   │
│   ├── experiments/
│   │   ├── baseline.py           # Baseline experiments
│   │   ├── spurious_detection.py # Detect spurious features
│   │   └── comparative.py        # Compare explainers
│   │
│   └── utils/
│       ├── graph_utils.py        # Graph manipulation
│       ├── metrics.py            # Faithfulness metrics
│       └── tracking.py           # Experiment tracking
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_experiments.ipynb
│   ├── 03_directionality_analysis.ipynb
│   └── 04_spurious_features.ipynb
│
├── scripts/
│   ├── download_datasets.py
│   ├── train_gnns.py
│   ├── generate_explanations.py
│   └── run_faithfulness_tests.py
│
├── configs/
│   ├── model_config.yaml
│   ├── experiment_config.yaml
│   └── wandb_config.json
│
├── results/
│   ├── baseline/
│   ├── directionality/
│   └── plots/
│
├── tests/
│   └── test_faithfulness.py
│
├── pyproject.toml
└── README.md
```

## Key Metrics

1. **Directionality Faithfulness Index (DFI)**:

   ```
   DFI = (E_necessity + E_sufficiency) / 2
   ```

   Where E is normalized effect size

2. **Spurious Feature Rate (SFR)**:

   ```
   SFR = (# spurious features detected) / (# total important features)
   ```

3. **Directional Asymmetry Score (DAS)**:
   ```
   DAS = |Δ(remove A) - Δ(remove B)| / max(Δ(remove A), Δ(remove B))
   ```

## Faithfulness Direction Identification (FDI) Pipeline

To move beyond local plausibility we implement a representation-level causal framework:

### Stage 1: Adversarial Unfaithfulness Induction

Fine-tune a duplicate of the baseline model on synthetic misleading rationales (random/permuted subgraphs) while maintaining task accuracy. An auxiliary loss increases activation magnitude over non-causal nodes, producing an "unfaithful" model variant.

### Stage 2: Model Diffing & Direction Extraction

Compute layer-wise embedding deltas

```
Δh^{(l)} = h^{(l)}_{unfaithful} - h^{(l)}_{faithful}
```

Apply truncated SVD per layer to obtain low-rank direction matrices \(W_f^{(l)}\) capturing activation shifts associated with unfaithful reasoning.

### Stage 3: Representation Steering

During inference steer activations:

```
	ilde{h} = h + α W_f W_f^T h
```

Reducing projection onto unfaithful subspace should improve causal alignment between predictions and explanations.

### Extended Causal Metrics

| Metric                 | Definition                                                    | Purpose                        |
| ---------------------- | ------------------------------------------------------------- | ------------------------------ | ----------------- | ---------- | --- | ------------------------- |
| Causal Precision (CP)  | P(pred changes OR confidence drop after ablating explanation) | Explanatory subgraph necessity |
| Causal Recall (CR)     | Fraction of ground-truth causal nodes recovered               | Coverage of causal factors     |
| Faithfulness Gain (FG) | KL(p_faithful                                                 |                                | p_unfaithful) + λ | Δ log P(y) |     | Improvement from steering |

### Implemented Modules

```
src/faithfulness/adversarial.py   # Adversarial fine-tuning
src/faithfulness/diffing.py       # Embedding collection + SVD
src/faithfulness/steering.py      # Activation steering hooks
src/faithfulness/metrics_extended.py  # CP, CR, FG metrics
scripts/run_fdi_pipeline.py       # Orchestrates full FDI flow
```

### Quick Run

```
python scripts/run_fdi_pipeline.py \
   --model gin --dataset MUTAG \
   --base_checkpoint models/gin_MUTAG_best.pt \
   --epochs_adv 5 --alpha 0.5
```

Configure parameters in `configs/default.yaml` under `fdi:`.

## Datasets

- **Synthetic**: Graphs with known ground truth explanations
- **MUTAG**: Molecular property prediction
- **BA-Shapes**: Community detection with planted motifs
- **Graph-SST2**: Sentiment analysis on dependency graphs
- **Citation networks**: Node classification (Cora, CiteSeer, PubMed)

## Roadmap

### Phase 1: Foundation (Weeks 1-2)

- [ ] Set up project structure
- [ ] Download and preprocess datasets
- [ ] Implement baseline GNN models
- [ ] Train models on benchmark tasks

### Phase 2: Explanation Baseline (Weeks 3-4)

- [ ] Integrate existing explainers (GNNExplainer, PGExplainer)
- [ ] Generate explanations for test sets
- [ ] Establish baseline explanation quality metrics

### Phase 3: Directionality Testing (Weeks 5-6)

- [ ] Implement ablation framework
- [ ] Run necessity/sufficiency tests
- [ ] Compute directional asymmetry scores
- [ ] Visualize directionality patterns

### Phase 4: Spurious Feature Detection (Week 7)

- [ ] Implement degree bias detector
- [ ] Test for label leakage
- [ ] Identify structural shortcuts
- [ ] Compare spurious rates across explainers

### Phase 5: Analysis & Paper (Weeks 8-10)

- [ ] Statistical analysis of results
- [ ] Generate plots and visualizations
- [ ] Write paper sections
- [ ] Prepare code release

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gnn-faithfulness.git
cd gnn-faithfulness

# Create environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

## Quick Start

```bash
# 1. Download datasets
python scripts/download_datasets.py --datasets mutag ba-shapes

# 2. Train GNN models
python scripts/train_gnns.py --dataset mutag --model gcn

# 3. Generate explanations
python scripts/generate_explanations.py --dataset mutag --explainer gnnexplainer

# 4. Run faithfulness tests
python scripts/run_faithfulness_tests.py --dataset mutag --test directionality
```

## Experiment Tracking

We use Weights & Biases (wandb) for experiment tracking:

```python
from src.utils.tracking import init_experiment, log_metrics

# Initialize experiment
run = init_experiment(
    experiment_name="directionality_test",
    config={"model": "gcn", "dataset": "mutag"}
)

# Log metrics
log_metrics({
    "faithfulness_score": 0.73,
    "directional_asymmetry": 0.45
})
```

## Related Work

- **CoT Faithfulness**: Per-step ablation in language model reasoning
- **GNN Explainability**: GNNExplainer, PGExplainer, SubgraphX
- **Causal Inference in GNNs**: Intervention-based evaluation
- **Constitutional AI**: Detecting unfaithful reasoning patterns

## Citation

```bibtex
@inproceedings{gnn-directionality-faithfulness,
  title={A Directional Approach to Faithfulness in Graph Neural Networks},
  author={Your Name},
  booktitle={CS 512 - Project},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or collaboration: [your email]
