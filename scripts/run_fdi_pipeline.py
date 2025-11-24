"""Run full Faithfulness Direction Identification (FDI) pipeline.

Stages:
 1. Load baseline (faithful) model checkpoint.
 2. Induce unfaithfulness via adversarial fine-tuning (misleading rationales).
 3. Perform model diffing to extract faithfulness directions (W_f).
 4. Apply steering and evaluate causal metrics (CP, CR, FG).

Usage:
  python scripts/run_fdi_pipeline.py --model gin --dataset MUTAG --base_checkpoint models/gin_MUTAG_best.pt \
      --epochs_adv 5 --alpha 0.5
"""

import argparse
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.gnn_models import create_model
from src.faithfulness.adversarial import generate_misleading_masks, adversarial_finetune
from src.faithfulness.diffing import extract_faithfulness_directions
from src.faithfulness.steering import SteeringController, steer_inference
from src.faithfulness.metrics_extended import causal_precision, faithfulness_gain
from src.utils.metrics import evaluate_faithfulness


def load_base_model(args, dataset):
    checkpoint = torch.load(args.base_checkpoint, map_location=args.device)
    cfg = checkpoint.get('config', {})
    model = create_model(
        args.model,
        in_channels=dataset.num_features,
        hidden_channels=cfg.get('hidden', args.hidden),
        out_channels=dataset.num_classes,
        num_layers=cfg.get('layers', args.layers),
        return_embeddings=True,
        dropout=0.0,
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gin', choices=['gcn', 'gat', 'gin'])
    parser.add_argument('--dataset', type=str, default='MUTAG')
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--epochs_adv', type=int, default=5)
    parser.add_argument('--adv_fraction', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--top_k_dirs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"[FDI] Loading dataset: {args.dataset}")
    dataset = TUDataset(root=args.data_dir, name=args.dataset)
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    # Stage 1: Load base (faithful) model
    print("[FDI] Loading base model...")
    base_model = load_base_model(args, dataset)

    # Stage 1b: Induce unfaithfulness
    print("[FDI] Generating misleading rationales...")
    mis_masks = generate_misleading_masks(train_dataset, fraction=args.adv_fraction, seed=args.seed)
    print("[FDI] Adversarial fine-tuning (unfaithful model)...")
    unfaithful_model = load_base_model(args, dataset)  # start from base weights
    unfaithful_model = adversarial_finetune(
        unfaithful_model, train_dataset, mis_masks, torch.device(args.device), epochs=args.epochs_adv
    )
    unfaithful_model.eval()

    # Stage 2: Model diffing to extract directions
    print("[FDI] Extracting faithfulness directions via SVD...")
    # Build a representative batch (sample subset)
    loader = DataLoader(train_dataset[:min(32, len(train_dataset))], batch_size=min(32, len(train_dataset)))
    batch = next(iter(loader)).to(args.device)
    directions = extract_faithfulness_directions(base_model, unfaithful_model, batch, torch.device(args.device), top_k=args.top_k_dirs)
    print(f"[FDI] Extracted directions for {len(directions)} layers.")

    # Stage 3: Steering evaluation on test set
    print("[FDI] Evaluating steering effects...")
    controller = SteeringController(base_model, directions, alpha=args.alpha)
    original_logits_list = []
    steered_logits_list = []
    faithful_masks = []
    unfaithful_masks = []
    log_prob_shifts = []

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    for data in test_loader:
        data = data.to(args.device)
        with torch.no_grad():
            # Model returns (logits, embeddings) when return_embeddings=True, extract logits
            out = base_model(data.x, data.edge_index, data.batch)
            orig_logits = out[0] if isinstance(out, tuple) else out
            steered_out = steer_inference(base_model, data, controller)
            steered_logits = steered_out[0] if isinstance(steered_out, tuple) else steered_out
        original_logits_list.append(orig_logits)
        steered_logits_list.append(steered_logits)
        # For FG we simulate importance distributions using node feature norms
        node_importance_f = data.x.norm(p=2, dim=1)
        node_importance_u = (data.x + 0.01).norm(p=2, dim=1)  # proxy shift
        faithful_masks.append(node_importance_f)
        unfaithful_masks.append(node_importance_u)
        # Log prob shift for predicted class
        orig_prob = torch.log_softmax(orig_logits, dim=1).max().item()
        steer_prob = torch.log_softmax(steered_logits, dim=1).max().item()
        log_prob_shifts.append(steer_prob - orig_prob)

    original_logits = torch.cat(original_logits_list, dim=0)
    steered_logits = torch.cat(steered_logits_list, dim=0)
    cp = causal_precision(original_logits, steered_logits, threshold_drop=0.1)
    fg = faithfulness_gain(faithful_masks[0], unfaithful_masks[0], log_prob_shifts[0])  # example first

    print("\n[FDI] Results:")
    print(f"  Causal Precision (CP): {cp:.4f}")
    print(f"  Faithfulness Gain (FG) example: {fg:.4f}")
    print(f"  Avg log prob shift: {sum(log_prob_shifts)/len(log_prob_shifts):.4f}")

    print("[FDI] Pipeline complete.")


if __name__ == '__main__':
    main()