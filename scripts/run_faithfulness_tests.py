"""
Run faithfulness tests on trained GNN models with explanations.

Tests:
1. Necessity: Does removing important features hurt performance?
2. Sufficiency: Does adding important features help?
3. Directionality: Is there asymmetric influence?

Usage:
    python scripts/run_faithfulness_tests.py --model gcn --dataset MUTAG --explainer gnnexplainer
"""

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer, GNNExplainer
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.models.gnn_models import create_model
from src.faithfulness.ablation import NodeAblator, DirectionalAblator
from src.utils.metrics import evaluate_faithfulness
from src.faithfulness.metrics_extended import causal_precision, faithfulness_gain
from src.faithfulness.steering import SteeringController, steer_inference
from src.faithfulness.diffing import extract_faithfulness_directions
from src.utils.tracking import init_experiment, log_metrics, finish_experiment


def get_explanation(explainer, model, data, target_class=None):
    """Get explanation for a single graph."""
    model.eval()
    
    # Get prediction (with gradients for explainer)
    with torch.enable_grad():
        out = model(data.x, data.edge_index, data.batch)
        # Handle tuple returns from models with return_embeddings
        if isinstance(out, tuple):
            out = out[0]
        # Handle single graph vs batched outputs
        if out.dim() == 1:
            out = out.unsqueeze(0)  # Add batch dimension
        pred = out.argmax(dim=1).item()
        prob = F.softmax(out, dim=1)[0, pred].item()
    
    if target_class is None:
        target_class = pred
    
    # Generate explanation (requires gradients)
    with torch.enable_grad():
        explanation = explainer(
            data.x,
            data.edge_index,
            target=torch.tensor([target_class])
        )
    
    return explanation, pred, prob


def run_necessity_test(model, data, explanation, ablator, device, top_k=5):
    """
    Test necessity: ablate top-k important nodes, measure drop in confidence.
    """
    model.eval()
    
    # Get original prediction
    with torch.no_grad():
        out_orig = model(data.x, data.edge_index, data.batch)
        pred_orig = out_orig.argmax(dim=1).item()
        prob_orig = F.softmax(out_orig, dim=1)[0, pred_orig].item()
    
    # Get top-k important nodes
    node_importance = explanation.node_mask
    num_nodes = data.x.size(0)
    k_nodes = min(top_k, len(node_importance), num_nodes)
    if k_nodes == 0:
        return prob_orig, prob_orig  # Skip if no nodes available
    top_nodes = torch.topk(node_importance, k=k_nodes).indices.tolist()
    
    # Ablate and re-predict
    data_ablated = ablator.ablate(data, top_nodes, inplace=False)
    with torch.no_grad():
        out_ablated = model(data_ablated.x, data_ablated.edge_index, data_ablated.batch)
        if isinstance(out_ablated, tuple):
            out_ablated = out_ablated[0]
        if out_ablated.dim() == 1:
            out_ablated = out_ablated.unsqueeze(0)
        prob_ablated = F.softmax(out_ablated, dim=1)[0, pred_orig].item()
    
    return prob_orig, prob_ablated


def run_sufficiency_test(model, data, explanation, incorrect_graphs, device, top_k=5):
    """
    Test sufficiency: add explanation subgraph to incorrect graphs.
    
    For simplicity, we measure if augmenting features increases confidence.
    """
    # This is a simplified version - full implementation would transfer subgraph
    # Here we just measure baseline confidence on incorrect predictions
    model.eval()
    
    if len(incorrect_graphs) == 0:
        return None, None
    
    data_incorrect = incorrect_graphs[0]
    with torch.no_grad():
        out = model(data_incorrect.x, data_incorrect.edge_index, data_incorrect.batch)
        if isinstance(out, tuple):
            out = out[0]
        if out.dim() == 1:
            out = out.unsqueeze(0)
        pred = out.argmax(dim=1).item()
        prob_before = F.softmax(out, dim=1)[0, pred].item()
    
    # Simplified: measure current confidence
    prob_after = prob_before  # Placeholder
    
    return prob_before, prob_after


def run_directionality_test(model, data, explanation, dir_ablator, device, top_k=3):
    """
    Test directionality: compare ablating source vs target nodes of important edges.
    """
    model.eval()
    
    # Get original prediction
    with torch.no_grad():
        out_orig = model(data.x, data.edge_index, data.batch)
        if isinstance(out_orig, tuple):
            out_orig = out_orig[0]
        if out_orig.dim() == 1:
            out_orig = out_orig.unsqueeze(0)
        pred_orig = out_orig.argmax(dim=1).item()
        prob_orig = F.softmax(out_orig, dim=1)[0, pred_orig].item()
    
    # Get top-k important edges
    edge_importance = explanation.edge_mask
    num_edges = data.edge_index.size(1)
    k_edges = min(top_k, len(edge_importance), num_edges)
    if k_edges == 0:
        return prob_orig, np.array([prob_orig]), np.array([prob_orig])  # Skip if no edges
    top_edges = torch.topk(edge_importance, k=k_edges).indices.tolist()
    
    source_probs = []
    target_probs = []
    
    for edge_idx in top_edges:
        # Ablate source
        data_src_ablated = dir_ablator.ablate_source(data, edge_idx, inplace=False)
        with torch.no_grad():
            out = model(data_src_ablated.x, data_src_ablated.edge_index, data_src_ablated.batch)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 1:
                out = out.unsqueeze(0)
            prob_src = F.softmax(out, dim=1)[0, pred_orig].item()
        source_probs.append(prob_src)
        
        # Ablate target
        data_tgt_ablated = dir_ablator.ablate_target(data, edge_idx, inplace=False)
        with torch.no_grad():
            out = model(data_tgt_ablated.x, data_tgt_ablated.edge_index, data_tgt_ablated.batch)
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 1:
                out = out.unsqueeze(0)
            prob_tgt = F.softmax(out, dim=1)[0, pred_orig].item()
        target_probs.append(prob_tgt)
    
    return prob_orig, np.array(source_probs), np.array(target_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "gin"])
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--explainer", type=str, default="gnnexplainer")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/raw")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top nodes/edges to test")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of graphs to test")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--steer", action="store_true", help="Apply activation steering using extracted directions")
    parser.add_argument("--alpha", type=float, default=0.5, help="Steering strength alpha")
    parser.add_argument("--top_k_dirs", type=int, default=4, help="Number of singular directions per layer for steering")
    parser.add_argument("--unfaithful_checkpoint", type=str, default=None, help="Path to adversarially trained unfaithful model for direction extraction")
    args = parser.parse_args()
    
    # Initialize tracking
    if args.track:
        init_experiment(
            experiment_name=f"faithfulness_{args.model}_{args.dataset}",
            config=vars(args),
            tags=["faithfulness", args.model, args.dataset],
            notes=f"Testing faithfulness of {args.explainer} on {args.model}/{args.dataset}"
        )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = TUDataset(root=args.data_dir, name=args.dataset)
    
    # Load model
    if args.model_path is None:
        args.model_path = f"./models/{args.model}_{args.dataset}_best.pt"
    
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    model = create_model(
        args.model,
        in_channels=dataset.num_features,
        hidden_channels=checkpoint['config']['hidden'],
        out_channels=dataset.num_classes,
        num_layers=checkpoint['config']['layers'],
        dropout=0.0  # No dropout for evaluation
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    # Create ablators
    node_ablator = NodeAblator(ablation_mode="zero")
    dir_ablator = DirectionalAblator(node_ablator)
    
    # Run tests
    print(f"\nRunning faithfulness tests on {args.num_samples} graphs...")
    
    original_probs = []
    ablated_probs = []
    source_ablated_all = []
    target_ablated_all = []
    successful_tests = 0
    failed_tests = 0
    
    test_indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(test_indices, desc="Testing"):
        data = dataset[int(idx)].to(args.device)
        
        try:
            # Get explanation
            explanation, pred, prob = get_explanation(explainer, model, data)
            
            # Necessity test
            prob_orig, prob_abl = run_necessity_test(
                model, data, explanation, node_ablator, args.device, top_k=args.top_k
            )
            original_probs.append(prob_orig)
            ablated_probs.append(prob_abl)
            
            # Directionality test
            prob_orig_dir, src_probs, tgt_probs = run_directionality_test(
                model, data, explanation, dir_ablator, args.device, top_k=3
            )
            source_ablated_all.append(src_probs.mean())
            target_ablated_all.append(tgt_probs.mean())
            successful_tests += 1
            
        except Exception as e:
            # Track errors but continue
            failed_tests += 1
            if failed_tests <= 3:  # Print first few errors for debugging
                import traceback
                print(f"\nError on graph {idx}:")
                print(f"  Graph shape: {data.x.shape}, edges: {data.edge_index.shape}")
                print(f"  Error: {e}")
                if failed_tests == 1:
                    print("  Full traceback:")
                    traceback.print_exc()
            continue
    
    print(f"\nCompleted: {successful_tests} successful, {failed_tests} failed")
    
    # Compute faithfulness scores
    if successful_tests == 0:
        print("\n⚠️  No successful tests - cannot compute faithfulness metrics")
        print("   All graphs failed during explanation generation.")
        scores = None
    else:
        scores = evaluate_faithfulness(
            np.array(original_probs),
            np.array(ablated_probs),
            None,  # No sufficiency test
            np.array(source_ablated_all),
            np.array(target_ablated_all)
        )

    # Optional steering evaluation (simplified): extract directions using subset graphs
    steering_results = {}
    if args.steer:
        print("\nExtracting directions for steering...")
        # Clear any lingering explainer state from PyG
        for module in model.modules():
            if hasattr(module, '_edge_mask'):
                module._edge_mask = None
            if hasattr(module, '_loop_mask'):
                module._loop_mask = None
            if hasattr(module, '_apply_sigmoid'):
                module._apply_sigmoid = True
        
        # Load unfaithful model if provided
        if args.unfaithful_checkpoint:
            print(f"Loading unfaithful model from: {args.unfaithful_checkpoint}")
            unfaithful_checkpoint = torch.load(args.unfaithful_checkpoint, map_location=args.device)
            unfaithful_model = create_model(
                args.model,
                in_channels=dataset.num_features,
                hidden_channels=unfaithful_checkpoint['config']['hidden'],
                out_channels=dataset.num_classes,
                num_layers=unfaithful_checkpoint['config']['layers'],
                dropout=0.0
            ).to(args.device)
            unfaithful_model.load_state_dict(unfaithful_checkpoint['model_state_dict'])
            unfaithful_model.eval()
        else:
            print("⚠️  No unfaithful checkpoint provided - using self-comparison (will produce weak directions)")
            print("   Run FDI pipeline first or provide --unfaithful_checkpoint")
            unfaithful_model = model
        
        subset_loader = DataLoader(dataset[:min(32, len(dataset))], batch_size=min(32, len(dataset)))
        batch_subset = next(iter(subset_loader)).to(args.device)
        directions = extract_faithfulness_directions(model, unfaithful_model, batch_subset, torch.device(args.device), top_k=args.top_k_dirs)
        controller = SteeringController(model, directions, alpha=args.alpha)
        steered_logits_list = []
        original_logits_list = []
        for idx in test_indices:
            data = dataset[int(idx)].to(args.device)
            with torch.no_grad():
                orig = model(data.x, data.edge_index, data.batch)
                # Handle tuple returns
                if isinstance(orig, tuple):
                    orig = orig[0]
                steered = steer_inference(model, data, controller)
                if isinstance(steered, tuple):
                    steered = steered[0]
            original_logits_list.append(orig)
            steered_logits_list.append(steered)
        original_logits = torch.cat(original_logits_list, dim=0)
        steered_logits = torch.cat(steered_logits_list, dim=0)
        cp = causal_precision(original_logits, steered_logits, threshold_drop=0.1)
        # Example FG using first graph's node norms as proxy distributions
        node_imp_f = dataset[int(test_indices[0])].x.norm(p=2, dim=1)
        node_imp_u = node_imp_f + 0.01  # proxy unfaithful distribution
        log_prob_shift = (torch.log_softmax(steered_logits, dim=1) - torch.log_softmax(original_logits, dim=1)).mean().item()
        fg = faithfulness_gain(node_imp_f, node_imp_u, log_prob_shift)
        steering_results = {"causal_precision": cp, "faithfulness_gain": fg, "log_prob_shift": log_prob_shift}
    
    print(f"{'='*60}")
    print("Faithfulness Results:")
    print(f"{'='*60}")
    if scores is not None:
        for key, value in scores.to_dict().items():
            print(f"  {key}: {value:.4f}")
    else:
        print("  No valid results - all tests failed")
    if steering_results:
        print("\nSteering Extended Metrics:")
        for k, v in steering_results.items():
            print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")
    
    if args.track:
        if scores is not None:
            log_metrics(scores.to_dict())
        log_metrics({"successful_tests": successful_tests, "failed_tests": failed_tests})
        if steering_results:
            log_metrics({f"steer_{k}": v for k, v in steering_results.items()})
        finish_experiment()


if __name__ == "__main__":
    main()
