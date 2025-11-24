"""
Experiment tracking utilities for GNN faithfulness experiments.

Integrates with Weights & Biases for tracking metrics, configurations, and artifacts.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def init_experiment(
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None,
    project: str = "gnn-faithfulness",
    entity: Optional[str] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialize experiment tracking.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration dictionary
        project: WandB project name
        entity: WandB entity (username/team)
        tags: List of tags for the experiment
        notes: Experiment description
        
    Returns:
        wandb run object if available, None otherwise
    """
    if WANDB_AVAILABLE:
        run = wandb.init(
            project=project,
            entity=entity,
            name=experiment_name,
            config=config or {},
            tags=tags or [],
            notes=notes,
        )
        print(f"✓ Experiment tracking initialized: {run.url}")
        return run
    else:
        print(f"✓ Experiment: {experiment_name}")
        if config:
            print(f"  Config: {config}")
        return None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Log metrics to experiment tracker.
    
    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(metrics, step=step)
    else:
        step_str = f" (step {step})" if step is not None else ""
        print(f"Metrics{step_str}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


def log_graph_example(
    graph_id: str,
    prediction: str,
    explanation: Dict[str, Any],
    faithfulness_score: float,
    step: Optional[int] = None
):
    """
    Log a graph example with its explanation and faithfulness score.
    
    Args:
        graph_id: Graph identifier
        prediction: Model prediction
        explanation: Explanation dictionary (nodes, edges, scores)
        faithfulness_score: Faithfulness score (0-1)
        step: Optional step number
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        table = wandb.Table(
            columns=["Graph ID", "Prediction", "Explanation", "Faithfulness"],
            data=[[graph_id, prediction, str(explanation), faithfulness_score]]
        )
        wandb.log({"examples": table}, step=step)
    else:
        step_str = f" (step {step})" if step is not None else ""
        print(f"\nExample{step_str}:")
        print(f"  Graph: {graph_id}")
        print(f"  Prediction: {prediction}")
        print(f"  Faithfulness: {faithfulness_score:.4f}")


def save_artifact(filepath: str, name: str, artifact_type: str = "model"):
    """
    Save an artifact (model, results, etc).
    
    Args:
        filepath: Path to file
        name: Artifact name
        artifact_type: Type of artifact ('model', 'dataset', 'results')
    """
    if WANDB_AVAILABLE and wandb.run is not None:
        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_file(filepath)
        wandb.log_artifact(artifact)
        print(f"✓ Artifact saved: {name}")
    else:
        print(f"✓ Artifact {name} saved locally at {filepath}")


def finish_experiment():
    """Finish experiment tracking."""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
        print("✓ Experiment finished")
    else:
        print("✓ Experiment completed")


# Example usage
if __name__ == "__main__":
    # Initialize
    run = init_experiment(
        experiment_name="test_gnn_faithfulness",
        config={"model": "gcn", "dataset": "mutag", "explainer": "gnnexplainer"},
        tags=["test", "baseline"],
        notes="Testing faithfulness metrics on MUTAG dataset"
    )
    
    # Log metrics
    for epoch in range(5):
        log_metrics({
            "train_loss": 1.0 / (epoch + 1),
            "train_acc": 0.5 + epoch * 0.1,
            "faithfulness_score": 0.6 + epoch * 0.05
        }, step=epoch)
    
    # Log example
    log_graph_example(
        graph_id="graph_42",
        prediction="toxic",
        explanation={"important_nodes": [0, 3, 7], "edge_weights": [0.9, 0.7, 0.6]},
        faithfulness_score=0.82
    )
    
    # Finish
    finish_experiment()
