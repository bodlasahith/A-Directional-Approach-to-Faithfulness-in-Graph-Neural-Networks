"""Activation steering utilities.

Implements layer hooks applying:
    h_tilde = h + alpha * W_f @ h
where W_f is a learned/extracted faithfulness direction matrix per layer.

Supports scheduling alpha over steps for gradual steering strength.
"""

from typing import Dict, Callable, Optional
import torch
import torch.nn as nn


class SteeringController:
    """Manage steering parameters and hooks for a model."""

    def __init__(
        self,
        model: nn.Module,
        direction_matrices: Dict[int, torch.Tensor],
        alpha: float = 0.5,
        layer_map: Optional[Dict[int, str]] = None,
    ):
        """Args:
            model: GNN model with named_modules.
            direction_matrices: layer_index -> W_f [hidden_dim, k]
            alpha: Steering strength scalar.
            layer_map: Optional mapping from index to module name if indices differ.
        """
        self.model = model
        self.directions = direction_matrices
        self.alpha = alpha
        self.layer_map = layer_map or {}
        self.handles = []

    def _hook_factory(self, layer_idx: int) -> Callable:
        W = self.directions[layer_idx]  # [hidden_dim, k]
        # Precompute projector W W^T for efficiency.
        projector = (W @ W.T).detach()  # [hidden_dim, hidden_dim]

        def hook(module, inputs, output):
            # Assume output is node embedding tensor [num_nodes, hidden_dim]
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if out.dim() != 2:
                return output  # skip non-standard shapes
            steered = out + self.alpha * (out @ projector)
            if isinstance(output, tuple):
                new_output = (steered,) + output[1:]
                return new_output
            return steered

        return hook

    def apply(self):
        """Register forward hooks on layers with available directions."""
        for idx, W in self.directions.items():
            # Determine module name: use explicit map or ordinal matching
            target_name = self.layer_map.get(idx)
            if target_name is None:
                # Fallback: iterate and match by order of convs
                # Expect model has attribute 'convs'
                if hasattr(self.model, 'convs') and idx < len(self.model.convs):
                    module = self.model.convs[idx]
                else:
                    continue
            else:
                # Find module by name
                module = dict(self.model.named_modules()).get(target_name, None)
                if module is None:
                    continue
            handle = module.register_forward_hook(self._hook_factory(idx))
            self.handles.append(handle)

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def set_alpha(self, alpha: float):
        self.alpha = alpha


def steer_inference(model: nn.Module, batch, controller: SteeringController):
    """Convenience wrapper to run model inference with steering applied."""
    controller.apply()
    try:
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
    finally:
        controller.clear()
    return out


if __name__ == "__main__":
    print("steering module loaded. Use SteeringController within pipeline.")