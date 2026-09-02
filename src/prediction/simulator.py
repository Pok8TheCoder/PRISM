"""
K-Step Forward Rollout Simulator: Autoregressively simulates future network states
and estimates forward infiltration risk trajectories.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from src.utils.constants import MITRE_STAGES_INV
from src.utils.logger import setup_logger

logger = setup_logger("Simulator")


def compute_entropy(probs: torch.Tensor) -> float:
    """Computes Shannon entropy of a probability distribution tensor."""
    eps = 1e-12
    p = torch.clamp(probs, min=eps, max=1.0)
    entropy = -torch.sum(p * torch.log2(p), dim=-1)
    return float(entropy.mean().item())


class RolloutSimulator:
    """
    Simulates K future time steps forward in time using the PRISM World Model.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def rollout(
        self,
        current_sequence: torch.Tensor,
        K: int = 10,
        stochastic: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rolls out K future states autoregressively from the given context window.

        Args:
            current_sequence: Tensor of shape (1, L, D_state) or (L, D_state)
            K: Number of future time windows to simulate forward
            stochastic: If True, samples states from predicted Gaussian distribution;
                        if False, takes deterministic mean rollout.

        Returns:
            List of K dictionaries containing future predictions at each step t+k.
        """
        if current_sequence.ndim == 2:
            current_sequence = current_sequence.unsqueeze(0)

        buffer = current_sequence.clone().to(self.device)
        trajectory = []

        with torch.no_grad():
            for k in range(1, K + 1):
                (
                    pred_mean,
                    pred_logvar,
                    pred_attack_logits,
                    pred_mitre_logits,
                    pred_frac,
                    attn_weights
                ) = self.model(buffer)

                # Next state simulation
                if stochastic:
                    std = torch.exp(0.5 * pred_logvar)
                    eps = torch.randn_like(std)
                    next_state = pred_mean + eps * std
                else:
                    next_state = pred_mean  # (1, D_state)

                # Probabilities & Confidence
                attack_probs = torch.softmax(pred_attack_logits, dim=-1)[0]
                mitre_probs = torch.softmax(pred_mitre_logits, dim=-1)[0]
                
                infil_prob = float(attack_probs[1].item())
                mitre_code = int(torch.argmax(mitre_probs).item())
                mitre_name = MITRE_STAGES_INV.get(mitre_code, "Unknown")
                attack_fraction = float(pred_frac[0, 0].item())
                
                # Confidence score based on prediction certainty
                confidence = max(0.0, 1.0 - (compute_entropy(attack_probs) / 1.0))

                step_record = {
                    "step": k,
                    "infiltration_prob": infil_prob,
                    "attack_fraction": attack_fraction,
                    "mitre_code": mitre_code,
                    "mitre_stage": mitre_name,
                    "mitre_probs": mitre_probs.cpu().numpy(),
                    "confidence": confidence,
                    "predicted_state": next_state[0].cpu().numpy(),
                    "attention_weights": attn_weights[0].cpu().numpy() if attn_weights is not None else None
                }
                trajectory.append(step_record)

                # Autoregressive shift: drop oldest state, append newly simulated state
                buffer = torch.cat([
                    buffer[:, 1:, :],
                    next_state.unsqueeze(1)
                ], dim=1)

        return trajectory

    def ensemble_rollout(
        self,
        current_sequence: torch.Tensor,
        K: int = 10,
        num_simulations: int = 10
    ) -> Dict[str, Any]:
        """
        Runs multiple stochastic simulations to quantify future trajectory uncertainty.
        """
        all_trajectories = [
            self.rollout(current_sequence, K=K, stochastic=True)
            for _ in range(num_simulations)
        ]

        # Aggregate across simulations
        mean_probs = []
        p05_probs = []
        p95_probs = []

        for step_idx in range(K):
            step_infil_probs = [
                traj[step_idx]["infiltration_prob"]
                for traj in all_trajectories
            ]
            mean_probs.append(float(np.mean(step_infil_probs)))
            p05_probs.append(float(np.percentile(step_infil_probs, 5)))
            p95_probs.append(float(np.percentile(step_infil_probs, 95)))

        return {
            "K": K,
            "num_simulations": num_simulations,
            "mean_infil_probs": mean_probs,
            "p05_probs": p05_probs,
            "p95_probs": p95_probs,
            "trajectories": all_trajectories
        }
