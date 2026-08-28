"""
PRISM - K-Step Forward Simulation Engine
Autoregressively rolls out predicted future network states using the world
model and produces structured infiltration forecasts.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.prediction.attack_mapper import AttackStageMapper
from src.prediction.scoring import InfiltrationScorer, StepScore, RolloutResult
from src.utils.constants import K_STEP_DEFAULT, MITRE_STAGES_INV

logger = logging.getLogger("prism.prediction.simulator")


class KStepSimulator:
    """
    K-Step Forward Simulation Engine.

    Given a lookback window of observed network states [S_{t-L+1}, ..., S_t],
    autoregressively predicts K future states and computes infiltration
    probability, MITRE stage, and contributing features at each step.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        k_steps: int = K_STEP_DEFAULT,
        num_rollouts: int = 10,
        deterministic: bool = False,
        thresholds: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        model        : trained world model (StateTransformerWorldModel, etc.)
        device       : "cpu" | "cuda"
        k_steps      : number of future steps to simulate
        num_rollouts : number of stochastic rollouts for uncertainty estimation
        deterministic: if True, use predicted mean (no sampling)
        thresholds   : dict of alert thresholds
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.k_steps = k_steps
        self.num_rollouts = num_rollouts
        self.deterministic = deterministic

        self.mapper = AttackStageMapper()
        self.scorer = InfiltrationScorer(thresholds)

        logger.info(
            "KStepSimulator ready: k=%d, rollouts=%d, deterministic=%s, device=%s",
            k_steps, num_rollouts, deterministic, device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def simulate(
        self,
        state_sequence: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> RolloutResult:
        """
        Run K-step forward simulation from an observed state sequence.

        Parameters
        ----------
        state_sequence : np.ndarray, shape (L, D_state) or (1, L, D_state)
            Lookback window of normalised network state vectors.
        feature_names : list of feature names (for explainability)

        Returns
        -------
        RolloutResult with full forecast timeline.
        """
        state_seq = self._prepare_input(state_sequence)

        if self.deterministic or self.num_rollouts == 1:
            steps = self._single_rollout(state_seq, feature_names, stochastic=False)
            return self.scorer.score_rollout(steps)
        else:
            all_rollouts = []
            for r in range(self.num_rollouts):
                steps = self._single_rollout(
                    state_seq.clone(), feature_names, stochastic=True
                )
                all_rollouts.append(steps)
            return self.scorer.aggregate_ensemble(all_rollouts)

    def simulate_batch(
        self,
        state_sequences: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> list[RolloutResult]:
        """
        Simulate a batch of state sequences.

        Parameters
        ----------
        state_sequences : np.ndarray, shape (B, L, D_state)
        """
        results = []
        for i in range(state_sequences.shape[0]):
            result = self.simulate(
                state_sequences[i], feature_names
            )
            results.append(result)
        return results

    def simulate_from_file(
        self,
        states_npz_path: str,
        start_window: int = 0,
        feature_names: Optional[list[str]] = None,
    ) -> RolloutResult:
        """
        Load states from a .npz file and simulate from a given window.

        Parameters
        ----------
        states_npz_path : path to .npz file from StateBuilder
        start_window    : index of the window to start simulation from
        """
        data = np.load(states_npz_path)
        states = data["states"]
        lookback = self.model.lookback if hasattr(self.model, "lookback") else 20

        if start_window < lookback:
            start_window = lookback

        seq = states[start_window - lookback : start_window]  # (L, D)
        return self.simulate(seq, feature_names)

    # ------------------------------------------------------------------
    # Core rollout
    # ------------------------------------------------------------------
    def _single_rollout(
        self,
        state_seq: torch.Tensor,
        feature_names: Optional[list[str]],
        stochastic: bool = False,
    ) -> list[StepScore]:
        """
        Perform one K-step autoregressive rollout.

        At each step:
          1. Forward pass through world model
          2. Get infiltration prob, MITRE stage
          3. Sample next predicted state
          4. Shift buffer: drop oldest, append predicted state
          5. Repeat

        Returns
        -------
        list of StepScore, length K
        """
        steps = []
        buffer = state_seq.clone()  # (1, L, D_state)

        with torch.no_grad():
            for k in range(self.k_steps):
                out = self.model(buffer, return_attention=True)

                # --- Infiltration probability ---
                inf_probs = torch.softmax(out["pred_binary"], dim=-1)
                inf_prob = float(inf_probs[0, 1].item())

                # --- MITRE stage probabilities ---
                mitre_probs = torch.softmax(out["pred_mitre"], dim=-1)
                mitre_probs_np = mitre_probs[0].cpu().numpy().tolist()

                # --- Confidence (1 - entropy of binary prediction) ---
                entropy = -(
                    inf_probs * (inf_probs + 1e-10).log()
                ).sum(dim=-1)
                confidence = float(1.0 - entropy[0].item() / np.log(2))

                # --- Attention weights ---
                attn = (
                    out["attention_weights"][0].cpu().numpy()
                    if out.get("attention_weights") is not None
                    else None
                )

                # --- Predicted next state ---
                mean = out["pred_state_mean"]
                if stochastic:
                    logvar = out["pred_state_logvar"].clamp(-10.0, 2.0)
                    std = (0.5 * logvar).exp()
                    pred_state = mean + torch.randn_like(std) * std
                else:
                    pred_state = mean  # deterministic

                pred_state_np = pred_state[0].cpu().numpy()

                # --- Feature attribution (gradient-based) ---
                top_features = self._gradient_attribution(
                    buffer, out, feature_names, k
                )

                step_score = self.scorer.score_step(
                    step=k + 1,
                    infiltration_prob=inf_prob,
                    mitre_probs=mitre_probs_np,
                    confidence=confidence,
                    top_features=top_features,
                    attention_weights=attn,
                    predicted_state=pred_state_np,
                )
                steps.append(step_score)

                # --- Shift buffer ---
                buffer = self._shift_buffer(buffer, pred_state)

        return steps

    # ------------------------------------------------------------------
    # Gradient-based feature attribution
    # ------------------------------------------------------------------
    def _gradient_attribution(
        self,
        state_seq: torch.Tensor,
        out: dict,
        feature_names: Optional[list[str]],
        step: int,
    ) -> list[dict]:
        """
        Compute gradient of infiltration probability w.r.t. input features.
        Returns top-10 contributing features sorted by |gradient|.
        """
        if not state_seq.requires_grad:
            x = state_seq.clone().detach().requires_grad_(True)
            try:
                out2 = self.model(x)
                inf_prob = torch.softmax(out2["pred_binary"], dim=-1)[:, 1].sum()
                inf_prob.backward()
                if x.grad is not None:
                    # Average gradient magnitude over sequence length
                    grads = x.grad[0].abs().mean(dim=0).cpu().numpy()  # (D_state,)
                    return self._format_feature_attrs(grads, feature_names)
            except Exception:
                pass
        return []

    @staticmethod
    def _format_feature_attrs(
        grads: np.ndarray,
        feature_names: Optional[list[str]],
    ) -> list[dict]:
        """Format gradient attributions into top-10 list."""
        top_idx = np.argsort(grads)[::-1][:10]
        result = []
        for i, idx in enumerate(top_idx):
            name = (
                feature_names[idx]
                if feature_names and idx < len(feature_names)
                else f"feature_{idx}"
            )
            result.append({
                "rank": i + 1,
                "feature": name,
                "attribution": float(grads[idx]),
            })
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _prepare_input(self, state_sequence: np.ndarray) -> torch.Tensor:
        """Convert numpy array to (1, L, D) float32 tensor on device."""
        arr = np.array(state_sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]   # (1, L, D)
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(
                f"state_sequence must be 2D or 3D, got shape {arr.shape}"
            )
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    @staticmethod
    def _shift_buffer(
        buffer: torch.Tensor, new_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Slide the lookback window: drop oldest state, append new_state.

        buffer    : (1, L, D_state)
        new_state : (1, D_state)
        Returns   : (1, L, D_state)
        """
        return torch.cat(
            [buffer[:, 1:, :], new_state.unsqueeze(1)],
            dim=1,
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def format_report(
        self, result: RolloutResult, feature_names: Optional[list[str]] = None
    ) -> str:
        """
        Generate a human-readable text report from a RolloutResult.
        """
        lines = [
            "=" * 70,
            "  PRISM — Infiltration Forecast Report",
            "=" * 70,
            f"  Overall Alert  : {result.overall_alert.upper()}",
            f"  Trajectory      : {result.trajectory_summary}",
            f"  Peak Prob       : {result.peak_infiltration_prob:.1%} at step {result.peak_step}",
            f"  Escalating      : {'YES' if result.is_escalating else 'No'}",
            f"  Early Warning   : {result.early_warning_steps} steps ahead",
            "-" * 70,
            "  Step-by-Step Forecast:",
        ]

        for step in result.steps:
            alert_str = f"[{step.alert_level.upper()}]" if step.alert_level != "none" else ""
            lines.append(
                f"  Step {step.step:2d}: P(attack)={step.infiltration_prob:.3f} "
                f"| Stage: {step.mitre_stage:<20} "
                f"| Conf: {step.confidence:.2f} {alert_str}"
            )
            if step.top_features:
                top3 = step.top_features[:3]
                feat_str = ", ".join(
                    f"{f['feature']}({f['attribution']:.3f})" for f in top3
                )
                lines.append(f"          Top features: {feat_str}")

        lines.append("=" * 70)
        return "\n".join(lines)
