"""
PRISM - Ablation Study
Systematically measures the contribution of each model component.
"""

import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd

logger = logging.getLogger("prism.evaluation.ablation")


class AblationStudy:
    """
    Ablation study for PRISM world model components.

    Tests:
      1. Full model (baseline)
      2. Without dynamics loss (lambda_dynamics=0)
      3. Without MITRE head (no stage supervision)
      4. Without packet-level features (flow-only)
      5. Reduced lookback (L=5 vs L=20)
      6. LSTM vs Transformer architecture
    """

    def __init__(self, base_config, device: str = "cpu"):
        self.base_config = base_config
        self.device = device
        self.results = {}

    def run_ablation(
        self,
        train_loader,
        val_loader,
        test_loader,
        n_epochs: int = 10,
        output_dir: str = "results/ablation",
    ) -> pd.DataFrame:
        """
        Train and evaluate each ablation variant.

        Parameters
        ----------
        train_loader, val_loader, test_loader : DataLoaders
        n_epochs  : training epochs per variant (keep short for ablation)
        output_dir: where to save results

        Returns
        -------
        pd.DataFrame with one row per ablation variant
        """
        from src.models.world_model import StateTransformerWorldModel, LSTMWorldModel
        from src.models.components import MultiTaskLoss
        from src.evaluation.metrics import evaluate_model

        os.makedirs(output_dir, exist_ok=True)

        ablations = self._define_ablations()

        for name, config_overrides in ablations.items():
            logger.info("Running ablation: %s", name)
            try:
                cfg = deepcopy(self.base_config)
                for k, v in config_overrides.items():
                    if hasattr(cfg.model, k):
                        setattr(cfg.model, k, v)
                    elif hasattr(cfg.train, k):
                        setattr(cfg.train, k, v)

                # Build model
                arch = config_overrides.get("architecture", cfg.model.architecture)
                if arch == "lstm":
                    model = LSTMWorldModel(
                        d_state=cfg.model.d_state,
                        d_model=cfg.model.d_model,
                        lstm_layers=cfg.model.lstm_layers,
                        lookback=cfg.data.lookback,
                        dropout=cfg.model.dropout,
                    )
                else:
                    model = StateTransformerWorldModel(
                        d_state=cfg.model.d_state,
                        d_model=cfg.model.d_model,
                        n_layers=cfg.model.n_layers,
                        n_heads=cfg.model.n_heads,
                        lookback=cfg.data.lookback,
                        dropout=cfg.model.dropout,
                    )

                # Loss config
                lambda_d = config_overrides.get("lambda_dynamics", cfg.train.lambda_dynamics)
                lambda_i = config_overrides.get("lambda_infiltration", cfg.train.lambda_infiltration)
                lambda_m = config_overrides.get("lambda_mitre", cfg.train.lambda_mitre)
                loss_fn = MultiTaskLoss(lambda_d, lambda_i, lambda_m)

                # Quick training
                metrics = self._train_and_evaluate(
                    model, loss_fn, train_loader, val_loader,
                    test_loader, n_epochs, cfg
                )
                self.results[name] = metrics

                # Save partial results
                self._save_partial(output_dir, name, metrics)

            except Exception as e:
                logger.error("Ablation '%s' failed: %s", name, e)
                self.results[name] = {"error": str(e)}

        df = self._build_table()
        df.to_csv(os.path.join(output_dir, "ablation_results.csv"))
        logger.info("Ablation complete. Results saved to %s", output_dir)
        return df

    # ------------------------------------------------------------------
    # Ablation definitions
    # ------------------------------------------------------------------
    def _define_ablations(self) -> dict:
        """Each entry: name -> dict of config overrides."""
        return {
            "Full Model": {},
            "No Dynamics Loss": {"lambda_dynamics": 0.0},
            "No MITRE Head": {"lambda_mitre": 0.0},
            "No Stage Supervision": {"lambda_mitre": 0.0, "lambda_infiltration": 0.0},
            "LSTM Architecture": {"architecture": "lstm"},
            "Short Lookback (L=5)": {"lookback": 5},
            "Shallow (2 layers)": {"n_layers": 2},
            "No Dropout": {"dropout": 0.0},
        }

    def _train_and_evaluate(
        self, model, loss_fn, train_loader, val_loader,
        test_loader, n_epochs, cfg
    ) -> dict:
        """Train a model variant and return test metrics."""
        import torch
        from src.evaluation.metrics import evaluate_model

        device = self.device
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )

        best_val_f1 = 0.0
        best_state = None

        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                state_seq = batch["state_seq"].to(device)
                next_state = batch["next_state"].to(device)
                y_bin = batch["label_binary"].to(device)
                y_mit = batch["label_mitre"].to(device)

                optimizer.zero_grad()
                out = model(state_seq)
                losses = loss_fn(
                    out["pred_state_mean"], out["pred_state_logvar"], next_state,
                    out["pred_binary"], y_bin,
                    out["pred_mitre"], y_mit,
                )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Quick val check
            model.eval()
            val_metrics = evaluate_model(model, val_loader, device)
            val_f1 = val_metrics["binary"]["f1"]
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = deepcopy(model.state_dict())

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        return evaluate_model(model, test_loader, device)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_table(self) -> pd.DataFrame:
        rows = []
        for name, metrics in self.results.items():
            if "error" in metrics:
                rows.append({"Variant": name, "Error": metrics["error"]})
                continue
            rows.append({
                "Variant": name,
                "F1": round(metrics["binary"].get("f1", 0), 4),
                "Precision": round(metrics["binary"].get("precision", 0), 4),
                "Recall": round(metrics["binary"].get("recall", 0), 4),
                "FPR": round(metrics["binary"].get("fpr", 0), 4),
                "MITRE_F1": round(metrics["mitre"].get("f1_macro", 0), 4),
                "Dynamics_MSE": round(metrics["dynamics"].get("mse", 0) or 0, 6),
            })
        return pd.DataFrame(rows).set_index("Variant")

    def _save_partial(self, output_dir: str, name: str, metrics: dict) -> None:
        path = os.path.join(output_dir, f"ablation_{name.replace(' ', '_')}.json")
        import json
        safe = {
            "f1": metrics.get("binary", {}).get("f1"),
            "precision": metrics.get("binary", {}).get("precision"),
            "recall": metrics.get("binary", {}).get("recall"),
            "fpr": metrics.get("binary", {}).get("fpr"),
            "mitre_f1": metrics.get("mitre", {}).get("f1_macro"),
            "dynamics_mse": metrics.get("dynamics", {}).get("mse"),
        }
        with open(path, "w") as f:
            json.dump(safe, f, indent=2)
