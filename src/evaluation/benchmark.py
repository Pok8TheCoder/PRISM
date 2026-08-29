"""
PRISM - Benchmark Runner
Compares world model against static baselines on the same test set.
Produces comparison tables, plots, and saves results to disk.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root in sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_mitre_metrics,
    compute_dynamics_metrics,
    evaluate_model,
)
from src.models.baseline import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    flatten_state_sequences,
)
from src.data.dataset import load_states_from_npz, create_dataloaders

logger = logging.getLogger("prism.evaluation.benchmark")


class BenchmarkRunner:
    """
    Runs world model and static baselines on the same test set and produces
    side-by-side metric comparison.
    """

    def __init__(
        self,
        world_model,
        baselines: Optional[dict] = None,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        world_model : trained world model (nn.Module)
        baselines   : dict mapping name -> fitted baseline object
                      e.g. {"LR": LogisticRegressionBaseline, "RF": RandomForestBaseline}
        device      : "cpu" | "cuda"
        threshold   : binary classification threshold
        """
        self.world_model = world_model
        self.baselines = baselines or {}
        self.device = device
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run(
        self,
        test_states_npz: str,
        lookback: int = 20,
        batch_size: int = 64,
    ) -> dict:
        """
        Load test data and evaluate all models.

        Parameters
        ----------
        test_states_npz : path to test split .npz file
        lookback        : sequence lookback for world model

        Returns
        -------
        dict mapping model_name -> metrics dict
        """
        logger.info("Loading test data from %s", test_states_npz)
        data = load_states_from_npz(test_states_npz)
        states = data["states"]
        labels_binary = data["labels_binary"]
        labels_mitre = data["labels_mitre"]

        results = {}

        # --- World model evaluation ---
        logger.info("Evaluating world model...")
        import torch
        from src.data.dataset import StateSequenceDataset
        from torch.utils.data import DataLoader

        test_ds = StateSequenceDataset(
            states=states,
            labels_binary=labels_binary,
            labels_mitre=labels_mitre,
            lookback=lookback,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )

        wm_metrics = evaluate_model(
            self.world_model, test_loader, device=self.device, threshold=self.threshold
        )
        results["World Model"] = wm_metrics
        logger.info(
            "World Model | F1=%.4f | Prec=%.4f | Rec=%.4f | FPR=%.4f",
            wm_metrics["binary"]["f1"],
            wm_metrics["binary"]["precision"],
            wm_metrics["binary"]["recall"],
            wm_metrics["binary"]["fpr"],
        )

        # --- Static baseline evaluations ---
        # Flatten states for static models (no temporal context, lookback=1)
        X_flat, y_binary_flat, y_mitre_flat = flatten_state_sequences(
            states, labels_binary, labels_mitre, lookback=1
        )

        for name, baseline in self.baselines.items():
            logger.info("Evaluating baseline: %s", name)
            try:
                y_pred = baseline.predict(X_flat)
                y_prob = None
                if hasattr(baseline, "predict_proba"):
                    try:
                        proba = baseline.predict_proba(X_flat)
                        y_prob = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
                    except Exception:
                        pass

                bin_metrics = compute_binary_metrics(y_binary_flat, y_pred, y_prob)
                mitre_metrics = compute_mitre_metrics(y_mitre_flat, y_pred)

                results[name] = {
                    "binary": bin_metrics,
                    "mitre": mitre_metrics,
                    "dynamics": {
                        "mse": None,
                        "rmse": None,
                        "mae": None,
                    },
                }
                logger.info(
                    "%s | F1=%.4f | Prec=%.4f | Rec=%.4f | FPR=%.4f",
                    name,
                    bin_metrics["f1"],
                    bin_metrics["precision"],
                    bin_metrics["recall"],
                    bin_metrics["fpr"],
                )
            except Exception as e:
                logger.error("Baseline %s failed: %s", name, e)

        return results

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    def compare(self, results: dict) -> pd.DataFrame:
        """
        Build a comparison DataFrame from results dict.

        Columns: F1, Precision, Recall, FPR, MITRE_F1_Macro, Dynamics_MSE,
                 Early_Warning (world model only)
        """
        rows = []
        for model_name, metrics in results.items():
            row = {
                "Model": model_name,
                "F1": metrics["binary"].get("f1", None),
                "Precision": metrics["binary"].get("precision", None),
                "Recall": metrics["binary"].get("recall", None),
                "FPR": metrics["binary"].get("fpr", None),
                "ROC_AUC": metrics["binary"].get("roc_auc", None),
                "MITRE_F1_Macro": metrics["mitre"].get("f1_macro", None),
                "Dynamics_MSE": (
                    metrics["dynamics"].get("mse", None)
                    if metrics.get("dynamics") else None
                ),
            }
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        # Round for readability
        numeric_cols = df.select_dtypes(include=[float]).columns
        df[numeric_cols] = df[numeric_cols].round(4)
        return df

    def print_table(self, df: pd.DataFrame) -> None:
        """Print a formatted comparison table to stdout."""
        sep = "=" * 90
        print(sep)
        print("  PRISM BENCHMARK COMPARISON")
        print(sep)
        print(df.to_string())
        print(sep)
        print("\n  KEY:  F1=Macro F1 | FPR=False Positive Rate | "
              "MITRE_F1=MITRE Stage Classification | MSE=State Prediction Error")
        print(sep)

    def save_results(
        self,
        results: dict,
        df: pd.DataFrame,
        output_dir: str = "results",
    ) -> None:
        """Save JSON + CSV results to output_dir."""
        os.makedirs(output_dir, exist_ok=True)

        # JSON (full metrics)
        json_path = os.path.join(output_dir, "benchmark_results.json")
        # Make serialisable
        serialisable = {}
        for model, metrics in results.items():
            serialisable[model] = {
                "binary": {
                    k: v for k, v in metrics["binary"].items()
                    if k not in ("report", "confusion_matrix")
                },
                "mitre": {
                    k: v for k, v in metrics["mitre"].items()
                    if k not in ("report", "confusion_matrix")
                },
                "dynamics": metrics.get("dynamics", {}),
            }
        with open(json_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        logger.info("Saved benchmark JSON -> %s", json_path)

        # CSV
        csv_path = os.path.join(output_dir, "benchmark_table.csv")
        df.to_csv(csv_path)
        logger.info("Saved benchmark CSV -> %s", csv_path)

        # Classification reports
        for model, metrics in results.items():
            if "report" in metrics.get("binary", {}):
                report_path = os.path.join(
                    output_dir, f"report_{model.replace(' ', '_')}.txt"
                )
                with open(report_path, "w") as f:
                    f.write(f"=== {model} Binary Classification Report ===\n")
                    f.write(metrics["binary"]["report"])
                    if "mitre" in metrics and "report" in metrics["mitre"]:
                        f.write(f"\n\n=== {model} MITRE Stage Report ===\n")
                        f.write(metrics["mitre"]["report"])

    def plot_comparison(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """Grouped bar chart comparing models across key metrics."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        metrics_to_plot = ["F1", "Precision", "Recall"]
        plot_df = df[
            [c for c in metrics_to_plot if c in df.columns]
        ].dropna(how="all")

        n_models = len(plot_df)
        n_metrics = len(plot_df.columns)
        x = np.arange(n_metrics)
        width = 0.8 / max(n_models, 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

        for i, (model_name, row) in enumerate(plot_df.iterrows()):
            offset = (i - n_models / 2 + 0.5) * width
            vals = [row.get(m, 0) or 0 for m in plot_df.columns]
            bars = ax.bar(
                x + offset, vals, width * 0.9,
                label=model_name, color=colors[i % len(colors)], alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                if val and val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(list(plot_df.columns), fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("PRISM — World Model vs Baseline Comparison", fontsize=13)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved benchmark plot -> %s", save_path)

        return fig


def main():
    """CLI entry point for running benchmark directly from src/evaluation/benchmark.py."""
    from src.model_comparison import run_model_comparison
    run_model_comparison()


if __name__ == "__main__":
    main()
