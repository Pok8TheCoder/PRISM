#!/usr/bin/env python3
"""
PRISM Gen 10 - Full Test Set Evaluation & Confusion Matrix Generator
Evaluates the Spatio-Temporal Graph-Transformer World Model on the complete 
universal 4-dataset test split (38,357 sequences / 76,732 windows).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import StateSequenceDataset
from src.models.gen10_world_model import Gen10SpatioTemporalWorldModel
from src.utils.constants import MITRE_STAGES

STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
    "Impact",
]

ARTIFACT_DIR = Path(r"C:\Users\aryan\.gemini\antigravity-ide\brain\c32d9e31-39ba-4472-94bc-6a76bcbb645f")


def run_test_evaluation():
    print("=" * 95)
    print("      PRISM GEN 10: FULL TEST SET BENCHMARK EVALUATION")
    print("=" * 95)

    test_npz_path = ROOT / "data" / "splits_universal_gen10_5s" / "test.npz"
    ckpt_path = ROOT / "weights" / "universal_gen10" / "world_model_best.pt"
    out_dir = ROOT / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print(f"[*] Loading test data from: {test_npz_path}")
    t0 = time.time()
    data = np.load(test_npz_path)
    states = data["states"]
    labels_bin = data["labels_binary"]
    labels_mit = data["labels_mitre"]

    lookback = 20
    stride = 2
    batch_size = 512

    dataset = StateSequenceDataset(
        states, labels_bin, labels_mit, lookback=lookback, stride=stride
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    num_samples = len(dataset)
    print(f"[*] Built test sequence dataset: {num_samples:,d} test sequences in {time.time()-t0:.2f}s")

    # 2. Load Gen 10 Model
    device = torch.device("cpu")
    print(f"[*] Loading model checkpoint on {device}: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ck.get("model_state_dict", ck)

    model = Gen10SpatioTemporalWorldModel(
        d_state=249,
        d_model=256,
        n_layers=4,
        n_heads=8,
        lookback=lookback,
        mlp_hidden=512,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Model Inference Loop
    print(f"[*] Running inference across {len(loader)} batches (batch_size={batch_size})...")
    all_y_bin_true = []
    all_y_bin_prob = []
    all_y_mit_true = []
    all_y_mit_pred = []
    all_y_mit_probs = []
    all_dynamics_mse = []

    t_inf_start = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x = batch["state_seq"].to(device)
            y_next = batch["next_state"].to(device)
            y_b = batch["label_binary"].numpy()
            y_m = batch["label_mitre"].numpy()

            out = model(x)

            # Binary
            prob_b = torch.softmax(out["pred_binary"], dim=-1)[:, 1].cpu().numpy()
            all_y_bin_true.extend(y_b)
            all_y_bin_prob.extend(prob_b)

            # MITRE
            prob_m = torch.softmax(out["pred_mitre"], dim=-1).cpu().numpy()
            pred_m = np.argmax(prob_m, axis=-1)
            all_y_mit_true.extend(y_m)
            all_y_mit_pred.extend(pred_m)
            all_y_mit_probs.extend(prob_m)

            # Dynamics
            diff = out["pred_state_mean"] - y_next
            mse = diff.pow(2).mean(dim=-1).cpu().numpy()
            all_dynamics_mse.extend(mse)

            if (b_idx + 1) % 15 == 0 or (b_idx + 1) == len(loader):
                elapsed = time.time() - t_inf_start
                rate = (len(all_y_bin_true)) / elapsed
                print(f"    Batch [{b_idx+1:02d}/{len(loader):02d}] -> Processed {len(all_y_bin_true):,d} sequences ({rate:.1f} seq/s)")

    total_inf_time = time.time() - t_inf_start
    print(f"[+] Inference complete: {num_samples:,d} sequences in {total_inf_time:.2f}s ({num_samples/total_inf_time:.1f} seq/s)")

    y_true_b = np.array(all_y_bin_true)
    y_prob_b = np.array(all_y_bin_prob)
    y_pred_b = (y_prob_b >= 0.50).astype(int)

    y_true_m = np.array(all_y_mit_true)
    y_pred_m = np.array(all_y_mit_pred)

    # 4. Binary Metrics & Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b).ravel()
    b_acc = accuracy_score(y_true_b, y_pred_b)
    b_prec = precision_score(y_true_b, y_pred_b, zero_division=0)
    b_rec = recall_score(y_true_b, y_pred_b, zero_division=0)
    b_f1 = f1_score(y_true_b, y_pred_b, zero_division=0)
    b_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    b_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "=" * 95)
    print("                    BINARY ATTACK DETECTION PERFORMANCE")
    print("=" * 95)
    print(f"  Test Accuracy           : {b_acc * 100:.2f}%")
    print(f"  Precision               : {b_prec * 100:.2f}%")
    print(f"  Recall (Sensitivity)    : {b_rec * 100:.2f}%")
    print(f"  F1-Score                : {b_f1 * 100:.2f}%")
    print(f"  False Positive Rate     : {b_fpr * 100:.2f}%")
    print(f"  Specificity (TN Rate)   : {b_spec * 100:.2f}%")
    print("-" * 95)
    print("  Binary Confusion Matrix:")
    print(f"                   Predicted Benign     Predicted Attack")
    print(f"    Actual Benign:    TN = {tn:6,d}           FP = {fp:6,d}   (Total = {tn+fp:,d})")
    print(f"    Actual Attack:    FN = {fn:6,d}           TP = {tp:6,d}   (Total = {fn+tp:,d})")

    # 5. MITRE Multi-Class Metrics & Confusion Matrix
    cm_mitre = confusion_matrix(y_true_m, y_pred_m, labels=list(range(len(STAGE_NAMES))))
    m_acc = accuracy_score(y_true_m, y_pred_m)
    m_macro_f1 = f1_score(y_true_m, y_pred_m, average="macro", zero_division=0)
    m_weighted_f1 = f1_score(y_true_m, y_pred_m, average="weighted", zero_division=0)

    per_prec = precision_score(y_true_m, y_pred_m, average=None, labels=list(range(len(STAGE_NAMES))), zero_division=0)
    per_rec = recall_score(y_true_m, y_pred_m, average=None, labels=list(range(len(STAGE_NAMES))), zero_division=0)
    per_f1 = f1_score(y_true_m, y_pred_m, average=None, labels=list(range(len(STAGE_NAMES))), zero_division=0)
    stage_support = np.bincount(y_true_m, minlength=len(STAGE_NAMES))

    print("\n" + "=" * 95)
    print("              MITRE ATT&CK 7-STAGE ATTRIBUTION PERFORMANCE")
    print("=" * 95)
    print(f"  Multi-Class Accuracy    : {m_acc * 100:.2f}%")
    print(f"  MITRE Macro F1          : {m_macro_f1 * 100:.2f}%")
    print(f"  MITRE Weighted F1       : {m_weighted_f1 * 100:.2f}%")
    print(f"  Dynamics Pred MSE       : {np.mean(all_dynamics_mse):.4f}")
    print("-" * 95)
    print(f"{'Stage ID':<8} | {'MITRE Stage Name':<20} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 95)
    for i, name in enumerate(STAGE_NAMES):
        print(f"S{i:<7} | {name:<20} | {stage_support[i]:<8,d} | {per_prec[i]*100:>8.2f}% | {per_rec[i]*100:>8.2f}% | {per_f1[i]*100:>8.2f}%")
    print("-" * 95)

    print("\nMITRE ATT&CK 7x7 Confusion Matrix (Rows: Ground Truth, Columns: Predicted):")
    header = "       " + " ".join([f"  S{i}  " for i in range(len(STAGE_NAMES))])
    print(header)
    print("       " + "-" * len(header))
    for i, row in enumerate(cm_mitre):
        row_str = " ".join([f"{val:6d}" for val in row])
        print(f"  S{i} | {row_str}  ({STAGE_NAMES[i]})")

    # 6. Generate High-Resolution Visual Heatmap PNG
    print("\n[*] Generating high-resolution visual confusion matrix plots...")
    fig = plt.figure(figsize=(20, 6.5), dpi=300)

    # Subplot 1: Binary Confusion Matrix Heatmap
    ax1 = plt.subplot(1, 3, 1)
    cm_bin = np.array([[tn, fp], [fn, tp]])
    cm_bin_norm = cm_bin.astype("float") / cm_bin.sum(axis=1)[:, np.newaxis]
    bin_annot = np.array([
        [f"{tn:,d}\n({cm_bin_norm[0,0]*100:.1f}%)", f"{fp:,d}\n({cm_bin_norm[0,1]*100:.1f}%)"],
        [f"{fn:,d}\n({cm_bin_norm[1,0]*100:.1f}%)", f"{tp:,d}\n({cm_bin_norm[1,1]*100:.1f}%)"],
    ])
    sns.heatmap(
        cm_bin_norm,
        annot=bin_annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        ax=ax1,
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"],
        linewidths=1.5,
        linecolor="#e2e8f0",
    )
    ax1.set_title(f"Binary Detection Confusion Matrix\nAccuracy: {b_acc*100:.2f}% | F1: {b_f1*100:.2f}% | FPR: {b_fpr*100:.2f}%", fontsize=11, fontweight="bold", pad=12)
    ax1.set_xlabel("Predicted Label", fontweight="bold")
    ax1.set_ylabel("True Label", fontweight="bold")

    # Subplot 2: MITRE 7-Class Normalized Confusion Matrix Heatmap
    ax2 = plt.subplot(1, 3, 2)
    cm_mitre_norm = cm_mitre.astype("float") / (cm_mitre.sum(axis=1)[:, np.newaxis] + 1e-9)
    short_names = ["S0 Benign", "S1 Recon", "S2 InitAcc", "S3 Lateral", "S4 C2", "S5 Exfil", "S6 Impact"]
    sns.heatmap(
        cm_mitre_norm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar=True,
        ax=ax2,
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=1.0,
        linecolor="#e2e8f0",
    )
    ax2.set_title(f"MITRE ATT&CK 7-Stage Confusion Matrix\nWeighted F1: {m_weighted_f1*100:.2f}% | Macro F1: {m_macro_f1*100:.2f}%", fontsize=11, fontweight="bold", pad=12)
    ax2.set_xlabel("Predicted Stage", fontweight="bold")
    ax2.set_ylabel("True Stage", fontweight="bold")
    plt.setp(ax2.get_xticklabels(), rotation=40, ha="right")

    # Subplot 3: Per-Class F1 / Precision / Recall Bar Chart
    ax3 = plt.subplot(1, 3, 3)
    x_indices = np.arange(len(STAGE_NAMES))
    bar_width = 0.26
    ax3.bar(x_indices - bar_width, per_prec * 100, width=bar_width, label="Precision", color="#3b82f6", alpha=0.9)
    ax3.bar(x_indices, per_rec * 100, width=bar_width, label="Recall", color="#10b981", alpha=0.9)
    ax3.bar(x_indices + bar_width, per_f1 * 100, width=bar_width, label="F1-Score", color="#8b5cf6", alpha=0.9)
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(short_names, rotation=40, ha="right", fontsize=9)
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("Score (%)", fontweight="bold")
    ax3.set_title("Per-Stage Detection Fidelity\nPrecision, Recall & F1-Score", fontsize=11, fontweight="bold", pad=12)
    ax3.legend(loc="lower right", frameon=True)
    ax3.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    # Save to benchmark directory
    plot_file = out_dir / "gen10_test_confusion_matrices.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"[+] Saved visual plot to: {plot_file}")

    # Copy to Artifact directory
    if ARTIFACT_DIR.is_dir():
        artifact_plot_file = ARTIFACT_DIR / "gen10_test_confusion_matrices.png"
        shutil.copy2(plot_file, artifact_plot_file)
        print(f"[+] Copied visual plot to artifact directory: {artifact_plot_file}")

    # 7. Save JSON Report
    report = {
        "dataset": "splits_universal_gen10_5s/test.npz",
        "total_test_sequences": num_samples,
        "lookback": lookback,
        "stride": stride,
        "binary_metrics": {
            "accuracy": float(b_acc),
            "precision": float(b_prec),
            "recall": float(b_rec),
            "f1": float(b_f1),
            "fpr": float(b_fpr),
            "specificity": float(b_spec),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        },
        "mitre_metrics": {
            "accuracy": float(m_acc),
            "macro_f1": float(m_macro_f1),
            "weighted_f1": float(m_weighted_f1),
            "dynamics_mse": float(np.mean(all_dynamics_mse)),
            "per_stage": {
                STAGE_NAMES[i]: {
                    "stage_id": i,
                    "support": int(stage_support[i]),
                    "precision": float(per_prec[i]),
                    "recall": float(per_rec[i]),
                    "f1": float(per_f1[i]),
                }
                for i in range(len(STAGE_NAMES))
            },
            "confusion_matrix": cm_mitre.tolist(),
        }
    }
    json_path = out_dir / "gen10_test_evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved structured metrics JSON report to: {json_path}")

    print("=" * 95)
    print("                     EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 95)


if __name__ == "__main__":
    run_test_evaluation()
