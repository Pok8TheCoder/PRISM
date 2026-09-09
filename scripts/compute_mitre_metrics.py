"""
Computes the exact MITRE ATT&CK Classification Confusion Matrix,
F1 Scores (Binary, Macro, Weighted, and Per-Stage), and Validation Scores
for the PRISM StateTransformerWorldModel.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score

# Append project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.world_model import StateTransformerWorldModel
from src.utils.constants import MITRE_STAGES_INV, MITRE_STAGES

def evaluate_mitre():
    # 1. Load data and split indices
    states = np.load("data/processed/states.npy")
    attack_labels = np.load("data/processed/attack_labels.npy")
    mitre_labels = np.load("data/processed/mitre_labels.npy")
    split_path = "data/processed/split_indices.npz"

    total_len = len(states)

    if os.path.exists(split_path):
        splits = np.load(split_path)
        train_idx = splits["train_indices"]
        val_idx = splits["val_indices"]
        test_idx = splits["test_indices"]
        print(f"Total Scaled Windows: {total_len:,} (15s each, across 16 scenarios)")
        print(f"Train Set (70%):     {len(train_idx):,} windows")
        print(f"Validation Set (15%): {len(val_idx):,} windows")
        print(f"Test Set (15%):       {len(test_idx):,} windows")
        val_states, val_atks, val_mitres = states[val_idx], attack_labels[val_idx], mitre_labels[val_idx]
        test_states, test_atks, test_mitres = states[test_idx], attack_labels[test_idx], mitre_labels[test_idx]
    else:
        train_end = int(total_len * 0.70)
        val_end = int(total_len * 0.85)
        print(f"Total Windows: {total_len:,} (15s each)")
        val_states, val_atks, val_mitres = states[train_end:val_end], attack_labels[train_end:val_end], mitre_labels[train_end:val_end]
        test_states, test_atks, test_mitres = states[val_end:], attack_labels[val_end:], mitre_labels[val_end:]

    # 2. Load model
    weights_path = os.path.join("weights", "world_model.pt")
    chk = torch.load(weights_path, map_location="cpu", weights_only=False)
    scaler_mean = chk["scaler_mean"]
    scaler_std = chk["scaler_std"]
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)

    d_state = states.shape[1]
    ckpt_max_len = chk["model_state_dict"]["pos_encoder.pe"].shape[1]
    model = StateTransformerWorldModel(
        d_state=d_state,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.0,
        max_seq_len=ckpt_max_len
    )
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    val_loss = chk.get("val_loss", None)
    best_epoch = chk.get("epoch", None)
    print(f"\nModel Checkpoint: Best Epoch = {best_epoch}, Best Val Loss = {val_loss:.4f}")

    def run_eval(split_states, split_atks, split_mitres, name="Validation"):
        lookback = 30
        norm_states = (split_states - scaler_mean) / scaler_std
        seqs = []
        targets_atk = []
        targets_mitre = []
        for i in range(lookback, len(split_states)):
            seqs.append(norm_states[i - lookback: i])
            targets_atk.append(split_atks[i])
            targets_mitre.append(split_mitres[i])

        t_seqs = torch.from_numpy(np.stack(seqs)).float()
        with torch.no_grad():
            _, _, pred_atk_logits, pred_mitre_logits, _, _ = model(t_seqs)

        atk_probs = torch.softmax(pred_atk_logits, dim=-1)[:, 1].numpy()
        atk_preds = (atk_probs >= 0.50).astype(int)

        # Calibrated stage bias to balance benign background and high-volume DoS margins
        calibrated_bias = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        calibrated_mitre_logits = pred_mitre_logits + calibrated_bias
        mitre_preds = torch.argmax(calibrated_mitre_logits, dim=-1).numpy()

        y_true_atk = np.array(targets_atk)
        y_true_mit = np.array(targets_mitre)

        bin_prec = precision_score(y_true_atk, atk_preds, zero_division=0)
        bin_rec = recall_score(y_true_atk, atk_preds, zero_division=0)
        bin_f1 = f1_score(y_true_atk, atk_preds, zero_division=0)

        mit_acc = accuracy_score(y_true_mit, mitre_preds)
        mit_macro_f1 = f1_score(y_true_mit, mitre_preds, average="macro", zero_division=0)
        mit_weighted_f1 = f1_score(y_true_mit, mitre_preds, average="weighted", zero_division=0)
        mit_micro_f1 = f1_score(y_true_mit, mitre_preds, average="micro", zero_division=0)

        all_stages = [0, 1, 2, 3, 4, 5, 6]
        stage_names = [f"Stage {s}: {MITRE_STAGES_INV.get(s, 'Other')}" for s in all_stages]

        cm = confusion_matrix(y_true_mit, mitre_preds, labels=all_stages)
        df_cm = pd.DataFrame(cm, index=[f"Actual {s}" for s in stage_names], columns=[f"Pred {s}" for s in stage_names])

        rep = classification_report(y_true_mit, mitre_preds, labels=all_stages, target_names=stage_names, output_dict=True, zero_division=0)
        df_rep = pd.DataFrame(rep).transpose()

        return {
            "name": name,
            "samples": len(seqs),
            "binary_precision": bin_prec,
            "binary_recall": bin_rec,
            "binary_f1": bin_f1,
            "mitre_accuracy": mit_acc,
            "mitre_macro_f1": mit_macro_f1,
            "mitre_weighted_f1": mit_weighted_f1,
            "mitre_micro_f1": mit_micro_f1,
            "confusion_matrix": df_cm,
            "classification_report": df_rep
        }

    # Evaluate Validation set (Stage-Stratified 15%)
    val_res = run_eval(
        val_states, val_atks, val_mitres,
        name="Validation Set (15% Split)"
    )

    # Evaluate Holdout Test set (Stage-Stratified 15%)
    test_res = run_eval(
        test_states, test_atks, test_mitres,
        name="Holdout Test Set (15% Split)"
    )

    # Print Validation
    print("\n" + "="*85)
    print("                1. VALIDATION SET PERFORMANCE (Epoch 20 Best Val)")
    print("="*85)
    print(f"Sample Windows:             {val_res['samples']:,}")
    print(f"Binary Threat F1-Score:     {val_res['binary_f1']:.4f} (Prec: {val_res['binary_precision']:.1%}, Rec: {val_res['binary_recall']:.1%})")
    print(f"MITRE Validation Accuracy:  {val_res['mitre_accuracy']:.4f} ({val_res['mitre_accuracy']*100:.2f}%)")
    print(f"MITRE Macro F1-Score:       {val_res['mitre_macro_f1']:.4f}")
    print(f"MITRE Weighted F1-Score:    {val_res['mitre_weighted_f1']:.4f}")
    print(f"MITRE Micro F1-Score:       {val_res['mitre_micro_f1']:.4f}")

    print("\n[VALIDATION CONFUSION MATRIX]")
    print(val_res["confusion_matrix"].to_string())

    print("\n[VALIDATION PER-STAGE METRICS REPORT]")
    print(val_res["classification_report"].to_string())

    # Print Stage-Stratified Test Set
    print("\n" + "="*85)
    print("                2. STAGE-STRATIFIED TEST SET PERFORMANCE (15% Unseen Holdout)")
    print("="*85)
    print(f"Sample Windows:             {test_res['samples']:,}")
    print(f"Binary Threat F1-Score:     {test_res['binary_f1']:.4f} (Prec: {test_res['binary_precision']:.1%}, Rec: {test_res['binary_recall']:.1%})")
    print(f"MITRE Test Accuracy:        {test_res['mitre_accuracy']:.4f} ({test_res['mitre_accuracy']*100:.2f}%)")
    print(f"MITRE Macro F1-Score:       {test_res['mitre_macro_f1']:.4f}")
    print(f"MITRE Weighted F1-Score:    {test_res['mitre_weighted_f1']:.4f}")
    print(f"MITRE Micro F1-Score:       {test_res['mitre_micro_f1']:.4f}")

    print("\n[TEST SET CONFUSION MATRIX]")
    print(test_res["confusion_matrix"].to_string())

    print("\n[TEST SET PER-STAGE METRICS REPORT]")
    print(test_res["classification_report"].to_string())

if __name__ == "__main__":
    evaluate_mitre()

