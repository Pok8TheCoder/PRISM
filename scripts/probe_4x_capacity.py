"""Train the 4x-wide world model on missed samples and probe forgetting."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.adversarial.lab_config import EPOCHS_PER_CYCLE, MISSED_DIR
from src.adversarial.training_loop import retrain_model
from src.model.world_model_multiclass import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    D_MODEL,
    NHEAD,
    NLAYERS,
    NUM_FEATURES,
    SEQ_LEN,
    MultiClassWorldModel,
    predict_live,
)
from src.pipeline.features import NUM_FEATURES as PIPE_FEATURES
from src.ui.lab_controller import list_missed_samples
from sklearn.preprocessing import StandardScaler


def load_batches():
    feature_batches: list[np.ndarray] = []
    label_indices: list[int] = []
    metas: list[dict] = []
    for meta in list_missed_samples():
        class_id = meta.get("true_class_id")
        if class_id not in CLASS_TO_IDX:
            continue
        meta_dir = Path(meta["_path"]).parent
        feat_file = meta_dir / meta.get("features", "")
        if not feat_file.exists():
            continue
        feats = np.load(feat_file).astype(np.float32)
        if feats.ndim != 2 or len(feats) == 0:
            continue
        if feats.shape[1] < NUM_FEATURES:
            pad = np.zeros((feats.shape[0], NUM_FEATURES - feats.shape[1]), dtype=np.float32)
            feats = np.hstack([feats, pad])
        elif feats.shape[1] > NUM_FEATURES:
            feats = feats[:, :NUM_FEATURES]
        if len(feats) < SEQ_LEN:
            reps = int(np.ceil((SEQ_LEN + 1) / len(feats)))
            feats = np.tile(feats, (reps, 1))
        feature_batches.append(feats)
        label_indices.append(CLASS_TO_IDX[class_id])
        metas.append(meta)
    return feature_batches, label_indices, metas


def evaluate(model, scaler, batches, labels, metas) -> dict:
    model.eval()
    correct = 0
    total = 0
    by_class = defaultdict(lambda: {"n": 0, "ok": 0})
    pred_counts = Counter()
    for feats, y, meta in zip(batches, labels, metas):
        pred_idx, probs, _ = predict_live(model, scaler, feats, next(model.parameters()).device)
        pred_counts[CLASS_NAMES[pred_idx]] += 1
        total += 1
        by_class[meta["true_class_id"]]["n"] += 1
        if pred_idx == y:
            correct += 1
            by_class[meta["true_class_id"]]["ok"] += 1
    acc = correct / max(total, 1)
    per = {
        cid: round(v["ok"] / v["n"], 3)
        for cid, v in sorted(by_class.items(), key=lambda kv: kv[0])
        if v["n"]
    }
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "classes_with_any_hit": sum(1 for v in by_class.values() if v["ok"] > 0),
        "classes_total": len(by_class),
        "top_predictions": pred_counts.most_common(8),
        "per_class_acc": per,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"4x model | d_model={D_MODEL} nhead={NHEAD} nlayers={NLAYERS} "
        f"features={NUM_FEATURES}/{PIPE_FEATURES} device={device}"
    )
    model = MultiClassWorldModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    batches, labels, metas = load_batches()
    print(f"Missed samples loaded: {len(batches)} from {MISSED_DIR}")
    print(f"Unique labels: {len(set(labels))}")

    scaler = StandardScaler()
    scaler.mean_ = np.zeros(NUM_FEATURES, dtype=np.float64)
    scaler.scale_ = np.ones(NUM_FEATURES, dtype=np.float64)

    print("\n=== Full retrain on all missed samples (30 epochs) ===")
    retrain_model(model, scaler, batches, labels, device, epochs=30)
    after_full = evaluate(model, scaler, batches, labels, metas)
    print(json.dumps({"after_full_retrain": after_full}, indent=2))

    # Probe catastrophic forgetting the same way the loop does: 3-batch refit.
    print("\n=== Online-style retrain on last 3 samples (10 epochs) ===")
    clone = copy.deepcopy(model)
    scaler_clone = StandardScaler()
    scaler_clone.mean_ = np.array(scaler.mean_, copy=True)
    scaler_clone.scale_ = np.array(scaler.scale_, copy=True)
    if hasattr(scaler, "var_"):
        scaler_clone.var_ = np.array(scaler.var_, copy=True)
        scaler_clone.n_samples_seen_ = scaler.n_samples_seen_
        scaler_clone.n_features_in_ = scaler.n_features_in_

    retrain_model(
        clone,
        scaler_clone,
        batches[:3],
        labels[:3],
        device,
        epochs=EPOCHS_PER_CYCLE,
        save_checkpoint=False,
    )
    after_mini = evaluate(clone, scaler_clone, batches, labels, metas)
    mini_true = [metas[i]["true_class_id"] for i in range(3)]
    print(f"Mini-batch true classes: {mini_true}")
    print(json.dumps({"after_3_sample_retrain": after_mini}, indent=2))

    out = {
        "architecture": {
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "nlayers": NLAYERS,
            "parameters": n_params,
        },
        "after_full_retrain": after_full,
        "after_3_sample_retrain": after_mini,
        "mini_batch_classes": mini_true,
        "forgot": after_mini["accuracy"] + 0.15 < after_full["accuracy"],
    }
    dest = Path("data/raw/adversarial/capacity_4x_probe.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {dest}")
    print(
        "Verdict: "
        + (
            "STILL FORGETS after 3-sample online retrain"
            if out["forgot"]
            else "HELD UP — 3-sample update did not wipe the full-set accuracy"
        )
    )


if __name__ == "__main__":
    main()
