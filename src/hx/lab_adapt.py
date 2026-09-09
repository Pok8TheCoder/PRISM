"""Incremental HX-C lab adaptation from forecast-demo kill-chain ground truth.

After each demo run (IPS block, kill-chain complete, or Ctrl+C), labeled windows
from the live scorer are finetuned with a small LR and pullback toward the base
checkpoint so technique labels track the Harborline phases without drifting far
from the offline-trained weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.hx.causal import CausalHXClassifier, HX_C_CKPT, LOOKBACK, SHAUN_DIM, load_hxc_bundle
from src.hx.heads import focal_binary_fp_penalty

ROOT = Path(__file__).resolve().parent.parent.parent
AUTOMODE = ROOT.parent / "Automode"
LAB_CKPT = AUTOMODE / "train" / "checkpoints" / "hx_c_w5s_lab.pt"
BUFFER_PATH = ROOT / "data" / "lab_events" / "forecast_adapt_buffer.npz"
META_PATH = LAB_CKPT.with_suffix(".json")

# Harborline kill-chain phase -> catalog class id (ground truth for lab adaptation).
FORECAST_PHASE_LABELS: dict[str, str] = {
    "recon": "T1046_service_scan",
    "enum": "T1190_web_exploit_probe",
    "spray": "T1187_password_spray",
    "loot": "T1041_exfil_c2",
}

MAX_BUFFER = 512
DEFAULT_EPOCHS = 5
DEFAULT_LR = 5e-5
DEFAULT_PULLBACK = 0.02


def resolve_hxc_ckpt() -> Path:
    """Prefer lab-adapted weights when present."""
    if LAB_CKPT.exists():
        return LAB_CKPT
    return HX_C_CKPT


def phase_to_class_id(phase: str) -> str | None:
    return FORECAST_PHASE_LABELS.get(phase)


def class_id_to_index(class_names: list[str], class_id: str) -> int:
    try:
        return class_names.index(class_id)
    except ValueError:
        return 0


def append_run_samples(
    samples: list[dict[str, Any]],
    *,
    class_names: list[str],
    run_id: str | None = None,
) -> int:
    """Merge in-memory run samples into the persistent replay buffer."""
    if not samples:
        return 0
    xs, yb, yt, phases, windows = [], [], [], [], []
    for row in samples:
        phase = str(row.get("phase") or "")
        class_id = phase_to_class_id(phase)
        if not class_id:
            continue
        xs.append(np.asarray(row["seq"], dtype=np.float32))
        yb.append(1)
        yt.append(class_id_to_index(class_names, class_id))
        phases.append(phase)
        windows.append(int(row.get("window", 0)))
    if not xs:
        return 0

    new_x = np.stack(xs)
    new_yb = np.asarray(yb, dtype=np.int64)
    new_yt = np.asarray(yt, dtype=np.int64)

    if BUFFER_PATH.exists():
        d = np.load(BUFFER_PATH, allow_pickle=True)
        old_x = d["X"]
        old_yb = d["y_bin"]
        old_yt = d["y_tech"]
        old_phases = list(d["phases"].tolist())
        old_windows = list(d["windows"].tolist())
        x_all = np.concatenate([old_x, new_x], axis=0)
        yb_all = np.concatenate([old_yb, new_yb])
        yt_all = np.concatenate([old_yt, new_yt])
        phases_all = old_phases + phases
        windows_all = old_windows + windows
    else:
        x_all, yb_all, yt_all = new_x, new_yb, new_yt
        phases_all, windows_all = phases, windows

    if len(x_all) > MAX_BUFFER:
        x_all = x_all[-MAX_BUFFER:]
        yb_all = yb_all[-MAX_BUFFER:]
        yt_all = yt_all[-MAX_BUFFER:]
        phases_all = phases_all[-MAX_BUFFER:]
        windows_all = windows_all[-MAX_BUFFER:]

    BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        BUFFER_PATH,
        X=x_all,
        y_bin=yb_all,
        y_tech=yt_all,
        phases=np.asarray(phases_all, dtype=object),
        windows=np.asarray(windows_all, dtype=np.int64),
        class_names=np.asarray(class_names, dtype=object),
        last_run=str(run_id or ""),
    )
    return len(xs)


def finetune_lab_adapt(
    bundle: dict[str, Any] | None = None,
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    pullback: float = DEFAULT_PULLBACK,
    reason: str = "run_end",
) -> dict[str, Any]:
    """Finetune HX-C on the replay buffer; save to hx_c_w5s_lab.pt."""
    if not BUFFER_PATH.exists():
        return {"ok": False, "reason": "no_buffer", "message": "No lab adapt buffer yet."}

    d = np.load(BUFFER_PATH, allow_pickle=True)
    x = d["X"].astype(np.float32)
    yb = d["y_bin"].astype(np.int64)
    yt = d["y_tech"].astype(np.int64)
    if len(x) < 2:
        return {"ok": False, "reason": "too_few", "message": f"Need >=2 samples, have {len(x)}."}

    bundle = bundle or load_hxc_bundle(ckpt_path=resolve_hxc_ckpt())
    base_bundle = load_hxc_bundle(ckpt_path=HX_C_CKPT)
    model: CausalHXClassifier = bundle["model"]
    base_model: CausalHXClassifier = base_bundle["model"]
    device = bundle["device"]
    class_names = list(bundle["class_names"])
    base_params = [p.detach().clone() for p in base_model.parameters()]

    # SGD avoids AdamW -> torch._dynamo import issues on some Windows installs.
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    model.train()
    losses: list[float] = []
    for _epoch in range(epochs):
        perm = np.random.permutation(len(x))
        for i in perm:
            xb = torch.from_numpy(x[i : i + 1]).to(device)
            ybin = torch.tensor([yb[i]], device=device, dtype=torch.long)
            ytech = torch.tensor([yt[i]], device=device, dtype=torch.long)
            out = model(xb)
            loss_b = focal_binary_fp_penalty(out["binary_logits"], ybin, fp_weight=4.0)
            loss_c = ce(out["class_logits"], ytech)
            delta = out["pred_state_mean"] - xb[:, -1, :]
            loss_d = delta.pow(2).mean() * 0.05
            reg = sum(
                ((p - b) ** 2).sum()
                for p, b in zip(model.parameters(), base_params)
            )
            loss = loss_b + loss_c + loss_d + pullback * reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))["class_logits"]
        pred = logits.argmax(dim=-1).cpu().numpy()
    acc = float((pred == yt).mean())

    LAB_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": bundle["mean"],
            "scaler_std": bundle["std"],
            "class_names": class_names,
            "d_state": SHAUN_DIM,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "lookback": LOOKBACK,
            "lab_adapt": True,
            "adapt_reason": reason,
            "buffer_n": len(x),
            "buffer_acc": acc,
        },
        LAB_CKPT,
    )
    meta = {
        "reason": reason,
        "epochs": epochs,
        "lr": lr,
        "pullback": pullback,
        "buffer_n": len(x),
        "buffer_acc": acc,
        "loss_mean": float(np.mean(losses)) if losses else 0.0,
        "ckpt": str(LAB_CKPT),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"ok": True, **meta}


def hot_reload_streaming(hx: Any, bundle: dict[str, Any] | None = None) -> None:
    """Reload weights into a live StreamingHXC after adaptation."""
    bundle = bundle or load_hxc_bundle(ckpt_path=resolve_hxc_ckpt())
    hx.model.load_state_dict(bundle["model"].state_dict())
    hx.model.eval()
