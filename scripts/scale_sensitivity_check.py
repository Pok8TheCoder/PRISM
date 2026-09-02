#!/usr/bin/env python3
"""Quick check: do lab model scores rise when traffic density matches CIC training scale?"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
sys.path.insert(0, str(SHAUN_ROOT))
from src.models.world_model import StateTransformerWorldModel  # noqa: E402

for _k in list(sys.modules):
    if _k == "src" or _k.startswith("src."):
        del sys.modules[_k]
sys.path.insert(0, str(ROOT))

from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.pipeline.extract import pcap_to_rows  # noqa: E402
from scripts.shaun_pcap_ingest import prism_rows_to_cic_dataframe  # noqa: E402

SCALE_KEYS = [
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s",
]

SAMPLE_PCAPS = [
    "live_http_flood_none.pcap",
    "r1_a1_ssh_bruteforce_none.pcap",
    "r2_a1_port_scan_sequential_none.pcap",
    "r3_a1_http_flood_none.pcap",
    "live_port_scan_sequential_none.pcap",
]


def scale_rows(rows: list[dict], factor: float, replicate: int = 1) -> list[dict]:
    """Boost volume: multiply rate/count cols and optionally replicate flows."""
    out: list[dict] = []
    cic = prism_rows_to_cic_dataframe(rows)
    if cic.empty:
        return []
    for rep in range(replicate):
        chunk = cic.copy()
        for col in SCALE_KEYS:
            if col in chunk.columns:
                chunk[col] = chunk[col] * factor
        # Spread replicas slightly in time within same second bucket
        if rep and "Timestamp" in chunk.columns:
            chunk["Timestamp"] = chunk["Timestamp"] + pd_timedelta_ms(rep * 50)
        out.extend(chunk.to_dict("records"))
    return out


def pd_timedelta_ms(ms: int):
    import pandas as pd
    return pd.Timedelta(milliseconds=ms)


def rows_from_cic_records(records: list[dict]) -> np.ndarray:
    import pandas as pd

    if not records:
        return np.zeros((0, 292), dtype=np.float32)

    prev = os.getcwd()
    prev_path = list(sys.path)
    os.chdir(SHAUN_ROOT)
    for k in list(sys.modules):
        if k == "src" or k.startswith("src."):
            del sys.modules[k]
    sys.path = [str(SHAUN_ROOT)] + [p for p in sys.path if Path(p).resolve() != ROOT.resolve()]
    from src.data.schema_aligner import SchemaAligner
    from src.data.state_builder import StateBuilder

    try:
        aligned = SchemaAligner().align_dataframe(pd.DataFrame(records))
        states, _, _, _, _ = StateBuilder(window_size_seconds=15).build_states_from_dataframe(aligned)
    finally:
        os.chdir(prev)
        sys.path[:] = prev_path
    return states.astype(np.float32)


def load_shaun_model():
    prev = os.getcwd()
    os.chdir(SHAUN_ROOT)
    try:
        ck = torch.load("weights/world_model.pt", map_location="cpu", weights_only=False)
        sd = ck["model_state_dict"]
        d = sd["input_embed.0.weight"].shape[1]
        pe = sd.get("pos_encoder.pe", torch.zeros(1, 50, 256)).shape[1]
        model = StateTransformerWorldModel(d_state=d, d_model=256, nhead=8, num_layers=4, max_seq_len=pe)
        model.load_state_dict(sd)
        model.eval()
        mean = np.asarray(ck.get("scaler_mean", np.zeros((1, d))), dtype=np.float32)
        std = np.asarray(ck.get("scaler_std", np.ones((1, d))), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return model, mean, std
    finally:
        os.chdir(prev)


def score_states(model, mean, std, states: np.ndarray, lookback: int = 30) -> dict:
    if len(states) == 0:
        return {"n": 0, "mean_p": 0.0, "max_p": 0.0, "flows_per_window_med": 0.0}
    m, s = mean.reshape(-1), std.reshape(-1)
    buf: list[np.ndarray] = []
    ps: list[float] = []
    for raw in states:
        buf.append(((raw - m) / s).astype(np.float32))
        win = buf[-lookback:]
        if len(win) < lookback:
            win = [win[0]] * (lookback - len(win)) + win
        seq = np.stack(win)
        x = torch.from_numpy(seq).float().unsqueeze(0)
        with torch.no_grad():
            _, _, logits, _, _, _ = model(x)
            p = float(torch.softmax(logits, dim=-1)[0, 1].item())
        ps.append(p)
    flows = np.expm1(states[:, 276])
    return {
        "n": len(ps),
        "mean_p": float(np.mean(ps)),
        "max_p": float(np.max(ps)),
        "flows_per_window_med": float(np.median(flows)),
    }


def main() -> int:
    os.chdir(ROOT)
    model, mean, std = load_shaun_model()

    # CIC reference (attack windows from holdout)
    cic_attack = np.load(SHAUN_ROOT / "data/processed/states.npy")
    cic_labels = np.load(SHAUN_ROOT / "data/processed/attack_labels.npy")
    atk_idx = np.where(cic_labels == 1)[0][:200]
    cic_ref = score_states(model, mean, std, cic_attack[atk_idx])

    rows_out = {"cic_attack_reference": cic_ref, "pcaps": []}

    for name in SAMPLE_PCAPS:
        pcap = SAVE_DIR / name
        if not pcap.exists():
            continue
        raw_rows = pcap_to_rows(pcap)
        entry = {"pcap": name, "raw_flows": len(raw_rows), "conditions": []}

        conditions = [
            ("lab_raw", 1.0, 1),
            ("scale_10x", 10.0, 1),
            ("scale_50x", 50.0, 1),
            ("replicate_5x", 1.0, 5),
            ("scale_10x_replicate_5x", 10.0, 5),
        ]
        for label, factor, rep in conditions:
            cic_recs = scale_rows(raw_rows, factor, replicate=rep)
            states = rows_from_cic_records(cic_recs)
            sc = score_states(model, mean, std, states)
            sc["label"] = label
            sc["scale_factor"] = factor
            sc["replicate"] = rep
            entry["conditions"].append(sc)

        rows_out["pcaps"].append(entry)
        print(f"\n{name} ({len(raw_rows)} flows)")
        for c in entry["conditions"]:
            print(f"  {c['label']:28s} flows/win~{c['flows_per_window_med']:6.0f}  mean_P={c['mean_p']:.3f}  max_P={c['max_p']:.3f}")

    print(f"\nCIC attack reference (n={cic_ref['n']}): flows/win~{cic_ref['flows_per_window_med']:.0f}  mean_P={cic_ref['mean_p']:.3f}  max_P={cic_ref['max_p']:.3f}")

    out = ROOT / "results" / "shaun_vs_ary5" / "scale_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
