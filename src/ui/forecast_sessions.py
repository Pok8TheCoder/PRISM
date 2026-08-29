"""Session/recording builder for the SOC dashboard's Forecast Player tab.

Builds a synthetic multi-attack timeline from Aryan's real held-out
CIC-IDS-2018 windows (`data/aryan_splits/*.npz`), runs his real trained
Transformer World Model checkpoint through it two ways -- frozen (control)
and wrapped in RAM's receding-horizon test-time-training + episodic memory
(RAM-A.01) -- and packages the result into a `Recording` that the player
(`src/ui/forecast_player.py`) can scrub through interactively.

This is a dashboard-facing port of the research in
`results/forecast/v8_ram_aryan_killchain/` (originally
`scripts/ram_aryan_killchain_eval.py` on the `aryan` worktree). See
`docs/WORKSPACE_AND_ARYAN_BRANCH.md` for the branch/model background.

Every step in the timeline is annotated with TWO kinds of signal, computed
differently depending on whether the step is behind or ahead of the current
playhead (the player picks which one applies -- see `Recording.anomaly_at`
/ `Recording.intrusion_at`):

  retrospective (step already revealed) -- forecast error vs the REAL value,
    and the infiltration head run on the REAL context window ending there.
  prospective (step still in the forecast cone, ahead of playhead) -- the
    model has no ground truth yet, so anomalousness is the standardized
    magnitude of its OWN predicted state, and intrusion suspicion is the
    infiltration head run on its OWN rolled-out predicted trajectory.

Both use the exact same calibrated model / thresholds, so the "self-heal"
behaviour the player shows (a forecast getting corrected once the real step
arrives) falls out naturally: nothing is recomputed, the player just reads
the retrospective arrays instead of the prospective ones past that point.
"""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.aryan.world_model import TemporalTransformerWorldModel

ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_PATH = ROOT / "models" / "checkpoints" / "aryan_world_model_best.pt"
SPLITS_DIR = ROOT / "data" / "aryan_splits"

STAGE_NAMES = {0: "Benign", 1: "Reconnaissance", 2: "InitialAccess", 3: "LateralMovement",
               4: "C2", 5: "Exfiltration", 6: "Impact"}

CONTEXT = 20      # = Aryan's trained lookback
HORIZON = 20      # receding-horizon chunk size
GAP_COUNT = 4

ANOMALY_PCTL = 97.0            # retrospective forecast-error percentile -> blue
UNUSUAL_PCTL = 97.0            # prospective state-magnitude percentile -> blue (future cone)
INTRUSION_PROB_THRESH = 0.5    # infiltration-head P(attack) threshold -> red
MEM_RADIUS = 5
MATCH_MAX_DIST_FALLBACK = 30.0
BLEND_MAX_WEIGHT = 0.6
ADAPT_LR = 3e-4
ADAPT_STEPS = 3
PULLBACK = 5e-3

MODEL_CHOICES = {
    "none": "\u2014 none \u2014",
    "aryan_frozen": "Aryan-WM (frozen)",
    "ram_a01": "RAM-A.01 (Aryan-WM + TTT + memory)",
}


# --------------------------------------------------------------------------
# Model wrapper + RAM core (ported from ram_aryan_killchain_eval.py, extended
# to also surface the infiltration head at no extra inference cost)
# --------------------------------------------------------------------------
class ModelAdapter(nn.Module):
    """Wraps Aryan's TemporalTransformerWorldModel (dict-of-heads output) so
    it exposes both the plain forward(seq) -> next_state contract RAM's
    OnlineAdaptive/rollout code expects, AND the infiltration-probability
    head -- both come out of the same forward pass, so the player's red
    "suspected intrusion" signal is free."""

    def __init__(self, inner: TemporalTransformerWorldModel):
        super().__init__()
        self.inner = inner

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.inner(seq)["pred_state_mean"]

    def forward_full(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (next_state, p_attack), each (batch, ...)."""
        out = self.inner(seq)
        p_attack = torch.softmax(out["pred_binary"], dim=-1)[:, 1]
        return out["pred_state_mean"], p_attack


class EpisodicMemoryBank:
    def __init__(self):
        self.vecs: list[np.ndarray] = []
        self.entries: list[dict] = []

    def add(self, vec, label, step, continuation):
        self.vecs.append(vec)
        self.entries.append({"label": label, "step": step, "continuation": continuation})

    def query(self, vec, k=1):
        if not self.vecs:
            return []
        M = np.stack(self.vecs)
        d = np.linalg.norm(M - vec[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(self.entries[i], float(d[i])) for i in idx]

    def __len__(self):
        return len(self.vecs)


class OnlineAdaptive:
    def __init__(self, base_model: nn.Module, lr: float, pullback: float):
        self.model = copy.deepcopy(base_model)
        self.model.train()
        self.base_params = [p.clone().detach() for p in base_model.parameters()]
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.pullback = pullback
        self.crit = nn.MSELoss()

    def adapt(self, seqs, targets, steps):
        loss_val = 0.0
        for _ in range(steps):
            self.opt.zero_grad()
            pred = self.model(seqs)
            loss = self.crit(pred, targets)
            reg = sum(((p - b) ** 2).sum() for p, b in zip(self.model.parameters(), self.base_params))
            (loss + self.pullback * reg).backward()
            self.opt.step()
            loss_val = loss.item()
        return loss_val

    def rollout_with_infiltration(self, window_buf: list[np.ndarray], h: int) -> tuple[np.ndarray, np.ndarray]:
        buf = list(window_buf)
        preds, p_attacks = [], []
        self.model.eval()
        with torch.no_grad():
            for _ in range(h):
                x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
                nxt, p_att = self.model.forward_full(x)
                nxt = nxt.squeeze(0).numpy()
                preds.append(nxt)
                p_attacks.append(float(p_att.squeeze(0)))
                buf = buf[1:] + [nxt]
        self.model.train()
        return np.stack(preds), np.array(p_attacks, dtype=np.float32)


def frozen_rollout_with_infiltration(model: ModelAdapter, window_buf: list[np.ndarray], h: int) -> tuple[np.ndarray, np.ndarray]:
    buf = list(window_buf)
    preds, p_attacks = [], []
    with torch.no_grad():
        for _ in range(h):
            x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
            nxt, p_att = model.forward_full(x)
            nxt = nxt.squeeze(0).numpy()
            preds.append(nxt)
            p_attacks.append(float(p_att.squeeze(0)))
            buf = buf[1:] + [nxt]
    return np.stack(preds), np.array(p_attacks, dtype=np.float32)


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------
def load_split(name: str):
    d = np.load(SPLITS_DIR / f"{name}.npz")
    return d["states"].astype(np.float32), d["labels_binary"], d["labels_mitre"]


def _longest_run(labels_mitre, stage_id):
    runs, i = [], 0
    while i < len(labels_mitre):
        j = i
        while j < len(labels_mitre) and labels_mitre[j] == labels_mitre[i]:
            j += 1
        if labels_mitre[i] == stage_id:
            runs.append((i, j))
        i = j
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def build_timeline(train, val, test, target_len: int):
    """Splice real held-out attack windows into a benign background, exactly
    as in the v8 kill-chain eval: Initial Access from val (held-out),
    Lateral Movement from test (held-out), Impact from train (the only long
    contiguous run available -- seen during training, disclosed via label)."""
    tr_s, _, tr_m = train
    va_s, _, va_m = val
    te_s, _, te_m = test

    ia_lo, ia_hi = _longest_run(va_m, 2)
    lm_lo, lm_hi = _longest_run(te_m, 3)
    im_lo, im_hi = _longest_run(tr_m, 6)

    seg_initial_access = va_s[ia_lo:ia_hi]
    seg_lateral_move = te_s[lm_lo:lm_hi]
    seg_impact = tr_s[im_lo:im_hi]

    benign_pool = np.concatenate([tr_s[tr_m == 0], va_s[va_m == 0], te_s[te_m == 0]])
    attack_total = len(seg_initial_access) + len(seg_lateral_move) + len(seg_impact)
    remaining = max(target_len - attack_total, GAP_COUNT * 10)
    gap_len = remaining // GAP_COUNT
    n_needed = gap_len * GAP_COUNT
    reps = int(np.ceil(n_needed / len(benign_pool)))
    benign_tiled = np.tile(benign_pool, (reps, 1))[:n_needed]
    gaps = [benign_tiled[i * gap_len:(i + 1) * gap_len] for i in range(GAP_COUNT)]

    pieces = [
        ("Benign", gaps[0]), ("InitialAccess", seg_initial_access),
        ("Benign", gaps[1]), ("LateralMovement", seg_lateral_move),
        ("Benign", gaps[2]), ("Impact", seg_impact),
        ("Benign", gaps[3]),
    ]
    arrs, labels, segments = [], [], []
    pos = 0
    for lbl, arr in pieces:
        arrs.append(arr)
        labels.extend([lbl] * len(arr))
        segments.append((lbl, pos, pos + len(arr)))
        pos += len(arr)
    full = np.concatenate(arrs)
    return full, labels, segments


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def calibrate_anomaly_threshold(model: ModelAdapter, val_states: np.ndarray,
                                 sc_mean, sc_scale_report, context: int) -> float:
    s = val_states
    errs = []
    for t in range(context - 1, len(s) - 1):
        seq = torch.from_numpy(s[t - context + 1:t + 1].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pred = model(seq).squeeze(0).numpy()
        pred_z = (pred - sc_mean) / sc_scale_report
        true_z = (s[t + 1] - sc_mean) / sc_scale_report
        errs.append(np.mean((pred_z - true_z) ** 2))
    return float(np.percentile(np.array(errs), ANOMALY_PCTL))


def calibrate_unusualness_threshold(train_states: np.ndarray, sc_mean, sc_scale_report,
                                     pctl: float = UNUSUAL_PCTL) -> float:
    """Prospective blue-anomaly threshold: standardized magnitude of a state
    vector relative to the train distribution. Needs no ground truth, so it
    can flag the (not-yet-revealed) forecast cone too."""
    mag = _state_unusualness(train_states, sc_mean, sc_scale_report)
    return float(np.percentile(mag, pctl))


def _state_unusualness(states: np.ndarray, sc_mean, sc_scale_report) -> np.ndarray:
    z = (states - sc_mean) / sc_scale_report
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mean(z ** 2, axis=1)


def seed_memory(model: ModelAdapter, train_states: np.ndarray, train_mitre: np.ndarray,
                 sc_mean, sc_scale_calc, sc_scale_report, context: int, horizon: int,
                 anomaly_thresh: float) -> EpisodicMemoryBank:
    bank = EpisodicMemoryBank()
    s = train_states
    n = len(s)
    t = context - 1
    while t + 1 < n:
        h = min(horizon, n - 1 - t)
        buf = [s[t - context + 1 + i] for i in range(context)]
        preds, _ = frozen_rollout_with_infiltration(model, buf, h)
        true_future = s[t + 1:t + 1 + h]
        pred_z = (preds - sc_mean) / sc_scale_report
        true_z = (true_future - sc_mean) / sc_scale_report
        step_err = np.mean((pred_z - true_z) ** 2, axis=1)
        peak_idx = int(np.argmax(step_err))
        if step_err[peak_idx] > anomaly_thresh:
            center = t + 1 + peak_idx
            lo, hi = center - MEM_RADIUS, center + MEM_RADIUS + 1
            if lo >= 0 and hi <= n:
                vec = s[lo:hi].reshape(-1)
                label = STAGE_NAMES[int(train_mitre[center])]
                cont_end = min(n, center + 1 + horizon)
                continuation = (s[center + 1:cont_end] - sc_mean) / sc_scale_calc - (s[center] - sc_mean) / sc_scale_calc
                bank.add(vec, label, int(center), continuation)
        t += h
    return bank


def calibrate_match_thresh(bank: EpisodicMemoryBank) -> float:
    if len(bank) < 10:
        return MATCH_MAX_DIST_FALLBACK
    rng = np.random.default_rng(0)
    idx = rng.choice(len(bank), size=min(200, len(bank)), replace=False)
    same_d, diff_d = [], []
    for i in idx:
        vec, lbl = bank.vecs[i], bank.entries[i]["label"]
        others = [j for j in range(len(bank)) if j != i]
        if not others:
            continue
        d = np.linalg.norm(np.stack([bank.vecs[j] for j in others]) - vec[None, :], axis=1)
        labels = [bank.entries[j]["label"] for j in others]
        for dist, lb in zip(d, labels):
            (same_d if lb == lbl else diff_d).append(dist)
    if not same_d or not diff_d:
        return MATCH_MAX_DIST_FALLBACK
    return float((np.percentile(same_d, 40) + np.percentile(diff_d, 10)) / 2)


def compute_retro_infiltration(model: ModelAdapter, full: np.ndarray, context: int,
                                batch_size: int = 256) -> np.ndarray:
    """P(attack) at every real step, using the true context window ending
    there -- vectorized (batched forward passes) since it needs no
    autoregression."""
    n = len(full)
    probs = np.full(n, np.nan, dtype=np.float32)
    idxs = list(range(context - 1, n))
    for start in range(0, len(idxs), batch_size):
        chunk_idxs = idxs[start:start + batch_size]
        batch = np.stack([full[t - context + 1:t + 1] for t in chunk_idxs]).astype(np.float32)
        x = torch.from_numpy(batch)
        with torch.no_grad():
            _, p_att = model.forward_full(x)
        probs[chunk_idxs] = p_att.numpy()
    return probs


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------
@dataclass
class Recording:
    model_id: str
    display_name: str
    has_memory: bool
    context: int
    horizon: int
    full_actual: np.ndarray            # (T, D)
    full_predicted: np.ndarray         # (T, D), NaN outside forecast coverage
    retro_anomaly_mask: np.ndarray     # (T,) bool
    prospective_anomaly_mask: np.ndarray  # (T,) bool
    retro_intrusion_prob: np.ndarray   # (T,) float, NaN before context
    prospective_intrusion_prob: np.ndarray  # (T,) float, NaN outside forecast coverage
    retro_mse: np.ndarray              # (T,) float, NaN outside forecast coverage -- standardized MSE vs actual, all features
    memory_write_mask: np.ndarray      # (T,) bool
    segments: list[tuple[str, int, int]]
    labels: list[str]
    retrievals: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.full_actual)

    def anomaly_mask_at(self, playhead: int) -> np.ndarray:
        idx = np.arange(self.n_steps)
        past = idx <= playhead
        return np.where(past, self.retro_anomaly_mask, self.prospective_anomaly_mask)

    def intrusion_mask_at(self, playhead: int) -> np.ndarray:
        idx = np.arange(self.n_steps)
        past = idx <= playhead
        prob = np.where(past, self.retro_intrusion_prob, self.prospective_intrusion_prob)
        return np.nan_to_num(prob, nan=0.0) > INTRUSION_PROB_THRESH

    def cumulative_error_at(self, playhead: int) -> float:
        """Mean standardized MSE (all features) over every step revealed so
        far -- i.e. "how wrong has this model's forecast been, on average,
        up to now". NaN if nothing has been revealed yet."""
        vals = self.retro_mse[: playhead + 1]
        vals = vals[~np.isnan(vals)]
        return float(np.mean(vals)) if len(vals) else float("nan")

    def displayed_series(self, feature_idx: int, playhead: int, past_window: int, future_window: int):
        """Returns (x, actual_y, predicted_y) arrays for one feature, windowed
        around playhead: actual shown up to playhead, predicted shown only
        beyond it (the "self-heal" cut -- once revealed, only actual remains)."""
        lo = max(0, playhead - past_window)
        hi = min(self.n_steps, playhead + future_window + 1)
        x = np.arange(lo, hi)
        actual_y = self.full_actual[lo:hi, feature_idx].astype(np.float64)
        actual_y = np.where(x <= playhead, actual_y, np.nan)
        pred_y = self.full_predicted[lo:hi, feature_idx].astype(np.float64)
        pred_y = np.where(x > playhead, pred_y, np.nan)
        return x, actual_y, pred_y


def _run_chunks(base_model: ModelAdapter, full: np.ndarray, labels: list[str],
                 context: int, horizon: int, use_ram: bool,
                 mem_bank: EpisodicMemoryBank | None, anomaly_thresh: float,
                 unusual_thresh: float, match_thresh: float,
                 sc_mean, sc_scale_calc, sc_scale_report):
    n = len(full)
    full_predicted = np.full_like(full, np.nan, dtype=np.float32)
    retro_anomaly_mask = np.zeros(n, dtype=bool)
    prospective_anomaly_mask = np.zeros(n, dtype=bool)
    prospective_intrusion_prob = np.full(n, np.nan, dtype=np.float32)
    retro_mse = np.full(n, np.nan, dtype=np.float32)
    memory_write_mask = np.zeros(n, dtype=bool)
    retrievals: list[dict] = []

    online = OnlineAdaptive(base_model, ADAPT_LR, PULLBACK) if use_ram else None
    pending_match = None
    t = context - 1
    while t + horizon < n:
        window_buf = [full[t - context + 1 + i] for i in range(context)]
        if use_ram:
            raw_preds, p_att_preds = online.rollout_with_infiltration(window_buf, horizon)
        else:
            raw_preds, p_att_preds = frozen_rollout_with_infiltration(base_model, window_buf, horizon)

        preds = raw_preds
        if use_ram and pending_match is not None:
            entry, dist = pending_match
            w = min(BLEND_MAX_WEIGHT, max(0.0, 1.0 - dist / match_thresh)) * 0.5 + 0.15
            cont = entry["continuation"]
            k = min(len(cont), horizon)
            anchor_z = (full[t] - sc_mean) / sc_scale_calc
            recalled_z = anchor_z[None, :] + cont[:k]
            recalled = recalled_z * sc_scale_calc + sc_mean
            preds = raw_preds.copy()
            preds[:k] = (1 - w) * raw_preds[:k] + w * recalled
        pending_match = None

        true_future = full[t + 1:t + 1 + horizon]
        full_predicted[t + 1:t + 1 + horizon] = preds

        # prospective (no ground truth used): the model's own predicted
        # trajectory's unusualness + infiltration head on its own rollout
        pred_unusual = _state_unusualness(preds, sc_mean, sc_scale_report)
        prospective_anomaly_mask[t + 1:t + 1 + horizon] = pred_unusual > unusual_thresh
        prospective_intrusion_prob[t + 1:t + 1 + horizon] = p_att_preds

        # retrospective (ground truth now available in this offline recording)
        pred_z = (raw_preds - sc_mean) / sc_scale_report
        true_z = (true_future - sc_mean) / sc_scale_report
        step_err = np.mean((pred_z - true_z) ** 2, axis=1)
        retro_anomaly_mask[t + 1:t + 1 + horizon] = step_err > anomaly_thresh
        retro_mse[t + 1:t + 1 + horizon] = step_err

        if use_ram:
            peak_idx = int(np.argmax(step_err))
            if step_err[peak_idx] > anomaly_thresh:
                center = t + 1 + peak_idx
                lo, hi = center - MEM_RADIUS, center + MEM_RADIUS + 1
                if lo >= 0 and hi <= n:
                    vec = full[lo:hi].reshape(-1)
                    true_label = labels[center]
                    memory_write_mask[max(0, lo):min(n, hi)] = True
                    matches = mem_bank.query(vec, k=1)
                    if matches:
                        entry, dist = matches[0]
                        is_match = dist < match_thresh
                        retrievals.append({"step": int(center), "true_label": true_label,
                                            "retrieved_label": entry["label"], "distance": dist,
                                            "match": bool(is_match)})
                        if is_match:
                            pending_match = (entry, dist)
                    cont_end = min(n, center + 1 + horizon)
                    cont_z = (full[center + 1:cont_end] - sc_mean) / sc_scale_calc - (full[center] - sc_mean) / sc_scale_calc
                    mem_bank.add(vec, true_label, int(center), cont_z)

            seqs, targets = [], []
            for k in range(horizon):
                idx = t + k
                if idx - context + 1 < 0:
                    continue
                seqs.append(full[idx - context + 1:idx + 1])
                targets.append(full[idx + 1])
            if seqs:
                online.adapt(torch.from_numpy(np.stack(seqs).astype(np.float32)),
                              torch.from_numpy(np.stack(targets).astype(np.float32)), ADAPT_STEPS)

        t += horizon

    return (full_predicted, retro_anomaly_mask, prospective_anomaly_mask,
            prospective_intrusion_prob, retro_mse, memory_write_mask, retrievals)


def build_recording(model_choice: str, target_len: int) -> Recording:
    """Loads Aryan's real checkpoint + data, builds the synthetic timeline,
    and runs the requested model (frozen or RAM-A.01) across it."""
    if model_choice not in ("aryan_frozen", "ram_a01"):
        raise ValueError(f"Unknown model_choice: {model_choice}")
    use_ram = model_choice == "ram_a01"

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    tr_s, _, tr_m = train
    va_s, _, _ = val

    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    d_state = ck["model_state_dict"]["embedding.proj.weight"].shape[1]
    inner = TemporalTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=CONTEXT)
    inner.load_state_dict(ck["model_state_dict"])
    inner.eval()
    base_model = ModelAdapter(inner)
    n_params = sum(p.numel() for p in base_model.parameters())

    sc_mean = tr_s.mean(axis=0)
    sc_scale_calc = np.clip(tr_s.std(axis=0), 1e-6, None)
    sc_scale_report = np.where(sc_scale_calc > 1e-2, sc_scale_calc, np.inf)

    full, labels, segments = build_timeline(train, val, test, target_len)

    anomaly_thresh = calibrate_anomaly_threshold(base_model, va_s, sc_mean, sc_scale_report, CONTEXT)
    unusual_thresh = calibrate_unusualness_threshold(tr_s, sc_mean, sc_scale_report)

    mem_bank = None
    match_thresh = MATCH_MAX_DIST_FALLBACK
    mem_stats = {}
    if use_ram:
        mem_bank = seed_memory(base_model, tr_s, tr_m, sc_mean, sc_scale_calc, sc_scale_report,
                                CONTEXT, HORIZON, anomaly_thresh)
        match_thresh = calibrate_match_thresh(mem_bank)
        mem_stats = {"size": len(mem_bank), "by_class": dict(Counter(e["label"] for e in mem_bank.entries).most_common(10))}

    (full_predicted, retro_anomaly_mask, prospective_anomaly_mask,
     prospective_intrusion_prob, retro_mse, memory_write_mask, retrievals) = _run_chunks(
        base_model, full, labels, CONTEXT, HORIZON, use_ram, mem_bank,
        anomaly_thresh, unusual_thresh, match_thresh, sc_mean, sc_scale_calc, sc_scale_report,
    )

    retro_intrusion_prob = compute_retro_infiltration(base_model, full, CONTEXT)

    return Recording(
        model_id=model_choice,
        display_name=MODEL_CHOICES[model_choice],
        has_memory=use_ram,
        context=CONTEXT,
        horizon=HORIZON,
        full_actual=full,
        full_predicted=full_predicted,
        retro_anomaly_mask=retro_anomaly_mask,
        prospective_anomaly_mask=prospective_anomaly_mask,
        retro_intrusion_prob=retro_intrusion_prob,
        prospective_intrusion_prob=prospective_intrusion_prob,
        retro_mse=retro_mse,
        memory_write_mask=memory_write_mask,
        segments=segments,
        labels=labels,
        retrievals=retrievals,
        meta={
            "n_params": n_params,
            "checkpoint_epoch": ck.get("epoch"),
            "checkpoint_metrics": ck.get("metrics"),
            "anomaly_thresh": anomaly_thresh,
            "unusual_thresh": unusual_thresh,
            "match_thresh": match_thresh,
            "memory": mem_stats,
        },
    )


def top_variance_features(k: int = 12) -> list[int]:
    tr_s, _, _ = load_split("train")
    variances = tr_s.var(axis=0)
    return [int(i) for i in np.argsort(variances)[::-1][:k]]
