"""Gen8 RXI — RAMX immune v2 episodic seeding for streaming inference.

RXI uses the *same* Gen8 weights as base/RAMX. At session start it:
  1. Ingests labeled benign warmup windows into episodic memory
  2. Ingests a labeled benign donor PCAP (self-tolerance)
  3. Ingests a labeled same-class attack exemplar (or benign self-prefix)
  4. Recalibrates detect threshold from warmup replay
  5. Scores live/test windows unlabeled (normal RAMX step)

See R.A.M/src/prediction/ramx_immune.py for the Shaun 292-d analogue.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.aryan.gen8_streaming import StreamingAryanGen8Ramx, preprocess_state

CAL_MARGIN = 1e-4
DETECT_CLAMP = (0.99515, 0.999)

# Saved RXI profiles — v1_unified is the original shipped configuration.
RXI_PROFILES: dict[str, dict[str, Any]] = {
    "v1_unified": {
        "memory_mode": "unified",
        "immune_v2": True,
        "system_id": "aryan_gen8_rxi",
        "description": "Original RXI: single RAMX bank, immune v2 seeding, blended MITRE from k-NN.",
        "blend_mode": "unified",
        "knn_k": 3,
    },
    "v2_staged": {
        "memory_mode": "staged",
        "immune_v2": True,
        "system_id": "aryan_gen8_rxi_staged",
        "description": "Per-MITRE-stage RAMX banks; winner-take-all stage selection at query time.",
        "blend_mode": "staged",
        "knn_k": 3,
    },
    "v3_gated": {
        "memory_mode": "staged",
        "immune_v2": True,
        "system_id": "aryan_gen8_rxi_gated",
        "description": "Gen8-led MITRE: RXI only confirms/boosts Gen8 attack hypotheses. Strict memory for anomaly.",
        "blend_mode": "gated",
        "knn_k": 3,
        "mitre_confirm_floor": 0.15,
        "mitre_boost": 0.25,
        "anomaly_strict_factor": 0.35,
        "attack_prior_floor": 0.20,
    },
    "v4_targeted": {
        "memory_mode": "staged",
        "immune_v2": True,
        "system_id": "aryan_gen8_rxi_targeted",
        "description": "RXI only on Gen8 failure surfaces: binary uncertainty band + weak MITRE stages.",
        "blend_mode": "targeted",
        "knn_k": 3,
        "target_stages": [2, 3, 5],
        "mitre_pass_through_stages": [0, 1, 4, 6],
        "seed_stages": [0, 2, 3, 5],
        "mitre_confirm_floor": 0.10,
        "mitre_weak_confidence_ceiling": 0.55,
        "mitre_boost": 0.30,
        "anomaly_strict_factor": 0.35,
        "binary_uncertainty_low": 0.25,
        "binary_uncertainty_high": 0.75,
    },
}
DEFAULT_RXI_PROFILE = "v1_unified"

_BLEND_KWARGS = (
    "mitre_confirm_floor",
    "mitre_boost",
    "anomaly_strict_factor",
    "attack_prior_floor",
    "target_stages",
    "mitre_pass_through_stages",
    "mitre_weak_confidence_ceiling",
    "binary_uncertainty_low",
    "binary_uncertainty_high",
    "blend_cap",
    "weak_stages",
    "weak_confidence_ceiling",
    "strong_confidence_floor",
    "rescue_boost",
    "rescue_min",
)

# Gen8 bulk training mix (Automode/train/generate_bulk_windows_gpu.py)
BULK_GEN8_PATH = Path(__file__).resolve().parents[2].parent / "Automode" / "train" / "data" / "bulk_5s_gen8.npz"
BULK_GEN8_META_PATH = BULK_GEN8_PATH.parent / "bulk_5s_gpu_meta.json"
TRAIN_DONOR_WINDOWS = 50


def calibrate_threshold_from_scores(
    scores: list[float],
    margin: float = CAL_MARGIN,
    clamp: tuple[float, float] = DETECT_CLAMP,
) -> float:
    if not scores:
        return clamp[0]
    raw = float(max(scores) + margin)
    return float(min(max(raw, clamp[0]), clamp[1]))


class StreamingAryanGen8Rxi(StreamingAryanGen8Ramx):
    """Gen8 + RAMX with immune v2 episodic seeding."""

    system_id = "aryan_gen8_rxi"

    def __init__(
        self,
        *args,
        immune_v2: bool = True,
        rxi_profile: str = DEFAULT_RXI_PROFILE,
        memory_mode: str | None = None,
        **kwargs,
    ):
        profile = RXI_PROFILES.get(rxi_profile, RXI_PROFILES[DEFAULT_RXI_PROFILE])
        mode = memory_mode or profile["memory_mode"]
        blend_mode = profile.get("blend_mode", "unified")
        blend_kwargs = {
            k: (tuple(v) if k in ("target_stages", "mitre_pass_through_stages") else v)
            for k, v in ((k, profile[k]) for k in _BLEND_KWARGS if k in profile)
        }
        self.seed_stages = list(profile.get("seed_stages", [0, 1, 2, 3, 4, 5, 6]))
        super().__init__(
            *args,
            memory_mode=mode,
            blend_mode=blend_mode,
            blend_kwargs=blend_kwargs,
            **kwargs,
        )
        self.immune_v2 = bool(immune_v2 if immune_v2 is not None else profile.get("immune_v2", True))
        self.rxi_profile = rxi_profile if rxi_profile in RXI_PROFILES else DEFAULT_RXI_PROFILE
        self.system_id = profile.get("system_id", self.system_id)
        self.knn_k = int(profile.get("knn_k", self.knn_k))
        self.seed_stats: dict[str, Any] = {"profile": self.rxi_profile}

    def reset_session(self) -> None:
        """Fresh buffer + empty episodic bank."""
        self.buffer = []
        self._pending = None
        if hasattr(self.bank, "clone_empty"):
            self.bank = self.bank.clone_empty()
        else:
            self.bank = self._make_bank()
        self.seed_stats = {"profile": self.rxi_profile}

    def ingest_labeled_states(
        self,
        states: list[np.ndarray],
        true_bin: int,
        true_mit: int = 0,
    ) -> int:
        n = 0
        for st in states:
            self.step(st, true_bin=int(true_bin), true_mit=int(true_mit))
            n += 1
        return n

    def replay_unlabeled_states(self, states: list[np.ndarray]) -> list[float]:
        ps: list[float] = []
        for st in states:
            ps.append(float(self.step(st, true_bin=None, true_mit=None)["p_att"]))
        return ps

    def seed_immune_session(
        self,
        *,
        warmup_states: list[np.ndarray],
        benign_donor_states: list[np.ndarray] | None = None,
        attack_exemplar_states: list[np.ndarray] | None = None,
        attack_mitre_stage: int = 1,
        self_tolerance_states: list[np.ndarray] | None = None,
        self_tolerance_fraction: float = 0.5,
    ) -> tuple[float, list[np.ndarray] | None]:
        """
        Full immune v2 seeding. Returns (detect_threshold, held_out_suffix_or_none).

        For benign scoring with self_tolerance_states, returns suffix to score.
        """
        self.reset_session()
        self.ingest_labeled_states(warmup_states, 0, 0)
        self.seed_stats["benign_warmup_windows"] = len(warmup_states)

        if self.immune_v2 and benign_donor_states:
            self.ingest_labeled_states(benign_donor_states, 0, 0)
            self.seed_stats["benign_donor_windows"] = len(benign_donor_states)

        score_suffix: list[np.ndarray] | None = None
        if attack_exemplar_states:
            self.ingest_labeled_states(attack_exemplar_states, 1, attack_mitre_stage)
            self.seed_stats["attack_exemplar_windows"] = len(attack_exemplar_states)
        elif self_tolerance_states:
            n = len(self_tolerance_states)
            cal_n = max(1, int(n * self_tolerance_fraction))
            if cal_n >= n:
                cal_n = max(1, n - 1)
            prefix, suffix = self_tolerance_states[:cal_n], self_tolerance_states[cal_n:]
            self.ingest_labeled_states(prefix, 0, 0)
            self.seed_stats["self_tolerance_windows"] = cal_n
            score_suffix = suffix

        self.seed_stats["memory_entries"] = len(self.bank)

        saved_buf, saved_pending = self.buffer, self._pending
        self.buffer, self._pending = [], None
        cal_ps = self.replay_unlabeled_states(warmup_states)
        thresh = calibrate_threshold_from_scores(cal_ps)
        self.detect_threshold = thresh
        self.buffer, self._pending = saved_buf, saved_pending
        self.buffer, self._pending = [], None

        self.seed_stats["detect_threshold"] = thresh
        return thresh, score_suffix


def states_from_raw_list(raw_states: list[np.ndarray], scaler: Any) -> list[np.ndarray]:
    return [preprocess_state(np.asarray(s, dtype=np.float32), scaler) for s in raw_states]


def pcap_to_all_gen8_states(pcap: Path, scaler: Any, window_sec: float = 5.0) -> list[np.ndarray]:
    """All 5s windows from a PCAP as preprocessed Gen8 states."""
    import pandas as pd

    from src.aryan.ingest import _states_from_flow_df
    from src.pipeline.extract import pcap_to_rows

    rows = pcap_to_rows(pcap)
    if not rows:
        return []
    states, _, _ = _states_from_flow_df(pd.DataFrame(rows), window_sec=window_sec)
    return [preprocess_state(np.asarray(s, dtype=np.float32), scaler) for s in states]


@lru_cache(maxsize=1)
def build_bulk_train_class_slices(
    npz_path: str | None = None,
    meta_path: str | None = None,
) -> dict[str, tuple[int, int]]:
    """Map attack class_id -> [start, end) row indices in bulk_5s_gen8.npz."""
    npz = Path(npz_path) if npz_path else BULK_GEN8_PATH
    meta = Path(meta_path) if meta_path else BULK_GEN8_META_PATH
    if not npz.exists() or not meta.exists():
        return {}

    plan = json.loads(meta.read_text(encoding="utf-8"))["plan"]["classes"]
    n_states = int(np.load(npz, mmap_mode="r")["states"].shape[0])
    slices: dict[str, tuple[int, int]] = {}
    off = 0
    for entry in plan:
        cls = entry["class"]
        tw = int(entry["target_windows"])
        if off >= n_states:
            break
        end = min(off + tw, n_states)
        if cls != "Benign" and end > off:
            slices[cls] = (off, end)
        off += tw
    return slices


def build_train_attack_donor_states(
    scaler: Any,
    *,
    max_windows: int = TRAIN_DONOR_WINDOWS,
    npz_path: Path | None = None,
    meta_path: Path | None = None,
    seed: int = 0,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    """
    Per-class attack donor windows sampled from Gen8 bulk training NPZ.

    States are preprocessed (log1p + scaler) for direct RAMX ingest.
    """
    npz = npz_path or BULK_GEN8_PATH
    slices = build_bulk_train_class_slices(
        str(npz),
        str(meta_path or BULK_GEN8_META_PATH),
    )
    if not slices:
        return {}, {"error": "bulk train NPZ or meta missing", "path": str(npz)}

    rng = np.random.default_rng(seed)
    donors: dict[str, list[np.ndarray]] = {}
    with np.load(npz, mmap_mode="r") as data:
        states = data["states"]
        for cls, (start, end) in slices.items():
            pool = states[start:end]
            n = int(end - start)
            k = min(max_windows, n)
            if k <= 0:
                continue
            if k >= n:
                idx = np.arange(n)
            else:
                idx = np.sort(rng.choice(n, size=k, replace=False))
            donors[cls] = [
                preprocess_state(np.asarray(pool[i], dtype=np.float32), scaler) for i in idx
            ]

    from src.model.attack_catalog import get_class_names

    attack_classes = [c for c in get_class_names() if c != "Benign"]
    missing = [c for c in attack_classes if c not in donors]
    return donors, {
        "npz_path": str(npz),
        "n_classes_in_npz": len(slices),
        "n_donor_classes": len(donors),
        "missing_classes": missing,
        "windows_per_class": max_windows,
        "class_slices": {k: [v[0], v[1]] for k, v in slices.items()},
    }


def build_donor_index(
    *roots: Path,
) -> dict[str, list[Path]]:
    from src.model.attack_catalog import resolve_pcap_class

    idx: dict[str, list[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for pcap in sorted(root.rglob("*.pcap")):
            cls = resolve_pcap_class(pcap.name)
            if cls and cls != "Benign":
                idx.setdefault(cls, []).append(pcap)
    return idx


def pick_attack_donor(
    test_class: str,
    donors: dict[str, list[Path]],
    exclude: Path | None = None,
) -> Path | None:
    pool = [p for p in donors.get(test_class, []) if exclude is None or p.resolve() != exclude.resolve()]
    if not pool:
        return None
    pool.sort(key=lambda p: (0 if "bulk_parallel_weak" in str(p) else 1, p.name))
    return pool[0]


def pick_benign_donor_pcaps(benign_dir: Path, exclude: Path | None = None) -> list[Path]:
    pool = sorted(benign_dir.glob("*.pcap"))
    if exclude is not None:
        pool = [p for p in pool if p.resolve() != exclude.resolve()]
    return pool


def seed_rxi_scorer(
    scorer: StreamingAryanGen8Rxi,
    *,
    warmup_states: list[np.ndarray],
    attack_class: str | None = None,
    donor_roots: tuple[Path, ...] = (),
    benign_dir: Path | None = None,
    attack_mitre_stage: int = 1,
    train_attack_donors: dict[str, list[np.ndarray]] | None = None,
) -> float:
    """Seed an RXI scorer from warmup + optional donor PCAPs."""
    donors = build_donor_index(*donor_roots) if donor_roots else {}
    benign_donor_states: list[np.ndarray] | None = None
    attack_states: list[np.ndarray] | None = None

    if benign_dir and benign_dir.is_dir():
        bpool = pick_benign_donor_pcaps(benign_dir)
        if bpool:
            benign_donor_states = pcap_to_all_gen8_states(bpool[0], scorer.scaler)

    if attack_class:
        if train_attack_donors and attack_class in train_attack_donors:
            attack_states = train_attack_donors[attack_class]
        elif donors:
            donor_pcap = pick_attack_donor(attack_class, donors)
            if donor_pcap is not None:
                attack_states = pcap_to_all_gen8_states(donor_pcap, scorer.scaler)

    thresh, _ = scorer.seed_immune_session(
        warmup_states=warmup_states,
        benign_donor_states=benign_donor_states,
        attack_exemplar_states=attack_states,
        attack_mitre_stage=attack_mitre_stage,
    )
    return thresh
