#!/usr/bin/env python3
"""Copy trained checkpoints from Automode (and local PRISM) into models/checkpoints/ + MANIFEST.json."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "checkpoints"
AUTOMODE = ROOT.parent / "Automode"
AUTOMODE_CKPT = AUTOMODE / "train" / "checkpoints"

# basename -> (source relative to repo root or automode train/checkpoints, description, train_cmd)
CATALOG: dict[str, tuple[str, str, str]] = {
    "hx_c_v4_w5s.pt": (
        "Automode/train/checkpoints/hx_c_v4_w5s.pt",
        "HX-C v4 causal classifier (primary)",
        "cd Automode && python train/build_hx_bulk_mix.py && python train/train_hx_c_v4.py",
    ),
    "hx_c_v3_w5s.pt": (
        "Automode/train/checkpoints/hx_c_v3_w5s.pt",
        "HX-C v3",
        "cd Automode && python train/train_hx_c_v3.py",
    ),
    "hx_c_v2_w5s.pt": (
        "Automode/train/checkpoints/hx_c_v2_w5s.pt",
        "HX-C v2",
        "cd Automode && python train/train_hx_c_v2.py",
    ),
    "hx_c_w5s.pt": (
        "Automode/train/checkpoints/hx_c_w5s.pt",
        "HX-C v1 causal",
        "cd Automode && python train/train_hx_c.py",
    ),
    "hx_w5s.pt": (
        "Automode/train/checkpoints/hx_w5s.pt",
        "HX v1 Shaun-wrap",
        "cd Automode && python train/train_hx.py",
    ),
    "shaun_branch_fairfit.pt": (
        "Automode/train/checkpoints/shaun_branch_fairfit.pt",
        "Shaun fair-fit branch",
        "cd Automode && python train/finetune_shaun_quiet_lab.py",
    ),
    "shaun_quietlab_e6.pt": (
        "Automode/train/checkpoints/shaun_quietlab_e6.pt",
        "Shaun quiet-lab e6",
        "cd Automode && python train/finetune_shaun_quiet_lab.py",
    ),
    "ary_5sv01.pt": ("models/checkpoints/ary_5sv01.pt", "ARY 5s v01", "PRISM training scripts"),
    "ary5s_v01.pt": ("models/checkpoints/ary5s_v01.pt", "ARY 5s v01 alt name", "PRISM training scripts"),
    "aryan_world_model_best.pt": (
        "models/checkpoints/aryan_world_model_best.pt",
        "ARY world model best",
        "PRISM / Aryan train pipeline",
    ),
    "xmt_world_model_best.pt": (
        "models/checkpoints/xmt_world_model_best.pt",
        "XMT world model",
        "PRISM training scripts",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_src(rel: str) -> Path:
    if rel.startswith("Automode/"):
        return ROOT.parent / rel.replace("Automode/", "Automode/", 1)
    return ROOT / rel


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    missing: list[str] = []

    for name, (rel, desc, train_cmd) in CATALOG.items():
        src = resolve_src(rel)
        dst = OUT / name
        if src.is_file() and src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        elif not dst.is_file():
            missing.append(name)
            continue

        if not dst.is_file():
            missing.append(name)
            continue

        meta_path = dst.with_suffix(".json")
        sidecar_src = src.with_suffix(".json")
        if sidecar_src.is_file() and sidecar_src.resolve() != meta_path.resolve():
            meta_path.write_bytes(sidecar_src.read_bytes())

        meta: dict = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}

        entries.append(
            {
                "file": name,
                "description": desc,
                "train": train_cmd,
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "metrics": meta,
            }
        )

    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": entries,
        "missing_sources": missing,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUT / 'MANIFEST.json'} ({len(entries)} checkpoints, {len(missing)} missing)")
    if missing:
        print("Missing:", ", ".join(missing))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
