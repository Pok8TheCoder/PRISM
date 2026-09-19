#!/usr/bin/env python3
"""Quick duplicate audit for bulk_5s PCAP corpus."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/pok8/prism/PRISM/data/raw/bulk_5s")
MANIFEST = ROOT / "manifest.jsonl"
SAMPLE_PER_CLASS = 15
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_stats() -> dict:
    rows = []
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    ok = [r for r in rows if r.get("ok")]
    dup_paths = [p for p, c in Counter(r["file"] for r in ok).items() if c > 1]
    parallel = [r for r in ok if r.get("project")]
    by_class = Counter(r["class"] for r in ok)
    return {
        "manifest_lines": len(rows),
        "ok_rows": len(ok),
        "failed_rows": len(rows) - len(ok),
        "duplicate_manifest_paths": dup_paths[:10],
        "duplicate_manifest_path_count": len(dup_paths),
        "parallel_rows": len(parallel),
        "ok_by_class": dict(by_class),
    }


def sample_pcaps() -> list[Path]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for pcap in ROOT.rglob("*.pcap"):
        by_class[pcap.parent.name].append(pcap)
    picked: list[Path] = []
    rng = random.Random(SEED)
    for cls, files in sorted(by_class.items()):
        files = sorted(files)
        if len(files) <= SAMPLE_PER_CLASS:
            picked.extend(files)
        else:
            picked.extend(rng.sample(files, SAMPLE_PER_CLASS))
    return picked


def pcap_byte_audit(pcaps: list[Path]) -> dict:
    by_hash: dict[str, list[str]] = defaultdict(list)
    sizes = []
    for p in pcaps:
        sizes.append(p.stat().st_size)
        by_hash[sha256_file(p)].append(str(p.relative_to(ROOT)))
    dup_groups = {h: paths for h, paths in by_hash.items() if len(paths) > 1}
    return {
        "sampled_pcaps": len(pcaps),
        "unique_sha256": len(by_hash),
        "byte_duplicate_groups": len(dup_groups),
        "byte_duplicate_examples": [
            {"hash": h[:12], "files": paths[:4]} for h, paths in list(dup_groups.items())[:8]
        ],
        "size_min": min(sizes) if sizes else 0,
        "size_max": max(sizes) if sizes else 0,
        "size_mean": round(sum(sizes) / len(sizes)) if sizes else 0,
    }


def flow_fingerprint_audit(pcaps: list[Path]) -> dict:
    sys.path.insert(0, "/pok8/prism/PRISM")
    from src.pipeline.extract import pcap_to_rows

    whole_dups: dict[str, str] = {}
    whole_dup_pairs = []
    row_hashes: dict[str, str] = {}
    row_collisions = 0
    empty = 0
    errors = 0
    row_counts = []

    for pcap in pcaps:
        try:
            rows = pcap_to_rows(pcap)
            if not rows:
                empty += 1
                continue
            row_counts.append(len(rows))
            # Stable fingerprint from sorted flow keys + numeric fields.
            parts = []
            for row in sorted(rows, key=lambda r: (r.get("src_ip", ""), r.get("dst_ip", ""), r.get("sport", 0), r.get("dport", 0))):
                nums = []
                for k, v in sorted(row.items()):
                    if isinstance(v, (int, float)):
                        nums.append(f"{k}={v}")
                parts.append("|".join(nums))
            whole_h = hashlib.sha256("\n".join(parts).encode()).hexdigest()
            rel = str(pcap.relative_to(ROOT))
            if whole_h in whole_dups:
                whole_dup_pairs.append((rel, whole_dups[whole_h], whole_h[:12]))
            else:
                whole_dups[whole_h] = rel
            for row in rows:
                rh = hashlib.sha256(repr(tuple(sorted((k, row[k]) for k in row if isinstance(row[k], (int, float))))).encode()).hexdigest()
                if rh in row_hashes and row_hashes[rh] != rel:
                    row_collisions += 1
                else:
                    row_hashes[rh] = rel
        except Exception:
            errors += 1

    return {
        "sampled_pcaps": len(pcaps),
        "empty_ingest": empty,
        "errors": errors,
        "unique_flow_fingerprints": len(whole_dups),
        "whole_pcap_flow_duplicate_pairs": whole_dup_pairs[:12],
        "unique_flow_rows": len(row_hashes),
        "cross_pcap_flow_row_collisions": row_collisions,
        "flow_row_count_min": min(row_counts) if row_counts else 0,
        "flow_row_count_max": max(row_counts) if row_counts else 0,
        "flow_row_count_mean": round(sum(row_counts) / len(row_counts), 1) if row_counts else 0,
    }


def state_audit(pcaps: list[Path]) -> dict:
    sys.path.insert(0, "/pok8/prism/PRISM")
    from src.aryan.ingest import pcap_to_states

    by_class: dict[str, dict] = {}
    for cls in sorted({p.parent.name for p in pcaps}):
        class_pcaps = [p for p in pcaps if p.parent.name == cls]
        whole_dups: dict[str, str] = {}
        whole_dup_pairs = []
        window_hashes: dict[str, str] = {}
        window_collisions = 0
        empty = 0
        errors = 0
        first_error = ""
        window_counts = []

        for pcap in class_pcaps:
            try:
                states = pcap_to_states(pcap, window_sec=5.0)
                if not states:
                    empty += 1
                    continue
                g = np.stack([np.asarray(s, dtype=np.float32) for s in states])
                window_counts.append(g.shape[0])
                whole_h = hashlib.sha256(g.tobytes()).hexdigest()
                rel = str(pcap.relative_to(ROOT))
                if whole_h in whole_dups:
                    whole_dup_pairs.append((rel, whole_dups[whole_h], whole_h[:12]))
                else:
                    whole_dups[whole_h] = rel
                for row in g:
                    wh = hashlib.sha256(row.tobytes()).hexdigest()
                    if wh in window_hashes and window_hashes[wh] != rel:
                        window_collisions += 1
                    else:
                        window_hashes[wh] = rel
            except Exception as e:
                errors += 1
                if not first_error:
                    first_error = f"{pcap.name}: {e!r}"

        by_class[cls] = {
            "sampled": len(class_pcaps),
            "empty_ingest": empty,
            "errors": errors,
            "first_error": first_error,
            "unique_whole_state_sets": len(whole_dups),
            "whole_state_duplicate_pairs": whole_dup_pairs[:5],
            "unique_windows": len(window_hashes),
            "cross_pcap_window_collisions": window_collisions,
            "window_count_mean": round(sum(window_counts) / len(window_counts), 2) if window_counts else 0,
        }
    return by_class


def main() -> None:
    print("=== MANIFEST ===")
    print(json.dumps(manifest_stats(), indent=2))
    pcaps = sample_pcaps()
    print("\n=== PCAP BYTE DUPES (sample) ===")
    print(json.dumps(pcap_byte_audit(pcaps), indent=2))
    print("\n=== FLOW FINGERPRINT DUPES (sample) ===")
    print(json.dumps(flow_fingerprint_audit(pcaps), indent=2))
    print("\n=== GEN8 STATE DUPES BY CLASS (sample) ===")
    print(json.dumps(state_audit(pcaps), indent=2))


if __name__ == "__main__":
    main()
