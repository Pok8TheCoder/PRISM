"""Build mitre_network_catalog.json from attack_catalog.yaml + mitre_attack.json."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "configs" / "attack_catalog.yaml"
MITRE_PATH = ROOT / "data" / "mitre_attack.json"
OUTPUT_PATH = ROOT / "data" / "mitre_network_catalog.json"


def load_yaml() -> dict:
    with open(YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mitre() -> list[dict]:
    with open(MITRE_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_catalog() -> dict:
    cfg = load_yaml()
    mitre_techniques = load_mitre()
    mitre_by_id = {t["id"]: t for t in mitre_techniques}

    network_distinct = cfg.get("network_distinct", [])
    network_families = cfg.get("network_families", [])

    class_entries = []
    technique_to_class: dict[str, str] = {}
    family_members: dict[str, list[str]] = {}

    for entry in network_distinct:
        class_id = entry["class_id"]
        mitre_ids = entry["mitre_ids"]
        for tid in mitre_ids:
            technique_to_class[tid] = class_id

        class_entries.append({
            "class_id": class_id,
            "name": entry["name"],
            "mitre_ids": mitre_ids,
            "primary_tactic": entry.get("primary_tactic", ""),
            "bot_module": entry.get("bot_module", ""),
            "evasion_chain": entry.get("evasion_chain", []),
            "detectable_from_network": True,
        })

    family_entries = []
    for fam in network_families:
        primary = fam["primary_class_id"]
        members = fam["mitre_ids"]
        family_entries.append({
            "family_id": fam["family_id"],
            "primary_class_id": primary,
            "mitre_ids": members,
            "reason": fam.get("reason", ""),
        })
        family_members[fam["family_id"]] = members
        for tid in members:
            if tid not in technique_to_class:
                technique_to_class[tid] = primary

    techniques_out = []
    stats = {"network_distinct": 0, "network_family": 0, "host_only": 0}

    for tech in mitre_techniques:
        tid = tech["id"]
        if tid in technique_to_class:
            mapped_class = technique_to_class[tid]
            is_primary = any(
                tid in e["mitre_ids"] for e in network_distinct
                if e["class_id"] == mapped_class
            )
            if is_primary:
                bucket = "network_distinct"
                stats["network_distinct"] += 1
            else:
                bucket = "network_family"
                stats["network_family"] += 1
            family_id = next(
                (f["family_id"] for f in network_families if tid in f["mitre_ids"]),
                None,
            )
        else:
            bucket = "host_only"
            mapped_class = None
            family_id = None
            stats["host_only"] += 1

        techniques_out.append({
            "id": tid,
            "name": tech["name"],
            "tactics": tech.get("tactics", []),
            "description": tech.get("description", ""),
            "example": tech.get("example", ""),
            "bucket": bucket,
            "mapped_class_id": mapped_class,
            "family_id": family_id,
            "detectable_from_network": bucket != "host_only",
        })

    trainable_classes = ["Benign"] + [e["class_id"] for e in network_distinct]

    catalog = {
        "version": 1,
        "summary": {
            "total_mitre_techniques": len(mitre_techniques),
            "trainable_classes": len(trainable_classes),
            "network_distinct_bots": len(network_distinct),
            "network_family_groups": len(network_families),
            **stats,
        },
        "trainable_classes": trainable_classes,
        "network_distinct": class_entries,
        "network_families": family_entries,
        "legacy_strategy_map": cfg.get("legacy_strategy_map", {}),
        "techniques": techniques_out,
    }
    return catalog


def main():
    catalog = build_catalog()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    s = catalog["summary"]
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  MITRE techniques: {s['total_mitre_techniques']}")
    print(f"  Trainable classes (incl. Benign): {s['trainable_classes']}")
    print(f"  Network-distinct: {s['network_distinct']}")
    print(f"  Network-family mapped: {s['network_family']}")
    print(f"  Host-only: {s['host_only']}")


if __name__ == "__main__":
    main()
