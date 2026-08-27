"""Load PRISM attack catalog for model training and labeling."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_JSON = ROOT / "data" / "mitre_network_catalog.json"
CATALOG_YAML = ROOT / "configs" / "attack_catalog.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    if not CATALOG_JSON.exists():
        import runpy
        runpy.run_path(
            str(ROOT / "scripts" / "build_mitre_network_catalog.py"),
            run_name="__main__",
        )
    with open(CATALOG_JSON, encoding="utf-8") as f:
        return json.load(f)


def get_trainable_classes() -> list[str]:
    return load_catalog()["trainable_classes"]


def get_class_names() -> list[str]:
    return get_trainable_classes()


def get_num_classes() -> int:
    return len(get_trainable_classes())


def get_class_to_idx() -> dict[str, int]:
    return {name: i for i, name in enumerate(get_trainable_classes())}


def get_mitre_map() -> dict[str, tuple[str, str]]:
    """Map class_id -> (primary_tactic, mitre_label)."""
    catalog = load_catalog()
    mapping: dict[str, tuple[str, str]] = {"Benign": ("Normal", "Benign traffic")}

    for entry in catalog["network_distinct"]:
        class_id = entry["class_id"]
        tactic = entry.get("primary_tactic", "")
        mitre_ids = entry.get("mitre_ids", [])
        label = mitre_ids[0] if mitre_ids else class_id
        name = entry.get("name", class_id)
        mapping[class_id] = (tactic, f"{label} – {name}")

    return mapping


def get_bot_class_ids() -> list[str]:
    return [e["class_id"] for e in load_catalog()["network_distinct"]]


def get_legacy_strategy_map() -> dict[str, str]:
    return load_catalog().get("legacy_strategy_map", {})


def resolve_pcap_class(pcap_name: str) -> str | None:
    """Infer trainable class from PCAP filename."""
    name = pcap_name.lower()
    if "benign" in name:
        return "Benign"

    for class_id in get_bot_class_ids():
        key = class_id.lower()
        if key in name:
            return class_id

    for legacy, class_id in get_legacy_strategy_map().items():
        if legacy.lower() in name:
            return class_id

    return None


def get_evasion_chains() -> dict[str, list[str]]:
    chains = {}
    for entry in load_catalog()["network_distinct"]:
        chains[entry["class_id"]] = entry.get("evasion_chain", [])
    return chains


def get_technique_bucket(technique_id: str) -> str:
    for tech in load_catalog()["techniques"]:
        if tech["id"] == technique_id:
            return tech["bucket"]
    return "host_only"
