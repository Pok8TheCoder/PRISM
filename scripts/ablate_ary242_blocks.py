"""242-d ARY world-model feature-block ablation (ARY.01 architecture, CIC splits).

Trains TemporalTransformerWorldModel variants with different feature subsets,
ranks by composite score, optionally retrains the winner and a FEATURES.md combo.

Results -> results/ary242_ablation/ablation.json
Checkpoints -> models/checkpoints/ary242_ablation/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_aryan_wm import (  # noqa: E402
    CKPT_DIR,
    RESULTS_DIR,
    composite_score,
    resolve_columns,
    save_checkpoint,
    train_one,
)
from src.aryan.feature_schema242 import describe_subset  # noqa: E402

SWEEP_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2)
MAX_TRAIN_MIN = 20  # if sweep finishes faster, run combo variant too

# Blocks map to FEATURES.md §7 aggregated state vector groups.
VARIANTS: dict[str, dict] = {
    "full_242": {},
    "minus_dead_meta": {"drop_blocks": ["dead_meta"]},
    "minus_B_mean": {"drop_blocks": ["B_mean"]},
    "minus_B_std": {"drop_blocks": ["B_std"]},
    "minus_B_dispersion": {"drop_blocks": ["B_dispersion"]},
    "minus_C_flags": {"drop_blocks": ["C_flags"]},
    "minus_D_proto": {"drop_blocks": ["D_proto"]},
    "minus_E_top_ports": {"drop_blocks": ["E_top_ports"]},
    "B_std_only": {"keep_blocks": ["A_meta", "B_std", "C_flags", "D_proto", "E_top_ports"]},
    "B_mean_only": {"keep_blocks": ["A_meta", "B_mean", "C_flags", "D_proto", "E_top_ports"]},
    "meta_only_ACDE": {"keep_blocks": ["A_meta", "C_flags", "D_proto", "E_top_ports"]},
    "core_std_ACD": {"keep_blocks": ["A_meta", "B_std", "C_flags", "D_proto"]},
    "lean_no_ports": {"drop_blocks": ["E_top_ports", "dead_meta"]},
    "lean_no_mean": {"drop_blocks": ["B_mean", "dead_meta"]},
    "dispersion_only_B": {"keep_blocks": ["B_dispersion"]},
    # YMT-like lean set: composition + flags + proto (no bulk flow stats / ports)
    "ymt_like_lean": {"keep_blocks": ["A_meta", "C_flags", "D_proto"]},
}


def run_variant(name: str, spec: dict, device: torch.device, seeds: tuple[int, ...]) -> dict:
    cols = resolve_columns(spec)
    runs = []
    for seed in seeds:
        print(f"\n  seed={seed}  features={len(cols)}", flush=True)
        model, meta = train_one(name, cols, seed, device)
        meta["composite_score"] = composite_score(meta)
        ckpt = CKPT_DIR / f"{name}_seed{seed}.pt"
        save_checkpoint(model, meta, ckpt)
        runs.append(meta)
        print(
            f"    {meta['train_sec']:.0f}s  testF1={meta['test_binary_f1']:.3f} "
            f"FPR={meta['test_binary_fpr']:.3f} mitre={meta['test_mitre_f1_macro']:.3f} "
            f"score={meta['composite_score']:.3f}",
            flush=True,
        )
    keys = ["test_binary_f1", "test_binary_fpr", "test_mitre_f1_macro",
            "test_binary_precision", "test_binary_recall", "composite_score", "train_sec"]
    return {
        "variant": name,
        "spec": spec,
        **describe_subset(cols),
        **{k: sum(r[k] for r in runs) / len(runs) for k in keys},
        **{f"{k}_std": (sum(r[k] ** 2 for r in runs) / len(runs) - (sum(r[k] for r in runs) / len(runs)) ** 2) ** 0.5
            for k in keys},
        "per_seed": [{k: r[k] for k in keys + ["epochs_run", "n_params"]} for r in runs],
    }


def build_combo_spec(ranked: list[dict]) -> dict:
    """Union blocks from top-3 variants (excluding full baseline duplicates)."""
    keep_blocks: set[str] = set()
    for r in ranked[:4]:
        if r["variant"] == "full_242":
            continue
        spec = VARIANTS.get(r["variant"], {})
        if spec.get("keep_blocks"):
            keep_blocks.update(spec["keep_blocks"])
        elif spec.get("drop_blocks"):
            # invert: keep everything except dropped
            from src.aryan.feature_schema242 import FEATURE_BLOCKS_242
            all_b = set(FEATURE_BLOCKS_242)
            keep_blocks.update(all_b - set(spec["drop_blocks"]))
    if not keep_blocks:
        keep_blocks = {"A_meta", "B_std", "C_flags", "D_proto"}
    return {"keep_blocks": sorted(keep_blocks)}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    variants = {k: VARIANTS[k] for k in (only or VARIANTS)}

    results: dict[str, dict] = {}
    t0 = time.time()
    for name, spec in variants.items():
        print(f"\n{'=' * 72}\n  {name}  {spec or 'full'}", flush=True)
        results[name] = run_variant(name, spec, device, SWEEP_SEEDS)

    ranked = sorted(results.values(), key=lambda r: r["composite_score"], reverse=True)
    elapsed_min = (time.time() - t0) / 60

    combo_name = None
    if elapsed_min < MAX_TRAIN_MIN and len(variants) > 1:
        combo_spec = build_combo_spec(ranked)
        combo_name = "combo_best_blocks"
        print(f"\nSweep finished in {elapsed_min:.1f} min — running {combo_name}: {combo_spec}", flush=True)
        results[combo_name] = run_variant(combo_name, combo_spec, device, SWEEP_SEEDS)
        ranked = sorted(results.values(), key=lambda r: r["composite_score"], reverse=True)

    best = ranked[0]
    print(f"\nRetraining best '{best['variant']}' with {len(FINAL_SEEDS)} seeds...", flush=True)
    final = run_variant(best["variant"], VARIANTS.get(best["variant"], build_combo_spec(ranked)
                        if best["variant"] == "combo_best_blocks" else {}),
                        device, FINAL_SEEDS)

    OUT = RESULTS_DIR / "ablation.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ranking": [{k: r[k] for k in ("variant", "num_features", "composite_score",
                                        "test_binary_f1", "test_binary_fpr", "test_mitre_f1_macro")}
                    for r in ranked],
        "variants": results,
        "best_variant": best["variant"],
        "final_retrain": final,
        "elapsed_min": elapsed_min,
        "combo_ran": combo_name,
        "feature_comparison": {
            "ary01_trained_dims": 242,
            "ymt01_dims": 64,
            "amt_named_subset_dims": 82,
            "note": "ARY.01 uses full padded CIC StateBuilder; YMT uses 64-d 8-flow v2 schema",
        },
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 88)
    print(f"{'Variant':<22}{'Feat':>5}{'BinF1':>8}{'FPR':>8}{'MitreF1':>9}{'Score':>8}{'Sec':>8}")
    print("-" * 88)
    for r in ranked:
        print(
            f"{r['variant']:<22}{r['num_features']:>5}"
            f"{r['test_binary_f1']:>8.3f}{r['test_binary_fpr']:>8.3f}"
            f"{r['test_mitre_f1_macro']:>9.3f}{r['composite_score']:>8.3f}"
            f"{r['train_sec']:>8.0f}"
        )
    print("=" * 88)
    print(f"Best: {best['variant']} ({best['num_features']} feats) -> {OUT}")


if __name__ == "__main__":
    main()
