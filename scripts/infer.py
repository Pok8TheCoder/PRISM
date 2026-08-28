"""
PRISM - Inference Script
Run world model inference on a PCAP or CSV input.
Produces infiltration probability timeline, MITRE stage predictions,
and SHAP feature attributions.

Usage:
    python scripts/infer.py --input data/sample.csv --model weights/world_model_best.pt
    python scripts/infer.py --input capture.pcap --model weights/world_model_best.pt --k 15
    python scripts/infer.py --states data/splits/test.npz --window 100
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.constants import MITRE_STAGES_INV

logger = setup_logger("prism.infer", log_dir="results/logs")


def load_model(model_path: str, cfg, device: str):
    """Load trained world model from checkpoint."""
    import torch
    from src.models.world_model import build_world_model, load_checkpoint

    model = build_world_model(cfg.model).to(device)
    load_checkpoint(model, model_path, device=device)
    model.eval()
    return model


def load_input_data(args, cfg) -> tuple[np.ndarray, list[str]]:
    """
    Load input data from PCAP, CSV, or pre-built states .npz.

    Returns
    -------
    state_sequence : np.ndarray, shape (L, D_state)
    feature_names  : list of feature name strings
    """
    if args.states:
        data = np.load(args.states)
        states = data["states"]
        lookback = cfg.data.lookback
        window_idx = args.window if args.window is not None else len(states) - lookback
        window_idx = max(lookback, min(window_idx, len(states) - 1))
        seq = states[window_idx - lookback : window_idx]
        logger.info(
            "Loaded states.npz: window %d-%d, shape=%s",
            window_idx - lookback, window_idx, seq.shape,
        )
        return seq, []

    elif args.input and args.input.endswith(".csv"):
        return _load_from_csv(args.input, cfg)

    elif args.input and (args.input.endswith(".pcap") or args.input.endswith(".pcapng")):
        return _load_from_pcap(args.input, cfg)

    else:
        raise ValueError(
            "Provide one of: --states <npz>, --input <csv>, --input <pcap>"
        )


def _load_from_csv(csv_path: str, cfg) -> tuple[np.ndarray, list[str]]:
    """Full pipeline: CSV -> FlowExtractor -> StateBuilder -> state sequence."""
    from src.data.flow_extractor import FlowExtractor
    from src.data.feature_merger import FeatureMerger
    from src.data.state_builder import StateBuilder

    logger.info("Processing CSV: %s", csv_path)
    extractor = FlowExtractor(dataset_type=cfg.data.dataset)
    df = extractor.extract(file_path=csv_path, fit_scaler=False)

    merger = FeatureMerger()
    merged = merger.merge(df)

    builder = StateBuilder(window_size_seconds=cfg.data.window_size_seconds)
    result = builder.build_states(merged)

    states = result["states"]
    feature_names = result.get("feature_names", [])
    lookback = cfg.data.lookback

    if len(states) < lookback:
        padding = np.zeros((lookback - len(states), states.shape[1]), dtype=np.float32)
        states = np.vstack([padding, states])

    seq = states[-lookback:]
    logger.info("CSV pipeline complete: state_seq shape=%s", seq.shape)
    return seq, feature_names


def _load_from_pcap(pcap_path: str, cfg) -> tuple[np.ndarray, list[str]]:
    """Full pipeline: PCAP -> PacketExtractor + FlowExtractor -> state sequence."""
    from src.data.packet_extractor import PacketExtractor
    from src.data.feature_merger import FeatureMerger
    from src.data.state_builder import StateBuilder

    logger.info("Processing PCAP: %s", pcap_path)
    pkt_extractor = PacketExtractor(window_size_seconds=cfg.data.window_size_seconds)
    pkt_df = pkt_extractor.extract_from_pcap(pcap_path)

    merger = FeatureMerger()
    merged = merger.merge(None, pkt_df)  # No flow CSV; use packet-only

    builder = StateBuilder(window_size_seconds=cfg.data.window_size_seconds)
    result = builder.build_states(merged)

    states = result["states"]
    feature_names = result.get("feature_names", [])
    lookback = cfg.data.lookback

    if len(states) < lookback:
        padding = np.zeros((lookback - len(states), states.shape[1]), dtype=np.float32)
        states = np.vstack([padding, states])

    seq = states[-lookback:]
    logger.info("PCAP pipeline complete: state_seq shape=%s", seq.shape)
    return seq, feature_names


def run_inference(args):
    """Main inference pipeline."""
    import torch

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    logger.info("PRISM Inference | device=%s | k_steps=%d", device, args.k)

    # 1. Load input data first to infer d_state dynamically
    state_seq, feature_names = load_input_data(args, cfg)
    cfg.model.d_state = state_seq.shape[-1]

    # 2. Load model with matching d_state
    model = load_model(args.model, cfg, device)

    # 3. Run K-step simulation
    from src.prediction.simulator import KStepSimulator
    from src.prediction.attack_mapper import AttackStageMapper

    simulator = KStepSimulator(
        model=model,
        device=device,
        k_steps=args.k,
        num_rollouts=args.rollouts,
        deterministic=args.deterministic,
    )

    logger.info("Running %d-step forward simulation (%d rollouts)...", args.k, args.rollouts)
    result = simulator.simulate(state_seq, feature_names=feature_names or None)

    # 4. Print report
    report = simulator.format_report(result, feature_names)
    print(report)

    # 5. SHAP explanation (optional, slower)
    if args.explain:
        logger.info("Computing SHAP explanations...")
        try:
            from src.explainability.shap_explainer import PRISMShapExplainer
            # Use the input sequence itself as background (single-sample explanation)
            bg = np.tile(state_seq[np.newaxis], (10, 1, 1))
            explainer = PRISMShapExplainer(
                model=model,
                background_data=bg,
                feature_names=feature_names,
                device=device,
            )
            explain_result = explainer.explain(state_seq)
            print("\n" + "=" * 60)
            print("SHAP FEATURE ATTRIBUTION")
            print("=" * 60)
            print(explain_result["explanation_text"])
            print("\nTop Features:")
            for f in explain_result["top_features"][:10]:
                print(
                    f"  {f['rank']:2d}. {f['feature']:<35} "
                    f"SHAP={f['shap_value']:+.4f} "
                    f"val={f['feature_value']:.4f}"
                )
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)

    # 6. Save output JSON
    if args.output:
        output = {
            "overall_alert": result.overall_alert,
            "peak_infiltration_prob": result.peak_infiltration_prob,
            "peak_step": result.peak_step,
            "is_escalating": result.is_escalating,
            "early_warning_steps": result.early_warning_steps,
            "trajectory_summary": result.trajectory_summary,
            "steps": [
                {
                    "step": s.step,
                    "infiltration_prob": s.infiltration_prob,
                    "mitre_stage": s.mitre_stage,
                    "confidence": s.confidence,
                    "alert_level": s.alert_level,
                    "top_features": s.top_features[:5],
                }
                for s in result.steps
            ],
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Saved inference result -> %s", args.output)

    return result


def main():
    parser = argparse.ArgumentParser(description="PRISM Inference")
    parser.add_argument("--model", default="weights/world_model_best.pt")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--input", default=None, help="CSV or PCAP file")
    parser.add_argument("--states", default=None, help="Pre-built states.npz")
    parser.add_argument("--window", type=int, default=None, help="Window index for --states mode")
    parser.add_argument("--k", type=int, default=10, help="K-step horizon")
    parser.add_argument("--rollouts", type=int, default=10, help="Ensemble rollouts")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--explain", action="store_true", help="Run SHAP explanation")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--output", default=None, help="Save JSON output to file")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
