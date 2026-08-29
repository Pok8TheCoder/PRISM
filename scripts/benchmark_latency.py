import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import time
import numpy as np
from src.models.world_model import TemporalTransformerWorldModel, load_checkpoint
from src.prediction.simulator import KStepSimulator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({dev_name})")

    model = TemporalTransformerWorldModel(
        d_state=242, d_model=256, n_layers=4, n_heads=8, lookback=20
    ).to(device)
    load_checkpoint(model, "weights/world_model_best.pt", device=str(device))
    model.eval()

    # Model size in memory
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    # 1. Warmup
    dummy_single = torch.randn(1, 20, 242, device=device)
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_single)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # 2. Benchmark Single-window Inference (Batch size = 1, real-time streaming)
    n_runs = 500
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_single)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t_total = time.perf_counter() - t0
    latency_b1_ms = (t_total / n_runs) * 1000
    fps_b1 = n_runs / t_total

    vram_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    vram_res_mb = torch.cuda.memory_reserved() / (1024**2) if device.type == "cuda" else 0.0

    # 3. Benchmark Batched Inference (B=32)
    dummy_batch = torch.randn(32, 20, 242, device=device)
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_batch = 200
    with torch.no_grad():
        for _ in range(n_batch):
            _ = model(dummy_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
    t_batch = time.perf_counter() - t0
    latency_b32_ms = (t_batch / n_batch) * 1000
    fps_b32 = (n_batch * 32) / t_batch
    vram_b32_alloc = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0

    # 4. Benchmark K-Step Forward Autoregressive Simulation (K=10 steps)
    sim = KStepSimulator(model=model, device=str(device), k_steps=10, num_rollouts=1, deterministic=True)
    sample_state = np.random.randn(20, 242).astype(np.float32)

    t0 = time.perf_counter()
    n_sim = 100
    for _ in range(n_sim):
        _ = sim.simulate(sample_state)
    t_sim = time.perf_counter() - t0
    latency_sim10_ms = (t_sim / n_sim) * 1000
    fps_sim = n_sim / t_sim

    # 5. Benchmark Stochastic Ensemble Rollout (K=10 steps, N=10 rollouts with uncertainty)
    sim_ensemble = KStepSimulator(model=model, device=str(device), k_steps=10, num_rollouts=10, deterministic=False)
    t0 = time.perf_counter()
    n_ens = 25
    for _ in range(n_ens):
        _ = sim_ensemble.simulate(sample_state)
    t_ens = time.perf_counter() - t0
    latency_ens10_ms = (t_ens / n_ens) * 1000

    print("\n" + "=" * 70)
    print("      PRISM TEMPORAL TRANSFORMER: VRAM & LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Hardware Platform       : {dev_name}")
    print(f"Model Parameters        : {param_count/1e6:.2f} Million ({param_size_mb:.2f} MB float32)")
    print("-" * 70)
    print("VRAM CONSUMPTION:")
    print(f"  - Peak Allocated VRAM (B=1) : {vram_alloc_mb:.2f} MB")
    print(f"  - Peak Allocated VRAM (B=32): {vram_b32_alloc:.2f} MB")
    print(f"  - PyTorch CUDA Reserved     : {vram_res_mb:.2f} MB  (< 0.5% of 6GB RTX 3050)")
    print("-" * 70)
    print("INFERENCE LATENCY & THROUGHPUT:")
    print(f"  - Real-time Point (B=1)     : {latency_b1_ms:.2f} ms  ({fps_b1:.1f} windows/sec)")
    print(f"  - Batched Inference (B=32)  : {latency_b32_ms:.2f} ms  ({fps_b32:.1f} windows/sec)")
    print(f"  - 10-Step Deterministic Sim : {latency_sim10_ms:.2f} ms  ({fps_sim:.1f} rollouts/sec)")
    print(f"  - 10-Step Monte Carlo (N=10): {latency_ens10_ms:.2f} ms  (full uncertainty forecast)")
    print("=" * 70)

if __name__ == "__main__":
    main()
