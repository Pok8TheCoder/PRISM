# PRISM World Model: Multi-Step Horizon Stress-Test Report

## Executive Summary
This stress-test evaluates the predictive horizon limits of the **StateTransformerWorldModel**.
PRISM autoregressively rolls out continuous network states up to **8 steps forward (+120 seconds)**
to evaluate how early preemptive intrusion warnings can be reliably generated before physical breach.

- **Test Sequences**: 449 rolling windows across holdout telemetry.
- **Lookback Context**: 30 windows (7.5 minutes).
- **Calibrated Threshold**: `0.60`

## Horizon Performance Decay Table

|   Step (k) | Horizon   |   F1-Score | Precision   | Recall   | ROC-AUC   | PR-AUC   | FPR   | MITRE Acc   |   MITRE F1 |   State MSE |   Cosine Sim | Preemptive Viability   |
|-----------:|:----------|-----------:|:------------|:---------|:----------|:---------|:------|:------------|-----------:|------------:|-------------:|:-----------------------|
|          1 | +15s      |     0.9438 | 92.6%       | 96.2%    | 99.0%     | 95.8%    | 3.14% | 95.5%       |     0.8222 |      0.4457 |       0.7696 | HIGH                   |
|          2 | +30s      |     0.8949 | 89.8%       | 89.1%    | 98.0%     | 95.9%    | 4.06% | 93.1%       |     0.7818 |      0.4928 |       0.7636 | HIGH                   |
|          3 | +45s      |     0.8685 | 85.2%       | 88.6%    | 97.4%     | 94.6%    | 5.83% | 92.0%       |     0.716  |      0.6603 |       0.7475 | HIGH                   |
|          4 | +60s      |     0.8372 | 84.4%       | 83.1%    | 95.6%     | 91.3%    | 6.27% | 91.1%       |     0.751  |      0.5878 |       0.7467 | HIGH                   |
|          5 | +75s      |     0.8359 | 83.6%       | 83.6%    | 94.6%     | 84.1%    | 6.54% | 90.4%       |     0.7407 |      0.6445 |       0.7447 | HIGH                   |
|          6 | +90s      |     0.83   | 80.2%       | 86.1%    | 94.7%     | 83.6%    | 7.95% | 89.8%       |     0.6913 |      0.8144 |       0.7293 | HIGH                   |
|          7 | +105s     |     0.8045 | 78.1%       | 82.9%    | 92.7%     | 80.0%    | 9.38% | 87.8%       |     0.6988 |      0.7468 |       0.7295 | HIGH                   |
|          8 | +120s     |     0.8015 | 77.8%       | 82.7%    | 91.8%     | 76.0%    | 9.32% | 87.8%       |     0.7038 |      0.8195 |       0.7281 | HIGH                   |

## Key Technical Takeaways

1. **Preemptive Action Window (+30s to +60s)**:
   - The model maintains high fidelity within the **+30s to +60s window**, providing optimal lead time for autonomous firewall drops and human Co-Pilot verification.

2. **Graceful Autoregressive Degradation (+75s to +120s)**:
   - As the autoregressive rollout compounds state drift across 2 minutes, the state Cosine Similarity and F1-score exhibit predictable, graceful decay rather than catastrophic divergence.

3. **MITRE Kill-Chain Projection**:
   - The multi-task MITRE classification head reliably identifies multi-phase progression ahead of volumetric surges, enabling graduated preemption (Rate-Limit vs. Drop).
