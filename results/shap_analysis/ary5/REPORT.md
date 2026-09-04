# SHAP / Attribution Report: ARY-5sV01

Architecture: `ary` | lookback=20
Method: `gradient` | threshold=0.5

## Bucket counts (sampled eval set)

| Bucket | Count |
|---|---:|
| tp | 193 |
| tn | 2083 |
| fp | 5 |
| fn | 1 |

## Mean |attribution| by feature block

### tp

- **A_meta**: 0.000158
- **dead_meta**: 0.000000
- **B_mean**: 0.000007
- **B_std**: 0.000002
- **B_dispersion**: 0.000005
- **C_flags**: 0.000002
- **D_proto**: 0.000000
- **E_top_ports**: 0.000000

### tn

- **A_meta**: 0.000073
- **dead_meta**: 0.000000
- **B_mean**: 0.000013
- **B_std**: 0.000008
- **B_dispersion**: 0.000010
- **C_flags**: 0.000009
- **D_proto**: 0.000000
- **E_top_ports**: 0.000000

### fp

- **A_meta**: 0.002729
- **dead_meta**: 0.000000
- **B_mean**: 0.001833
- **B_std**: 0.000879
- **B_dispersion**: 0.001356
- **C_flags**: 0.001126
- **D_proto**: 0.000000
- **E_top_ports**: 0.000000

### fn

- **A_meta**: 0.006186
- **dead_meta**: 0.000000
- **B_mean**: 0.001097
- **B_std**: 0.000998
- **B_dispersion**: 0.001047
- **C_flags**: 0.004500
- **D_proto**: 0.000000
- **E_top_ports**: 0.000000

## Failure analysis (FN vs TP attribution delta)

Features with largest |FN − TP| mean |attribution| — candidates for root cause.

| Feature | FN−TP delta | FN imp | TP imp |
|---|---:|---:|---:|
| std_flow_71 | +0.044810 | 0.044811 | 0.000001 |
| num_flows | +0.017402 | 0.018159 | 0.000757 |
| std_flow_23 | +0.013145 | 0.013166 | 0.000021 |
| num_unique_dst_ports | +0.012737 | 0.012772 | 0.000035 |
| std_flow_59 | +0.012280 | 0.012306 | 0.000027 |
| flag_frac_PSH | +0.011313 | 0.011315 | 0.000002 |
| mean_flow_22 | +0.010479 | 0.010482 | 0.000003 |
| mean_flow_72 | +0.009674 | 0.009674 | 0.000000 |
| flag_frac_ACK | +0.009321 | 0.009324 | 0.000003 |
| mean_flow_6 | +0.006882 | 0.006898 | 0.000016 |
| mean_flow_94 | +0.005984 | 0.005986 | 0.000002 |
| mean_flow_70 | +0.005709 | 0.005713 | 0.000004 |
| mean_flow_32 | +0.005582 | 0.005641 | 0.000058 |
| mean_flow_84 | +0.005570 | 0.005573 | 0.000004 |
| mean_flow_80 | +0.005050 | 0.005053 | 0.000003 |

## Notes

TimesFM hybrid attack classifier reuses this same ARY base head.