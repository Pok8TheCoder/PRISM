# SHAP / Attribution Report: Shaun PRISM V2

Architecture: `shaun` | lookback=30
Method: `gradient` | threshold=0.5

## Bucket counts (sampled eval set)

| Bucket | Count |
|---|---:|
| tp | 168 |
| tn | 618 |
| fp | 14 |
| fn | 1 |

## Mean |attribution| by feature block

### tp

- **flow_means**: 0.010528
- **flow_stds**: 0.014171
- **flow_maxs**: 0.016948
- **flow_mins**: 0.000741
- **flow_medians**: 0.010668
- **macro_graph**: 0.054561

### tn

- **flow_means**: 0.000092
- **flow_stds**: 0.000134
- **flow_maxs**: 0.000109
- **flow_mins**: 0.000006
- **flow_medians**: 0.000050
- **macro_graph**: 0.000396

### fp

- **flow_means**: 0.061150
- **flow_stds**: 0.086320
- **flow_maxs**: 0.097194
- **flow_mins**: 0.000905
- **flow_medians**: 0.044056
- **macro_graph**: 0.315377

### fn

- **flow_means**: 0.017939
- **flow_stds**: 0.017886
- **flow_maxs**: 0.031202
- **flow_mins**: 0.000041
- **flow_medians**: 0.009376
- **macro_graph**: 0.048482

## Failure analysis (FN vs TP attribution delta)

Features with largest |FN − TP| mean |attribution| — candidates for root cause.

| Feature | FN−TP delta | FN imp | TP imp |
|---|---:|---:|---:|
| flow_9_mean | +0.321736 | 0.347070 | 0.025335 |
| flow_8_max | +0.310724 | 0.328255 | 0.017531 |
| flow_11_max | +0.227952 | 0.244358 | 0.016406 |
| flow_9_max | +0.179119 | 0.193100 | 0.013981 |
| flow_49_max | +0.176293 | 0.202761 | 0.026468 |
| flow_11_mean | +0.114344 | 0.123805 | 0.009461 |
| flow_8_std | +0.104370 | 0.119966 | 0.015596 |
| flow_18_max | +0.076567 | 0.104706 | 0.028139 |
| flow_11_std | +0.069076 | 0.069474 | 0.000398 |
| flow_8_mean | +0.058964 | 0.066360 | 0.007396 |
| flow_62_min | -0.044743 | 0.000000 | 0.044743 |
| flow_19_max | -0.042841 | 0.034527 | 0.077368 |
| flow_32_max | -0.038338 | 0.045872 | 0.084210 |
| max_src_out_degree_log | -0.038321 | 0.026549 | 0.064870 |
| flow_55_max | +0.036083 | 0.052196 | 0.016113 |

## Notes

Uses Shaun processed holdout states (attack_labels.npy). RAMX fusion not included.