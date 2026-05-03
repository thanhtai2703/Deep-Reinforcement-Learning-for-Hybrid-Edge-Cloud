# Sim Calibration Report (decomposed model)

- Data: `E:\Research\Project_RL\database\dispatcher.db` table `execution_logs`
- Successful rows used: 249

## Per-role exec_time fit

  exec_time = α × workload_proxy × (1 + β × cpu_during/100)

| Role | n | α | β | R² | exec RMSE (ms) |
|---|---|---|---|---|---|
| cloud | 81 | 0.52 | 0.47 | 0.136 | 1296 |
| edge_1 | 85 | 0.66 | 0.37 | 0.024 | 4173 |
| edge_2 | 83 | 0.62 | 0.26 | 0.008 | 2818 |

## Per-role overhead constants (avg)

| Role | submit (ms) | startup (ms) | poll (ms) |
|---|---|---|---|
| cloud | 259 | 679 | 3006 |
| edge_1 | 141 | 1035 | 3189 |
| edge_2 | 133 | 1084 | 3164 |

## Sim-to-real gap on total_ms

| Variant | MAE (ms) | RMSE (ms) | MAPE (%) | KS distance |
|---|---|---|---|---|
| uncalibrated | 11989 | 13589 | 99.5 | 1.000 |
| calibrated | 2097 | 3461 | 13.8 | 0.100 |

**Calibration reduces MAE by 82.5%** (11989ms → 2097ms).
**KS distance reduced by 90.0%** (1.000 → 0.100).

## Output files

- `calibration/calibrated_constants.py`
- `calibration/plots/distribution_comparison.png`
- `calibration/plots/scatter_pred_vs_real.png`
- `calibration/plots/exec_fit_per_role.png`