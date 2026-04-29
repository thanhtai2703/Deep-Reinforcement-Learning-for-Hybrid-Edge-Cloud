# Sim Calibration Report (decomposed model)

- Data: `/home/ubuntu/Deep-Reinforcement-Learning-for-Hybrid-Edge-Cloud/database/dispatcher.db` table `execution_logs`
- Successful rows used: 184

## Per-role exec_time fit

exec_time = α × workload_proxy × (1 + β × cpu_during/100)

| Role   | n   | α    | β     | R²    | exec RMSE (ms) |
| ------ | --- | ---- | ----- | ----- | -------------- |
| cloud  | 56  | 1.04 | -0.03 | 0.038 | 371            |
| edge_1 | 73  | 0.98 | 0.05  | 0.046 | 416            |
| edge_2 | 55  | 0.99 | 0.05  | 0.067 | 450            |

## Per-role overhead constants (avg)

| Role   | submit (ms) | startup (ms) | poll (ms) |
| ------ | ----------- | ------------ | --------- |
| cloud  | 143         | 679          | 3034      |
| edge_1 | 2932        | 1644         | 3519      |
| edge_2 | 782         | 1327         | 3459      |

## Sim-to-real gap on total_ms

| Variant      | MAE (ms) | RMSE (ms) | MAPE (%) | KS distance |
| ------------ | -------- | --------- | -------- | ----------- |
| uncalibrated | 17228    | 19208     | 99.6     | 1.000       |
| calibrated   | 2430     | 6713      | 11.5     | 0.087       |

**Calibration reduces MAE by 85.9%** (17228ms → 2430ms).
**KS distance reduced by 91.3%** (1.000 → 0.087).

## Output files

- `calibration/calibrated_constants.py`
- `calibration/plots/distribution_comparison.png`
- `calibration/plots/scatter_pred_vs_real.png`
