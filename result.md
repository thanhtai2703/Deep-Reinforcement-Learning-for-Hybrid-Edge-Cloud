======================================================================
  SECTION 1: DATA OVERVIEW
======================================================================

Total rows in execution_logs: 700

Rows by exec_status:
  succeeded            : 700

Rows by target_role:
  edge_2               : 240
  edge_1               : 233
  cloud                : 227

Rows by policy_name:
  round_robin          : 350
  random               : 200
  least_connection     : 150

Usable rows (for calibration): 259 / 700 (37.0%)

NULL counts in key columns (all rows):
  cpu_during_exec           : 441 NULLs (63.0%)
  ram_during_exec           : 1 NULLs (0.1%)
  total_ms                  : 0 NULLs (0.0%)
  exec_time_ms              : 0 NULLs (0.0%)
  submit_overhead_ms        : 0 NULLs (0.0%)
  container_startup_ms      : 0 NULLs (0.0%)
  poll_overhead_ms          : 0 NULLs (0.0%)
  target_role               : 0 NULLs (0.0%)
  cpu_requirement           : 0 NULLs (0.0%)

======================================================================
  SECTION 2: TIMING BREAKDOWN PER ROLE (succeeded only)
======================================================================

  ── cloud (n=227) ──
    total_ms          : avg=10798  min=5102  max=30649
    submit_overhead   : avg=454ms
    container_startup : avg=907ms
    exec_time         : avg=5991ms  min=1000  max=26000
    poll_overhead     : avg=3208ms
    sum(components)   : 10560ms  vs  total=10798ms  (gap=238ms)
    cpu_during_exec   : avg=75.0%
    ram_during_exec   : avg=24.9%
    cpu_requirement   : avg=33.0
    deadline_ms       : avg=10535
    %% breakdown      : submit=4.2% startup=8.4% exec=55.5% poll=29.7%

  ── edge_1 (n=233) ──
    total_ms          : avg=10718  min=5098  max=33485
    submit_overhead   : avg=232ms
    container_startup : avg=1421ms
    exec_time         : avg=5399ms  min=1000  max=28000
    poll_overhead     : avg=3459ms
    sum(components)   : 10510ms  vs  total=10718ms  (gap=208ms)
    cpu_during_exec   : avg=71.5%
    ram_during_exec   : avg=35.2%
    cpu_requirement   : avg=32.1
    deadline_ms       : avg=9532
    %% breakdown      : submit=2.2% startup=13.3% exec=50.4% poll=32.3%

  ── edge_2 (n=240) ──
    total_ms          : avg=11652  min=5116  max=31686
    submit_overhead   : avg=279ms
    container_startup : avg=1429ms
    exec_time         : avg=6238ms  min=1000  max=27000
    poll_overhead     : avg=3487ms
    sum(components)   : 11433ms  vs  total=11652ms  (gap=219ms)
    cpu_during_exec   : avg=74.8%
    ram_during_exec   : avg=32.3%
    cpu_requirement   : avg=32.4
    deadline_ms       : avg=10919
    %% breakdown      : submit=2.4% startup=12.3% exec=53.5% poll=29.9%

======================================================================
  SECTION 3: CALIBRATED CONSTANTS VALIDATION
======================================================================

  calibrated_constants.py loaded OK ✓

  ── cloud ──
    α (alpha) = 1.0407  (hardware scaling, ideal ≈ 1.0)
    β (beta)  = -0.0046   (contention slope, 0 = no contention)
    R²        = 0.000407  (⚠ VERY LOW)
    RMSE      = 476ms
    n samples = 80
    submit_overhead    = 488ms
    container_startup  = 900ms
    poll_overhead      = 3129ms
    fixed_overhead     = 4516ms

  ── edge_1 ──
    α (alpha) = 0.9996  (hardware scaling, ideal ≈ 1.0)
    β (beta)  = 0.0283   (contention slope, 0 = no contention)
    R²        = 0.012584  (⚠ VERY LOW)
    RMSE      = 483ms
    n samples = 78
    submit_overhead    = 179ms
    container_startup  = 1526ms
    poll_overhead      = 3469ms
    fixed_overhead     = 5174ms

  ── edge_2 ──
    α (alpha) = 1.0112  (hardware scaling, ideal ≈ 1.0)
    β (beta)  = 0.0334   (contention slope, 0 = no contention)
    R²        = 0.017537  (⚠ VERY LOW)
    RMSE      = 468ms
    n samples = 97
    submit_overhead    = 247ms
    container_startup  = 1557ms
    poll_overhead      = 3494ms
    fixed_overhead     = 5298ms

======================================================================
  SECTION 4: PREDICTION ACCURACY TEST
======================================================================

  uncalibrated_env:
    MAE  = 15765ms
    RMSE = 16590ms
    MAPE = 99.5%
    Mean real = 15838ms, Mean pred = 73ms
    Pred range: [22, 165]ms
    Real range: [8160, 33485]ms

  calibrated:
    MAE  = 941ms
    RMSE = 1428ms
    MAPE = 6.3%
    Mean real = 15838ms, Mean pred = 15665ms
    Pred range: [7500, 34130]ms
    Real range: [8160, 33485]ms

  ── Per-role accuracy ──

    cloud (n=81):
      Real      : mean=15421ms  std=4958ms  [9502, 30649]
      Calibrated: MAE=962ms  pred_mean=15257ms
      Uncal env : MAE=15283ms  pred_mean=138ms
      est_latency (at dispatch): MAE=10393ms  n=81

    edge_1 (n=79):
      Real      : mean=15767ms  std=5342ms  [8160, 33485]
      Calibrated: MAE=832ms  pred_mean=15587ms
      Uncal env : MAE=15724ms  pred_mean=43ms
      est_latency (at dispatch): MAE=10980ms  n=79

    edge_2 (n=99):
      Real      : mean=16235ms  std=5160ms  [9105, 31686]
      Calibrated: MAE=1012ms  pred_mean=16061ms
      Uncal env : MAE=16192ms  pred_mean=44ms
      est_latency (at dispatch): MAE=11022ms  n=99

======================================================================
  SECTION 5: WORKLOAD PROXY vs ACTUAL EXEC_TIME
======================================================================

  cloud (n=227):
    workload_proxy : mean=5639ms  [240, 25874]
    exec_time      : mean=5991ms  [1000, 26000]
    u=exec/proxy   : mean=1.170  std=0.402  [0.692, 4.171]
    correlation(proxy, exec) = 0.9955

  edge_1 (n=233):
    workload_proxy : mean=5171ms  [285, 28430]
    exec_time      : mean=5399ms  [1000, 28000]
    u=exec/proxy   : mean=1.181  std=0.408  [0.590, 3.505]
    correlation(proxy, exec) = 0.9956

  edge_2 (n=240):
    workload_proxy : mean=5937ms  [367, 26796]
    exec_time      : mean=6238ms  [1000, 27000]
    u=exec/proxy   : mean=1.168  std=0.434  [0.615, 5.466]
    correlation(proxy, exec) = 0.9960

======================================================================
  SECTION 6: β ≈ 0 INVESTIGATION (CPU vs exec_time)
======================================================================

  cloud (n=81):
    cpu_during_exec: mean=75.0%  std=24.5%  [22.6%, 99.7%]
    exec_time_ms   : mean=10654ms  std=4889ms
    correlation(cpu, exec_time) = 0.0411
    CPU range span = 77.1%  (✓ sufficient)
    exec_time when CPU≤64%: mean=10429ms (n=21)
    exec_time when CPU≥94%: mean=10619ms (n=21)
    Difference: +1.8%  (⚠ NO contention effect)

  edge_1 (n=79):
    cpu_during_exec: mean=71.5%  std=23.4%  [12.2%, 99.4%]
    exec_time_ms   : mean=10342ms  std=5141ms
    correlation(cpu, exec_time) = -0.1175
    CPU range span = 87.2%  (✓ sufficient)
    exec_time when CPU≤56%: mean=11700ms (n=20)
    exec_time when CPU≥92%: mean=9667ms (n=21)
    Difference: -17.4%  (✓ contention visible)

  edge_2 (n=99):
    cpu_during_exec: mean=74.8%  std=22.7%  [18.9%, 99.5%]
    exec_time_ms   : mean=10707ms  std=5383ms
    correlation(cpu, exec_time) = -0.0544
    CPU range span = 80.7%  (✓ sufficient)
    exec_time when CPU≤64%: mean=10720ms (n=25)
    exec_time when CPU≥94%: mean=10600ms (n=25)
    Difference: -1.1%  (⚠ NO contention effect)

======================================================================
  SECTION 7: SLA ANALYSIS (real vs estimated)
======================================================================

  cloud (n=227):
    SLA rate (real)     : 38.3%  (87/227)
    SLA rate (estimated): 75.8%  (172/227)
    avg total_ms=10798  avg deadline=10535ms
    ratio total/deadline = 1.02  (⚠ tasks usually exceed deadline)

  edge_1 (n=233):
    SLA rate (real)     : 27.9%  (65/233)
    SLA rate (estimated): 84.5%  (197/233)
    avg total_ms=10718  avg deadline=9532ms
    ratio total/deadline = 1.12  (⚠ tasks usually exceed deadline)

  edge_2 (n=240):
    SLA rate (real)     : 34.2%  (82/240)
    SLA rate (estimated): 82.9%  (199/240)
    avg total_ms=11652  avg deadline=10919ms
    ratio total/deadline = 1.07  (⚠ tasks usually exceed deadline)

======================================================================
  SECTION 8: SUMMARY & ISSUES
======================================================================

  Issues to investigate:
  1. R² ≈ 0 for all roles → Is cgroup isolation hiding contention?
  2. Are CPU ranges during execution diverse enough?
  3. Does workload_proxy correlate well with exec_time?
  4. How large is the sim-to-real gap for uncalibrated env?
  5. SLA: real vs estimated accuracy

Done.