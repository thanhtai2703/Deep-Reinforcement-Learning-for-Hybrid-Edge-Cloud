"""
analyze_calibration.py
======================
Phân tích chi tiết calibration pipeline:
1. Data quality trong execution_logs
2. Timing breakdown per role
3. Kiểm tra calibrated_constants
4. So sánh uncalibrated vs calibrated predictions
5. Phát hiện vấn đề

Chạy:
    python calibration/analyze_calibration.py
"""

import sqlite3
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "database" / "dispatcher.db"

# ── Section 1: Data Overview ──────────────────────────────────────────────

print("=" * 70)
print("  SECTION 1: DATA OVERVIEW")
print("=" * 70)

conn = sqlite3.connect(str(DB_PATH))
conn.row_factory = sqlite3.Row

# Total rows
total = conn.execute("SELECT COUNT(*) FROM execution_logs").fetchone()[0]
print(f"\nTotal rows in execution_logs: {total}")

# By status
print("\nRows by exec_status:")
for row in conn.execute(
    "SELECT exec_status, COUNT(*) as cnt FROM execution_logs GROUP BY exec_status ORDER BY cnt DESC"
):
    print(f"  {row['exec_status'] or 'NULL':20s} : {row['cnt']}")

# By role
print("\nRows by target_role:")
for row in conn.execute(
    "SELECT target_role, COUNT(*) as cnt FROM execution_logs GROUP BY target_role ORDER BY cnt DESC"
):
    print(f"  {row['target_role'] or 'NULL':20s} : {row['cnt']}")

# By policy
print("\nRows by policy_name:")
for row in conn.execute(
    "SELECT policy_name, COUNT(*) as cnt FROM execution_logs GROUP BY policy_name ORDER BY cnt DESC"
):
    print(f"  {row['policy_name'] or 'NULL':20s} : {row['cnt']}")

# Usable rows
usable = conn.execute("""
    SELECT COUNT(*) FROM execution_logs
    WHERE exec_status='succeeded'
      AND cpu_during_exec IS NOT NULL
      AND total_ms IS NOT NULL
      AND exec_time_ms IS NOT NULL
      AND target_role IS NOT NULL
      AND cpu_requirement IS NOT NULL
      AND deadline_ms IS NOT NULL
""").fetchone()[0]
print(f"\nUsable rows (for calibration): {usable} / {total} ({usable/total*100:.1f}%)")

# NULL analysis
print("\nNULL counts in key columns (all rows):")
for col in ["cpu_during_exec", "ram_during_exec", "total_ms", "exec_time_ms",
            "submit_overhead_ms", "container_startup_ms", "poll_overhead_ms",
            "target_role", "cpu_requirement"]:
    null_cnt = conn.execute(
        f"SELECT COUNT(*) FROM execution_logs WHERE {col} IS NULL"
    ).fetchone()[0]
    print(f"  {col:25s} : {null_cnt} NULLs ({null_cnt/total*100:.1f}%)")


# ── Section 2: Timing Breakdown Per Role ──────────────────────────────────

print("\n" + "=" * 70)
print("  SECTION 2: TIMING BREAKDOWN PER ROLE (succeeded only)")
print("=" * 70)

roles_data = {}
for row in conn.execute("""
    SELECT target_role,
           COUNT(*) as n,
           AVG(total_ms) as avg_total,
           AVG(submit_overhead_ms) as avg_submit,
           AVG(container_startup_ms) as avg_startup,
           AVG(exec_time_ms) as avg_exec,
           AVG(poll_overhead_ms) as avg_poll,
           AVG(cpu_during_exec) as avg_cpu_during,
           AVG(ram_during_exec) as avg_ram_during,
           AVG(cpu_requirement) as avg_cpu_req,
           AVG(deadline_ms) as avg_deadline,
           MIN(total_ms) as min_total,
           MAX(total_ms) as max_total,
           MIN(exec_time_ms) as min_exec,
           MAX(exec_time_ms) as max_exec
    FROM execution_logs
    WHERE exec_status='succeeded'
      AND target_role IS NOT NULL
    GROUP BY target_role
"""):
    role = row["target_role"]
    roles_data[role] = dict(row)
    print(f"\n  ── {role} (n={row['n']}) ──")
    print(f"    total_ms          : avg={row['avg_total']:.0f}  min={row['min_total']}  max={row['max_total']}")
    print(f"    submit_overhead   : avg={row['avg_submit']:.0f}ms")
    print(f"    container_startup : avg={row['avg_startup']:.0f}ms")
    print(f"    exec_time         : avg={row['avg_exec']:.0f}ms  min={row['min_exec']}  max={row['max_exec']}")
    print(f"    poll_overhead     : avg={row['avg_poll']:.0f}ms")
    
    # Sanity check: components should sum to ~total
    comp_sum = (row['avg_submit'] or 0) + (row['avg_startup'] or 0) + (row['avg_exec'] or 0) + (row['avg_poll'] or 0)
    print(f"    sum(components)   : {comp_sum:.0f}ms  vs  total={row['avg_total']:.0f}ms  (gap={row['avg_total']-comp_sum:.0f}ms)")
    print(f"    cpu_during_exec   : avg={row['avg_cpu_during']:.1f}%")
    print(f"    ram_during_exec   : avg={row['avg_ram_during']:.1f}%")
    print(f"    cpu_requirement   : avg={row['avg_cpu_req']:.1f}")
    print(f"    deadline_ms       : avg={row['avg_deadline']:.0f}")

    # Percentage breakdown
    if row['avg_total'] and row['avg_total'] > 0:
        t = row['avg_total']
        print(f"    %% breakdown      : submit={((row['avg_submit'] or 0)/t*100):.1f}% "
              f"startup={((row['avg_startup'] or 0)/t*100):.1f}% "
              f"exec={((row['avg_exec'] or 0)/t*100):.1f}% "
              f"poll={((row['avg_poll'] or 0)/t*100):.1f}%")


# ── Section 3: Calibrated Constants Validation ────────────────────────────

print("\n" + "=" * 70)
print("  SECTION 3: CALIBRATED CONSTANTS VALIDATION")
print("=" * 70)

try:
    from calibration.calibrated_constants import (
        EXEC_CALIBRATION, OVERHEAD_CALIBRATION,
        workload_proxy_ms, predict_total_ms,
    )
    print("\n  calibrated_constants.py loaded OK ✓")
    
    for role in sorted(EXEC_CALIBRATION.keys()):
        ec = EXEC_CALIBRATION[role]
        oh = OVERHEAD_CALIBRATION[role]
        print(f"\n  ── {role} ──")
        print(f"    α (alpha) = {ec['alpha']:.4f}  (hardware scaling, ideal ≈ 1.0)")
        print(f"    β (beta)  = {ec['beta']:.4f}   (contention slope, 0 = no contention)")
        print(f"    R²        = {ec['r2']:.6f}  ({'⚠ VERY LOW' if ec['r2'] < 0.1 else '✓'})")
        print(f"    RMSE      = {ec['rmse_ms']:.0f}ms")
        print(f"    n samples = {ec['n']}")
        print(f"    submit_overhead    = {oh['submit_overhead_ms']:.0f}ms")
        print(f"    container_startup  = {oh['container_startup_ms']:.0f}ms")
        print(f"    poll_overhead      = {oh['poll_overhead_ms']:.0f}ms")
        print(f"    fixed_overhead     = {oh['submit_overhead_ms'] + oh['container_startup_ms'] + oh['poll_overhead_ms']:.0f}ms")
        
except ImportError as e:
    print(f"\n  ⚠ Cannot import calibrated_constants: {e}")


# ── Section 4: Prediction Test ────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SECTION 4: PREDICTION ACCURACY TEST")
print("=" * 70)

try:
    # Load actual data for comparison
    rows = conn.execute("""
        SELECT target_role, cpu_requirement, deadline_ms,
               cpu_during_exec, total_ms, exec_time_ms,
               submit_overhead_ms, container_startup_ms, poll_overhead_ms,
               est_latency_ms
        FROM execution_logs
        WHERE exec_status='succeeded'
          AND cpu_during_exec IS NOT NULL
          AND total_ms IS NOT NULL
          AND exec_time_ms IS NOT NULL
          AND target_role IS NOT NULL
          AND cpu_requirement IS NOT NULL
          AND deadline_ms IS NOT NULL
    """).fetchall()
    
    results = {"uncalibrated_env": [], "calibrated": []}
    per_role = {}
    
    for r in rows:
        role = r["target_role"]
        real_total = r["total_ms"]
        
        # Calibrated prediction
        cal_pred = predict_total_ms(
            role=role,
            cpu_requirement=r["cpu_requirement"],
            deadline_ms=r["deadline_ms"],
            cpu_during_exec=r["cpu_during_exec"],
        )
        
        # Uncalibrated env prediction (from edge_cloud_env.py logic)
        UNCAL_EDGE_BASE = 17.5   # avg of uniform(5,30)
        UNCAL_CLOUD_BASE = 55.0  # avg of uniform(30,80)
        UNCAL_LOAD_COEF = 2.0
        base = UNCAL_CLOUD_BASE if role == "cloud" else UNCAL_EDGE_BASE
        uncal_pred = base * (1.0 + r["cpu_during_exec"] / 100.0 * UNCAL_LOAD_COEF)
        
        results["uncalibrated_env"].append((real_total, uncal_pred))
        results["calibrated"].append((real_total, cal_pred))
        
        if role not in per_role:
            per_role[role] = {"real": [], "cal": [], "uncal": [], "est": []}
        per_role[role]["real"].append(real_total)
        per_role[role]["cal"].append(cal_pred)
        per_role[role]["uncal"].append(uncal_pred)
        if r["est_latency_ms"] is not None:
            per_role[role]["est"].append((real_total, r["est_latency_ms"]))
    
    # Overall metrics
    for label, pairs in results.items():
        reals = np.array([p[0] for p in pairs])
        preds = np.array([p[1] for p in pairs])
        errors = reals - preds
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        mape = np.mean(abs_errors / np.maximum(reals, 1)) * 100
        
        print(f"\n  {label}:")
        print(f"    MAE  = {mae:.0f}ms")
        print(f"    RMSE = {rmse:.0f}ms")
        print(f"    MAPE = {mape:.1f}%")
        print(f"    Mean real = {np.mean(reals):.0f}ms, Mean pred = {np.mean(preds):.0f}ms")
        print(f"    Pred range: [{np.min(preds):.0f}, {np.max(preds):.0f}]ms")
        print(f"    Real range: [{np.min(reals):.0f}, {np.max(reals):.0f}]ms")
    
    # Per-role breakdown
    print(f"\n  ── Per-role accuracy ──")
    for role in sorted(per_role.keys()):
        d = per_role[role]
        reals = np.array(d["real"])
        cals = np.array(d["cal"])
        uncals = np.array(d["uncal"])
        
        cal_mae = np.mean(np.abs(reals - cals))
        uncal_mae = np.mean(np.abs(reals - uncals))
        
        print(f"\n    {role} (n={len(reals)}):")
        print(f"      Real      : mean={np.mean(reals):.0f}ms  std={np.std(reals):.0f}ms  "
              f"[{np.min(reals):.0f}, {np.max(reals):.0f}]")
        print(f"      Calibrated: MAE={cal_mae:.0f}ms  "
              f"pred_mean={np.mean(cals):.0f}ms")
        print(f"      Uncal env : MAE={uncal_mae:.0f}ms  "
              f"pred_mean={np.mean(uncals):.0f}ms")
        
        if len(d["est"]) > 0:
            est_reals = np.array([e[0] for e in d["est"]])
            est_preds = np.array([e[1] for e in d["est"]])
            est_mae = np.mean(np.abs(est_reals - est_preds))
            print(f"      est_latency (at dispatch): MAE={est_mae:.0f}ms  n={len(d['est'])}")

except Exception as e:
    print(f"\n  ⚠ Error in prediction test: {e}")
    import traceback
    traceback.print_exc()


# ── Section 5: Workload Proxy Analysis ────────────────────────────────────

print("\n" + "=" * 70)
print("  SECTION 5: WORKLOAD PROXY vs ACTUAL EXEC_TIME")
print("=" * 70)

try:
    rows2 = conn.execute("""
        SELECT target_role, cpu_requirement, deadline_ms,
               exec_time_ms, cpu_during_exec
        FROM execution_logs
        WHERE exec_status='succeeded'
          AND exec_time_ms IS NOT NULL
          AND cpu_requirement IS NOT NULL
          AND deadline_ms IS NOT NULL
    """).fetchall()
    
    for role in sorted(set(r["target_role"] for r in rows2 if r["target_role"])):
        sub = [r for r in rows2 if r["target_role"] == role]
        
        wps = []
        execs = []
        us = []
        for r in sub:
            wp = workload_proxy_ms(r["cpu_requirement"], r["deadline_ms"])
            if wp > 100:  # same filter as calibrate_env.py
                wps.append(wp)
                execs.append(r["exec_time_ms"])
                us.append(r["exec_time_ms"] / wp)
        
        if wps:
            wps = np.array(wps)
            execs = np.array(execs)
            us = np.array(us)
            corr = np.corrcoef(wps, execs)[0, 1] if len(wps) > 1 else 0
            
            print(f"\n  {role} (n={len(wps)}):")
            print(f"    workload_proxy : mean={np.mean(wps):.0f}ms  [{np.min(wps):.0f}, {np.max(wps):.0f}]")
            print(f"    exec_time      : mean={np.mean(execs):.0f}ms  [{np.min(execs):.0f}, {np.max(execs):.0f}]")
            print(f"    u=exec/proxy   : mean={np.mean(us):.3f}  std={np.std(us):.3f}  [{np.min(us):.3f}, {np.max(us):.3f}]")
            print(f"    correlation(proxy, exec) = {corr:.4f}")

except Exception as e:
    print(f"  ⚠ Error: {e}")


# ── Section 6: β ≈ 0 Investigation ───────────────────────────────────────

print("\n" + "=" * 70)
print("  SECTION 6: β ≈ 0 INVESTIGATION (CPU vs exec_time)")
print("=" * 70)

try:
    for role in sorted(set(r["target_role"] for r in rows2 if r["target_role"])):
        sub = [r for r in rows2 if r["target_role"] == role and r["cpu_during_exec"] is not None]
        
        if len(sub) < 5:
            continue
            
        cpus = np.array([r["cpu_during_exec"] for r in sub])
        execs = np.array([r["exec_time_ms"] for r in sub])
        
        corr = np.corrcoef(cpus, execs)[0, 1] if len(cpus) > 1 else 0
        
        print(f"\n  {role} (n={len(sub)}):")
        print(f"    cpu_during_exec: mean={np.mean(cpus):.1f}%  std={np.std(cpus):.1f}%  "
              f"[{np.min(cpus):.1f}%, {np.max(cpus):.1f}%]")
        print(f"    exec_time_ms   : mean={np.mean(execs):.0f}ms  std={np.std(execs):.0f}ms")
        print(f"    correlation(cpu, exec_time) = {corr:.4f}")
        
        # Check if CPU range is too narrow
        cpu_range = np.max(cpus) - np.min(cpus)
        print(f"    CPU range span = {cpu_range:.1f}%  "
              f"({'⚠ NARROW - may explain low R²' if cpu_range < 30 else '✓ sufficient'})")
        
        # Quartile analysis
        q25 = np.percentile(cpus, 25)
        q75 = np.percentile(cpus, 75)
        low_cpu = execs[cpus <= q25]
        high_cpu = execs[cpus >= q75]
        if len(low_cpu) > 0 and len(high_cpu) > 0:
            print(f"    exec_time when CPU≤{q25:.0f}%: mean={np.mean(low_cpu):.0f}ms (n={len(low_cpu)})")
            print(f"    exec_time when CPU≥{q75:.0f}%: mean={np.mean(high_cpu):.0f}ms (n={len(high_cpu)})")
            diff_pct = (np.mean(high_cpu) - np.mean(low_cpu)) / np.mean(low_cpu) * 100
            print(f"    Difference: {diff_pct:+.1f}%  "
                  f"({'⚠ NO contention effect' if abs(diff_pct) < 10 else '✓ contention visible'})")

except Exception as e:
    print(f"  ⚠ Error: {e}")


# ── Section 7: SLA Analysis ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SECTION 7: SLA ANALYSIS (real vs estimated)")
print("=" * 70)

try:
    sla_rows = conn.execute("""
        SELECT target_role, 
               COUNT(*) as n,
               SUM(CASE WHEN sla_met_real = 1 THEN 1 ELSE 0 END) as sla_ok,
               SUM(CASE WHEN est_sla_met = 1 THEN 1 ELSE 0 END) as est_sla_ok,
               AVG(total_ms) as avg_total,
               AVG(deadline_ms) as avg_deadline
        FROM execution_logs
        WHERE exec_status='succeeded'
          AND sla_met_real IS NOT NULL
          AND target_role IS NOT NULL
        GROUP BY target_role
    """).fetchall()
    
    for r in sla_rows:
        real_rate = r["sla_ok"] / r["n"] * 100 if r["n"] > 0 else 0
        est_rate = r["est_sla_ok"] / r["n"] * 100 if r["n"] > 0 else 0
        print(f"\n  {r['target_role']} (n={r['n']}):")
        print(f"    SLA rate (real)     : {real_rate:.1f}%  ({r['sla_ok']}/{r['n']})")
        print(f"    SLA rate (estimated): {est_rate:.1f}%  ({r['est_sla_ok']}/{r['n']})")
        print(f"    avg total_ms={r['avg_total']:.0f}  avg deadline={r['avg_deadline']:.0f}ms")
        if r['avg_total'] and r['avg_deadline']:
            print(f"    ratio total/deadline = {r['avg_total']/r['avg_deadline']:.2f}  "
                  f"({'⚠ tasks usually exceed deadline' if r['avg_total'] > r['avg_deadline'] else '✓ mostly within deadline'})")

except Exception as e:
    print(f"  ⚠ Error: {e}")


# ── Section 8: Summary ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SECTION 8: SUMMARY & ISSUES")
print("=" * 70)

print("""
  Issues to investigate:
  1. R² ≈ 0 for all roles → Is cgroup isolation hiding contention?
  2. Are CPU ranges during execution diverse enough?
  3. Does workload_proxy correlate well with exec_time?
  4. How large is the sim-to-real gap for uncalibrated env?
  5. SLA: real vs estimated accuracy
""")

conn.close()
print("Done.")
