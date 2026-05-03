"""
run_sweep.py
============
Sweep concurrency × policy để thu thập dataset đa dạng cho thesis.

Trục:
  - concurrency ∈ {1, 3, 5, 10, 15, 20}
  - policy ∈ {random, round_robin, least_connection, edge_only, cloud_only, dqn}

Mỗi cell: 1 run × 50 task, seed=42, mix=default.
Tổng: 36 runs × ~3-5 phút = 2-3h.

Append summary vào CSV; có thể resume nếu interrupt (skip cells đã có).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

CONCURRENCIES = [3, 5, 10, 20]
POLICIES = [
    "random",
    "round_robin",
    "least_connection",
    "edge_only",
    "cloud_only",
    "dqn",        # calibrated DQN
    "dqn_uncal",  # uncalibrated DQN (synthetic label — runs --policy dqn + uncal model, CSV row relabeled)
]
DEFAULT_DQN_MODEL = "models/checkpoints/dqn_calibrated/dqn_best.pth"
DEFAULT_DQN_UNCAL_MODEL = "models/checkpoints/dqn_uncalibrated/dqn_best.pth"
DEFAULT_CSV = "experiments/logs/sweep_concurrency.csv"
SETTLE_SECONDS = 45  # cho cluster về idle giữa các run


def already_done(csv_path: str, policy: str, concurrency: int, seed: int, n_tasks: int) -> bool:
    """Check resume: cell này đã có trong CSV chưa."""
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("policy") == policy
                        and int(row.get("concurrency", -1)) == concurrency
                        and int(row.get("seed", -1)) == seed
                        and int(row.get("n_tasks", -1)) == n_tasks):
                    return True
    except Exception:
        return False
    return False


def _row_count(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="") as f:
        return sum(1 for _ in f)


def _relabel_last_row(csv_path: str, expected_prev_count: int, new_label: str):
    """Rewrite the policy field of the last data row to `new_label`.

    Used cho dqn_uncal: dispatcher_cli ghi policy='dqn', cần đổi thành 'dqn_uncal'.
    Safe-guarded: chỉ relabel nếu CSV tăng đúng 1 row so với trước run.
    """
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) != expected_prev_count + 1:
        print(f"  [WARN] expected {expected_prev_count + 1} rows, got {len(rows)} — skip relabel",
              flush=True)
        return
    header = rows[0]
    if "policy" not in header:
        return
    pol_idx = header.index("policy")
    rows[-1][pol_idx] = new_label
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def run_one(policy: str, concurrency: int, n_tasks: int, seed: int,
            prometheus: str, csv_path: str,
            dqn_model: str, dqn_uncal_model: str) -> bool:
    # Resolve actual --policy + model. dqn_uncal là label tổng hợp.
    is_uncal = (policy == "dqn_uncal")
    real_policy = "dqn" if is_uncal else policy

    cmd = [
        sys.executable, "-m", "dispatcher.dispatcher_cli",
        "--policy", real_policy,
        "--num-tasks", str(n_tasks),
        "--concurrency", str(concurrency),
        "--seed", str(seed),
        "--prometheus", prometheus,
        "--save-summary", csv_path,
    ]
    if real_policy == "dqn":
        cmd += ["--dqn-model", dqn_uncal_model if is_uncal else dqn_model]

    print(f"  CMD: {' '.join(cmd)}", flush=True)
    rows_before = _row_count(csv_path)
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            print(f"  [FAIL] returncode={result.returncode} after {elapsed:.0f}s", flush=True)
            return False
        if is_uncal:
            _relabel_last_row(csv_path, rows_before, "dqn_uncal")
        print(f"  [OK] in {elapsed:.0f}s", flush=True)
        return True
    except Exception as e:
        print(f"  [EXC] {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prometheus", default="http://localhost:9090")
    ap.add_argument("--dqn-model", default=DEFAULT_DQN_MODEL)
    ap.add_argument("--dqn-uncal-model", default=DEFAULT_DQN_UNCAL_MODEL)
    ap.add_argument("--settle", type=int, default=SETTLE_SECONDS,
                    help="Seconds to wait between runs for cluster to idle.")
    ap.add_argument("--concurrencies", type=int, nargs="+", default=CONCURRENCIES)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    cells = [(p, c) for c in args.concurrencies for p in args.policies]
    total = len(cells)

    print(f"Sweep plan: {total} cells = {len(args.concurrencies)} concurrency × "
          f"{len(args.policies)} policy")
    print(f"  concurrencies = {args.concurrencies}")
    print(f"  policies      = {args.policies}")
    print(f"  csv           = {args.csv}")
    print(f"  settle        = {args.settle}s between runs")
    print()

    done = 0
    skipped = 0
    failed = 0
    t_sweep_start = time.perf_counter()

    for idx, (policy, concurrency) in enumerate(cells, 1):
        prefix = f"[{idx}/{total}] policy={policy} c={concurrency}"

        if already_done(args.csv, policy, concurrency, args.seed, args.n_tasks):
            print(f"{prefix} — SKIP (already in CSV)", flush=True)
            skipped += 1
            done += 1
            continue

        elapsed_sweep = time.perf_counter() - t_sweep_start
        if done > skipped:
            avg_per_run = elapsed_sweep / max(done - skipped, 1)
            remaining = (total - idx + 1) * (avg_per_run + args.settle)
            eta = datetime.now() + timedelta(seconds=remaining)
            eta_str = f" | ETA {eta.strftime('%H:%M:%S')}"
        else:
            eta_str = ""
        print(f"{prefix}{eta_str}", flush=True)

        ok = run_one(
            policy=policy,
            concurrency=concurrency,
            n_tasks=args.n_tasks,
            seed=args.seed,
            prometheus=args.prometheus,
            csv_path=args.csv,
            dqn_model=args.dqn_model,
            dqn_uncal_model=args.dqn_uncal_model,
        )
        if ok:
            done += 1
        else:
            failed += 1

        # settle giữa các run cho cluster về idle
        if idx < total:
            print(f"  settling {args.settle}s...", flush=True)
            time.sleep(args.settle)

    total_elapsed = time.perf_counter() - t_sweep_start
    print()
    print(f"Done. {done}/{total} succeeded ({skipped} skipped, {failed} failed) "
          f"in {total_elapsed/60:.1f} min.")
    print(f"CSV: {args.csv}")


if __name__ == "__main__":
    main()
