"""
dispatcher_cli.py
=================
CLI tool để chạy và test Smart Dispatcher.

Usage:
    # Demo mode với baseline
    python -m dispatcher.dispatcher_cli --policy round_robin --num-tasks 50 --demo

    # Demo mode với DQN
    python -m dispatcher.dispatcher_cli --policy dqn --model models/checkpoints/dqn_best.pth --num-tasks 100 --demo

    # Production mode (cần Prometheus)
    python -m dispatcher.dispatcher_cli --policy dqn --model models/checkpoints/dqn_best.pth --prometheus http://prom:9090

    # So sánh tất cả policies
    python -m dispatcher.dispatcher_cli --compare --num-tasks 100 --demo
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

# Thêm project root vào path
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dispatcher.smart_dispatcher import SmartDispatcher
from dispatcher.state_builder import TaskInfo
from dispatcher.model_loader import ALL_POLICY_NAMES, BASELINE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CLI")


def _sample_deadline_ms(rng) -> float:
    """
    Mixture distribution mô phỏng workload thật trên K8s.
    K8s overhead (submit+startup+poll) ≈ 5s, nên deadline tối thiểu
    phải > 5s để task CÓ THỂ đạt SLA.
      30% short  (8-15s)  — task gấp, chỉ kịp nếu chọn đúng node
      50% medium (15-25s) — task vừa, hầu hết đạt SLA nếu dispatch tốt
      20% long   (25-45s) — task thoải mái, signal cho cost optimization
    """
    tier = rng.choice(["short", "medium", "long"], p=[0.3, 0.5, 0.2])
    if tier == "short":
        return float(rng.uniform(8000, 15000))
    if tier == "medium":
        return float(rng.uniform(15000, 25000))
    return float(rng.uniform(25000, 45000))


def generate_tasks(count: int, seed: int = 42, mix: str = "default") -> list:
    """
    Sinh danh sách tasks để test.

    mix:
      - "default": uniform random (như cũ)
      - "tiered": 3 class rõ rệt (40% light, 40% medium, 20% heavy)
                  Để demo task-aware routing.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(count):
        if mix == "tiered":
            r = rng.random()
            if r < 0.40:  # light
                cpu = float(rng.uniform(5, 15))
                ram = float(rng.uniform(5, 15))
                deadline = float(rng.uniform(20000, 40000))
                klass = "light"
            elif r < 0.80:  # medium
                cpu = float(rng.uniform(20, 40))
                ram = float(rng.uniform(20, 30))
                deadline = float(rng.uniform(10000, 25000))
                klass = "medium"
            else:  # heavy
                cpu = float(rng.uniform(45, 60))
                ram = float(rng.uniform(40, 50))
                deadline = float(rng.uniform(8000, 15000))
                klass = "heavy"
            tasks.append(TaskInfo(
                task_id=f"{klass}_{i + 1:06d}",
                cpu_requirement=cpu,
                ram_requirement=ram,
                deadline_ms=deadline,
                priority="high" if klass == "heavy" else "medium",
                payload_type=klass,
            ))
        else:  # default
            tasks.append(TaskInfo(
                task_id=f"task_{i + 1:06d}",
                cpu_requirement=float(rng.uniform(5, 60)),
                ram_requirement=float(rng.uniform(5, 50)),
                deadline_ms=_sample_deadline_ms(rng),
                priority=rng.choice(["low", "medium", "high"]),
                payload_type=rng.choice(["compute", "image", "io"]),
            ))
    return tasks


def run_single_policy(args):
    """Chạy dispatcher với một policy."""
    dispatcher = SmartDispatcher(
        policy_name=args.policy,
        model_path=args.model,
        n_edge_nodes=args.edges,
        prometheus_url=args.prometheus,
        demo_mode=args.demo,
    )

    tasks = generate_tasks(args.num_tasks, seed=args.seed, mix=args.mix)
    logger.info(
        "Dispatching %d tasks with policy=%s concurrency=%d ...",
        len(tasks), args.policy, args.concurrency,
    )

    t0 = time.perf_counter()

    if args.concurrency > 1:
        results = dispatcher.dispatch_concurrent(tasks, max_workers=args.concurrency)
        for i, (task, result) in enumerate(zip(tasks, results)):
            sla_icon = "OK" if result.sla_met else "MISS"
            logger.info(
                "[%3d/%d] %s -> %-8s | lat=%6.1fms | SLA=%s",
                i + 1, len(tasks), task.task_id, result.selected_node,
                result.latency_est_ms, sla_icon,
            )
    else:
        for i, task in enumerate(tasks):
            result = dispatcher.dispatch(task)
            if (i + 1) % max(1, args.num_tasks // 10) == 0 or i == len(tasks) - 1:
                sla_icon = "OK" if result.sla_met else "MISS"
                logger.info(
                    "[%3d/%d] %s -> %-8s | lat=%6.1fms | cost=%.4f | SLA=%s",
                    i + 1, len(tasks), task.task_id, result.selected_node,
                    result.latency_est_ms, result.cost_est, sla_icon,
                )

    elapsed = time.perf_counter() - t0

    dispatcher.print_summary()
    logger.info("Total time: %.2fs (%.1f tasks/sec)", elapsed, len(tasks) / elapsed)

    summary = dispatcher.get_summary()

    # Save summary to CSV (append mode — gom nhiều run vào 1 file)
    if args.save_summary:
        _append_summary_csv(args.save_summary, args, summary, elapsed)
        logger.info("Summary appended to: %s", args.save_summary)

    return summary


def _append_summary_csv(path: str, args, summary: dict, elapsed: float):
    """Flatten summary dict + append vào CSV. Tự tạo header nếu chưa có."""
    n_edge = args.edges
    fieldnames = ["timestamp", "policy", "n_tasks", "concurrency", "seed",
                  "elapsed_sec", "tasks_per_sec",
                  "sla_rate", "sla_count",
                  "lat_mean", "lat_median", "lat_p95", "lat_p99", "lat_std",
                  "cost_total", "cost_avg", "infer_avg_ms"]
    for i in range(n_edge):
        fieldnames += [f"edge_{i+1}_pct", f"edge_{i+1}_count",
                       f"edge_{i+1}_sla", f"edge_{i+1}_avg_lat"]
    fieldnames += ["cloud_pct", "cloud_count", "cloud_sla", "cloud_avg_lat",
                   "rejected_pct", "rejected_count"]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy": args.policy,
        "n_tasks": args.num_tasks,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "elapsed_sec": round(elapsed, 2),
        "tasks_per_sec": round(args.num_tasks / max(elapsed, 1e-6), 2),
    }
    for k in ["sla_rate", "sla_count", "lat_mean", "lat_median",
              "lat_p95", "lat_p99", "lat_std",
              "cost_total", "cost_avg", "infer_avg_ms"]:
        row[k] = summary.get(k)
    for i in range(n_edge):
        pn = summary["per_node"][f"edge_{i+1}"]
        row[f"edge_{i+1}_pct"]     = pn["pct"]
        row[f"edge_{i+1}_count"]   = pn["count"]
        row[f"edge_{i+1}_sla"]     = pn["sla"]
        row[f"edge_{i+1}_avg_lat"] = pn["avg_latency_ms"]
    pn = summary["per_node"]["cloud"]
    row["cloud_pct"]     = pn["pct"]
    row["cloud_count"]   = pn["count"]
    row["cloud_sla"]     = pn["sla"]
    row["cloud_avg_lat"] = pn["avg_latency_ms"]
    pn = summary["per_node"]["rejected"]
    row["rejected_pct"]   = pn["pct"]
    row["rejected_count"] = pn["count"]

    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_comparison(args):
    """So sánh tất cả policies trên cùng workload."""
    tasks = generate_tasks(args.num_tasks, seed=args.seed, mix=args.mix)
    policies = sorted(BASELINE_NAMES)

    # Map policy_name → model_path cho các RL model
    rl_models: dict = {}
    if args.ppo_model and os.path.exists(args.ppo_model):
        rl_models["ppo"] = args.ppo_model
    if args.dqn_model and os.path.exists(args.dqn_model):
        rl_models["dqn"] = args.dqn_model
    # Legacy: --model tự detect theo extension
    if args.model and os.path.exists(args.model):
        if args.model.endswith(".zip") and "ppo" not in rl_models:
            rl_models["ppo"] = args.model
        elif args.model.endswith(".pth") and "dqn" not in rl_models:
            rl_models["dqn"] = args.model

    # RL models đứng đầu bảng
    rl_order = [p for p in ("ppo", "dqn") if p in rl_models]
    policies = rl_order + policies

    results = {}

    for policy_name in policies:
        model_path = rl_models.get(policy_name)

        dispatcher = SmartDispatcher(
            policy_name=policy_name,
            model_path=model_path,
            n_edge_nodes=args.edges,
            demo_mode=True,
        )

        for task in tasks:
            dispatcher.dispatch(task)

        summary = dispatcher.get_summary()
        results[policy_name] = summary

        edges_str = " ".join([
            f"E{i+1}={summary['per_node'][f'edge_{i+1}']['pct']:.0f}%"
            for i in range(args.edges)
        ])
        logger.info(
            "%-18s | SLA=%5.1f%% | Mean=%6.0fms P95=%6.0fms | "
            "Cost=%.4f | %s Cloud=%.0f%% Rej=%.0f%%",
            policy_name, summary["sla_rate"],
            summary["lat_mean"], summary["lat_p95"],
            summary["cost_avg"],
            edges_str,
            summary["per_node"]["cloud"]["pct"],
            summary["per_node"]["rejected"]["pct"],
        )

    # Print comparison table — per-node breakdown
    rl_names = set(rl_models.keys())
    n_edge = args.edges
    edge_cols = "".join([f" {'E'+str(i+1)+'%':>6}" for i in range(n_edge)])
    table_w = 22 + 7 + 11 + 11 + 10 + 6 * n_edge + 7 + 6 + 8

    bar = "═" * table_w
    print(f"\n{bar}")
    header = (f"{'Policy':<22}{'SLA%':>7}"
              f"{'Mean(ms)':>11}{'P95(ms)':>11}{'Cost':>10}"
              f"{edge_cols}{'Cloud%':>7}{'Rej%':>6}{'Infer(ms)':>10}")
    print(header)
    print("─" * table_w)

    for name, s in results.items():
        marker = " ⭐" if name in rl_names else ""
        edge_pcts = "".join([
            f" {s['per_node'][f'edge_{i+1}']['pct']:>6.1f}"
            for i in range(n_edge)
        ])
        print(
            f"{name:<22}{s['sla_rate']:>7.1f}"
            f"{s['lat_mean']:>11.1f}{s['lat_p95']:>11.1f}{s['cost_avg']:>10.4f}"
            f"{edge_pcts}"
            f"{s['per_node']['cloud']['pct']:>7.1f}"
            f"{s['per_node']['rejected']['pct']:>6.1f}"
            f"{s['infer_avg_ms']:>10.3f}{marker}"
        )
    print(f"{bar}\n")

    # Save CSV — flatten per_node
    os.makedirs("experiments/logs", exist_ok=True)
    csv_path = "experiments/logs/cli_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # Build flat schema
        fieldnames = ["policy", "total", "sla_rate", "sla_count",
                      "lat_mean", "lat_median", "lat_p95", "lat_p99", "lat_std",
                      "cost_total", "cost_avg", "infer_avg_ms"]
        for i in range(n_edge):
            fieldnames += [f"edge_{i+1}_pct", f"edge_{i+1}_count",
                           f"edge_{i+1}_sla", f"edge_{i+1}_avg_lat"]
        fieldnames += ["cloud_pct", "cloud_count", "cloud_sla", "cloud_avg_lat",
                       "rejected_pct", "rejected_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, s in results.items():
            row = {"policy": name}
            for k in ["total", "sla_rate", "sla_count",
                      "lat_mean", "lat_median", "lat_p95", "lat_p99", "lat_std",
                      "cost_total", "cost_avg", "infer_avg_ms"]:
                row[k] = s.get(k)
            for i in range(n_edge):
                pn = s["per_node"][f"edge_{i+1}"]
                row[f"edge_{i+1}_pct"]      = pn["pct"]
                row[f"edge_{i+1}_count"]    = pn["count"]
                row[f"edge_{i+1}_sla"]      = pn["sla"]
                row[f"edge_{i+1}_avg_lat"]  = pn["avg_latency_ms"]
            for role, label in [("cloud", "cloud"), ("rejected", "rejected")]:
                pn = s["per_node"][role]
                row[f"{label}_pct"]   = pn["pct"]
                row[f"{label}_count"] = pn["count"]
                if role == "cloud":
                    row["cloud_sla"]      = pn["sla"]
                    row["cloud_avg_lat"]  = pn["avg_latency_ms"]
            writer.writerow(row)
    logger.info("Comparison saved: %s", csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Dispatcher CLI - Hybrid Edge-Cloud Task Scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--policy", default="round_robin",
        help=f"Policy name: {sorted(ALL_POLICY_NAMES)} (default: round_robin)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model checkpoint — auto-detect ppo (.zip) or dqn (.pth)",
    )
    parser.add_argument(
        "--ppo-model", default=None,
        help="Path to PPO checkpoint (.zip) — dùng với --compare",
    )
    parser.add_argument(
        "--dqn-model", default=None,
        help="Path to DQN checkpoint (.pth) — dùng với --compare",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=50,
        help="Number of tasks to dispatch (default: 50)",
    )
    parser.add_argument(
        "--edges", type=int, default=2,
        help="Number of edge nodes (default: 2)",
    )
    parser.add_argument(
        "--prometheus", default="http://localhost:9090",
        help="Prometheus URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--save-summary", default=None,
        help="Append run summary to CSV file (gom nhiều policies vào 1 file để vẽ).",
    )
    parser.add_argument(
        "--mix", choices=("default", "tiered"), default="default",
        help="Workload mix: 'default'=random uniform, "
             "'tiered'=40%% light + 40%% medium + 20%% heavy "
             "(để demo task-aware routing).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Demo mode: use simulation instead of real Prometheus",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all policies on same workload",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Number of parallel dispatch workers. >1 enables concurrent K8s "
             "Jobs to create real CPU/RAM load on nodes (needed for "
             "calibration data with non-zero load).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.compare:
        run_comparison(args)
    else:
        run_single_policy(args)


if __name__ == "__main__":
    main()
