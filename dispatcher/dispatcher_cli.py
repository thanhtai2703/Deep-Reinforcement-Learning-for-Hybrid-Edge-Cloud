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


def generate_tasks(count: int, seed: int = 42) -> list:
    """Sinh danh sách tasks ngẫu nhiên để test."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(count):
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

    tasks = generate_tasks(args.num_tasks, seed=args.seed)
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

    return dispatcher.get_summary()


def run_comparison(args):
    """So sánh tất cả policies trên cùng workload."""
    tasks = generate_tasks(args.num_tasks, seed=args.seed)
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

        logger.info(
            "%-18s | SLA=%5.1f%% | Lat=%6.1fms | Cost=%.4f | "
            "Edge=%4.1f%% Cloud=%4.1f%% Rej=%4.1f%%",
            policy_name, summary["sla_rate"], summary["avg_latency_ms"],
            summary["avg_cost"],
            summary["edge_usage_pct"], summary["cloud_usage_pct"],
            summary["reject_usage_pct"],
        )

    # Print comparison table
    rl_names = set(rl_models.keys())
    print(f"\n{'=' * 96}")
    print(f"{'Policy':<20} {'SLA%':>8} {'Avg Lat(ms)':>12} "
          f"{'P95 Lat(ms)':>12} {'Avg Cost':>10} {'Edge%':>7} "
          f"{'Cloud%':>7} {'Rej%':>6}")
    print(f"{'-' * 96}")
    for name, s in results.items():
        marker = " <-- RL" if name in rl_names else ""
        print(
            f"{name:<20} {s['sla_rate']:>8.1f} {s['avg_latency_ms']:>12.1f} "
            f"{s['p95_latency_ms']:>12.1f} {s['avg_cost']:>10.5f} "
            f"{s['edge_usage_pct']:>7.1f} {s['cloud_usage_pct']:>7.1f} "
            f"{s['reject_usage_pct']:>6.1f}{marker}"
        )
    print(f"{'=' * 96}\n")

    # Save CSV
    os.makedirs("experiments/logs", exist_ok=True)
    csv_path = "experiments/logs/cli_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy"] + list(next(iter(results.values())).keys()))
        writer.writeheader()
        for name, s in results.items():
            writer.writerow({"policy": name, **s})
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
