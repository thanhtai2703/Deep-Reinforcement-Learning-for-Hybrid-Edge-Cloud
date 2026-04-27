"""
demo_runner.py
==============
Chạy dispatcher liên tục với task_generator, đồng thời expose metrics
ra Prometheus endpoint (port 8000) để Grafana scrape.

Usage:
    # Simulation mode (không cần infra thật)
    python demo_runner.py --policy ppo --model models/checkpoints/ppo/ppo_best.zip

    # So sánh policies tuần tự (mỗi policy chạy N giây)
    python demo_runner.py --compare --interval 60

    # Chỉ dùng baseline
    python demo_runner.py --policy round_robin
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import threading
import random
from datetime import datetime

# Thêm project root
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dispatcher.smart_dispatcher import SmartDispatcher
from dispatcher.state_builder import TaskInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DemoRunner")

# ── Prometheus metrics export ────────────────────────────────────────────────
try:
    from prometheus_client import (
        Gauge, Counter, Histogram, start_http_server, REGISTRY
    )
    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False
    logger.warning("prometheus-client not installed — metrics endpoint disabled")

if PROM_AVAILABLE:
    _task_total   = Counter("dispatcher_tasks_total",   "Total tasks dispatched", ["policy", "node"])
    _sla_met      = Counter("dispatcher_sla_met_total", "Tasks meeting SLA",      ["policy"])
    _sla_miss     = Counter("dispatcher_sla_miss_total","Tasks missing SLA",      ["policy"])
    _latency_hist = Histogram("dispatcher_latency_ms",  "Task latency histogram",  ["policy"],
                              buckets=[20, 50, 100, 150, 200, 300, 400, 500])
    _sla_rate_g   = Gauge("dispatcher_sla_rate",        "Current SLA rate %",      ["policy"])
    _edge1_cpu    = Gauge("sim_edge1_cpu_pct",           "Simulated edge-1 CPU %")
    _edge2_cpu    = Gauge("sim_edge2_cpu_pct",           "Simulated edge-2 CPU %")
    _cloud_cpu    = Gauge("sim_cloud_cpu_pct",           "Simulated cloud CPU %")


# ── Task generator ────────────────────────────────────────────────────────────
#
# Thiết kế để làm lộ điểm yếu của Round Robin và thể hiện sức mạnh của RL:
#
#  light            → deadline chặt (50-120ms) — Cloud base ~65ms → thường miss
#  heavy            → cpu cao (60-88%) — Edge overload → penalty ×3 → miss
#  mixed            → 50% light + 50% heavy — RL phân loại, RR không
#  bursty           → 70% light + 30% spike nặng đột ngột
#  deadline_critical→ deadline 30-70ms — chỉ Edge ít tải mới kịp
#  constant         → hỗn hợp cân bằng (dùng để baseline sanity check)

def _light():
    return (random.uniform(5, 18), random.uniform(5, 20), random.uniform(50, 120))

def _heavy():
    return (random.uniform(62, 88), random.uniform(50, 78), random.uniform(280, 500))

def _mixed():
    return _light() if random.random() < 0.5 else _heavy()

def _bursty():
    return (random.uniform(68, 92), random.uniform(55, 85), random.uniform(70, 180)) \
        if random.random() < 0.3 else _light()

def _deadline_critical():
    return (random.uniform(10, 38), random.uniform(10, 30), random.uniform(30, 70))

_PATTERNS = {
    "constant":          lambda: (random.uniform(10, 55), random.uniform(10, 45), random.uniform(80, 420)),
    "light":             _light,
    "heavy":             _heavy,
    "mixed":             _mixed,
    "bursty":            _bursty,
    "deadline_critical": _deadline_critical,
}

_task_counter = 0

def next_task(pattern: str = "constant") -> TaskInfo:
    global _task_counter
    _task_counter += 1
    cpu, ram, deadline = _PATTERNS.get(pattern, _PATTERNS["constant"])()
    return TaskInfo(
        task_id=f"task_{_task_counter:08d}",
        cpu_requirement=float(cpu),
        ram_requirement=float(ram),
        deadline_ms=float(deadline),
        priority=random.choice(["low", "medium", "high"]),
        payload_type=random.choice(["compute", "image", "io"]),
    )


# ── Runner ─────────────────────────────────────────────────────────────────

class PolicyRunner:
    def __init__(self, policy_name: str, model_path: str | None,
                 tasks_per_sec: float = 2.0, pattern: str = "constant",
                 demo_mode: bool = False, prometheus_url: str = "http://localhost:9090"):
        self.policy_name = policy_name
        self.tasks_per_sec = tasks_per_sec
        self.pattern = pattern
        self._demo_mode = demo_mode
        self._prometheus_url = prometheus_url
        self._stop = threading.Event()
        self._total = 0
        self._sla_ok = 0

        self.dispatcher = SmartDispatcher(
            policy_name=policy_name,
            model_path=model_path,
            n_edge_nodes=2,
            prometheus_url=self._prometheus_url,
            demo_mode=self._demo_mode,
        )
        logger.info("Runner ready: policy=%s  rate=%.1f tasks/s", policy_name, tasks_per_sec)

    def run(self):
        interval = 1.0 / self.tasks_per_sec
        while not self._stop.is_set():
            t0 = time.perf_counter()
            task = next_task(self.pattern)
            result = self.dispatcher.dispatch(task)
            self._total += 1
            if result.sla_met:
                self._sla_ok += 1

            sla_rate = self._sla_ok / self._total * 100

            # Update Prometheus gauges
            if PROM_AVAILABLE:
                _task_total.labels(self.policy_name, result.selected_node).inc()
                if result.sla_met:
                    _sla_met.labels(self.policy_name).inc()
                else:
                    _sla_miss.labels(self.policy_name).inc()
                _latency_hist.labels(self.policy_name).observe(result.latency_est_ms)
                _sla_rate_g.labels(self.policy_name).set(sla_rate)

                # Node CPU gauges (từ internal simulation state)
                m = self.dispatcher.state_builder.get_current_metrics_summary()
                _edge1_cpu.set(m.get("edge_1_cpu", 0))
                _edge2_cpu.set(m.get("edge_2_cpu", 0))
                _cloud_cpu.set(m.get("cloud_cpu", 0))

            # Log mỗi 10 tasks
            if self._total % 10 == 0:
                logger.info(
                    "[%s] tasks=%d  SLA=%.1f%%  avg_lat=%.0fms",
                    self.policy_name, self._total, sla_rate,
                    self.dispatcher.get_summary().get("avg_latency_ms", 0),
                )

            elapsed = time.perf_counter() - t0
            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self._stop.set()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Demo Runner — Hybrid Edge-Cloud Dispatcher")
    parser.add_argument("--policy", default="ppo",
                        help="Policy: ppo, dqn, round_robin, least_connection, ... (default: ppo)")
    parser.add_argument("--model", default=None,
                        help="Model checkpoint path (bắt buộc với ppo/dqn)")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="Tasks per second (default: 2.0)")
    parser.add_argument("--pattern", default="constant",
                        choices=["constant", "heavy", "bursty"],
                        help="Workload pattern (default: constant)")
    parser.add_argument("--metrics-port", type=int, default=8000,
                        help="Prometheus metrics port (default: 8000)")
    parser.add_argument("--compare", action="store_true",
                        help="Chạy lần lượt round_robin → least_connection → ppo mỗi 60s")
    parser.add_argument("--interval", type=int, default=60,
                        help="Giây mỗi policy khi --compare (default: 60)")
    parser.add_argument("--demo", action="store_true",
                        help="Simulation mode (không cần Prometheus thật)")
    parser.add_argument("--prometheus", default="http://localhost:9090",
                        help="Prometheus URL (default: http://localhost:9090)")
    args = parser.parse_args()

    # Bắt đầu Prometheus HTTP server
    if PROM_AVAILABLE:
        start_http_server(args.metrics_port)
        logger.info("Metrics endpoint: http://localhost:%d/metrics", args.metrics_port)

    mode_str = "SIMULATION" if args.demo else f"REAL PROMETHEUS ({args.prometheus})"
    logger.info("Mode: %s", mode_str)

    if args.compare:
        sequence = [
            ("round_robin",      None),
            ("least_connection", None),
            ("ppo",              args.model),
        ]
        for policy_name, model_path in sequence:
            if policy_name in ("ppo", "dqn") and model_path and not os.path.exists(model_path):
                logger.warning("Model not found: %s — skipping %s", model_path, policy_name)
                continue

            logger.info("=== Switching to policy: %s ===", policy_name)
            runner = PolicyRunner(policy_name, model_path, args.rate, args.pattern,
                                  demo_mode=args.demo, prometheus_url=args.prometheus)
            t = threading.Thread(target=runner.run, daemon=True)
            t.start()
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                runner.stop()
                break
            runner.stop()
            t.join(timeout=5)
            runner.dispatcher.print_summary()
    else:
        runner = PolicyRunner(args.policy, args.model, args.rate, args.pattern,
                              demo_mode=args.demo, prometheus_url=args.prometheus)
        logger.info("Starting... Press Ctrl+C to stop.")
        try:
            runner.run()
        except KeyboardInterrupt:
            runner.stop()
            runner.dispatcher.print_summary()


if __name__ == "__main__":
    main()
