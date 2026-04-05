"""
smart_dispatcher.py
===================
Core Smart Dispatcher cho hệ thống Hybrid Edge-Cloud.

Workflow mỗi task:
  1. Nhận task từ queue/generator
  2. StateBuilder query Prometheus → build state vector
  3. ModelLoader load DQN/PPO/Baseline → predict action
  4. Execute action (gửi task đến node qua K8s hoặc simulation)
  5. Log decision vào database
  6. Tính reward, update metrics

Chạy qua CLI:
    python -m dispatcher.dispatcher_cli --policy dqn --num-tasks 50 --demo

Hoặc import:
    from dispatcher.smart_dispatcher import SmartDispatcher
    d = SmartDispatcher(policy_name="dqn", demo_mode=True)
    result = d.dispatch(task_info)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from dispatcher.state_builder import (
    MAX_LATENCY,
    NodeMetrics,
    StateBuilder,
    TaskInfo,
)
from dispatcher.model_loader import ModelLoader
from dispatcher.error_handlers import with_retry, CircuitBreaker
from database.log_decisions import DatabaseManager

logger = logging.getLogger("SmartDispatcher")

# Cost constants (khớp với rl_env/edge_cloud_env.py)
EDGE_COST_PER_UNIT = 0.01
CLOUD_COST_PER_UNIT = 0.05


@dataclass
class DispatchResult:
    """Kết quả dispatch một task."""
    task_id: str
    selected_node: str
    action: int
    policy_name: str
    latency_est_ms: float
    cost_est: float
    sla_met: bool
    inference_ms: float
    q_values: Optional[list] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class SmartDispatcher:
    """
    Bộ điều phối thông minh kết hợp RL model + Prometheus metrics.

    Parameters
    ----------
    policy_name     : "dqn", "ppo", "round_robin", "least_connection", ...
    model_path      : đường dẫn model checkpoint (cho dqn/ppo)
    n_edge_nodes    : số Edge nodes (default 2)
    prometheus_url  : URL Prometheus server
    demo_mode       : True = simulation, không cần infra thật
    db_path         : đường dẫn SQLite database
    """

    def __init__(
        self,
        policy_name: str = "dqn",
        model_path: Optional[str] = None,
        n_edge_nodes: int = 2,
        prometheus_url: str = "http://localhost:9090",
        demo_mode: bool = False,
        db_path: Optional[str] = None,
    ):
        self.policy_name = policy_name
        self.n_edge_nodes = n_edge_nodes
        self.n_actions = n_edge_nodes + 1
        self.demo_mode = demo_mode
        self._dispatch_count = 0
        self._results: List[DispatchResult] = []

        # State builder
        self.state_builder = StateBuilder(
            n_edge_nodes=n_edge_nodes,
            prometheus_url=prometheus_url,
            use_prometheus=(not demo_mode),
        )
        if demo_mode:
            self.state_builder.reset_simulation()

        # Model loader
        obs_dim = n_edge_nodes * 3 + 3 + 3
        self.model_loader = ModelLoader(
            obs_dim=obs_dim,
            n_actions=self.n_actions,
            n_edge_nodes=n_edge_nodes,
        )

        # Load policy
        if model_path:
            self.model_loader.load(policy_name, model_path)
        else:
            self.model_loader.load(policy_name)

        # Database
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "database", "dispatcher.db",
            )
        self.db = DatabaseManager(db_path)

        # Circuit breaker cho K8s calls
        self._k8s_breaker = CircuitBreaker(
            failure_threshold=5, reset_timeout=30.0
        )

        logger.info(
            "SmartDispatcher ready | policy=%s | edges=%d | demo=%s",
            policy_name, n_edge_nodes, demo_mode,
        )

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def dispatch(self, task: TaskInfo) -> DispatchResult:
        """
        Dispatch một task: build state → predict → execute → log.

        Returns: DispatchResult
        """
        # 1. Build state vector
        state = self.state_builder.build_state(task)

        # 2. Model predict
        t0 = time.perf_counter()
        action, q_values = self.model_loader.predict(state)
        inference_ms = (time.perf_counter() - t0) * 1000

        # 3. Determine node
        node_name = self.state_builder.get_node_name(action)
        is_cloud = (action == self.n_edge_nodes)

        # 4. Estimate latency & cost
        latency_est, cost_est, sla_met = self._estimate_execution(
            action, is_cloud, task
        )

        # 5. Execute (real K8s hoặc simulation)
        if not self.demo_mode:
            self._execute_on_node(task, node_name)

        # 6. Update simulation state
        if self.demo_mode:
            self.state_builder.update_simulation_state(action, task)

        # 7. Build result
        result = DispatchResult(
            task_id=task.task_id,
            selected_node=node_name,
            action=action,
            policy_name=self.policy_name,
            latency_est_ms=round(latency_est, 2),
            cost_est=round(cost_est, 6),
            sla_met=sla_met,
            inference_ms=round(inference_ms, 3),
            q_values=[round(q, 4) for q in q_values] if q_values else None,
        )
        self._results.append(result)
        self._dispatch_count += 1

        # 8. Log to database
        self._log_to_db(task, result, state)

        return result

    def dispatch_batch(self, tasks: List[TaskInfo]) -> List[DispatchResult]:
        """Dispatch một batch tasks tuần tự."""
        results = []
        for task in tasks:
            results.append(self.dispatch(task))
        return results

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _estimate_execution(
        self, action: int, is_cloud: bool, task: TaskInfo
    ) -> tuple:
        """Ước tính latency, cost, SLA cho một dispatch decision."""
        metrics = self.state_builder.get_current_metrics_summary()

        if is_cloud:
            node_cpu = metrics["cloud_cpu"]
            base_latency = metrics["cloud_lat"]
            cost_rate = CLOUD_COST_PER_UNIT
        else:
            node_cpu = metrics[f"edge_{action + 1}_cpu"]
            base_latency = metrics[f"edge_{action + 1}_lat"]
            cost_rate = EDGE_COST_PER_UNIT

        # Latency tăng theo tải hiện tại
        load_factor = 1.0 + (node_cpu / 100.0) * 2.0
        latency = base_latency * load_factor

        # Penalty nếu node quá tải
        if node_cpu + task.cpu_requirement > 95:
            latency *= 3.0

        cost = (task.cpu_requirement + task.ram_requirement) * cost_rate
        sla_met = latency <= task.deadline_ms

        return float(latency), float(cost), bool(sla_met)

    def _execute_on_node(self, task: TaskInfo, node_name: str):
        """
        Gửi task đến node thật qua K8s API.
        Person 1 sẽ cung cấp pod_deployer.py để thay thế phần này.
        """
        if not self._k8s_breaker.allow_request():
            logger.warning(
                "Circuit breaker OPEN for K8s - task %s fallback to logging only",
                task.task_id,
            )
            return

        try:
            # Placeholder: sẽ integrate pod_deployer.py từ Person 1
            # from k8s_integration.pod_deployer import deploy_task_pod
            # deploy_task_pod(task.task_id, node_name)
            logger.info("Task %s → %s (K8s deploy pending integration)", task.task_id, node_name)
            self._k8s_breaker.record_success()
        except Exception as e:
            logger.error("K8s deploy failed for task %s: %s", task.task_id, e)
            self._k8s_breaker.record_failure()

    # ------------------------------------------------------------------
    # Database logging
    # ------------------------------------------------------------------

    def _log_to_db(self, task: TaskInfo, result: DispatchResult, state: np.ndarray):
        """Log task và decision vào database."""
        try:
            # Insert task
            self.db.insert_task(
                task_id=task.task_id,
                arrival_time=datetime.now(timezone.utc).isoformat(),
                deadline_ms=int(task.deadline_ms),
                cpu_requirement=task.cpu_requirement,
                ram_requirement_mb=int(task.ram_requirement),
                priority=task.priority,
                payload_type=task.payload_type,
            )

            # Update status
            self.db.update_task_status(
                task_id=task.task_id,
                status="dispatched",
                assigned_node=result.selected_node,
                execution_latency_ms=result.latency_est_ms,
                deadline_met=result.sla_met,
            )

            # Log decision
            self.db.log_decision(
                task_id=task.task_id,
                policy_name=result.policy_name,
                state_vector=state.tolist(),
                action=result.action,
                selected_node=result.selected_node,
                q_values=result.q_values,
                inference_latency_ms=result.inference_ms,
            )
        except Exception as e:
            logger.error("DB logging failed for task %s: %s", task.task_id, e)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """Thống kê tổng hợp từ kết quả dispatch trong memory."""
        if not self._results:
            return {"total": 0}

        sla_flags = [r.sla_met for r in self._results]
        latencies = [r.latency_est_ms for r in self._results]
        costs = [r.cost_est for r in self._results]
        nodes = [r.selected_node for r in self._results]

        cloud_count = sum(1 for n in nodes if n == "cloud")

        return {
            "total": self._dispatch_count,
            "sla_rate": round(sum(sla_flags) / len(sla_flags) * 100, 1),
            "avg_latency_ms": round(float(np.mean(latencies)), 1),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
            "avg_cost": round(float(np.mean(costs)), 5),
            "cloud_usage_pct": round(cloud_count / len(nodes) * 100, 1),
            "edge_usage_pct": round((len(nodes) - cloud_count) / len(nodes) * 100, 1),
            "policy": self.policy_name,
        }

    def print_summary(self):
        """In thống kê ra terminal."""
        s = self.get_summary()
        if s["total"] == 0:
            print("No tasks dispatched yet.")
            return

        print(f"\n{'=' * 55}")
        print(f"  Dispatcher Summary ({s['total']} tasks, policy={s['policy']})")
        print(f"  SLA Rate      : {s['sla_rate']:.1f}%")
        print(f"  Avg Latency   : {s['avg_latency_ms']:.1f} ms")
        print(f"  P95 Latency   : {s['p95_latency_ms']:.1f} ms")
        print(f"  Avg Cost      : {s['avg_cost']:.5f}")
        print(f"  Cloud Usage   : {s['cloud_usage_pct']:.1f}%")
        print(f"  Edge Usage    : {s['edge_usage_pct']:.1f}%")
        print(f"{'=' * 55}\n")

    def switch_policy(self, policy_name: str, model_path: Optional[str] = None):
        """Hot-switch sang policy khác mà không restart."""
        self.policy_name = policy_name
        if model_path:
            self.model_loader.load(policy_name, model_path)
        else:
            self.model_loader.load(policy_name)
        logger.info("Policy switched to: %s", policy_name)
