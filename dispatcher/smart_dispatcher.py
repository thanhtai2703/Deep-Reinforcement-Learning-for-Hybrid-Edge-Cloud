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
        instance_map: Optional[dict] = None,
    ):
        self.policy_name = policy_name
        self.n_edge_nodes = n_edge_nodes
        self.n_actions = n_edge_nodes + 2
        self.reject_action = n_edge_nodes + 1
        self.demo_mode = demo_mode
        self._dispatch_count = 0
        self._results: List[DispatchResult] = []

        # Load instance_map từ infra_config nếu không truyền vào
        if instance_map is None and not demo_mode:
            try:
                from dispatcher.infra_config import INSTANCE_MAP
                instance_map = INSTANCE_MAP
            except ImportError:
                pass

        # State builder
        self.state_builder = StateBuilder(
            n_edge_nodes=n_edge_nodes,
            prometheus_url=prometheus_url,
            use_prometheus=(not demo_mode),
            instance_map=instance_map or {},
        )
        if demo_mode:
            self.state_builder.reset_simulation()

        # Model loader
        obs_dim = n_edge_nodes * 4 + 4 + 3 + 2
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
        is_reject = (action == self.reject_action)

        # 4. Estimate latency & cost (skip for reject — task is dropped)
        if is_reject:
            latency_est, cost_est, sla_met = 0.0, 0.0, False
        else:
            latency_est, cost_est, sla_met = self._estimate_execution(
                action, is_cloud, task
            )

        # 5. Execute real backend.
        # In demo mode, execution is skipped by default.
        # Set FORCE_EXECUTE=true to create real K8s Jobs while still using demo metrics.
        import os as _os
        force_execute = _os.getenv("FORCE_EXECUTE", "false").lower() in ("1", "true", "yes")

        if not is_reject and ((not self.demo_mode) or force_execute):
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
        if action >= self.n_edge_nodes + 1:
            return 0.0, 0.0, False

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

    def _normalize_k8s_role(self, node_name: str) -> str:
        """
        Convert dispatcher node name to Kubernetes nodeSelector role.

        Kubernetes node labels:
        - role=edge_1
        - role=edge_2
        - role=cloud
        """
        mapping = {
            "edge1": "edge_1",
            "edge-1": "edge_1",
            "edge_1": "edge_1",

            "edge2": "edge_2",
            "edge-2": "edge_2",
            "edge_2": "edge_2",

            "cloud": "cloud",
            "cloud1": "cloud",
            "cloud-1": "cloud",
            "cloud_1": "cloud",
        }

        return mapping.get(str(node_name).lower(), str(node_name))

    def _to_k8s_cpu(self, cpu_requirement) -> str:
        """
        Convert project CPU requirement to Kubernetes CPU format.

        Examples:
        - 50      -> 500m
        - 30      -> 300m
        - 0.5     -> 0.5 core
        - "500m"  -> "500m"
        """
        value_str = str(cpu_requirement).strip()

        if value_str.endswith("m"):
            return value_str

        try:
            value = float(value_str)
        except Exception:
            return "500m"

        # Nếu task đã dùng dạng core nhỏ: 0.5, 1, 2
        if 0 < value <= 4:
            return f"{value:g}"

        # Nếu project dùng thang 0-100, map thành millicpu
        millicpu = int(max(100, min(value * 10, 1000)))
        return f"{millicpu}m"

    def _to_k8s_memory(self, ram_requirement) -> str:
        """
        Convert project RAM requirement to Kubernetes memory format.

        Examples:
        - 128      -> 128Mi
        - 256      -> 256Mi
        - "512Mi"  -> "512Mi"
        """
        value_str = str(ram_requirement).strip()

        if value_str.lower().endswith(("mi", "gi")):
            return value_str

        try:
            value = int(float(value_str))
        except Exception:
            return "128Mi"

        value = max(64, min(value, 1024))
        return f"{value}Mi"

    def _execute_on_node(self, task: TaskInfo, node_name: str):
        """
        Execute task on selected node.

        Default backend:
        - k8s: create real Kubernetes Job/Pod on K3s using nodeSelector.

        Fallback backend:
        - http: old task_worker.py HTTP mode.
        """
        import os
        import time

        backend = os.getenv("EXECUTION_BACKEND", "k8s").lower()

        # Fallback về HTTP worker cũ nếu cần debug nhanh:
        # EXECUTION_BACKEND=http python3 ...
        if backend == "http":
            return self._execute_on_node_http(task, node_name)

        if not self._k8s_breaker.allow_request():
            logger.warning("Circuit breaker OPEN — task %s logged only", task.task_id)
            return

        try:
            from dispatcher.pod_deployer import deploy_and_wait

            target_role = self._normalize_k8s_role(node_name)
            cpu_req = self._to_k8s_cpu(task.cpu_requirement)
            ram_req = self._to_k8s_memory(task.ram_requirement)

            # Giới hạn duration để demo/test không chạy quá lâu.
            # deadline_ms càng lớn thì task chạy lâu hơn một chút.
            try:
                duration_seconds = max(1.0, min(float(task.deadline_ms) / 1000.0 * 0.5, 5.0))
            except Exception:
                duration_seconds = 2.0

            start = time.perf_counter()

            job_name, status, latency_ms = deploy_and_wait(
                task_id=str(task.task_id),
                target_role=target_role,
                cpu_req=cpu_req,
                ram_req=ram_req,
                duration_seconds=duration_seconds,
                cleanup=False,
            )

            total_ms = int((time.perf_counter() - start) * 1000)

            if status == "succeeded":
                logger.info(
                    "Task %s → %s | backend=k8s job=%s status=%s latency_ms=%s total_ms=%s",
                    task.task_id,
                    target_role,
                    job_name,
                    status,
                    latency_ms,
                    total_ms,
                )
                self._k8s_breaker.record_success()
            else:
                logger.error(
                    "Task %s → %s | backend=k8s job=%s status=%s latency_ms=%s",
                    task.task_id,
                    target_role,
                    job_name,
                    status,
                    latency_ms,
                )
                self._k8s_breaker.record_failure()

        except Exception as e:
            logger.error("K8s Job execution failed task=%s node=%s: %s", task.task_id, node_name, e)
            self._k8s_breaker.record_failure()

    def _execute_on_node_http(self, task: TaskInfo, node_name: str):
        """
        Fallback: gửi task đến HTTP worker cũ trên node được chọn.
        Giữ lại để debug nhanh nếu K8s backend có lỗi.
        """
        if not self._k8s_breaker.allow_request():
            logger.warning("Circuit breaker OPEN — task %s logged only", task.task_id)
            return

        try:
            from dispatcher.infra_config import WORKER_URLS
            import requests as _requests

            url = WORKER_URLS.get(node_name)
            if not url:
                logger.warning("No worker URL for node=%s", node_name)
                return

            payload = {
                "task_id":         task.task_id,
                "cpu_requirement": task.cpu_requirement,
                "ram_requirement": task.ram_requirement,
                "deadline_ms":     task.deadline_ms,
            }

            resp = _requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()

            logger.info(
                "Task %s → %s | backend=http worker=%s",
                task.task_id,
                node_name,
                resp.json().get("status"),
            )

            self._k8s_breaker.record_success()

        except Exception as e:
            logger.error("Worker call failed task=%s node=%s: %s", task.task_id, node_name, e)
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