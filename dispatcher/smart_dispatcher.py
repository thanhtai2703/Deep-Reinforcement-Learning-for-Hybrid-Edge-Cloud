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
        import threading
        self._dispatch_count = 0
        self._results: List[DispatchResult] = []
        self._results_lock = threading.Lock()
        self._state_lock = threading.Lock()   # serialize state→predict→update

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
        # ── Phase 1: State → Predict → Update (serialized) ──
        # Lock ensures concurrent threads see updated state from
        # previous dispatches, preventing all-same-action collapse.
        with self._state_lock:
            # 1. Build state vector
            state = self.state_builder.build_state(task)
            metrics_summary = self.state_builder.get_current_metrics_summary()

            # 2. Model predict
            t0 = time.perf_counter()
            action, q_values = self.model_loader.predict(state)
            inference_ms = (time.perf_counter() - t0) * 1000

            # 3. Determine node
            node_name = self.state_builder.get_node_name(action)
            is_cloud = (action == self.n_edge_nodes)
            is_reject = (action == self.reject_action)

            # ── DEBUG: dump first 10 dispatches' state + Q values ──
            import os as _os
            if _os.getenv("DQN_DEBUG", "0") in ("1", "true") and self._dispatch_count < 10:
                logger.info(
                    "[DEBUG #%d] state=%s | metrics=%s | Q=%s | action=%d -> %s",
                    self._dispatch_count,
                    [round(float(x), 3) for x in state.tolist()],
                    metrics_summary,
                    [round(q, 3) for q in (q_values or [])],
                    action, node_name,
                )

            # 4. Update internal state IMMEDIATELY so next thread
            #    sees the load impact of this dispatch.
            self.state_builder.update_simulation_state(action, task)

        # ── Phase 2: Execute + Log (parallel, outside lock) ──

        # 5. Estimate latency & cost (skip for reject)
        if is_reject:
            est_latency_ms, est_cost, est_sla = 0.0, 0.0, False
        else:
            est_latency_ms, est_cost, est_sla = self._estimate_execution(
                action, is_cloud, task
            )

        # 6. Execute real backend.
        import os as _os
        force_execute = _os.getenv("FORCE_EXECUTE", "false").lower() in ("1", "true", "yes")

        exec_info = {
            "status": None, "total_ms": None, "timings": None,
            "target_role": None, "cpu_req_k8s": None, "ram_req_k8s": None,
            "exec_backend": None,
        }
        if not is_reject and ((not self.demo_mode) or force_execute):
            exec_info = self._execute_on_node(task, node_name)

        real_latency_ms = exec_info["total_ms"]
        exec_status = exec_info["status"]

        if real_latency_ms is not None and exec_status == "succeeded":
            reported_latency = float(real_latency_ms)
            reported_sla = (reported_latency <= task.deadline_ms)
        else:
            reported_latency = est_latency_ms
            reported_sla = est_sla

        # 7. Build result
        result = DispatchResult(
            task_id=task.task_id,
            selected_node=node_name,
            action=action,
            policy_name=self.policy_name,
            latency_est_ms=round(reported_latency, 2),
            cost_est=round(est_cost, 6),
            sla_met=reported_sla,
            inference_ms=round(inference_ms, 3),
            q_values=[round(q, 4) for q in q_values] if q_values else None,
        )
        with self._results_lock:
            self._results.append(result)
            self._dispatch_count += 1

        # 8. Log to database
        self._log_to_db(task, result, state)
        self._log_calibration(
            task=task,
            action=action,
            node_name=node_name,
            metrics_summary=metrics_summary,
            est_latency_ms=est_latency_ms,
            est_cost=est_cost,
            est_sla=est_sla,
            exec_info=exec_info,
        )

        return result

    def dispatch_batch(self, tasks: List[TaskInfo]) -> List[DispatchResult]:
        """Dispatch một batch tasks tuần tự."""
        results = []
        for task in tasks:
            results.append(self.dispatch(task))
        return results

    def dispatch_concurrent(
        self,
        tasks: List[TaskInfo],
        max_workers: int = 10,
    ) -> List[DispatchResult]:
        """
        Dispatch tasks concurrently via ThreadPoolExecutor.

        Multiple K8s Jobs run in parallel → real CPU/RAM contention on nodes.
        Required for collecting calibration data with non-zero load.

        Order of returned results matches order of input tasks.
        """
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(self.dispatch, tasks))
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

        ram_requirement is a percentage (0-100, matches env state vector).
        Scale to MB on a 512MB base: 5% → 26MB → clamped to 64Mi floor,
        50% → 256MB, 100% → 512MB.

        Examples:
        - 5       -> 64Mi   (floor)
        - 30      -> 154Mi
        - 50      -> 256Mi
        - "512Mi" -> "512Mi" (passthrough)
        """
        value_str = str(ram_requirement).strip()

        if value_str.lower().endswith(("mi", "gi")):
            return value_str

        try:
            pct = float(value_str)
        except Exception:
            return "128Mi"

        # Scale percentage to MB (base 512MB)
        value = int(pct * 5.12)
        value = max(64, min(value, 1024))
        return f"{value}Mi"

    def _execute_on_node(self, task: TaskInfo, node_name: str):
        """
        Execute task on selected node.

        Returns dict (thread-safe — no instance state mutation):
          {status, total_ms, timings, target_role, cpu_req_k8s, ram_req_k8s,
           exec_backend}
        Fields may be None if execution was skipped or failed early.
        """
        import os
        import time

        backend = os.getenv("EXECUTION_BACKEND", "k8s").lower()

        if backend == "http":
            return self._execute_on_node_http(task, node_name)

        result = {
            "status": None, "total_ms": None, "timings": None,
            "target_role": None, "cpu_req_k8s": None, "ram_req_k8s": None,
            "exec_backend": backend,
        }

        if not self._k8s_breaker.allow_request():
            logger.warning("Circuit breaker OPEN — task %s logged only", task.task_id)
            return result

        try:
            from dispatcher.pod_deployer import deploy_and_wait

            target_role = self._normalize_k8s_role(node_name)
            cpu_req = self._to_k8s_cpu(task.cpu_requirement)
            ram_req = self._to_k8s_memory(task.ram_requirement)

            result["target_role"] = target_role
            result["cpu_req_k8s"] = cpu_req
            result["ram_req_k8s"] = ram_req

            # Duration: 0.5 × deadline, clamp [1, 15]s.
            # Cap cao hơn (15s vs 5s cũ) → pods overlap nhiều khi concurrent
            # → CPU contention thật → calibration data có range load rộng.
            try:
                duration_seconds = max(1.0, min(float(task.deadline_ms) / 1000.0 * 0.5, 15.0))
            except   Exception:
                duration_seconds = 2.0

            start = time.perf_counter()

            job_name, status, timings = deploy_and_wait(
                task_id=str(task.task_id),
                target_role=target_role,
                cpu_req=cpu_req,
                ram_req=ram_req,
                duration_seconds=duration_seconds,
                cleanup=False,
                cpu_intensity=float(task.cpu_requirement),
                ram_intensity=float(task.ram_requirement),
            )

            total_ms = int((time.perf_counter() - start) * 1000)
            result["status"] = status
            result["total_ms"] = total_ms
            result["timings"] = timings

            if status == "succeeded":
                logger.info(
                    "Task %s → %s | backend=k8s job=%s status=%s total_ms=%s "
                    "submit=%s startup=%s exec=%s poll=%s",
                    task.task_id, target_role, job_name, status, total_ms,
                    timings.submit_overhead_ms,
                    timings.container_startup_ms,
                    timings.exec_time_ms,
                    timings.poll_overhead_ms,
                )
                self._k8s_breaker.record_success()
            else:
                logger.error(
                    "Task %s → %s | backend=k8s job=%s status=%s total_ms=%s "
                    "submit=%s startup=%s exec=%s",
                    task.task_id, target_role, job_name, status, total_ms,
                    timings.submit_overhead_ms,
                    timings.container_startup_ms,
                    timings.exec_time_ms,
                )
                self._k8s_breaker.record_failure()

            return result

        except Exception as e:
            logger.error("K8s Job execution failed task=%s node=%s: %s", task.task_id, node_name, e)
            self._k8s_breaker.record_failure()
            result["status"] = "failed"
            return result

    def _execute_on_node_http(self, task: TaskInfo, node_name: str):
        """Returns dict (same shape as _execute_on_node)."""
        result = {
            "status": None, "total_ms": None, "timings": None,
            "target_role": node_name, "cpu_req_k8s": None, "ram_req_k8s": None,
            "exec_backend": "http",
        }

        if not self._k8s_breaker.allow_request():
            logger.warning("Circuit breaker OPEN — task %s logged only", task.task_id)
            return result

        try:
            from dispatcher.infra_config import WORKER_URLS
            import requests as _requests

            url = WORKER_URLS.get(node_name)
            if not url:
                logger.warning("No worker URL for node=%s", node_name)
                return result

            payload = {
                "task_id":         task.task_id,
                "cpu_requirement": task.cpu_requirement,
                "ram_requirement": task.ram_requirement,
                "deadline_ms":     task.deadline_ms,
            }

            t0 = time.perf_counter()
            resp = _requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            real_latency_ms = int((time.perf_counter() - t0) * 1000)

            logger.info(
                "Task %s → %s | backend=http worker=%s latency_ms=%d",
                task.task_id, node_name,
                resp.json().get("status"), real_latency_ms,
            )

            self._k8s_breaker.record_success()
            result["status"] = "succeeded"
            result["total_ms"] = real_latency_ms
            return result

        except Exception as e:
            logger.error("Worker call failed task=%s node=%s: %s", task.task_id, node_name, e)
            self._k8s_breaker.record_failure()
            result["status"] = "failed"
            return result

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
    # Calibration logging
    # ------------------------------------------------------------------

    def _log_calibration(
        self,
        task: TaskInfo,
        action: int,
        node_name: str,
        metrics_summary: dict,
        est_latency_ms: float,
        est_cost: float,
        est_sla: bool,
        exec_info: dict,
    ):
        """
        Insert one row into execution_logs for sim calibration analysis.

        Captures: task spec, dispatch decision, node state at dispatch (raw),
        sim env's prediction, and real execution outcome with timing breakdown.
        Safe to call in demo mode (exec_* fields will be NULL).
        """
        import json as _json

        is_reject = (action == self.reject_action)
        is_cloud = (action == self.n_edge_nodes)

        # Pick the row of metrics_summary corresponding to the selected node.
        if is_reject:
            sel_cpu = sel_ram = sel_lat = sel_queue = None
        elif is_cloud:
            sel_cpu = metrics_summary.get("cloud_cpu")
            sel_ram = metrics_summary.get("cloud_ram")
            sel_lat = metrics_summary.get("cloud_lat")
            sel_queue = metrics_summary.get("cloud_queue")
        else:
            prefix = f"edge_{action + 1}"
            sel_cpu = metrics_summary.get(f"{prefix}_cpu")
            sel_ram = metrics_summary.get(f"{prefix}_ram")
            sel_lat = metrics_summary.get(f"{prefix}_lat")
            sel_queue = metrics_summary.get(f"{prefix}_queue")

        timings = exec_info.get("timings")
        real_latency_ms = exec_info.get("total_ms")
        exec_status = exec_info.get("status")
        sla_real = None
        if real_latency_ms is not None:
            sla_real = int(real_latency_ms <= task.deadline_ms)

        # Ground-truth load: avg CPU/RAM during pod lifetime.
        # selected_cpu (snapshot at dispatch) underestimates true load because
        # it's measured BEFORE the pod and its concurrent siblings start
        # burning CPU. cpu_during_exec captures load WHILE the pod was running.
        cpu_during = ram_during = None
        if (timings and timings.container_finished_at
                and timings.exec_time_ms
                and exec_info.get("target_role")):
            try:
                from datetime import datetime as _dt
                t_end = _dt.fromisoformat(
                    timings.container_finished_at.replace("Z", "+00:00")
                ).timestamp()
                if timings.pod_scheduled_at:
                    t_start = _dt.fromisoformat(
                        timings.pod_scheduled_at.replace("Z", "+00:00")
                    ).timestamp()
                    duration_s = max(t_end - t_start, 5.0)
                else:
                    duration_s = max(timings.exec_time_ms / 1000.0, 5.0)
                cpu_during, ram_during = self.state_builder.query_load_in_window(
                    role=exec_info["target_role"],
                    t_end_unix=t_end,
                    duration_s=duration_s,
                )
            except Exception as e:
                logger.warning("query_load_in_window failed task=%s: %s",
                               task.task_id, e)

        row = {
            "task_id": task.task_id,
            "cpu_requirement": float(task.cpu_requirement),
            "ram_requirement": float(task.ram_requirement),
            "deadline_ms": float(task.deadline_ms),
            "priority": task.priority,
            "payload_type": task.payload_type,
            "policy_name": self.policy_name,
            "action": int(action),
            "selected_node": node_name,
            "target_role": exec_info.get("target_role"),
            "cpu_req_k8s": exec_info.get("cpu_req_k8s"),
            "ram_req_k8s": exec_info.get("ram_req_k8s"),
            "metrics_summary_json": _json.dumps(metrics_summary),
            "selected_cpu": sel_cpu,
            "selected_ram": sel_ram,
            "selected_latency": sel_lat,
            "selected_queue": sel_queue,
            "est_latency_ms": float(est_latency_ms),
            "est_cost": float(est_cost),
            "est_sla_met": int(bool(est_sla)),
            "exec_backend": exec_info.get("exec_backend"),
            "exec_status": exec_status,
            "total_ms": (timings.total_ms if timings else real_latency_ms),
            "submit_overhead_ms": timings.submit_overhead_ms if timings else None,
            "container_startup_ms": timings.container_startup_ms if timings else None,
            "exec_time_ms": timings.exec_time_ms if timings else None,
            "poll_overhead_ms": timings.poll_overhead_ms if timings else None,
            "pod_node": timings.node_name if timings else None,
            "cpu_during_exec": cpu_during,
            "ram_during_exec": ram_during,
            "sla_met_real": sla_real,
        }

        try:
            self.db.insert_execution_log(row)
        except Exception as e:
            logger.error("Calibration log insert failed task=%s: %s", task.task_id, e)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """
        Per-node summary cho phân tích khoa học.

        Trả về dict gồm:
          - Overall: total, sla_rate, latency stats (mean/median/p95/p99/std), cost
          - Per-node: edge_i_pct/count, cloud_pct/count, reject_pct/count
          - SLA broken-down theo node (giúp debug node nào miss SLA nhiều)
        """
        if not self._results:
            return {"total": 0}

        latencies = np.array([r.latency_est_ms for r in self._results])
        costs     = np.array([r.cost_est for r in self._results])
        nodes     = [r.selected_node for r in self._results]
        sla_flags = np.array([r.sla_met for r in self._results])
        infer_ms  = np.array([r.inference_ms for r in self._results])
        n         = len(self._results)

        # Per-node breakdown
        per_node = {}
        for role in [f"edge_{i + 1}" for i in range(self.n_edge_nodes)] \
                    + ["cloud", "rejected"]:
            mask = np.array([x == role for x in nodes])
            count = int(mask.sum())
            per_node[role] = {
                "count": count,
                "pct":   round(count / n * 100, 1),
                "sla":   (round(float(sla_flags[mask].mean()) * 100, 1)
                          if count > 0 else None),
                "avg_latency_ms": (round(float(latencies[mask].mean()), 1)
                                   if count > 0 else None),
            }

        # Overall stats
        return {
            "total": n,
            "policy": self.policy_name,
            # SLA
            "sla_rate":   round(float(sla_flags.mean()) * 100, 1),
            "sla_count":  int(sla_flags.sum()),
            # Latency
            "lat_mean":   round(float(np.mean(latencies)), 1),
            "lat_median": round(float(np.median(latencies)), 1),
            "lat_p95":    round(float(np.percentile(latencies, 95)), 1),
            "lat_p99":    round(float(np.percentile(latencies, 99)), 1),
            "lat_std":    round(float(np.std(latencies)), 1),
            # Cost
            "cost_total": round(float(np.sum(costs)), 4),
            "cost_avg":   round(float(np.mean(costs)), 5),
            # Inference
            "infer_avg_ms": round(float(np.mean(infer_ms)), 3),
            # Per-node
            "per_node": per_node,
            # Backward-compat aliases
            "avg_latency_ms":   round(float(np.mean(latencies)), 1),
            "p95_latency_ms":   round(float(np.percentile(latencies, 95)), 1),
            "avg_cost":         round(float(np.mean(costs)), 5),
            "edge_usage_pct":   round(sum(per_node[f"edge_{i+1}"]["count"]
                                           for i in range(self.n_edge_nodes))
                                       / n * 100, 1),
            "cloud_usage_pct":  per_node["cloud"]["pct"],
            "reject_usage_pct": per_node["rejected"]["pct"],
        }

    def print_summary(self):
        """In thống kê khoa học ra terminal."""
        s = self.get_summary()
        if s["total"] == 0:
            print("No tasks dispatched yet.")
            return

        WIDTH = 64
        bar = "═" * WIDTH

        print()
        print(bar)
        print(f"  Smart Dispatcher Report — policy={s['policy']:<20s}")
        print(bar)
        print(f"  Tasks dispatched      : {s['total']}")
        print(f"  Avg inference latency : {s['infer_avg_ms']:.3f} ms / task")

        print(f"\n  ── SLA Compliance ──")
        sla_miss = s['total'] - s['sla_count'] - s['per_node']['rejected']['count']
        print(f"  SLA met               : {s['sla_rate']:>5.1f}%  "
              f"({s['sla_count']:>3d}/{s['total']})")
        print(f"  Deadline missed       : "
              f"{sla_miss/s['total']*100:>5.1f}%  ({sla_miss:>3d}/{s['total']})")
        print(f"  Rejected              : "
              f"{s['per_node']['rejected']['pct']:>5.1f}%  "
              f"({s['per_node']['rejected']['count']:>3d}/{s['total']})")

        print(f"\n  ── Latency (ms) ──")
        print(f"  Mean                  : {s['lat_mean']:>9.1f}")
        print(f"  Median                : {s['lat_median']:>9.1f}")
        print(f"  P95                   : {s['lat_p95']:>9.1f}")
        print(f"  P99                   : {s['lat_p99']:>9.1f}")
        print(f"  Std                   : {s['lat_std']:>9.1f}")

        print(f"\n  ── Cost ──")
        print(f"  Total                 : {s['cost_total']:>9.4f}")
        print(f"  Avg / task            : {s['cost_avg']:>9.5f}")

        print(f"\n  ── Action Distribution ──")
        for role in [f"edge_{i + 1}" for i in range(self.n_edge_nodes)] \
                    + ["cloud", "rejected"]:
            pn = s["per_node"][role]
            sla_str = (f"SLA={pn['sla']:>5.1f}%"
                       if pn["sla"] is not None else "SLA=  -  ")
            lat_str = (f"avg_lat={pn['avg_latency_ms']:>7.1f}ms"
                       if pn["avg_latency_ms"] is not None else "avg_lat=  -    ")
            print(f"  {role:<13s} : {pn['pct']:>5.1f}%  "
                  f"({pn['count']:>3d}/{s['total']})  | {sla_str}  {lat_str}")

        print(bar)
        print()

    def switch_policy(self, policy_name: str, model_path: Optional[str] = None):
        """Hot-switch sang policy khác mà không restart."""
        self.policy_name = policy_name
        if model_path:
            self.model_loader.load(policy_name, model_path)
        else:
            self.model_loader.load(policy_name)
        logger.info("Policy switched to: %s", policy_name)