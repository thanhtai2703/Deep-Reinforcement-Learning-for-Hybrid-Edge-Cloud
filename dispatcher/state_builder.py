"""
state_builder.py
================
Xây dựng và chuẩn hóa state vector cho Smart Dispatcher.

State vector gồm 12 chiều (với 2 edge nodes):
  [e0_cpu, e0_ram, e0_lat,
   e1_cpu, e1_ram, e1_lat,
   cloud_cpu, cloud_ram, cloud_lat,
   task_cpu, task_ram, task_deadline]

Tất cả giá trị được normalize về [0, 1].

Prometheus metrics được lấy qua PrometheusClient của Person 1
(week2/prometheus_client.py).

Hỗ trợ 2 chế độ:
  - Prometheus mode: query metrics thật qua Person 1 client
  - Simulation mode: dùng giá trị giả lập (demo/test)
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Thêm project root vào path để import week2
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger("StateBuilder")

# Normalization constants (khớp với rl_env/edge_cloud_env.py)
MAX_CPU = 100.0
MAX_RAM = 100.0
MAX_LATENCY = 200.0
MAX_DEADLINE = 500.0
MAX_QUEUE = 20


@dataclass
class NodeMetrics:
    """Metrics của một node (Edge hoặc Cloud)."""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    latency_ms: float = 0.0


@dataclass
class TaskInfo:
    """Thông tin task cần dispatch."""
    task_id: str = ""
    cpu_requirement: float = 0.0
    ram_requirement: float = 0.0
    deadline_ms: float = 0.0
    priority: str = "medium"
    payload_type: str = "compute"


class StateBuilder:
    """
    Xây dựng state vector normalized [0, 1] từ metrics thực
    hoặc giả lập.

    Parameters
    ----------
    n_edge_nodes    : số edge nodes (default 2)
    prometheus_url  : URL Prometheus server
    use_prometheus  : True = query Prometheus thật qua Person 1 client
    instance_map    : mapping Prometheus instance → node role
                      VD: {"192.168.1.100:9100": "edge_1",
                           "192.168.1.101:9100": "edge_2",
                           "10.0.0.50:9100": "cloud"}
                      Nếu không truyền sẽ tự gán theo thứ tự.
    """

    # Node names mapping: action index → node name
    NODE_NAMES = {0: "edge_1", 1: "edge_2", 2: "cloud"}

    def __init__(
        self,
        n_edge_nodes: int = 2,
        prometheus_url: str = "http://localhost:9090",
        use_prometheus: bool = False,
        instance_map: Optional[Dict[str, str]] = None,
    ):
        self.n_edge_nodes = n_edge_nodes
        self.use_prometheus = use_prometheus
        self.obs_dim = n_edge_nodes * 3 + 3 + 3  # edges + cloud + task

        # Mapping: Prometheus instance (IP:port) → role (edge_1, edge_2, cloud)
        self._instance_map = instance_map or {}

        # Person 1's Prometheus client (week2/prometheus_client.py)
        self._prom_client = None
        if use_prometheus:
            try:
                from week2.prometheus_client import PrometheusClient, PrometheusConfig
                config = PrometheusConfig(base_url=prometheus_url, timeout_seconds=5)
                self._prom_client = PrometheusClient(config)
                logger.info("Using Person 1 PrometheusClient: %s", prometheus_url)
            except ImportError:
                logger.warning(
                    "week2.prometheus_client not found — "
                    "falling back to simulation"
                )
                self.use_prometheus = False

        # Internal state cache (dùng cho simulation hoặc cache khi prom fail)
        self._edge_metrics: List[NodeMetrics] = [
            NodeMetrics() for _ in range(n_edge_nodes)
        ]
        self._cloud_metrics = NodeMetrics()
        self._rng = np.random.default_rng(42)

    def build_state(self, task: TaskInfo) -> np.ndarray:
        """
        Xây dựng state vector đã normalize.

        Returns: np.ndarray shape (obs_dim,) trong [0, 1]
        """
        if self.use_prometheus and self._prom is not None:
            self._fetch_prometheus_metrics()
        # Nếu không dùng Prometheus, dùng state đã cache (simulation)

        return self._compose_observation(task)

    def build_state_from_raw(
        self,
        edge_metrics: List[NodeMetrics],
        cloud_metrics: NodeMetrics,
        task: TaskInfo,
    ) -> np.ndarray:
        """Xây dựng state từ metrics đã cho (dùng cho testing)."""
        self._edge_metrics = edge_metrics
        self._cloud_metrics = cloud_metrics
        return self._compose_observation(task)

    def update_simulation_state(self, action: int, task: TaskInfo):
        """
        Cập nhật internal state sau khi dispatch (simulation mode).
        Mô phỏng tải tăng ở node được chọn, decay ở các node khác.
        """
        decay = 0.85

        for i in range(self.n_edge_nodes):
            m = self._edge_metrics[i]
            if i == action:
                m.cpu_percent = float(np.clip(
                    m.cpu_percent * decay + task.cpu_requirement * 0.5, 5, 99
                ))
                m.ram_percent = float(np.clip(
                    m.ram_percent * decay + task.ram_requirement * 0.5, 5, 99
                ))
            else:
                m.cpu_percent = float(np.clip(
                    m.cpu_percent * decay + self._rng.uniform(-3, 3), 5, 95
                ))
                m.ram_percent = float(np.clip(
                    m.ram_percent * decay + self._rng.uniform(-3, 3), 5, 95
                ))
            m.latency_ms = float(np.clip(
                self._rng.uniform(5, 30) + m.cpu_percent * 0.3, 1, MAX_LATENCY
            ))

        cm = self._cloud_metrics
        if action == self.n_edge_nodes:
            cm.cpu_percent = float(np.clip(
                cm.cpu_percent * decay + task.cpu_requirement * 0.3, 5, 99
            ))
            cm.ram_percent = float(np.clip(
                cm.ram_percent * decay + task.ram_requirement * 0.3, 5, 99
            ))
        else:
            cm.cpu_percent = float(np.clip(
                cm.cpu_percent * decay + self._rng.uniform(-3, 3), 5, 80
            ))
            cm.ram_percent = float(np.clip(
                cm.ram_percent * decay + self._rng.uniform(-3, 3), 5, 80
            ))
        cm.latency_ms = float(
            self._rng.uniform(30, 80) + cm.cpu_percent * 0.2
        )

    def reset_simulation(self, seed: int = 42):
        """Reset internal state cho simulation mode."""
        self._rng = np.random.default_rng(seed)
        for i in range(self.n_edge_nodes):
            self._edge_metrics[i] = NodeMetrics(
                cpu_percent=float(self._rng.uniform(20, 70)),
                ram_percent=float(self._rng.uniform(30, 80)),
                latency_ms=float(self._rng.uniform(10, 50)),
            )
        self._cloud_metrics = NodeMetrics(
            cpu_percent=float(self._rng.uniform(10, 50)),
            ram_percent=float(self._rng.uniform(10, 50)),
            latency_ms=float(self._rng.uniform(30, 80)),
        )

    def get_node_name(self, action: int) -> str:
        """Map action index sang tên node."""
        if action < self.n_edge_nodes:
            return f"edge_{action + 1}"
        return "cloud"

    def get_current_metrics_summary(self) -> dict:
        """Trả về summary metrics hiện tại (dùng cho logging/debug)."""
        summary = {}
        for i, m in enumerate(self._edge_metrics):
            prefix = f"edge_{i + 1}"
            summary[f"{prefix}_cpu"] = round(m.cpu_percent, 1)
            summary[f"{prefix}_ram"] = round(m.ram_percent, 1)
            summary[f"{prefix}_lat"] = round(m.latency_ms, 1)
        summary["cloud_cpu"] = round(self._cloud_metrics.cpu_percent, 1)
        summary["cloud_ram"] = round(self._cloud_metrics.ram_percent, 1)
        summary["cloud_lat"] = round(self._cloud_metrics.latency_ms, 1)
        return summary

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fetch_prometheus_metrics(self):
        """
        Query Prometheus qua Person 1 client (get_dispatcher_metrics).

        Person 1 trả về dict dạng:
          {"192.168.1.100:9100": {"cpu_usage_pct": 45.2, "ram_usage_pct": 67.8,
                                   "queue_length": 3, "network_latency_ms": 12.5}, ...}

        Dùng instance_map để map IP:port → edge_1/edge_2/cloud.
        """
        try:
            raw = self._prom_client.get_dispatcher_metrics()
        except Exception as e:
            logger.warning("Prometheus query failed: %s — using cached state", e)
            return

        if not raw:
            logger.debug("Prometheus returned empty, using cache")
            return

        # Đảo ngược: role → instance
        role_to_inst = {v: k for k, v in self._instance_map.items()}

        # Nếu không có instance_map, gán theo thứ tự alphabetical
        if not role_to_inst:
            instances = sorted(raw.keys())
            roles = [f"edge_{i + 1}" for i in range(self.n_edge_nodes)] + ["cloud"]
            for inst, role in zip(instances, roles):
                role_to_inst[role] = inst

        # Cập nhật edge metrics
        for i in range(self.n_edge_nodes):
            role = f"edge_{i + 1}"
            inst = role_to_inst.get(role)
            if inst and inst in raw:
                d = raw[inst]
                self._edge_metrics[i] = NodeMetrics(
                    cpu_percent=d.get("cpu_usage_pct") or 0.0,
                    ram_percent=d.get("ram_usage_pct") or 0.0,
                    latency_ms=d.get("network_latency_ms") or 0.0,
                )

        # Cập nhật cloud metrics
        inst = role_to_inst.get("cloud")
        if inst and inst in raw:
            d = raw[inst]
            self._cloud_metrics = NodeMetrics(
                cpu_percent=d.get("cpu_usage_pct") or 0.0,
                ram_percent=d.get("ram_usage_pct") or 0.0,
                latency_ms=d.get("network_latency_ms") or 0.0,
            )

    def _compose_observation(self, task: TaskInfo) -> np.ndarray:
        """Compose và normalize observation vector."""
        obs = []

        # Edge nodes: [cpu, ram, latency] per node
        for i in range(self.n_edge_nodes):
            m = self._edge_metrics[i]
            obs.extend([
                m.cpu_percent / MAX_CPU,
                m.ram_percent / MAX_RAM,
                m.latency_ms / MAX_LATENCY,
            ])

        # Cloud: [cpu, ram, latency]
        cm = self._cloud_metrics
        obs.extend([
            cm.cpu_percent / MAX_CPU,
            cm.ram_percent / MAX_RAM,
            cm.latency_ms / MAX_LATENCY,
        ])

        # Task: [cpu_req, ram_req, deadline]
        obs.extend([
            task.cpu_requirement / MAX_CPU,
            task.ram_requirement / MAX_RAM,
            task.deadline_ms / MAX_DEADLINE,
        ])

        arr = np.array(obs, dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

