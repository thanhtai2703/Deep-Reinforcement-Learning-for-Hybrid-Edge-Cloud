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

Hỗ trợ 2 chế độ:
  - Prometheus mode: query metrics thật từ Prometheus server
  - Simulation mode: dùng giá trị giả lập (demo/test)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

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


class PrometheusClient:
    """
    Client đọc metrics từ Prometheus server.

    Fallback sang giá trị mặc định nếu Prometheus không khả dụng.
    """

    def __init__(self, url: str = "http://localhost:9090", timeout: float = 2.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session = None
        self._last_metrics: Dict[str, NodeMetrics] = {}
        self._init_session()

    def _init_session(self):
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Accept": "application/json"})
        except ImportError:
            logger.warning("requests not installed - Prometheus queries disabled")

    def query(self, promql: str) -> Optional[float]:
        """Execute a PromQL instant query, return scalar value or None."""
        if self._session is None:
            return None
        try:
            r = self._session.get(
                f"{self.url}/api/v1/query",
                params={"query": promql},
                timeout=self.timeout,
            )
            r.raise_for_status()
            result = r.json()["data"]["result"]
            if result:
                return float(result[0]["value"][1])
            return None
        except Exception as e:
            logger.debug("Prometheus query failed (%s): %s", promql, e)
            return None

    def get_node_metrics(self, node_name: str) -> NodeMetrics:
        """
        Lấy CPU, RAM, Latency cho một node.

        PromQL labels chuẩn:
          node="edge_1", node="edge_2", node="cloud"
        """
        cpu = self.query(f'node_cpu_usage_percent{{node="{node_name}"}}')
        ram = self.query(f'node_memory_usage_percent{{node="{node_name}"}}')
        lat = self.query(f'node_latency_ms{{node="{node_name}"}}')

        metrics = NodeMetrics(
            cpu_percent=cpu if cpu is not None else 0.0,
            ram_percent=ram if ram is not None else 0.0,
            latency_ms=lat if lat is not None else 0.0,
        )

        self._last_metrics[node_name] = metrics
        return metrics

    def get_cached(self, node_name: str) -> NodeMetrics:
        """Trả về metrics đã cache nếu có, fallback sang default."""
        return self._last_metrics.get(node_name, NodeMetrics())


class StateBuilder:
    """
    Xây dựng state vector normalized [0, 1] từ metrics thực
    hoặc giả lập.

    Parameters
    ----------
    n_edge_nodes   : số edge nodes (default 2)
    prometheus_url : URL Prometheus server
    use_prometheus : True = query Prometheus thật
    """

    # Node names mapping: action index → node name
    NODE_NAMES = {0: "edge_1", 1: "edge_2", 2: "cloud"}

    def __init__(
        self,
        n_edge_nodes: int = 2,
        prometheus_url: str = "http://localhost:9090",
        use_prometheus: bool = False,
    ):
        self.n_edge_nodes = n_edge_nodes
        self.use_prometheus = use_prometheus
        self.obs_dim = n_edge_nodes * 3 + 3 + 3  # edges + cloud + task

        # Prometheus client
        self._prom = PrometheusClient(prometheus_url) if use_prometheus else None

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
        """Query Prometheus cho tất cả nodes."""
        for i in range(self.n_edge_nodes):
            node_name = f"edge_{i + 1}"
            m = self._prom.get_node_metrics(node_name)
            if m.cpu_percent > 0 or m.ram_percent > 0:
                self._edge_metrics[i] = m
            else:
                logger.debug("Prometheus returned empty for %s, using cache", node_name)

        m = self._prom.get_node_metrics("cloud")
        if m.cpu_percent > 0 or m.ram_percent > 0:
            self._cloud_metrics = m
        else:
            logger.debug("Prometheus returned empty for cloud, using cache")

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

    def handle_missing_data(self, node_name: str) -> NodeMetrics:
        """Fallback cho missing data: dùng cache hoặc default."""
        if self._prom is not None:
            cached = self._prom.get_cached(node_name)
            if cached.cpu_percent > 0:
                logger.debug("Using cached metrics for %s", node_name)
                return cached
        logger.warning("No metrics available for %s, using defaults", node_name)
        return NodeMetrics(cpu_percent=50.0, ram_percent=50.0, latency_ms=50.0)
