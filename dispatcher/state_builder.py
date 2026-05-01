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
import time
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
MAX_QUEUE = 20.0

# RTT probe config
RTT_TIMEOUT_S = 1.0     # Timeout HTTP probe (giây)
RTT_FAIL_MS   = 999.0   # Giá trị khi node không reachable
RTT_CACHE_TTL = 10.0    # Cache RTT trong 10s để tránh flood probe


@dataclass
class NodeMetrics:
    """Metrics của một node (Edge hoặc Cloud)."""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    latency_ms: float = 0.0
    queue_length: float = 0.0


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
        worker_urls: Optional[Dict[str, str]] = None,
    ):
        self.n_edge_nodes = n_edge_nodes
        self.use_prometheus = use_prometheus
        # Per-node: cpu, ram, lat, queue → 4 dims
        # Task: cpu_req, ram_req, deadline → 3 dims
        # Temporal: hour_sin, hour_cos → 2 dims
        self.obs_dim = n_edge_nodes * 4 + 4 + 3 + 2

        # Mapping: Prometheus instance (IP:port) → role (edge_1, edge_2, cloud)
        self._instance_map = instance_map or {}

        # Worker URLs cho RTT probe (lazy load từ infra_config nếu None)
        if worker_urls is None:
            try:
                from dispatcher.infra_config import WORKER_URLS
                worker_urls = WORKER_URLS
            except ImportError:
                worker_urls = {}
        self._worker_urls = worker_urls

        # Cache RTT đo được: {role: (rtt_ms, timestamp)}
        self._rtt_cache: Dict[str, tuple] = {}

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
        if self.use_prometheus and self._prom_client is not None:
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
        Mô phỏng tải tăng và queue tăng ở node được chọn, decay ở các node khác.
        Action == reject (n_edge_nodes + 1) → không cập nhật load.

        Decay/load factors PHẢI khớp với rl_env/edge_cloud_env.py để model
        thấy cùng dynamics lúc inference như lúc training.
        """
        decay = 0.92
        queue_decay = 0.85
        is_reject = (action == self.n_edge_nodes + 1)

        for i in range(self.n_edge_nodes):
            m = self._edge_metrics[i]
            if (not is_reject) and i == action:
                m.cpu_percent = float(np.clip(
                    m.cpu_percent * decay + task.cpu_requirement * 0.6, 5, 99
                ))
                m.ram_percent = float(np.clip(
                    m.ram_percent * decay + task.ram_requirement * 0.6, 5, 99
                ))
                m.queue_length = float(np.clip(
                    m.queue_length + 1, 0, MAX_QUEUE
                ))
            else:
                m.cpu_percent = float(np.clip(
                    m.cpu_percent * decay + self._rng.uniform(-3, 3), 5, 95
                ))
                m.ram_percent = float(np.clip(
                    m.ram_percent * decay + self._rng.uniform(-3, 3), 5, 95
                ))
                m.queue_length = float(np.clip(
                    m.queue_length * queue_decay, 0, MAX_QUEUE
                ))
            m.latency_ms = float(np.clip(
                self._rng.uniform(5, 30) + m.cpu_percent * 0.3, 1, MAX_LATENCY
            ))

        cm = self._cloud_metrics
        if (not is_reject) and action == self.n_edge_nodes:
            cm.cpu_percent = float(np.clip(
                cm.cpu_percent * decay + task.cpu_requirement * 0.4, 5, 99
            ))
            cm.ram_percent = float(np.clip(
                cm.ram_percent * decay + task.ram_requirement * 0.4, 5, 99
            ))
            cm.queue_length = float(np.clip(
                cm.queue_length + 1, 0, MAX_QUEUE
            ))
        else:
            cm.cpu_percent = float(np.clip(
                cm.cpu_percent * decay + self._rng.uniform(-3, 3), 5, 80
            ))
            cm.ram_percent = float(np.clip(
                cm.ram_percent * decay + self._rng.uniform(-3, 3), 5, 80
            ))
            cm.queue_length = float(np.clip(
                cm.queue_length * queue_decay, 0, MAX_QUEUE
            ))
        cm.latency_ms = float(
            self._rng.uniform(30, 80) + cm.cpu_percent * 0.2
        )

    def reset_simulation(self, seed: int = 42):
        """Reset internal state cho simulation mode."""
        self._rng = np.random.default_rng(seed)
        self._rtt_cache.clear()
        for i in range(self.n_edge_nodes):
            self._edge_metrics[i] = NodeMetrics(
                cpu_percent=float(self._rng.uniform(20, 70)),
                ram_percent=float(self._rng.uniform(30, 80)),
                latency_ms=float(self._rng.uniform(10, 50)),
                queue_length=0.0,
            )
        self._cloud_metrics = NodeMetrics(
            cpu_percent=float(self._rng.uniform(10, 50)),
            ram_percent=float(self._rng.uniform(10, 50)),
            latency_ms=float(self._rng.uniform(30, 80)),
            queue_length=0.0,
        )

    def get_node_name(self, action: int) -> str:
        """Map action index sang tên node. Action = n_edge_nodes+1 → 'rejected'."""
        if action < self.n_edge_nodes:
            return f"edge_{action + 1}"
        if action == self.n_edge_nodes:
            return "cloud"
        return "rejected"

    def query_load_in_window(
        self,
        role: str,
        t_end_unix: float,
        duration_s: float,
    ) -> tuple:
        """
        Query Prometheus for AVG CPU% / RAM% on the node `role` over the
        time window [t_end_unix - duration_s, t_end_unix].

        Used for sim calibration: ground-truth load while a pod was actually
        executing. Different from build_state's snapshot, which captures
        load AT dispatch time (before pod started burning CPU).

        Returns (cpu_pct, ram_pct), each may be None if query fails.
        """
        if not self.use_prometheus or self._prom_client is None:
            return None, None

        # Reverse instance_map: role -> instance
        role_to_inst = {v: k for k, v in (
            getattr(self, "_instance_map", {}) or {}
        ).items()}
        instance = role_to_inst.get(role)
        if not instance:
            return None, None

        # Min 5s for rate to have ≥2 samples at scrape_interval=5s.
        win_s = max(int(duration_s), 5)
        t_str = str(int(t_end_unix))

        cpu_query = (
            f'100 * (1 - avg by(instance)('
            f'rate(node_cpu_seconds_total{{mode="idle",instance="{instance}"}}[{win_s}s])'
            f'))'
        )
        ram_query = (
            f'100 * (1 - '
            f'node_memory_MemAvailable_bytes{{instance="{instance}"}} / '
            f'node_memory_MemTotal_bytes{{instance="{instance}"}}'
            f')'
        )

        from week2.prometheus_client import PrometheusClient as _PC
        cpu = ram = None
        try:
            resp = self._prom_client.instant_query(cpu_query, time=t_str)
            cpu = _PC.vector_to_map(resp).get(instance)
        except Exception as e:
            logger.warning("query_load_in_window cpu failed role=%s: %s", role, e)
        try:
            resp = self._prom_client.instant_query(ram_query, time=t_str)
            ram = _PC.vector_to_map(resp).get(instance)
        except Exception as e:
            logger.warning("query_load_in_window ram failed role=%s: %s", role, e)

        return cpu, ram

    def get_current_metrics_summary(self) -> dict:
        """Trả về summary metrics hiện tại (dùng cho logging/debug)."""
        summary = {}
        for i, m in enumerate(self._edge_metrics):
            prefix = f"edge_{i + 1}"
            summary[f"{prefix}_cpu"]   = round(m.cpu_percent, 1)
            summary[f"{prefix}_ram"]   = round(m.ram_percent, 1)
            summary[f"{prefix}_lat"]   = round(m.latency_ms, 1)
            summary[f"{prefix}_queue"] = round(m.queue_length, 1)
        summary["cloud_cpu"]   = round(self._cloud_metrics.cpu_percent, 1)
        summary["cloud_ram"]   = round(self._cloud_metrics.ram_percent, 1)
        summary["cloud_lat"]   = round(self._cloud_metrics.latency_ms, 1)
        summary["cloud_queue"] = round(self._cloud_metrics.queue_length, 1)
        return summary

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _measure_rtt(self, role: str) -> float:
        """
        Đo RTT thật bằng HTTP GET tới {worker_url}/health.
        Cache 10s để tránh flood probe.

        Returns: RTT (ms). Trả về RTT_FAIL_MS nếu node unreachable.
        """
        cached = self._rtt_cache.get(role)
        if cached and time.time() - cached[1] < RTT_CACHE_TTL:
            return cached[0]

        worker_url = self._worker_urls.get(role)
        if not worker_url:
            return RTT_FAIL_MS

        # Convert /task → /health (task_worker expose cả 2 endpoint)
        health_url = worker_url.replace("/task", "/health")

        try:
            import requests
            t0 = time.perf_counter()
            r = requests.get(health_url, timeout=RTT_TIMEOUT_S)
            r.raise_for_status()
            rtt = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            logger.debug("RTT probe failed for %s: %s", role, e)
            rtt = RTT_FAIL_MS

        self._rtt_cache[role] = (rtt, time.time())
        return rtt

    def _fetch_prometheus_metrics(self):
        """
        Query Prometheus qua Person 1 client (get_dispatcher_metrics).

        Person 1 trả về dict dạng:
          {"192.168.1.100:9100": {"cpu_usage_pct": 45.2, "ram_usage_pct": 67.8,
                                   "queue_length": 3, "network_latency_ms": 12.5}, ...}

        Dùng instance_map để map IP:port → edge_1/edge_2/cloud.
        Log rõ ràng khi không match được instance để tránh fallback im lặng.
        """
        try:
            raw = self._prom_client.get_dispatcher_metrics()
        except Exception as e:
            logger.warning("Prometheus query failed: %s — using cached state", e)
            return

        if not raw:
            logger.warning(
                "Prometheus trả về rỗng — không có node_exporter nào được scrape. "
                "Đang dùng cached state (có thể là zeros)."
            )
            return

        # Đảo ngược: role → instance
        role_to_inst = {v: k for k, v in self._instance_map.items()}

        # Nếu không có instance_map, gán theo thứ tự alphabetical
        if not role_to_inst:
            instances = sorted(raw.keys())
            roles = [f"edge_{i + 1}" for i in range(self.n_edge_nodes)] + ["cloud"]
            for inst, role in zip(instances, roles):
                role_to_inst[role] = inst
            logger.warning(
                "instance_map không được cấu hình — auto-assign theo thứ tự alphabet: %s",
                role_to_inst,
            )

        # Đếm số instance match được để phát hiện cấu hình sai
        matched = 0
        expected_roles = [f"edge_{i + 1}" for i in range(self.n_edge_nodes)] + ["cloud"]

        # Cập nhật edge metrics
        for i in range(self.n_edge_nodes):
            role = f"edge_{i + 1}"
            inst = role_to_inst.get(role)
            if inst is None:
                logger.warning("Role %s không có trong instance_map", role)
                continue
            if inst not in raw:
                logger.warning(
                    "Role %s mapped tới %s nhưng Prometheus không có instance này. "
                    "Available: %s",
                    role, inst, list(raw.keys()),
                )
                continue

            d = raw[inst]
            cpu = d.get("cpu_usage_pct") or 0.0
            # Latency từ Prometheus nếu có, ngược lại đo RTT thật qua HTTP probe
            lat = d.get("network_latency_ms") or 0.0
            if lat == 0.0:
                lat = self._measure_rtt(role)
            self._edge_metrics[i] = NodeMetrics(
                cpu_percent=cpu,
                ram_percent=d.get("ram_usage_pct") or 0.0,
                latency_ms=lat,
                queue_length=d.get("queue_length") or 0.0,
            )
            matched += 1

        # Cập nhật cloud metrics
        inst = role_to_inst.get("cloud")
        if inst is None:
            logger.warning("Role 'cloud' không có trong instance_map")
        elif inst not in raw:
            logger.warning(
                "Role 'cloud' mapped tới %s nhưng Prometheus không có instance này. "
                "Available: %s",
                inst, list(raw.keys()),
            )
        else:
            d = raw[inst]
            cpu = d.get("cpu_usage_pct") or 0.0
            lat = d.get("network_latency_ms") or 0.0
            if lat == 0.0:
                lat = self._measure_rtt("cloud")
            self._cloud_metrics = NodeMetrics(
                cpu_percent=cpu,
                ram_percent=d.get("ram_usage_pct") or 0.0,
                latency_ms=lat,
                queue_length=d.get("queue_length") or 0.0,
            )
            matched += 1

        # Cảnh báo nghiêm trọng nếu không match được instance nào
        if matched == 0:
            logger.error(
                "KHÔNG match được instance Prometheus nào với INSTANCE_MAP! "
                "Dispatcher đang dùng zeros. "
                "Expected roles=%s, INSTANCE_MAP=%s, Prometheus instances=%s. "
                "Cập nhật dispatcher/infra_config.py với đúng IP.",
                expected_roles, self._instance_map, list(raw.keys()),
            )
        elif matched < len(expected_roles):
            logger.warning(
                "Chỉ match được %d/%d instances. Một số node sẽ dùng giá trị cũ.",
                matched, len(expected_roles),
            )

    def _compose_observation(self, task: TaskInfo) -> np.ndarray:
        """Compose và normalize observation vector."""
        obs = []

        # Edge nodes: [cpu, ram, latency, queue] per node
        for i in range(self.n_edge_nodes):
            m = self._edge_metrics[i]
            obs.extend([
                m.cpu_percent  / MAX_CPU,
                m.ram_percent  / MAX_RAM,
                m.latency_ms   / MAX_LATENCY,
                m.queue_length / MAX_QUEUE,
            ])

        # Cloud: [cpu, ram, latency, queue]
        cm = self._cloud_metrics
        obs.extend([
            cm.cpu_percent  / MAX_CPU,
            cm.ram_percent  / MAX_RAM,
            cm.latency_ms   / MAX_LATENCY,
            cm.queue_length / MAX_QUEUE,
        ])

        # Task: [cpu_req, ram_req, deadline]
        obs.extend([
            task.cpu_requirement / MAX_CPU,
            task.ram_requirement / MAX_RAM,
            task.deadline_ms     / MAX_DEADLINE,
        ])

        # Temporal: sin/cos giờ trong ngày, map về [0, 1] để khớp obs space
        hour = (time.time() / 3600.0) % 24.0
        obs.extend([
            (np.sin(2 * np.pi * hour / 24.0) + 1.0) / 2.0,
            (np.cos(2 * np.pi * hour / 24.0) + 1.0) / 2.0,
        ])

        arr = np.array(obs, dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

