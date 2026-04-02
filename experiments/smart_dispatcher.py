"""
smart_dispatcher.py
===================
Bộ điều phối thông minh – "trung tâm điều khiển" của hệ thống thật.

Hoạt động:
  1. Đọc metrics thời gian thực từ Prometheus
  2. Nạp DQN model đã train
  3. Với mỗi task đến → xây dựng state vector → DQN chọn node
  4. Gửi task đến node tối ưu (gọi Kubernetes API hoặc HTTP)
  5. Log kết quả để Grafana dashboard hiển thị

Chạy:
    python experiments/smart_dispatcher.py \
        --model models/checkpoints/dqn_best.pth \
        --prometheus http://localhost:9090 \
        --interval 1.0

Chế độ demo (không cần infra thật):
    python experiments/smart_dispatcher.py --demo
"""

import os
import sys
import time
import json
import argparse
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_env.edge_cloud_env import EdgeCloudEnv
from models.dqn_agent import DQNAgent, DQNConfig


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiments/logs/dispatcher.log"),
    ],
)
logger = logging.getLogger("SmartDispatcher")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Task:
    task_id:   str
    cpu_req:   float   # % CPU cần thiết
    ram_req:   float   # % RAM cần thiết
    deadline:  float   # ms
    payload:   dict = None

    def to_json(self):
        return json.dumps(asdict(self))


@dataclass
class DispatchResult:
    task_id:      str
    selected_node: str   # "edge-0", "edge-1", "cloud", ...
    action:       int
    q_values:     list
    latency_est:  float
    cost_est:     float
    sla_met:      bool
    timestamp:    str


# ---------------------------------------------------------------------------
# Prometheus Client
# ---------------------------------------------------------------------------
class PrometheusClient:
    """Đọc metrics từ Prometheus server."""

    def __init__(self, url: str = "http://localhost:9090"):
        self.url = url
        try:
            import requests
            self._session = requests.Session()
        except ImportError:
            logger.warning("requests not installed → dùng fallback simulation")
            self._session = None

    def query(self, promql: str) -> float:
        if self._session is None:
            return 0.0
        try:
            r = self._session.get(
                f"{self.url}/api/v1/query",
                params={"query": promql},
                timeout=2
            )
            result = r.json()["data"]["result"]
            return float(result[0]["value"][1]) if result else 0.0
        except Exception as e:
            logger.debug(f"Prometheus query failed: {e}")
            return 0.0

    def get_node_metrics(self, n_edge_nodes: int = 3) -> dict:
        """
        Lấy CPU, RAM, Latency cho tất cả node.
        Trả về dict dễ dùng để build state vector.
        """
        metrics = {"edge": [], "cloud": {}}

        for i in range(n_edge_nodes):
            metrics["edge"].append({
                "cpu":     self.query(f'node_cpu_usage_percent{{node="edge-{i}"}}'),
                "ram":     self.query(f'node_memory_usage_percent{{node="edge-{i}"}}'),
                "latency": self.query(f'node_latency_ms{{node="edge-{i}"}}'),
            })

        metrics["cloud"] = {
            "cpu":     self.query('node_cpu_usage_percent{node="cloud"}'),
            "ram":     self.query('node_memory_usage_percent{node="cloud"}'),
            "latency": self.query('node_latency_ms{node="cloud"}'),
        }

        return metrics


# ---------------------------------------------------------------------------
# Smart Dispatcher
# ---------------------------------------------------------------------------
class SmartDispatcher:
    """
    Bộ điều phối thông minh kết hợp DQN + Prometheus.

    Parameters
    ----------
    model_path      : đường dẫn tới model checkpoint
    n_edge_nodes    : số lượng Edge node
    prometheus_url  : URL Prometheus server
    dispatch_interval: giây giữa các lần dispatch
    demo_mode       : True = dùng simulation thay vì Prometheus thật
    """

    NODE_NAMES = ["edge-0", "edge-1", "edge-2", "edge-3", "cloud"]

    def __init__(
        self,
        model_path:        str   = "models/checkpoints/dqn_best.pth",
        n_edge_nodes:      int   = 3,
        prometheus_url:    str   = "http://localhost:9090",
        dispatch_interval: float = 1.0,
        demo_mode:         bool  = False,
    ):
        self.n_edge_nodes       = n_edge_nodes
        self.dispatch_interval  = dispatch_interval
        self.demo_mode          = demo_mode
        self._dispatch_count    = 0
        self._results_log       = []

        # Môi trường để build state và tính estimated metrics
        self.env = EdgeCloudEnv(
            n_edge_nodes   = n_edge_nodes,
            use_prometheus = not demo_mode,
            prometheus_url = prometheus_url,
            max_steps      = 99999,
        )
        self.env.reset()

        # Prometheus client
        self.prometheus = PrometheusClient(prometheus_url)

        # Load DQN Agent
        obs_dim   = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.agent = DQNAgent(obs_dim, n_actions)

        if os.path.exists(model_path):
            self.agent.load(model_path)
            logger.info(f"DQN model loaded: {model_path}")
        else:
            logger.warning(f"Model not found: {model_path} – using untrained agent!")

        os.makedirs("experiments/logs", exist_ok=True)
        logger.info(
            f"SmartDispatcher initialized | "
            f"edge_nodes={n_edge_nodes} | demo={demo_mode}"
        )

    # -----------------------------------------------------------------------
    # Core dispatch
    # -----------------------------------------------------------------------

    def dispatch(self, task: Task) -> DispatchResult:
        """
        Xử lý một task: lấy state → DQN → chọn node → log kết quả.
        """
        # 1. Lấy state hiện tại
        obs = self._get_current_state(task)

        # 2. DQN chọn action
        action   = self.agent.select_action(obs, greedy=True)
        q_values = self.agent.get_q_values(obs).tolist()

        # 3. Xác định node
        is_cloud  = (action == self.n_edge_nodes)
        node_name = "cloud" if is_cloud else f"edge-{action}"

        # 4. Ước tính latency và cost
        lat, cost, sla = self.env._simulate_execution(action, is_cloud)

        # 5. Gửi task đến node (thật hoặc demo)
        if not self.demo_mode:
            self._send_to_node(task, node_name)

        result = DispatchResult(
            task_id       = task.task_id,
            selected_node = node_name,
            action        = action,
            q_values      = [round(q, 4) for q in q_values],
            latency_est   = round(lat, 2),
            cost_est      = round(cost, 6),
            sla_met       = sla,
            timestamp     = datetime.now().isoformat(),
        )

        # 6. Log
        self._log_result(task, result)
        self._dispatch_count += 1

        return result

    def _get_current_state(self, task: Task) -> np.ndarray:
        """Xây dựng state vector từ Prometheus metrics + task info."""
        if self.demo_mode:
            # Simulation: dùng env internal state
            self.env._generate_task()
            self.env._task = {
                "cpu_req":  task.cpu_req,
                "ram_req":  task.ram_req,
                "deadline": task.deadline,
            }
            return self.env._build_obs()

        # Real mode: lấy metrics từ Prometheus
        metrics = self.prometheus.get_node_metrics(self.n_edge_nodes)

        # Cập nhật env state với metrics thật
        for i, node_metrics in enumerate(metrics["edge"]):
            self.env._edge_cpu[i]     = node_metrics["cpu"]
            self.env._edge_ram[i]     = node_metrics["ram"]
            self.env._edge_latency[i] = node_metrics["latency"]

        self.env._cloud_cpu     = metrics["cloud"]["cpu"]
        self.env._cloud_ram     = metrics["cloud"]["ram"]
        self.env._cloud_latency = metrics["cloud"]["latency"]

        self.env._task = {
            "cpu_req":  task.cpu_req,
            "ram_req":  task.ram_req,
            "deadline": task.deadline,
        }

        return self.env._build_obs()

    def _send_to_node(self, task: Task, node_name: str):
        """
        Gửi task đến node được chọn (thật).
        Tùy kiến trúc: có thể gọi Kubernetes API, HTTP endpoint, message queue...
        """
        try:
            import requests
            endpoint = f"http://{node_name}:8080/tasks"
            requests.post(endpoint, json=asdict(task), timeout=5)
            logger.debug(f"Task {task.task_id} sent to {node_name}")
        except Exception as e:
            logger.error(f"Failed to send task to {node_name}: {e}")

    def _log_result(self, task: Task, result: DispatchResult):
        """Log dispatch result ra file JSONL và terminal."""
        log_entry = {
            "task_id":      task.task_id,
            "cpu_req":      task.cpu_req,
            "ram_req":      task.ram_req,
            "deadline_ms":  task.deadline,
            "node":         result.selected_node,
            "latency_est":  result.latency_est,
            "cost_est":     result.cost_est,
            "sla_met":      result.sla_met,
            "q_values":     result.q_values,
            "timestamp":    result.timestamp,
        }
        self._results_log.append(log_entry)

        sla_icon = "✅" if result.sla_met else "❌"
        logger.info(
            f"Task {task.task_id:>6} → {result.selected_node:<8} | "
            f"lat={result.latency_est:6.1f}ms | "
            f"cost={result.cost_est:.5f} | "
            f"SLA={sla_icon} | "
            f"Q={[f'{q:.2f}' for q in result.q_values]}"
        )

        # Append vào JSONL log
        with open("experiments/logs/dispatch_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    # -----------------------------------------------------------------------
    # Continuous dispatch loop
    # -----------------------------------------------------------------------

    def run(self, n_tasks: int = None):
        """
        Vòng lặp điều phối liên tục.
        n_tasks=None → chạy vô hạn (dùng Ctrl+C để dừng).
        """
        logger.info(f"🚀 SmartDispatcher running | interval={self.dispatch_interval}s")
        task_counter = 0

        try:
            while True:
                # Sinh task (thật: từ queue; demo: ngẫu nhiên)
                task = self._generate_demo_task(task_counter)
                result = self.dispatch(task)
                task_counter += 1

                if n_tasks and task_counter >= n_tasks:
                    break

                time.sleep(self.dispatch_interval)

        except KeyboardInterrupt:
            logger.info(f"\n⛔ Dispatcher stopped. Total dispatched: {self._dispatch_count}")
            self._print_summary()

    def _generate_demo_task(self, idx: int) -> Task:
        """Sinh task ngẫu nhiên cho demo mode."""
        rng = np.random.default_rng(idx)
        return Task(
            task_id  = f"task_{idx:05d}",
            cpu_req  = float(rng.uniform(5, 60)),
            ram_req  = float(rng.uniform(5, 50)),
            deadline = float(rng.uniform(50, 500)),
        )

    def _print_summary(self):
        """In thống kê tổng hợp sau khi dừng."""
        if not self._results_log:
            return
        sla_rate  = np.mean([r["sla_met"] for r in self._results_log]) * 100
        avg_lat   = np.mean([r["latency_est"] for r in self._results_log])
        avg_cost  = np.mean([r["cost_est"] for r in self._results_log])
        nodes     = [r["node"] for r in self._results_log]
        cloud_pct = nodes.count("cloud") / len(nodes) * 100

        print(f"\n{'='*50}")
        print(f"  Dispatcher Summary ({self._dispatch_count} tasks)")
        print(f"  SLA Rate    : {sla_rate:.1f}%")
        print(f"  Avg Latency : {avg_lat:.1f} ms")
        print(f"  Avg Cost    : {avg_cost:.5f}")
        print(f"  Cloud Usage : {cloud_pct:.1f}%")
        print(f"  Edge Usage  : {100-cloud_pct:.1f}%")
        print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Dispatcher – EdgeCloud DQN")
    parser.add_argument("--model",      default="models/checkpoints/dqn_best.pth")
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--interval",   type=float, default=1.0)
    parser.add_argument("--tasks",      type=int,   default=None,
                        help="Số task để chạy (None = vô hạn)")
    parser.add_argument("--demo",       action="store_true",
                        help="Chạy demo mode (simulation, không cần Prometheus)")
    args = parser.parse_args()

    dispatcher = SmartDispatcher(
        model_path        = args.model,
        prometheus_url    = args.prometheus,
        dispatch_interval = args.interval,
        demo_mode         = args.demo,
    )

    dispatcher.run(n_tasks=args.tasks)


if __name__ == "__main__":
    main()