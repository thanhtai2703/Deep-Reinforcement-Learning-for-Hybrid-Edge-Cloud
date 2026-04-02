import gymnasium as gym
import numpy as np
from gymnasium import spaces


EDGE_COST_PER_UNIT  = 0.01   # Chi phí rẻ hơn nhưng tài nguyên giới hạn
CLOUD_COST_PER_UNIT = 0.05   # Chi phí đắt hơn nhưng tài nguyên dồi dào
MAX_LATENCY         = 200.0  # ms – dùng để normalize
MAX_CPU             = 100.0  # % CPU
MAX_RAM             = 100.0  # % RAM usage
MAX_DEADLINE        = 500.0  # ms


class EdgeCloudEnv(gym.Env):
    """
    Môi trường Gymnasium mô phỏng bài toán điều phối tác vụ
    trong hệ thống Hybrid Edge-Cloud.

    Parameters
    ----------
    n_edge_nodes   : số lượng Edge node (mặc định 3 – mô phỏng cụm K3s)
    use_prometheus : True → đọc metrics thật từ Prometheus
    prometheus_url : URL của Prometheus server
    max_steps      : số bước tối đa mỗi episode
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_edge_nodes: int = 3,
        use_prometheus: bool = False,
        prometheus_url: str = "http://localhost:9090",
        max_steps: int = 200,
    ):
        super().__init__()
        self.n_edge_nodes    = n_edge_nodes
        self.use_prometheus  = use_prometheus
        self.prometheus_url  = prometheus_url
        self.max_steps       = max_steps
        self._step_count     = 0

        # ------------------------------------------------------------------
        # Không gian hành động: 0..n_edge_nodes-1 = Edge nodes, n = Cloud
        # ------------------------------------------------------------------
        self.n_actions = n_edge_nodes + 1          # Edge nodes + 1 Cloud
        self.action_space = spaces.Discrete(self.n_actions)

        # ------------------------------------------------------------------
        # Không gian trạng thái (tất cả normalize về [0, 1])
        # Với mỗi Edge node: [cpu, ram, latency]
        # Cloud            : [cpu, ram, latency]
        # Task             : [cpu_req, ram_req, deadline]
        # ------------------------------------------------------------------
        obs_dim = n_edge_nodes * 3 + 3 + 3        # nodes + cloud + task
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Khởi tạo state nội bộ
        self._edge_cpu      = np.zeros(n_edge_nodes)
        self._edge_ram      = np.zeros(n_edge_nodes)
        self._edge_latency  = np.zeros(n_edge_nodes)
        self._cloud_cpu     = 0.0
        self._cloud_ram     = 0.0
        self._cloud_latency = 0.0
        self._task          = {"cpu_req": 0.0, "ram_req": 0.0, "deadline": 0.0}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._refresh_node_metrics()
        self._generate_task()
        obs  = self._build_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self._step_count += 1

        # Xác định node được chọn
        is_cloud = (action == self.n_edge_nodes)

        # Tính toán kết quả khi gửi task đến node được chọn
        latency, cost, sla_met = self._simulate_execution(action, is_cloud)

        # Hàm thưởng
        reward = self._compute_reward(latency, cost, sla_met)

        # Cập nhật tải node sau khi nhận task
        self._update_node_load(action, is_cloud)

        # Sinh task mới cho bước tiếp theo
        self._generate_task()

        # Refresh metrics node (có noise để mô phỏng biến động mạng)
        self._refresh_node_metrics(add_noise=True)

        obs       = self._build_obs()
        terminated = False
        truncated  = self._step_count >= self.max_steps
        info = {
            "latency": latency,
            "cost": cost,
            "sla_met": sla_met,
            "action": action,
            "is_cloud": is_cloud,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        edge_str = " | ".join(
            f"E{i}(cpu={self._edge_cpu[i]:.1f}% ram={self._edge_ram[i]:.1f}%)"
            for i in range(self.n_edge_nodes)
        )
        print(
            f"[Step {self._step_count:3d}] {edge_str} | "
            f"Cloud(cpu={self._cloud_cpu:.1f}% ram={self._cloud_ram:.1f}%) | "
            f"Task(cpu={self._task['cpu_req']:.1f} ram={self._task['ram_req']:.1f} "
            f"deadline={self._task['deadline']:.0f}ms)"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _refresh_node_metrics(self, add_noise: bool = False):
        """Lấy metrics node – Simulation hoặc Prometheus thật."""
        if self.use_prometheus:
            self._fetch_prometheus_metrics()
        else:
            self._simulate_node_metrics(add_noise)

    def _simulate_node_metrics(self, add_noise: bool = False):
        """Sinh metrics giả lập cho Edge nodes và Cloud."""
        noise = 5.0 if add_noise else 0.0
        for i in range(self.n_edge_nodes):
            self._edge_cpu[i]     = np.clip(self._edge_cpu[i] + self.np_random.uniform(-noise, noise), 10, 90)
            self._edge_ram[i]     = np.clip(self._edge_ram[i] + self.np_random.uniform(-noise, noise), 10, 90)
            self._edge_latency[i] = np.clip(
                self.np_random.uniform(5, 30) + (self._edge_cpu[i] * 0.3), 1, MAX_LATENCY
            )

        if not add_noise:
            # Init lần đầu
            self._edge_cpu     = self.np_random.uniform(20, 70, self.n_edge_nodes)
            self._edge_ram     = self.np_random.uniform(30, 80, self.n_edge_nodes)
            self._cloud_cpu    = float(self.np_random.uniform(10, 50))
            self._cloud_ram    = float(self.np_random.uniform(10, 50))

        self._cloud_cpu     = np.clip(self._cloud_cpu + self.np_random.uniform(-noise, noise), 5, 80)
        self._cloud_ram     = np.clip(self._cloud_ram + self.np_random.uniform(-noise, noise), 5, 80)
        # Cloud latency cao hơn Edge do network overhead
        self._cloud_latency = float(self.np_random.uniform(30, 80) + self._cloud_cpu * 0.2)

    def _fetch_prometheus_metrics(self):
        """
        Đọc metrics thật từ Prometheus.
        Gọi khi use_prometheus=True (chế độ production).
        """
        try:
            import requests
            def query(q):
                r = requests.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={"query": q}, timeout=2
                )
                result = r.json()["data"]["result"]
                return float(result[0]["value"][1]) if result else 0.0

            for i in range(self.n_edge_nodes):
                self._edge_cpu[i]     = query(f'node_cpu_usage_percent{{node="edge-{i}"}}')
                self._edge_ram[i]     = query(f'node_memory_usage_percent{{node="edge-{i}"}}')
                self._edge_latency[i] = query(f'node_latency_ms{{node="edge-{i}"}}')

            self._cloud_cpu     = query('node_cpu_usage_percent{node="cloud"}')
            self._cloud_ram     = query('node_memory_usage_percent{node="cloud"}')
            self._cloud_latency = query('node_latency_ms{node="cloud"}')

        except Exception as e:
            print(f"[Prometheus] Lỗi: {e} → fallback sang simulation")
            self._simulate_node_metrics(add_noise=True)

    def _generate_task(self):
        """Sinh một tác vụ mới với yêu cầu tài nguyên và deadline ngẫu nhiên."""
        self._task = {
            "cpu_req":  float(self.np_random.uniform(5, 60)),   # % CPU cần
            "ram_req":  float(self.np_random.uniform(5, 50)),   # % RAM cần
            "deadline": float(self.np_random.uniform(50, MAX_DEADLINE)),  # ms
        }

    def _simulate_execution(self, action: int, is_cloud: bool):
        """
        Ước lượng latency và cost khi gửi task đến node được chọn.
        Returns: (latency_ms, cost, sla_met)
        """
        task_cpu = self._task["cpu_req"]
        task_ram = self._task["ram_req"]
        deadline  = self._task["deadline"]

        if is_cloud:
            node_cpu     = self._cloud_cpu
            node_ram     = self._cloud_ram
            base_latency = self._cloud_latency
            cost_rate    = CLOUD_COST_PER_UNIT
        else:
            node_cpu     = self._edge_cpu[action]
            node_ram     = self._edge_ram[action]
            base_latency = self._edge_latency[action]
            cost_rate    = EDGE_COST_PER_UNIT

        # Latency tăng tỉ lệ với mức tải hiện tại
        load_factor  = 1.0 + (node_cpu / 100.0) * 2.0
        latency      = base_latency * load_factor

        # Nếu node quá tải (cpu > 90%), latency tăng mạnh (penalty)
        if node_cpu + task_cpu > 95:
            latency *= 3.0

        cost    = (task_cpu + task_ram) * cost_rate
        sla_met = latency <= deadline

        return float(latency), float(cost), bool(sla_met)

    def _update_node_load(self, action: int, is_cloud: bool):
        """Cập nhật tải CPU/RAM sau khi node nhận task."""
        task_cpu = self._task["cpu_req"]
        task_ram = self._task["ram_req"]
        decay    = 0.85  # Tải giảm dần theo thời gian (decay)

        if is_cloud:
            self._cloud_cpu = np.clip(self._cloud_cpu * decay + task_cpu * 0.3, 5, 99)
            self._cloud_ram = np.clip(self._cloud_ram * decay + task_ram * 0.3, 5, 99)
        else:
            self._edge_cpu[action] = np.clip(
                self._edge_cpu[action] * decay + task_cpu * 0.5, 5, 99
            )
            self._edge_ram[action] = np.clip(
                self._edge_ram[action] * decay + task_ram * 0.5, 5, 99
            )

    def _compute_reward(self, latency: float, cost: float, sla_met: bool) -> float:
        """
        Hàm thưởng cân bằng 3 mục tiêu:
          1. Giảm latency
          2. Giảm chi phí
          3. Đảm bảo SLA (deadline)

        Reward dương khi SLA đạt, âm khi vi phạm.
        """
        latency_norm = latency / MAX_LATENCY          # [0, 1]
        cost_norm    = cost / (CLOUD_COST_PER_UNIT * 110)  # [0, 1]

        # Thưởng âm cho latency và cost
        reward = -0.6 * latency_norm - 0.2 * cost_norm

        # Bonus/Penalty SLA
        if sla_met:
            reward += 1.0
        else:
            reward -= 2.0   # Vi phạm deadline bị phạt nặng

        return float(reward)

    def _build_obs(self) -> np.ndarray:
        """Xây dựng vector quan sát đã normalize về [0, 1]."""
        edge_obs = []
        for i in range(self.n_edge_nodes):
            edge_obs.extend([
                self._edge_cpu[i]     / MAX_CPU,
                self._edge_ram[i]     / MAX_RAM,
                self._edge_latency[i] / MAX_LATENCY,
            ])

        cloud_obs = [
            self._cloud_cpu     / MAX_CPU,
            self._cloud_ram     / MAX_RAM,
            self._cloud_latency / MAX_LATENCY,
        ]

        task_obs = [
            self._task["cpu_req"]  / MAX_CPU,
            self._task["ram_req"]  / MAX_RAM,
            self._task["deadline"] / MAX_DEADLINE,
        ]

        obs = np.array(edge_obs + cloud_obs + task_obs, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)