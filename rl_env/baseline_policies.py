"""
baseline_policies.py
====================
Các thuật toán điều phối truyền thống dùng để so sánh với DQN:

  - RandomPolicy        : chọn node ngẫu nhiên
  - RoundRobinPolicy    : xoay vòng lần lượt qua các node
  - LeastConnectionPolicy: chọn node có CPU thấp nhất (ít tải nhất)
  - EdgeOnlyPolicy      : luôn gửi lên Edge node có tải thấp nhất
  - CloudOnlyPolicy     : luôn gửi lên Cloud
  - ThresholdPolicy     : edge nếu còn tải, cloud nếu quá tải

Lưu ý: Các baseline không bao giờ chọn action 'reject' — đó là khả năng riêng
của RL agent. Baselines chỉ dispatch tới {edges, cloud}.

State vector: mỗi node chiếm 4 dims (cpu, ram, lat, queue).
"""

import numpy as np

# Mỗi node chiếm 4 dims: cpu, ram, latency, queue
NODE_DIM = 4


class BasePolicy:
    """Interface chuẩn cho mọi policy."""

    def __init__(self, n_actions: int, n_edge_nodes: int):
        self.n_actions = n_actions             # tổng action (gồm reject)
        self.n_edge_nodes = n_edge_nodes
        # Số action dispatch (không gồm reject) = edges + cloud
        self.n_dispatch_actions = n_edge_nodes + 1
        self.name = "BasePolicy"

    def select_action(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    def reset(self):
        """Reset internal state (nếu có) khi bắt đầu episode mới."""
        pass

    def __repr__(self):
        return self.name


# ---------------------------------------------------------------------------
# 1. Random Policy
# ---------------------------------------------------------------------------
class RandomPolicy(BasePolicy):
    """Chọn node ngẫu nhiên (không bao giờ reject)."""

    def __init__(self, n_actions: int, n_edge_nodes: int, seed: int = 42):
        super().__init__(n_actions, n_edge_nodes)
        self.name = "Random"
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(0, self.n_dispatch_actions))


# ---------------------------------------------------------------------------
# 2. Round Robin Policy
# ---------------------------------------------------------------------------
class RoundRobinPolicy(BasePolicy):
    """
    Xoay vòng lần lượt qua tất cả node (Edge + Cloud).
    Không quan tâm đến trạng thái tài nguyên.
    """

    def __init__(self, n_actions: int, n_edge_nodes: int):
        super().__init__(n_actions, n_edge_nodes)
        self.name = "RoundRobin"
        self._cursor = 0

    def select_action(self, obs: np.ndarray) -> int:
        action = self._cursor
        # Xoay vòng qua dispatch actions (edges + cloud), bỏ qua reject
        self._cursor = (self._cursor + 1) % self.n_dispatch_actions
        return action

    def reset(self):
        self._cursor = 0


# ---------------------------------------------------------------------------
# 3. Least Connection (Least CPU Load) Policy
# ---------------------------------------------------------------------------
class LeastConnectionPolicy(BasePolicy):
    """
    Chọn node có CPU thấp nhất trong observation.

    Cấu trúc obs (mỗi node chiếm 4 giá trị: cpu, ram, lat, queue):
      [e0_cpu, e0_ram, e0_lat, e0_queue, ..., eN_*,
       cloud_cpu, cloud_ram, cloud_lat, cloud_queue,
       task_cpu, task_ram, task_deadline,
       hour_sin, hour_cos]
    """

    def __init__(self, n_actions: int, n_edge_nodes: int):
        super().__init__(n_actions, n_edge_nodes)
        self.name = "LeastConnection"

    def select_action(self, obs: np.ndarray) -> int:
        cpu_loads = []
        # CPU edge i tại index i * NODE_DIM
        for i in range(self.n_edge_nodes):
            cpu_loads.append(obs[i * NODE_DIM])
        # CPU cloud tại index n_edge_nodes * NODE_DIM
        cpu_loads.append(obs[self.n_edge_nodes * NODE_DIM])
        return int(np.argmin(cpu_loads))


# ---------------------------------------------------------------------------
# 4. Edge Only Policy
# ---------------------------------------------------------------------------
class EdgeOnlyPolicy(BasePolicy):
    """
    Luôn gửi task lên Edge node có CPU thấp nhất.
    Không bao giờ dùng Cloud – dùng để minh họa Edge-only scenario.
    """

    def __init__(self, n_actions: int, n_edge_nodes: int):
        super().__init__(n_actions, n_edge_nodes)
        self.name = "EdgeOnly"

    def select_action(self, obs: np.ndarray) -> int:
        edge_cpus = [obs[i * NODE_DIM] for i in range(self.n_edge_nodes)]
        return int(np.argmin(edge_cpus))   # trả về index Edge node


# ---------------------------------------------------------------------------
# 5. Cloud Only Policy
# ---------------------------------------------------------------------------
class CloudOnlyPolicy(BasePolicy):
    """
    Luôn gửi task lên Cloud.
    Dùng để minh họa Cloud-only scenario (chi phí cao).
    """

    def __init__(self, n_actions: int, n_edge_nodes: int):
        super().__init__(n_actions, n_edge_nodes)
        self.name = "CloudOnly"

    def select_action(self, obs: np.ndarray) -> int:
        return self.n_edge_nodes   # index cuối = Cloud


# ---------------------------------------------------------------------------
# 6. Threshold Policy (Heuristic thông minh hơn)
# ---------------------------------------------------------------------------
class ThresholdPolicy(BasePolicy):
    """
    Heuristic đơn giản:
      - Nếu Edge node nào có CPU < threshold → gửi lên Edge đó
      - Nếu tất cả Edge quá tải (CPU >= threshold) → gửi lên Cloud

    Đây là baseline heuristic "thông minh" hơn Round-Robin.
    """

    def __init__(
        self,
        n_actions: int,
        n_edge_nodes: int,
        cpu_threshold: float = 0.7,   # 70% CPU → xem là quá tải
    ):
        super().__init__(n_actions, n_edge_nodes)
        self.name = f"Threshold(cpu<{int(cpu_threshold*100)}%)"
        self.cpu_threshold = cpu_threshold

    def select_action(self, obs: np.ndarray) -> int:
        for i in range(self.n_edge_nodes):
            if obs[i * NODE_DIM] < self.cpu_threshold:
                return i   # Edge node còn tài nguyên

        # Tất cả Edge bị quá tải → Cloud
        return self.n_edge_nodes


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
def get_all_baselines(n_actions: int, n_edge_nodes: int) -> list:
    """Trả về danh sách tất cả baseline policies để tiện so sánh."""
    return [
        RandomPolicy(n_actions, n_edge_nodes),
        RoundRobinPolicy(n_actions, n_edge_nodes),
        LeastConnectionPolicy(n_actions, n_edge_nodes),
        EdgeOnlyPolicy(n_actions, n_edge_nodes),
        CloudOnlyPolicy(n_actions, n_edge_nodes),
        ThresholdPolicy(n_actions, n_edge_nodes, cpu_threshold=0.7),
    ]