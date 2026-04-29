"""
edge_cloud_env_calibrated.py
============================
Calibrated version of EdgeCloudEnv. Inherits everything from the original
env except _simulate_execution(), which is replaced with predict_total_ms()
from calibration/calibrated_constants.py.

The calibrated formula is:
  total_ms = submit_overhead[role]
           + container_startup[role]
           + α[role] * workload_proxy * (1 + β[role] * cpu/100)
           + poll_overhead[role]

Constants are fitted from real K8s execution logs (see
calibration/calibrate_env.py).

Drop-in replacement:
  from rl_env.edge_cloud_env_calibrated import EdgeCloudEnvCalibrated as Env
  # Train DQN/PPO normally; observation/action space unchanged.
"""

from __future__ import annotations

import numpy as np

from rl_env.edge_cloud_env import (
    EdgeCloudEnv,
    EDGE_COST_PER_UNIT,
    CLOUD_COST_PER_UNIT,
)

# Calibrated env operates at K8s timescale: 5-25s instead of 10-200ms.
# Override MAX_LATENCY and deadline range accordingly so reward gradient
# doesn't saturate and task distribution matches dispatcher_cli.
CAL_MAX_LATENCY_MS = 30000.0  # 30s
CAL_DEADLINE_MIN_MS = 2000.0
CAL_DEADLINE_MAX_MS = 30000.0

try:
    from calibration.calibrated_constants import (
        EXEC_CALIBRATION,
        OVERHEAD_CALIBRATION,
        predict_total_ms,
    )
    _CAL_AVAILABLE = True
except ImportError:
    _CAL_AVAILABLE = False


class EdgeCloudEnvCalibrated(EdgeCloudEnv):
    """
    EdgeCloudEnv with execution latency predicted from real K8s log fits.

    Behavior delta vs base env:
    - _simulate_execution(): replaced with calibrated formula.
    - Cost / SLA / queue logic: unchanged (cost is workload-only, doesn't
      depend on infra).
    - State / action / reward: unchanged. Same RL agent code works.

    Side effects of calibration:
    - Latency magnitudes are now in 5-25s range (real K8s) instead of
      10-200ms (uncal). Reward signal scale changes accordingly — but
      reward formula in EdgeCloudEnv normalizes by MAX_LATENCY=200, so
      latency_norm will saturate at 1.0 most of the time. **You may want
      to bump MAX_LATENCY to ~30000 (30s) when training on this env**,
      otherwise the gradient w.r.t. latency vanishes.
    - Per-role overhead means agent sees different "tax" per node, which
      is exactly what we want it to learn (e.g. edge_1 has 3s submit cost
      that's invisible in uncal env).

    Parameters
    ----------
    Same as EdgeCloudEnv. Calibrated constants are loaded automatically.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not _CAL_AVAILABLE:
            raise ImportError(
                "calibration.calibrated_constants not found. "
                "Run `python -m calibration.calibrate_env` first."
            )

        # Validate all roles needed by env are calibrated.
        required_roles = [f"edge_{i + 1}" for i in range(self.n_edge_nodes)] + ["cloud"]
        missing = [r for r in required_roles if r not in EXEC_CALIBRATION]
        if missing:
            raise ValueError(
                f"Missing calibration data for roles: {missing}. "
                f"Available: {list(EXEC_CALIBRATION.keys())}. "
                f"Either re-run calibrate_env.py with these roles in the "
                f"data, or use EdgeCloudEnv (uncalibrated)."
            )

    def _action_to_role(self, action: int, is_cloud: bool) -> str:
        if is_cloud:
            return "cloud"
        return f"edge_{action + 1}"

    def _generate_task(self):
        """
        Override: deadline in real K8s scale (2-30s) instead of 50-500ms.
        CPU/RAM ranges unchanged — they match dispatcher_cli's TaskInfo.
        """
        self._task = {
            "cpu_req":  float(self.np_random.uniform(5, 60)),
            "ram_req":  float(self.np_random.uniform(5, 50)),
            "deadline": float(self.np_random.uniform(
                CAL_DEADLINE_MIN_MS, CAL_DEADLINE_MAX_MS
            )),
        }

    def _compute_reward(
        self,
        latency: float,
        cost: float,
        sla_met: bool,
        deadline: float,
        is_reject: bool = False,
    ) -> float:
        """
        Override: normalize latency by CAL_MAX_LATENCY_MS=30s, not 200ms.
        Without this, latency_norm saturates at 100+ and gradient vanishes.
        Reward shape and weights match base env.
        """
        from rl_env.edge_cloud_env import REJECT_PENALTY
        if is_reject:
            return REJECT_PENALTY

        latency_norm = latency / CAL_MAX_LATENCY_MS
        cost_norm = cost / (CLOUD_COST_PER_UNIT * 110)
        slack = (deadline - latency) / max(deadline, 1.0)
        sla_signal = float(np.tanh(3.0 * slack))
        reward = -0.5 * latency_norm - 0.2 * cost_norm + 0.5 * sla_signal
        if not sla_met:
            reward -= 0.5
        return float(reward)

    def _build_obs(self) -> np.ndarray:
        """
        Override: normalize latency state by CAL_MAX_LATENCY_MS so the
        observation space [0,1] still applies to K8s-scale latencies.
        """
        from rl_env.edge_cloud_env import MAX_CPU, MAX_RAM, MAX_QUEUE

        edge_obs = []
        for i in range(self.n_edge_nodes):
            edge_obs.extend([
                self._edge_cpu[i]     / MAX_CPU,
                self._edge_ram[i]     / MAX_RAM,
                self._edge_latency[i] / CAL_MAX_LATENCY_MS,
                self._edge_queue[i]   / MAX_QUEUE,
            ])
        cloud_obs = [
            self._cloud_cpu     / MAX_CPU,
            self._cloud_ram     / MAX_RAM,
            self._cloud_latency / CAL_MAX_LATENCY_MS,
            self._cloud_queue   / MAX_QUEUE,
        ]
        task_obs = [
            self._task["cpu_req"]  / MAX_CPU,
            self._task["ram_req"]  / MAX_RAM,
            self._task["deadline"] / CAL_DEADLINE_MAX_MS,
        ]

        import time as _time
        hour = (_time.time() / 3600.0) % 24.0
        hour_sin = (np.sin(2 * np.pi * hour / 24.0) + 1.0) / 2.0
        hour_cos = (np.cos(2 * np.pi * hour / 24.0) + 1.0) / 2.0
        temporal_obs = [hour_sin, hour_cos]

        obs = np.array(edge_obs + cloud_obs + task_obs + temporal_obs,
                       dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _simulate_execution(self, action: int, is_cloud: bool):
        """
        Calibrated execution time using fitted constants per role.
        Returns: (latency_ms, cost, sla_met)
        """
        # Reject branch handled by caller (env.step), not here.
        if action >= self.n_edge_nodes + 1:
            return 0.0, 0.0, False

        role = self._action_to_role(action, is_cloud)

        # Pick task params and node CPU. In sim, we use current snapshot CPU
        # as proxy for cpu_during_exec — fine because β≈0 in our fit
        # (cgroup isolation), so this term contributes little.
        task_cpu_req = self._task["cpu_req"]
        task_ram_req = self._task["ram_req"]
        deadline = self._task["deadline"]

        if is_cloud:
            node_cpu = self._cloud_cpu
            cost_rate = CLOUD_COST_PER_UNIT
        else:
            node_cpu = self._edge_cpu[action]
            cost_rate = EDGE_COST_PER_UNIT

        # Calibrated end-to-end latency.
        # NOTE: env's task spec uses cpu_req / deadline in the same units
        # as dispatcher (TaskInfo), so workload_proxy formula matches.
        latency = predict_total_ms(
            role=role,
            cpu_requirement=task_cpu_req,
            deadline_ms=deadline,
            cpu_during_exec=float(node_cpu),
        )

        # Cost is workload-only; unchanged from base env.
        cost = (task_cpu_req + task_ram_req) * cost_rate
        sla_met = latency <= deadline

        return float(latency), float(cost), bool(sla_met)


__all__ = ["EdgeCloudEnvCalibrated"]
