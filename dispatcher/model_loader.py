"""
model_loader.py
===============
Load và quản lý RL models (DQN, PPO) và baseline policies.

Hỗ trợ:
  - DQN: custom DQNAgent (models/dqn_agent.py)
  - PPO: Stable-Baselines3 PPO.load()
  - Baselines: Round Robin, Least Connection, Threshold, ...
  - Hot-reload: reload model mới mà không restart dispatcher

Usage:
    loader = ModelLoader(obs_dim=12, n_actions=3, n_edge_nodes=2)
    loader.load("dqn", "models/checkpoints/dqn_best.pth")
    action, q_values = loader.predict(state_vector)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

# Thêm project root vào path
_PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.dqn_agent import DQNAgent, DQNConfig
from rl_env.baseline_policies import (
    BasePolicy,
    CloudOnlyPolicy,
    EdgeOnlyPolicy,
    LeastConnectionPolicy,
    RandomPolicy,
    RoundRobinPolicy,
    ThresholdPolicy,
    get_all_baselines,
)

logger = logging.getLogger("ModelLoader")

# Tên policy hợp lệ
BASELINE_NAMES = {
    "random", "round_robin", "least_connection",
    "edge_only", "cloud_only", "threshold",
}
RL_MODEL_NAMES = {"dqn", "ppo"}
ALL_POLICY_NAMES = BASELINE_NAMES | RL_MODEL_NAMES


class ModelLoader:
    """
    Load và quản lý policy/model cho dispatcher.

    Parameters
    ----------
    obs_dim      : kích thước observation (12 cho 2 edge nodes)
    n_actions    : số actions (3 = 2 edge + 1 cloud)
    n_edge_nodes : số edge nodes
    """

    def __init__(self, obs_dim: int, n_actions: int, n_edge_nodes: int = 2):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_edge_nodes = n_edge_nodes

        self._current_policy_name: str = ""
        self._dqn_agent: Optional[DQNAgent] = None
        self._ppo_model = None  # stable_baselines3.PPO
        self._baseline: Optional[BasePolicy] = None
        self._model_path: Optional[str] = None
        self._loaded_mtime: float = 0.0

    @property
    def policy_name(self) -> str:
        return self._current_policy_name

    def load(self, policy_name: str, model_path: Optional[str] = None) -> None:
        """
        Load một policy theo tên.

        Parameters
        ----------
        policy_name : "dqn", "ppo", "round_robin", "least_connection", ...
        model_path  : đường dẫn checkpoint (bắt buộc cho dqn/ppo)
        """
        policy_name = policy_name.lower().strip()
        self._current_policy_name = policy_name

        if policy_name == "dqn":
            self._load_dqn(model_path)
        elif policy_name == "ppo":
            self._load_ppo(model_path)
        elif policy_name in BASELINE_NAMES:
            self._load_baseline(policy_name)
        else:
            raise ValueError(
                f"Unknown policy: {policy_name}. "
                f"Valid: {sorted(ALL_POLICY_NAMES)}"
            )

    def predict(self, state: np.ndarray) -> Tuple[int, Optional[list]]:
        """
        Predict action từ state vector.

        Returns: (action, q_values_or_none)
        """
        name = self._current_policy_name

        if name == "dqn":
            return self._predict_dqn(state)
        elif name == "ppo":
            return self._predict_ppo(state)
        else:
            return self._predict_baseline(state)

    def hot_reload(self) -> bool:
        """
        Kiểm tra model file có thay đổi không, nếu có thì reload.
        Returns True nếu đã reload.
        """
        if self._model_path is None or not os.path.exists(self._model_path):
            return False

        current_mtime = os.path.getmtime(self._model_path)
        if current_mtime > self._loaded_mtime:
            logger.info("Model file changed, hot-reloading: %s", self._model_path)
            self.load(self._current_policy_name, self._model_path)
            return True
        return False

    def get_info(self) -> dict:
        """Thông tin về model hiện tại."""
        info = {
            "policy_name": self._current_policy_name,
            "model_path": self._model_path,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
        }
        if self._dqn_agent:
            info["epsilon"] = self._dqn_agent.epsilon
            info["type"] = "DQN (custom)"
        elif self._ppo_model:
            info["type"] = "PPO (stable-baselines3)"
        elif self._baseline:
            info["type"] = f"Baseline ({self._baseline.name})"
        return info

    # ------------------------------------------------------------------
    # Private: DQN
    # ------------------------------------------------------------------

    def _load_dqn(self, model_path: Optional[str]):
        """Load custom DQN agent."""
        self._dqn_agent = DQNAgent(self.obs_dim, self.n_actions)
        self._ppo_model = None
        self._baseline = None

        if model_path and os.path.exists(model_path):
            self._dqn_agent.load(model_path)
            self._model_path = model_path
            self._loaded_mtime = os.path.getmtime(model_path)
            logger.info("DQN model loaded: %s", model_path)
        else:
            logger.warning(
                "DQN model not found: %s - using untrained agent",
                model_path,
            )
            self._model_path = model_path

    def _predict_dqn(self, state: np.ndarray) -> Tuple[int, Optional[list]]:
        action = self._dqn_agent.select_action(state, greedy=True)
        q_values = self._dqn_agent.get_q_values(state).tolist()
        return action, q_values

    # ------------------------------------------------------------------
    # Private: PPO
    # ------------------------------------------------------------------

    def _load_ppo(self, model_path: Optional[str]):
        """Load PPO model via Stable-Baselines3."""
        self._dqn_agent = None
        self._baseline = None

        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError(
                "stable-baselines3 required for PPO. "
                "Install: pip install stable-baselines3"
            )

        if model_path and os.path.exists(model_path):
            self._ppo_model = PPO.load(model_path)
            self._model_path = model_path
            self._loaded_mtime = os.path.getmtime(model_path)
            logger.info("PPO model loaded: %s", model_path)
        else:
            logger.warning(
                "PPO model not found: %s - cannot predict without model",
                model_path,
            )
            self._ppo_model = None
            self._model_path = model_path

    def _predict_ppo(self, state: np.ndarray) -> Tuple[int, Optional[list]]:
        if self._ppo_model is None:
            logger.error("PPO model not loaded, falling back to random")
            return int(np.random.randint(0, self.n_actions)), None

        action, _states = self._ppo_model.predict(state, deterministic=True)
        return int(action), None

    # ------------------------------------------------------------------
    # Private: Baselines
    # ------------------------------------------------------------------

    def _load_baseline(self, policy_name: str):
        """Load một baseline policy."""
        self._dqn_agent = None
        self._ppo_model = None
        self._model_path = None

        policy_map = {
            "random": lambda: RandomPolicy(self.n_actions, self.n_edge_nodes),
            "round_robin": lambda: RoundRobinPolicy(self.n_actions, self.n_edge_nodes),
            "least_connection": lambda: LeastConnectionPolicy(self.n_actions, self.n_edge_nodes),
            "edge_only": lambda: EdgeOnlyPolicy(self.n_actions, self.n_edge_nodes),
            "cloud_only": lambda: CloudOnlyPolicy(self.n_actions, self.n_edge_nodes),
            "threshold": lambda: ThresholdPolicy(self.n_actions, self.n_edge_nodes),
        }

        factory = policy_map.get(policy_name)
        if factory is None:
            raise ValueError(f"Unknown baseline: {policy_name}")

        self._baseline = factory()
        logger.info("Baseline policy loaded: %s", self._baseline.name)

    def _predict_baseline(self, state: np.ndarray) -> Tuple[int, Optional[list]]:
        if self._baseline is None:
            raise RuntimeError("No baseline policy loaded")
        action = self._baseline.select_action(state)
        return action, None
