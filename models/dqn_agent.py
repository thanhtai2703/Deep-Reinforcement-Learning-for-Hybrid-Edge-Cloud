"""
dqn_agent.py
============
Triển khai DQN (Deep Q-Network) cho bài toán điều phối tác vụ.

Kiến trúc:
  - QNetwork      : MLP 3 lớp (obs → hidden → hidden → n_actions)
  - ReplayBuffer  : Experience Replay để phá vỡ correlation
  - DQNAgent      : Tích hợp epsilon-greedy, target network, training loop

Tham khảo: Mnih et al. (2015) - "Human-level control through deep RL"
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Cấu hình Hyperparameters
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    # Mạng neural
    hidden_dim: int   = 128      # Số neuron mỗi hidden layer
    n_layers:   int   = 3        # Số hidden layers

    # Training
    lr:              float = 1e-3    # Learning rate
    gamma:           float = 0.99   # Discount factor
    batch_size:      int   = 64
    buffer_size:     int   = 10_000  # Dung lượng Replay Buffer
    min_buffer_size: int   = 500    # Bắt đầu train khi buffer đủ lớn
    target_update_freq: int = 100   # Cập nhật target network mỗi N bước

    # Epsilon-greedy exploration
    eps_start: float = 1.0
    eps_end:   float = 0.05
    eps_decay: float = 0.995       # Giảm epsilon sau mỗi episode

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Q-Network: MLP đơn giản
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """
    Multi-Layer Perceptron ánh xạ (state) → Q-values cho mọi action.
    Dùng ReLU activation và BatchNorm để ổn định training.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()

        layers = []
        in_dim = obs_dim

        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, n_actions))
        self.net = nn.Sequential(*layers)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    """
    Experience Replay Buffer lưu trữ (s, a, r, s', done).
    Sample ngẫu nhiên để phá vỡ tương quan giữa các bước liên tiếp.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states      = np.array([t.state      for t in batch], dtype=np.float32)
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones       = np.array([t.done       for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) >= 500


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    DQN Agent với:
      - Epsilon-greedy exploration (annealing)
      - Experience Replay
      - Target Network (hard update định kỳ)
      - Huber Loss (ổn định hơn MSE)

    Parameters
    ----------
    obs_dim   : kích thước vector quan sát
    n_actions : số hành động
    config    : DQNConfig hyperparameters
    """

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig = None):
        self.config    = config or DQNConfig()
        self.n_actions = n_actions
        self.device    = torch.device(self.config.device)

        # Q-network chính và Target network
        self.q_net      = QNetwork(obs_dim, n_actions,
                                   self.config.hidden_dim,
                                   self.config.n_layers).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions,
                                   self.config.hidden_dim,
                                   self.config.n_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Target net không cần gradient

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber Loss

        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        # Epsilon cho exploration
        self.epsilon   = self.config.eps_start
        self._step_cnt = 0

        # Logging
        self.loss_history    = []
        self.reward_history  = []

    # -----------------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """
        Chọn action theo epsilon-greedy.
        greedy=True → luôn chọn action tốt nhất (dùng lúc evaluate).
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        return int(q_values.argmax(dim=1).item())

    # -----------------------------------------------------------------------
    # Learning
    # -----------------------------------------------------------------------

    def store_transition(self, state, action, reward, next_state, done):
        """Lưu experience vào Replay Buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """
        Thực hiện 1 bước gradient descent.
        Returns: loss value (hoặc None nếu buffer chưa đủ).
        """
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.config.batch_size)

        # Chuyển sang tensor
        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) – current Q-values
        q_values = self.q_net(states_t)
        q_sa     = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Q target: r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q      = self.target_net(next_states_t).max(dim=1)[0]
            q_target    = rewards_t + self.config.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_sa, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping để tránh exploding gradients
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step_cnt += 1

        # Hard update target network
        if self._step_cnt % self.config.target_update_freq == 0:
            self._update_target()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def decay_epsilon(self):
        """Giảm epsilon sau mỗi episode (gọi ở cuối episode)."""
        self.epsilon = max(
            self.config.eps_end,
            self.epsilon * self.config.eps_decay
        )

    def _update_target(self):
        """Hard copy weights từ Q-network sang Target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    # -----------------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------------

    def save(self, path: str):
        """Lưu model weights và config."""
        torch.save({
            "q_net_state":    self.q_net.state_dict(),
            "target_state":   self.target_net.state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "epsilon":        self.epsilon,
            "step_cnt":       self._step_cnt,
            "loss_history":   self.loss_history,
            "reward_history": self.reward_history,
        }, path)
        print(f"[DQNAgent] Model saved → {path}")

    def load(self, path: str):
        """Tải model weights đã lưu."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net_state"])
        self.target_net.load_state_dict(checkpoint["target_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon        = checkpoint.get("epsilon", self.config.eps_end)
        self._step_cnt      = checkpoint.get("step_cnt", 0)
        self.loss_history   = checkpoint.get("loss_history", [])
        self.reward_history = checkpoint.get("reward_history", [])
        print(f"[DQNAgent] Model loaded ← {path}")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Trả về Q-values cho tất cả action (dùng để debug/visualize)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(obs_t).cpu().numpy().flatten()

    def __repr__(self):
        total_params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"DQNAgent(actions={self.n_actions}, "
            f"params={total_params:,}, "
            f"eps={self.epsilon:.3f}, "
            f"device={self.config.device})"
        )