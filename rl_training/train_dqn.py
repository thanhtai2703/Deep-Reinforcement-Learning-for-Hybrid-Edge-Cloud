"""
train_dqn.py
============
Training loop chính cho DQN Agent trên môi trường EdgeCloudEnv.

Chạy:
    python -m rl_training.train_dqn

Kết quả:
    - Model checkpoint: models/checkpoints/dqn_best.pth
    - Training log:     experiments/logs/train_log.csv
    - Plots:            experiments/plots/training_curves.png
"""

import os
import sys
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (server)
import matplotlib.pyplot as plt

# Thêm root project vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_env.edge_cloud_env import EdgeCloudEnv
from models.dqn_agent import DQNAgent, DQNConfig


# ---------------------------------------------------------------------------
# Cấu hình Training
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    "n_episodes":    1000,     # Tổng số episode training
    "n_edge_nodes":  3,
    "max_steps":     200,
    "eval_interval": 50,       # Đánh giá mỗi N episode
    "eval_episodes": 20,       # Số episode dùng để đánh giá
    "save_dir":      "models/checkpoints",
    "log_dir":       "experiments/logs",
    "plot_dir":      "experiments/plots",
    "seed":          42,
}

DQN_HYPERPARAMS = DQNConfig(
    hidden_dim        = 128,
    n_layers          = 3,
    lr                = 1e-3,
    gamma             = 0.99,
    batch_size        = 64,
    buffer_size       = 10_000,
    min_buffer_size   = 500,
    target_update_freq= 100,
    eps_start         = 1.0,
    eps_end           = 0.05,
    eps_decay         = 0.995,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def evaluate_agent(agent: DQNAgent, env: EdgeCloudEnv, n_episodes: int = 20) -> dict:
    """
    Đánh giá agent trong chế độ greedy (không explore).
    Returns metrics: avg_reward, avg_latency, sla_rate, avg_cost
    """
    total_reward  = 0.0
    total_latency = 0.0
    total_cost    = 0.0
    sla_count     = 0
    total_steps   = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward  += reward
            total_latency += info["latency"]
            total_cost    += info["cost"]
            sla_count     += int(info["sla_met"])
            total_steps   += 1

    return {
        "avg_reward":  total_reward  / n_episodes,
        "avg_latency": total_latency / total_steps,
        "avg_cost":    total_cost    / total_steps,
        "sla_rate":    sla_count     / total_steps * 100,
    }


def setup_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def plot_training_curves(log_path: str, plot_dir: str):
    """Vẽ reward, latency, SLA rate theo episode."""
    episodes, rewards, latencies, sla_rates, epsilons = [], [], [], [], []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["eval_avg_reward"]))
            latencies.append(float(row["eval_avg_latency"]))
            sla_rates.append(float(row["eval_sla_rate"]))
            epsilons.append(float(row["epsilon"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Training Curves – EdgeCloud Task Scheduling", fontsize=14, fontweight="bold")

    axes[0, 0].plot(episodes, rewards, color="#2196F3", linewidth=1.5)
    axes[0, 0].set_title("Average Reward per Evaluation")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Avg Reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, latencies, color="#F44336", linewidth=1.5)
    axes[0, 1].set_title("Average Latency (ms)")
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Latency (ms)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, sla_rates, color="#4CAF50", linewidth=1.5)
    axes[1, 0].axhline(y=95, color="orange", linestyle="--", label="Target 95%")
    axes[1, 0].set_title("SLA Rate (%)")
    axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("SLA (%)")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, epsilons, color="#9C27B0", linewidth=1.5)
    axes[1, 1].set_title("Epsilon (Exploration Rate)")
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("Epsilon")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train():
    cfg = TRAIN_CONFIG
    setup_dirs(cfg["save_dir"], cfg["log_dir"], cfg["plot_dir"])

    # Khởi tạo môi trường
    env = EdgeCloudEnv(
        n_edge_nodes  = cfg["n_edge_nodes"],
        use_prometheus= False,     # Simulation mode
        max_steps     = cfg["max_steps"],
    )
    eval_env = EdgeCloudEnv(
        n_edge_nodes  = cfg["n_edge_nodes"],
        use_prometheus= False,
        max_steps     = cfg["max_steps"],
    )

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print(f"\n{'='*60}")
    print(f"  DQN Training – EdgeCloud Task Scheduling")
    print(f"  Obs dim: {obs_dim} | Actions: {n_actions}")
    print(f"  Edge nodes: {cfg['n_edge_nodes']} | Episodes: {cfg['n_episodes']}")
    print(f"{'='*60}\n")

    # Khởi tạo agent
    agent = DQNAgent(obs_dim, n_actions, DQN_HYPERPARAMS)
    print(agent)

    # CSV logger
    log_path = os.path.join(cfg["log_dir"], "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "train_reward", "steps", "loss",
            "epsilon", "eval_avg_reward", "eval_avg_latency",
            "eval_sla_rate", "eval_avg_cost", "elapsed_sec"
        ])

    best_reward = -np.inf
    start_time  = time.time()

    for episode in range(1, cfg["n_episodes"] + 1):
        obs, _     = env.reset()
        ep_reward  = 0.0
        ep_loss    = []
        steps      = 0

        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                ep_loss.append(loss)

            obs        = next_obs
            ep_reward += reward
            steps     += 1

        agent.decay_epsilon()
        agent.reward_history.append(ep_reward)

        avg_loss = np.mean(ep_loss) if ep_loss else 0.0

        # ── Evaluation checkpoint ──────────────────────────────────────────
        if episode % cfg["eval_interval"] == 0:
            metrics  = evaluate_agent(agent, eval_env, cfg["eval_episodes"])
            elapsed  = time.time() - start_time

            print(
                f"Ep {episode:4d}/{cfg['n_episodes']} | "
                f"train_r={ep_reward:6.2f} | "
                f"eval_r={metrics['avg_reward']:6.2f} | "
                f"lat={metrics['avg_latency']:6.1f}ms | "
                f"SLA={metrics['sla_rate']:5.1f}% | "
                f"eps={agent.epsilon:.3f} | "
                f"loss={avg_loss:.4f} | "
                f"{elapsed:.0f}s"
            )

            # Log CSV
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode, round(ep_reward, 4), steps, round(avg_loss, 6),
                    round(agent.epsilon, 4),
                    round(metrics["avg_reward"], 4),
                    round(metrics["avg_latency"], 2),
                    round(metrics["sla_rate"], 2),
                    round(metrics["avg_cost"], 6),
                    round(elapsed, 1),
                ])

            # Lưu best model
            if metrics["avg_reward"] > best_reward:
                best_reward = metrics["avg_reward"]
                agent.save(os.path.join(cfg["save_dir"], "dqn_best.pth"))

        # Lưu checkpoint định kỳ
        if episode % 200 == 0:
            agent.save(os.path.join(cfg["save_dir"], f"dqn_ep{episode}.pth"))

    # Lưu model cuối
    agent.save(os.path.join(cfg["save_dir"], "dqn_final.pth"))

    total_time = time.time() - start_time
    print(f"\n Training complete! Total time: {total_time:.0f}s")
    print(f"   Best eval reward: {best_reward:.4f}")
    print(f"   Model saved: {cfg['save_dir']}/dqn_best.pth")

    # Vẽ training curves
    plot_training_curves(log_path, cfg["plot_dir"])

    return agent


if __name__ == "__main__":
    train()