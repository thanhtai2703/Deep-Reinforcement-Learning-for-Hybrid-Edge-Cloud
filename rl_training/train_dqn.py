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
import torch
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (server)
import matplotlib.pyplot as plt

# Thêm root project vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_env.edge_cloud_env import EdgeCloudEnv
from models.dqn_agent import DQNAgent, DQNConfig


def make_env(env_kind: str, **kwargs):
    """Factory: env_kind in {'uncalibrated', 'calibrated'}."""
    if env_kind == "calibrated":
        from rl_env.edge_cloud_env_calibrated import EdgeCloudEnvCalibrated
        return EdgeCloudEnvCalibrated(**kwargs)
    return EdgeCloudEnv(**kwargs)


# ---------------------------------------------------------------------------
# Cấu hình Training
# ---------------------------------------------------------------------------
TRAIN_CONFIG = {
    "n_episodes":    2000,     # Tổng số episode training
    "n_edge_nodes":  2,
    "max_steps":     200,
    "eval_interval": 50,       # Đánh giá mỗi N episode
    "eval_episodes": 100,       # Số episode dùng để đánh giá
    "save_dir":      "models/checkpoints",
    "log_dir":       "experiments/logs",
    "plot_dir":      "experiments/plots",
    "seed":          42,
}

DQN_HYPERPARAMS = DQNConfig(
    hidden_dim      = 128,
    n_layers        = 3,
    lr              = 1e-3,
    gamma           = 0.99,
    batch_size      = 64,
    buffer_size     = 50_000,
    min_buffer_size = 2_000,
    tau             = 0.005,
    eps_start       = 1.0,
    eps_end         = 0.05,
    eps_decay       = 0.995,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def collect_reference_states(env: EdgeCloudEnv, n: int = 256, seed: int = 123) -> np.ndarray:
    """
    Sample một batch state cố định để theo dõi mean Q-value xuyên suốt training.
    Q ổn định trên batch này → bằng chứng convergence (Mnih et al. 2015 Fig. 2).
    """
    states = []
    obs, _ = env.reset(seed=seed)
    while len(states) < n:
        states.append(obs.copy())
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(states[:n], dtype=np.float32)


def compute_mean_q(agent: DQNAgent, ref_states: np.ndarray) -> float:
    """Mean of max-Q over reference states. Tăng đều rồi phẳng = đã hội tụ."""
    states_t = torch.FloatTensor(ref_states).to(agent.device)
    with torch.no_grad():
        q = agent.q_net(states_t)
    return float(q.max(dim=1).values.mean().item())


def evaluate_agent(agent: DQNAgent, env: EdgeCloudEnv, n_episodes: int = 20) -> dict:
    """
    Đánh giá agent trong chế độ greedy (không explore).
    Returns metrics: avg_reward, avg_latency, sla_rate, avg_cost,
                     edge_pct, cloud_pct, reject_pct.
    """
    total_reward  = 0.0
    total_latency = 0.0
    total_cost    = 0.0
    sla_count     = 0
    total_steps   = 0
    n_edge_actions = env.n_edge_nodes
    cloud_action   = n_edge_actions
    reject_action  = n_edge_actions + 1
    edge_count = cloud_count = reject_count = 0

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

            if action < n_edge_actions:
                edge_count += 1
            elif action == cloud_action:
                cloud_count += 1
            elif action == reject_action:
                reject_count += 1

    return {
        "avg_reward":  total_reward  / n_episodes,
        "avg_latency": total_latency / total_steps,
        "avg_cost":    total_cost    / total_steps,
        "sla_rate":    sla_count     / total_steps * 100,
        "edge_pct":    edge_count    / total_steps * 100,
        "cloud_pct":   cloud_count   / total_steps * 100,
        "reject_pct":  reject_count  / total_steps * 100,
    }


def setup_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _moving_average(x: list, window: int = 5) -> np.ndarray:
    """Simple moving average; trả về cùng độ dài (padding bằng NaN ở đầu)."""
    if len(x) < window:
        return np.array(x, dtype=float)
    arr = np.array(x, dtype=float)
    out = np.full_like(arr, np.nan)
    out[window - 1:] = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def plot_training_curves(log_path: str, plot_dir: str):
    """Vẽ reward (eval + train smoothed), latency, SLA rate, epsilon theo episode."""
    episodes, train_rewards, eval_rewards = [], [], []
    latencies, sla_rates, epsilons = [], [], []

    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            train_rewards.append(float(row["train_reward"]))
            eval_rewards.append(float(row["eval_avg_reward"]))
            latencies.append(float(row["eval_avg_latency"]))
            sla_rates.append(float(row["eval_sla_rate"]))
            epsilons.append(float(row["epsilon"]))

    train_smooth = _moving_average(train_rewards, window=5)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Training Curves – EdgeCloud Task Scheduling", fontsize=14, fontweight="bold")

    # Reward: train (raw + smooth) vs eval — overlay để check overfit
    axes[0, 0].plot(episodes, train_rewards, color="#90CAF9", linewidth=1.0,
                    alpha=0.5, label="Train (raw)")
    axes[0, 0].plot(episodes, train_smooth, color="#1976D2", linewidth=1.8,
                    label="Train (MA-5)")
    axes[0, 0].plot(episodes, eval_rewards, color="#D32F2F", linewidth=1.8,
                    label="Eval (greedy)")
    axes[0, 0].set_title("Reward — Train vs Eval (overfit check)")
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Avg Reward")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, latencies, color="#F44336", linewidth=1.5)
    axes[0, 1].set_title("Average Latency (ms)")
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Latency (ms)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, sla_rates, color="#4CAF50", linewidth=1.5)
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


def plot_loss_q_curves(log_path: str, plot_dir: str):
    """
    Bằng chứng hội tụ: TD-loss giảm + Mean-Q phẳng.
    - Loss (log scale): khi loss vẫn còn dao động lớn → chưa hội tụ.
    - Mean Q: tăng dần rồi bão hòa = Q-function đã ổn định trên reference states.
    """
    episodes, losses, mean_qs = [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            losses.append(max(float(row["loss"]), 1e-8))   # tránh log(0)
            mean_qs.append(float(row.get("mean_q", "nan")))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("DQN Convergence Diagnostics", fontsize=13, fontweight="bold")

    axes[0].plot(episodes, losses, color="#FF7043", linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_title("TD-Loss (Huber, log scale)")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].plot(episodes, mean_qs, color="#3949AB", linewidth=1.8)
    axes[1].set_title("Mean max-Q on reference states")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("E[max_a Q(s,a)]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "convergence_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_action_distribution(log_path: str, plot_dir: str):
    """
    Stacked area chart: % edge / cloud / reject theo episode.
    Đảm bảo agent không bị collapse (vd 100% reject hoặc 100% cloud).
    """
    episodes, edge_pct, cloud_pct, reject_pct = [], [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            edge_pct.append(float(row.get("edge_pct", 0)))
            cloud_pct.append(float(row.get("cloud_pct", 0)))
            reject_pct.append(float(row.get("reject_pct", 0)))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(episodes, edge_pct, cloud_pct, reject_pct,
                 labels=["Edge", "Cloud", "Reject"],
                 colors=["#4CAF50", "#2196F3", "#F44336"],
                 alpha=0.8)
    ax.set_title("Action Distribution Over Training (eval, greedy)",
                 fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("% of dispatched actions")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "action_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train():
    cfg = TRAIN_CONFIG
    setup_dirs(cfg["save_dir"], cfg["log_dir"], cfg["plot_dir"])

    env_kind = cfg.get("env_kind", "uncalibrated")
    print(f"  Env: {env_kind}")

    env = make_env(
        env_kind,
        n_edge_nodes  = cfg["n_edge_nodes"],
        use_prometheus= False,     # Simulation mode
        max_steps     = cfg["max_steps"],
    )
    eval_env = make_env(
        env_kind,
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

    # Reference states để track mean Q-value xuyên suốt training
    ref_states = collect_reference_states(eval_env, n=256, seed=cfg["seed"] + 1)
    print(f"  Reference states for Q-tracking: {ref_states.shape}")

    # CSV logger
    log_path = os.path.join(cfg["log_dir"], "train_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "train_reward", "steps", "loss",
            "epsilon", "eval_avg_reward", "eval_avg_latency",
            "eval_sla_rate", "eval_avg_cost", "mean_q",
            "edge_pct", "cloud_pct", "reject_pct", "elapsed_sec"
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
            mean_q   = compute_mean_q(agent, ref_states)
            elapsed  = time.time() - start_time

            print(
                f"Ep {episode:4d}/{cfg['n_episodes']} | "
                f"train_r={ep_reward:6.2f} | "
                f"eval_r={metrics['avg_reward']:6.2f} | "
                f"lat={metrics['avg_latency']:6.1f}ms | "
                f"SLA={metrics['sla_rate']:5.1f}% | "
                f"E/C/R={metrics['edge_pct']:4.1f}/{metrics['cloud_pct']:4.1f}/{metrics['reject_pct']:4.1f} | "
                f"meanQ={mean_q:6.3f} | "
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
                    round(mean_q, 4),
                    round(metrics["edge_pct"], 2),
                    round(metrics["cloud_pct"], 2),
                    round(metrics["reject_pct"], 2),
                    round(elapsed, 1),
                ])

            # Lưu best model
            if metrics["avg_reward"] > best_reward:
                best_reward = metrics["avg_reward"]
                agent.save(os.path.join(cfg["save_dir"], "dqn_best.pth"))

        # Lưu checkpoint định kỳ
        if episode % 500 == 0:
            agent.save(os.path.join(cfg["save_dir"], f"dqn_ep{episode}.pth"))

    # Lưu model cuối
    agent.save(os.path.join(cfg["save_dir"], "dqn_final.pth"))

    total_time = time.time() - start_time
    print(f"\n Training complete! Total time: {total_time:.0f}s")
    print(f"   Best eval reward: {best_reward:.4f}")
    print(f"   Model saved: {cfg['save_dir']}/dqn_best.pth")

    # Vẽ training curves
    plot_training_curves(log_path, cfg["plot_dir"])
    plot_loss_q_curves(log_path, cfg["plot_dir"])
    plot_action_distribution(log_path, cfg["plot_dir"])

    return agent


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=("uncalibrated", "calibrated"),
                   default="uncalibrated")
    p.add_argument("--save-dir", default=None,
                   help="Override save dir (default: models/checkpoints/<env>)")
    p.add_argument("--log-dir", default=None,
                   help="Override log dir")
    p.add_argument("--episodes", type=int, default=None,
                   help="Override n_episodes")
    args = p.parse_args()

    TRAIN_CONFIG["env_kind"] = args.env
    if args.save_dir:
        TRAIN_CONFIG["save_dir"] = args.save_dir
    else:
        TRAIN_CONFIG["save_dir"] = f"models/checkpoints/dqn_{args.env}"
    if args.log_dir:
        TRAIN_CONFIG["log_dir"] = args.log_dir
    else:
        TRAIN_CONFIG["log_dir"] = f"experiments/logs/dqn_{args.env}"
    if args.episodes:
        TRAIN_CONFIG["n_episodes"] = args.episodes

    train()