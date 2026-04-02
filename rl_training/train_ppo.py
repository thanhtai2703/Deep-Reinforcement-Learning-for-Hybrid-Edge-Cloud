"""
train_ppo.py
============
Training PPO (Proximal Policy Optimization) dùng Stable-Baselines3.

PPO thường stable hơn DQN cho continuous-like problems vì:
  - Clipped surrogate objective → tránh update quá lớn
  - On-policy → ít bị distribution shift
  - Actor-Critic → variance thấp hơn pure policy gradient

Chạy:
    python -m rl_training.train_ppo

So sánh sau đó:
    python -m rl_training.evaluate --model models/checkpoints/ppo_best/
"""

import os
import sys
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, CheckpointCallback
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[Warning] stable-baselines3 chưa cài. Chạy: pip install stable-baselines3")

from rl_env.edge_cloud_env import EdgeCloudEnv


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

PPO_CONFIG = {
    # Môi trường
    "n_edge_nodes":   2,
    "max_steps":      200,
    "n_envs":         4,           # Số envs song song (vectorized)

    # Training
    "total_timesteps": 300_000,
    "eval_freq":       10_000,     # Evaluate mỗi N timesteps
    "eval_episodes":   20,
    "save_dir":        "models/checkpoints/ppo",
    "log_dir":         "experiments/logs/ppo",
    "plot_dir":        "experiments/plots",
}

# PPO Hyperparameters – chuẩn cho discrete action spaces
PPO_HYPERPARAMS = dict(
    policy          = "MlpPolicy",
    learning_rate   = 3e-4,
    n_steps         = 512,         # Steps per env trước mỗi update
    batch_size      = 64,
    n_epochs        = 10,          # Số lần update mỗi batch
    gamma           = 0.99,
    gae_lambda      = 0.95,        # Generalized Advantage Estimation
    clip_range      = 0.2,         # PPO clipping epsilon
    ent_coef        = 0.01,        # Entropy bonus (khuyến khích explore)
    vf_coef         = 0.5,         # Value function loss coefficient
    max_grad_norm   = 0.5,
    verbose         = 0,
    tensorboard_log = "experiments/logs/ppo_tb",
    policy_kwargs   = dict(
        net_arch = [128, 128, 128],   # 3 hidden layers (match DQN)
    ),
)


# ──────────────────────────────────────────────────────────────────────────
# Custom Callback – ghi CSV log + theo dõi SLA
# ──────────────────────────────────────────────────────────────────────────

class EdgeCloudCallback(BaseCallback):
    """
    Callback tùy chỉnh:
      - Ghi CSV log mỗi eval_freq steps
      - Lưu best model dựa trên SLA rate
      - In progress bar đơn giản
    """

    def __init__(self, eval_env, log_path: str, eval_freq: int = 10_000,
                 eval_episodes: int = 20, save_dir: str = "models/checkpoints/ppo"):
        super().__init__()
        self.eval_env      = eval_env
        self.log_path      = log_path
        self.eval_freq     = eval_freq
        self.eval_episodes = eval_episodes
        self.save_dir      = save_dir
        self.best_sla      = -np.inf
        self._log_rows     = []
        self._start_time   = time.time()

        os.makedirs(save_dir, exist_ok=True)
        # Khởi tạo CSV header
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestep", "avg_reward", "avg_latency",
                "sla_rate", "avg_cost", "elapsed_sec"
            ])

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            metrics  = self._evaluate()
            elapsed  = time.time() - self._start_time

            print(
                f"  [{self.num_timesteps:>7,}/{PPO_CONFIG['total_timesteps']:,}] "
                f"reward={metrics['avg_reward']:6.2f} | "
                f"lat={metrics['avg_latency']:6.1f}ms | "
                f"SLA={metrics['sla_rate']:5.1f}% | "
                f"{elapsed:.0f}s"
            )

            # Lưu best model
            if metrics["sla_rate"] > self.best_sla:
                self.best_sla = metrics["sla_rate"]
                self.model.save(os.path.join(self.save_dir, "ppo_best"))

            # CSV log
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.num_timesteps,
                    round(metrics["avg_reward"],  4),
                    round(metrics["avg_latency"], 2),
                    round(metrics["sla_rate"],    2),
                    round(metrics["avg_cost"],    6),
                    round(elapsed, 1),
                ])

        return True

    def _evaluate(self) -> dict:
        rewards, latencies, costs, sla_flags = [], [], [], []
        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            ep_r   = 0.0
            done   = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = self.eval_env.step(int(action))
                done = terminated or truncated
                ep_r += r
                latencies.append(info["latency"])
                costs.append(info["cost"])
                sla_flags.append(int(info["sla_met"]))
            rewards.append(ep_r)

        return {
            "avg_reward":  np.mean(rewards),
            "avg_latency": np.mean(latencies),
            "avg_cost":    np.mean(costs),
            "sla_rate":    np.mean(sla_flags) * 100,
        }


# ──────────────────────────────────────────────────────────────────────────
# Hyperparameter Experiments (Reward Weight Tuning)
# ──────────────────────────────────────────────────────────────────────────

REWARD_EXPERIMENTS = [
    # (latency_w, cost_w, label)
    (0.6, 0.2, "default"),         # Default weights
    (0.8, 0.1, "latency_heavy"),   # Ưu tiên giảm latency
    (0.4, 0.4, "cost_heavy"),      # Ưu tiên giảm cost
    (0.7, 0.1, "sla_heavy"),       # Tập trung SLA (tăng SLA penalty ngầm)
    (0.5, 0.3, "balanced"),        # Cân bằng hơn
]


def run_reward_tuning(n_steps: int = 50_000) -> dict:
    """
    Thử 5 reward weight combinations, mỗi cái train nhanh 50K steps.
    Chọn best config để dùng cho full training.

    Returns: dict kết quả của từng experiment
    """
    if not SB3_AVAILABLE:
        print("[Error] Cần cài stable-baselines3")
        return {}

    results = {}
    os.makedirs(PPO_CONFIG["log_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Reward Weight Tuning – {len(REWARD_EXPERIMENTS)} experiments × {n_steps:,} steps")
    print(f"{'='*60}\n")

    for lat_w, cost_w, label in REWARD_EXPERIMENTS:
        print(f"  Experiment: {label} (lat_w={lat_w}, cost_w={cost_w})")

        # Patch reward weights vào env (monkey-patch đơn giản)
        def make_env(lw=lat_w, cw=cost_w):
            env = EdgeCloudEnv(n_edge_nodes=2, max_steps=200)
            # Override _compute_reward để test different weights
            original_reward = env._compute_reward

            def patched_reward(latency, cost, sla_met):
                from rl_env.edge_cloud_env import MAX_LATENCY, CLOUD_COST_PER_UNIT
                lat_norm  = latency / MAX_LATENCY
                cost_norm = cost / (CLOUD_COST_PER_UNIT * 110)
                r = -lw * lat_norm - cw * cost_norm
                r += 1.0 if sla_met else -2.0
                return float(r)

            env._compute_reward = patched_reward
            return Monitor(env)

        train_env = make_vec_env(make_env, n_envs=2)
        eval_env  = make_env()

        model = PPO(
            env            = train_env,
            device         = "cpu",
            seed           = 42,
            **{k: v for k, v in PPO_HYPERPARAMS.items()
               if k not in ["verbose", "tensorboard_log"]},
            verbose        = 0,
        )

        # Collect rewards mỗi 10K steps
        rewards_curve = []

        class QuickCallback(BaseCallback):
            def _on_step(self):
                if self.num_timesteps % 10_000 == 0:
                    r_list = []
                    for _ in range(10):
                        obs, _ = eval_env.reset()
                        ep_r, done = 0.0, False
                        while not done:
                            a, _ = self.model.predict(obs, deterministic=True)
                            obs, r, t, tr, _ = eval_env.step(int(a))
                            ep_r += r
                            done = t or tr
                        r_list.append(ep_r)
                    rewards_curve.append(np.mean(r_list))
                return True

        model.learn(total_timesteps=n_steps, callback=QuickCallback())

        results[label] = {
            "lat_w":        lat_w,
            "cost_w":       cost_w,
            "final_reward": rewards_curve[-1] if rewards_curve else 0,
            "rewards_curve": rewards_curve,
        }
        print(f"    Final reward: {results[label]['final_reward']:.3f}\n")
        train_env.close()

    # Best config
    best_label = max(results, key=lambda k: results[k]["final_reward"])
    print(f"\n  ✅ Best config: '{best_label}' "
          f"(reward={results[best_label]['final_reward']:.3f})")
    print(f"     lat_w={results[best_label]['lat_w']} | "
          f"cost_w={results[best_label]['cost_w']}\n")

    # Save results CSV
    csv_path = "experiments/reward_tuning_results.csv"
    os.makedirs("experiments", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "lat_w", "cost_w", "final_reward"])
        for label, data in results.items():
            writer.writerow([label, data["lat_w"], data["cost_w"], round(data["final_reward"], 4)])
    print(f"  Results saved → {csv_path}")

    return results


# ──────────────────────────────────────────────────────────────────────────
# Main Training
# ──────────────────────────────────────────────────────────────────────────

def train_ppo():
    if not SB3_AVAILABLE:
        print("[Error] Cần cài: pip install stable-baselines3")
        return

    cfg = PPO_CONFIG
    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["log_dir"],  exist_ok=True)
    os.makedirs(cfg["plot_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PPO Training – EdgeCloud Task Scheduling")
    print(f"  Total timesteps : {cfg['total_timesteps']:,}")
    print(f"  Parallel envs   : {cfg['n_envs']}")
    print(f"  Edge nodes      : {cfg['n_edge_nodes']}")
    print(f"{'='*60}\n")

    # Vectorized training envs
    def make_train_env():
        env = EdgeCloudEnv(
            n_edge_nodes  = cfg["n_edge_nodes"],
            max_steps     = cfg["max_steps"],
        )
        return Monitor(env, cfg["log_dir"])

    train_env = make_vec_env(make_train_env, n_envs=cfg["n_envs"], seed=42)

    # Single eval env
    eval_env = EdgeCloudEnv(
        n_edge_nodes = cfg["n_edge_nodes"],
        max_steps    = cfg["max_steps"],
    )

    # Khởi tạo PPO model
    model = PPO(
        env  = train_env,
        seed = 42,
        **PPO_HYPERPARAMS,
    )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}")
    print(f"  Device: {model.device}\n")

    # Custom callback
    log_path = os.path.join(cfg["log_dir"], "ppo_train_log.csv")
    callback = EdgeCloudCallback(
        eval_env      = eval_env,
        log_path      = log_path,
        eval_freq     = cfg["eval_freq"],
        eval_episodes = cfg["eval_episodes"],
        save_dir      = cfg["save_dir"],
    )

    # Checkpoint mỗi 50K timesteps
    checkpoint_cb = CheckpointCallback(
        save_freq   = 50_000,
        save_path   = cfg["save_dir"],
        name_prefix = "ppo_ckpt",
    )

    # Train!
    start = time.time()
    model.learn(
        total_timesteps  = cfg["total_timesteps"],
        callback         = [callback, checkpoint_cb],
        progress_bar     = False,
    )
    total_time = time.time() - start

    # Save final
    final_path = os.path.join(cfg["save_dir"], "ppo_final")
    model.save(final_path)
    print(f"\n✅ PPO training done! Time: {total_time:.0f}s")
    print(f"   Best model → {cfg['save_dir']}/ppo_best.zip")
    print(f"   Final model → {final_path}.zip")

    # Plot
    _plot_ppo_curves(log_path, cfg["plot_dir"])

    return model


def _plot_ppo_curves(log_path: str, plot_dir: str):
    """Vẽ training curves từ CSV log."""
    timesteps, rewards, latencies, sla_rates = [], [], [], []
    try:
        with open(log_path) as f:
            for row in csv.DictReader(f):
                timesteps.append(int(row["timestep"]))
                rewards.append(float(row["avg_reward"]))
                latencies.append(float(row["avg_latency"]))
                sla_rates.append(float(row["sla_rate"]))
    except Exception:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PPO Training Curves – EdgeCloud", fontsize=13, fontweight="bold")

    axes[0].plot(timesteps, rewards, color="#9C27B0", linewidth=1.5)
    axes[0].set_title("Avg Reward"); axes[0].set_xlabel("Timesteps")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps, latencies, color="#F44336", linewidth=1.5)
    axes[1].set_title("Avg Latency (ms)"); axes[1].set_xlabel("Timesteps")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(timesteps, sla_rates, color="#4CAF50", linewidth=1.5)
    axes[2].axhline(y=95, color="orange", linestyle="--", label="Target 95%")
    axes[2].set_title("SLA Rate (%)"); axes[2].set_xlabel("Timesteps")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "ppo_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO for EdgeCloud scheduling")
    parser.add_argument("--tune-rewards", action="store_true",
                        help="Chạy reward weight tuning trước (5 experiments × 50K steps)")
    parser.add_argument("--tune-steps", type=int, default=50_000,
                        help="Số steps cho mỗi tuning experiment")
    args = parser.parse_args()

    if args.tune_rewards:
        run_reward_tuning(n_steps=args.tune_steps)

    train_ppo()