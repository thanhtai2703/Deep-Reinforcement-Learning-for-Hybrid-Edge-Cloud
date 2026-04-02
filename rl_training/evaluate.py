"""
evaluate.py
===========
So sánh toàn diện DQN Agent vs tất cả Baseline Policies.

Chạy:
    python -m rl_training.evaluate --model models/checkpoints/dqn_best.pth

Output:
    - Bảng kết quả in ra terminal
    - experiments/plots/comparison_chart.png
    - experiments/logs/evaluation_results.csv
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_env.edge_cloud_env import EdgeCloudEnv
from rl_env.baseline_policies import get_all_baselines, BasePolicy
from models.dqn_agent import DQNAgent, DQNConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_CONFIG = {
    "n_episodes":   100,
    "n_edge_nodes": 2,
    "max_steps":    200,
    "log_dir":      "experiments/logs",
    "plot_dir":     "experiments/plots",
}


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def run_evaluation(policy, env: EdgeCloudEnv, n_episodes: int, is_dqn: bool = False) -> dict:
    """
    Chạy n_episodes episode với policy/agent và thu thập metrics.

    Returns dict với các metrics tổng hợp.
    """
    rewards, latencies, costs, sla_flags, action_dist = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()

        ep_reward = 0.0
        done      = False

        while not done:
            if is_dqn:
                action = policy.select_action(obs, greedy=True)
            else:
                action = policy.select_action(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            latencies.append(info["latency"])
            costs.append(info["cost"])
            sla_flags.append(int(info["sla_met"]))
            action_dist.append(info["action"])

        rewards.append(ep_reward)

    n_actions    = env.n_edge_nodes + 1
    action_counts = [action_dist.count(i) for i in range(n_actions)]
    total_actions = len(action_dist)

    return {
        "avg_reward":    np.mean(rewards),
        "std_reward":    np.std(rewards),
        "avg_latency":   np.mean(latencies),
        "p95_latency":   np.percentile(latencies, 95),
        "avg_cost":      np.mean(costs),
        "sla_rate":      np.mean(sla_flags) * 100,
        "edge_usage":    sum(action_counts[:-1]) / total_actions * 100,
        "cloud_usage":   action_counts[-1] / total_actions * 100,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(results: dict, plot_dir: str):
    """Vẽ bar chart so sánh các metrics giữa DQN và Baselines."""
    os.makedirs(plot_dir, exist_ok=True)

    names     = list(results.keys())
    metrics   = ["avg_reward", "avg_latency", "sla_rate", "avg_cost"]
    titles    = ["Avg Reward ↑", "Avg Latency (ms) ↓", "SLA Rate (%) ↑", "Avg Cost ↓"]
    colors_map = {
        "DQN":            "#2196F3",
        "Random":         "#9E9E9E",
        "RoundRobin":     "#FF9800",
        "LeastConnection":"#4CAF50",
        "EdgeOnly":       "#00BCD4",
        "CloudOnly":      "#F44336",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "DQN vs Baseline Policies – Hybrid Edge-Cloud Task Scheduling",
        fontsize=14, fontweight="bold"
    )

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        values = [results[n][metric] for n in names]
        bar_colors = [
            colors_map.get(n.split("(")[0].strip(), "#607D8B") for n in names
        ]
        bars = ax.bar(names, values, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Highlight DQN bar
        if "DQN" in names:
            bars[names.index("DQN")].set_edgecolor("black")
            bars[names.index("DQN")].set_linewidth(2.5)

        # Annotate values on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(max(values) * 0.01),
                f"{val:.2f}", ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    path = os.path.join(plot_dir, "comparison_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_action_distribution(results: dict, plot_dir: str):
    """Pie chart Edge vs Cloud usage cho mỗi policy."""
    names = list(results.keys())
    n     = len(names)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * n, 8))
    axes  = axes.flatten()

    for i, (name, data) in enumerate(results.items()):
        edge   = data["edge_usage"]
        cloud  = data["cloud_usage"]
        axes[i].pie(
            [edge, cloud],
            labels=["Edge", "Cloud"],
            colors=["#00BCD4", "#F44336"],
            autopct="%1.1f%%",
            startangle=140,
        )
        axes[i].set_title(name, fontweight="bold")

    # Ẩn subplot thừa
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Edge vs Cloud Usage Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(plot_dir, "action_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN vs Baselines")
    parser.add_argument("--model", default="models/checkpoints/dqn_best.pth",
                        help="Path to DQN model checkpoint")
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG["n_episodes"])
    args = parser.parse_args()

    cfg = EVAL_CONFIG
    os.makedirs(cfg["log_dir"],  exist_ok=True)
    os.makedirs(cfg["plot_dir"], exist_ok=True)

    env = EdgeCloudEnv(
        n_edge_nodes  = cfg["n_edge_nodes"],
        use_prometheus= False,
        max_steps     = cfg["max_steps"],
    )
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    results = {}

    # ── Evaluate DQN ────────────────────────────────────────────────────────
    if os.path.exists(args.model):
        print(f"\n[Eval] Loading DQN from {args.model}")
        agent = DQNAgent(obs_dim, n_actions)
        agent.load(args.model)
        results["DQN"] = run_evaluation(agent, env, args.episodes, is_dqn=True)
    else:
        print(f"[Warn] Model not found: {args.model} – skipping DQN eval")

    # ── Evaluate Baselines ──────────────────────────────────────────────────
    baselines = get_all_baselines(n_actions, cfg["n_edge_nodes"])
    for policy in baselines:
        print(f"[Eval] Running {policy.name} ...")
        results[policy.name] = run_evaluation(policy, env, args.episodes)

    # ── Print table ─────────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"{'Policy':<22} {'Avg Reward':>11} {'Avg Lat(ms)':>12} "
          f"{'SLA%':>8} {'Avg Cost':>10} {'Edge%':>7} {'Cloud%':>7}")
    print(f"{'-'*85}")
    for name, m in results.items():
        marker = " ← DQN" if name == "DQN" else ""
        print(
            f"{name:<22} {m['avg_reward']:>11.3f} {m['avg_latency']:>12.1f} "
            f"{m['sla_rate']:>8.1f} {m['avg_cost']:>10.5f} "
            f"{m['edge_usage']:>7.1f} {m['cloud_usage']:>7.1f}{marker}"
        )
    print(f"{'='*85}\n")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    log_path = os.path.join(cfg["log_dir"], "evaluation_results.csv")
    with open(log_path, "w", newline="") as f:
        fieldnames = ["policy", "avg_reward", "std_reward", "avg_latency",
                      "p95_latency", "sla_rate", "avg_cost", "edge_usage", "cloud_usage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in results.items():
            writer.writerow({"policy": name, **{k: round(v, 4) for k, v in m.items()}})
    print(f"[Log] Results saved → {log_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_comparison(results, cfg["plot_dir"])
    plot_action_distribution(results, cfg["plot_dir"])


if __name__ == "__main__":
    main()