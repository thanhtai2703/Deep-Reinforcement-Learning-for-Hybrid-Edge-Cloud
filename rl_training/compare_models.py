"""
compare_models.py
=================
So sánh toàn diện DQN vs PPO vs tất cả Baseline policies.
Bao gồm kiểm định thống kê (t-test, confidence intervals).

Chạy:
    python -m rl_training.compare_models \
        --dqn  models/checkpoints/dqn_best.pth \
        --ppo  models/checkpoints/ppo/ppo_best.zip

Output:
    - experiments/logs/model_comparison.csv
    - experiments/plots/final_comparison.png
    - experiments/plots/latency_boxplot.png
    - Terminal: bảng kết quả + t-test significance
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
from rl_env.baseline_policies import get_all_baselines
from models.dqn_agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
COMPARE_CONFIG = {
    "n_episodes":   100,
    "n_edge_nodes": 2,
    "max_steps":    200,
    "log_dir":      "experiments/logs",
    "plot_dir":     "experiments/plots",
}

SCENARIOS = {
    "constant":  {"task_cpu_range": (10, 40),  "task_deadline_range": (100, 400)},
    "bursty":    {"task_cpu_range": (30, 80),  "task_deadline_range": (50,  200)},
    "mixed":     {"task_cpu_range": (5,  70),  "task_deadline_range": (50,  500)},
}


# ──────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────

def collect_episode_rewards(policy_fn, env: EdgeCloudEnv,
                            n_episodes: int, is_sb3: bool = False) -> dict:
    """
    Chạy policy_fn trên env, thu về raw arrays cho statistical testing.

    Returns:
        dict với arrays: rewards, latencies, costs, sla_flags, action_dist
    """
    ep_rewards = []
    all_latencies, all_costs, all_sla, all_actions = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 7)
        ep_r, done = 0.0, False
        while not done:
            if is_sb3:
                action, _ = policy_fn(obs, deterministic=True)
                action = int(action)
            else:
                action = policy_fn(obs)
            obs, r, t, tr, info = env.step(action)
            done = t or tr
            ep_r += r
            all_latencies.append(info["latency"])
            all_costs.append(info["cost"])
            all_sla.append(int(info["sla_met"]))
            all_actions.append(action)

        ep_rewards.append(ep_r)

    return {
        "ep_rewards":  np.array(ep_rewards),
        "latencies":   np.array(all_latencies),
        "costs":       np.array(all_costs),
        "sla_flags":   np.array(all_sla),
        "action_dist": np.array(all_actions),
    }


def compute_metrics(data: dict, n_edge_nodes: int) -> dict:
    """Tính các metrics tổng hợp từ raw data."""
    n_act = len(data["action_dist"])
    return {
        "avg_reward":  float(np.mean(data["ep_rewards"])),
        "std_reward":  float(np.std(data["ep_rewards"])),
        "ci95_reward": float(1.96 * np.std(data["ep_rewards"]) / np.sqrt(len(data["ep_rewards"]))),
        "avg_latency": float(np.mean(data["latencies"])),
        "med_latency": float(np.median(data["latencies"])),
        "p95_latency": float(np.percentile(data["latencies"], 95)),
        "p99_latency": float(np.percentile(data["latencies"], 99)),
        "avg_cost":    float(np.mean(data["costs"])),
        "sla_rate":    float(np.mean(data["sla_flags"])) * 100,
        "edge_usage":  float(np.sum(data["action_dist"] < n_edge_nodes)) / n_act * 100,
        "cloud_usage": float(np.sum(data["action_dist"] == n_edge_nodes)) / n_act * 100,
    }


def ttest_vs_baseline(rl_rewards: np.ndarray, baseline_rewards: np.ndarray) -> dict:
    """
    Independent t-test: RL rewards vs baseline rewards.
    H0: means are equal. H1: RL mean > baseline mean (one-tailed).
    """
    try:
        from scipy import stats
        t_stat, p_two = stats.ttest_ind(rl_rewards, baseline_rewards)
        p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        effect = (np.mean(rl_rewards) - np.mean(baseline_rewards)) / (
            np.std(np.concatenate([rl_rewards, baseline_rewards])) + 1e-8
        )
        return {
            "t_stat":    round(float(t_stat), 4),
            "p_value":   round(float(p_one), 4),
            "effect_size": round(float(effect), 4),
            "significant": bool(p_one < 0.05),
        }
    except ImportError:
        return {"t_stat": 0, "p_value": 1, "effect_size": 0, "significant": False}


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

def plot_latency_boxplot(all_data: dict, plot_dir: str):
    """Box plot phân phối latency của tất cả models."""
    names = list(all_data.keys())
    latency_arrays = [all_data[n]["latencies"] for n in names]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(latency_arrays, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))

    colors = {
        "DQN":   "#2196F3", "PPO":  "#9C27B0",
        "Random": "#9E9E9E", "RoundRobin": "#FF9800",
        "LeastConnection": "#4CAF50", "EdgeOnly": "#00BCD4",
        "CloudOnly": "#F44336",
    }
    for patch, name in zip(bp["boxes"], names):
        c = colors.get(name.split("(")[0].strip(), "#607D8B")
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Latency Distribution – All Policies", fontweight="bold", fontsize=13)
    ax.set_ylabel("Latency (ms)")
    ax.grid(axis="y", alpha=0.3)

    # Highlight RL models
    for i, name in enumerate(names):
        if name in ("DQN", "PPO"):
            ax.get_xticklabels()[i].set_fontweight("bold")

    plt.tight_layout()
    path = os.path.join(plot_dir, "latency_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_final_comparison(metrics_dict: dict, plot_dir: str):
    """4-panel bar chart so sánh tất cả models."""
    names   = list(metrics_dict.keys())
    panels  = [
        ("avg_reward",  "Avg Reward ↑",     "#2196F3"),
        ("avg_latency", "Avg Latency (ms) ↓","#F44336"),
        ("sla_rate",    "SLA Rate (%) ↑",    "#4CAF50"),
        ("avg_cost",    "Avg Cost ↓",        "#FF9800"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Final Model Comparison – Hybrid Edge-Cloud Scheduling",
                 fontsize=14, fontweight="bold")

    bar_colors = {
        "DQN":   "#2196F3", "PPO":  "#9C27B0",
        "Random": "#BDBDBD", "RoundRobin": "#FFB74D",
        "LeastConnection": "#81C784", "EdgeOnly": "#4DD0E1",
        "CloudOnly": "#EF9A9A",
    }

    for ax, (metric, title, _) in zip(axes.flatten(), panels):
        vals = [metrics_dict[n][metric] for n in names]
        cols = [bar_colors.get(n.split("(")[0].strip(), "#90A4AE") for n in names]
        bars = ax.bar(names, vals, color=cols, edgecolor="white", linewidth=0.8)

        # Bold border for RL models
        for bar, name in zip(bars, names):
            if name in ("DQN", "PPO"):
                bar.set_edgecolor("black")
                bar.set_linewidth(2.5)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + abs(max(vals)) * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(plot_dir, "final_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_sla_over_time(all_data: dict, n_episodes: int, plot_dir: str):
    """SLA rate theo episode (sliding window)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = {"DQN": "#2196F3", "PPO": "#9C27B0"}

    for name, data in all_data.items():
        # Tính SLA per episode từ sla_flags (200 steps/ep)
        steps_per_ep = len(data["sla_flags"]) // max(n_episodes, 1)
        if steps_per_ep == 0:
            continue
        sla_per_ep = [
            np.mean(data["sla_flags"][i*steps_per_ep:(i+1)*steps_per_ep]) * 100
            for i in range(n_episodes)
        ]
        color = colors.get(name, "#BDBDBD")
        lw    = 2.0 if name in colors else 1.0
        alpha = 1.0 if name in colors else 0.5
        ax.plot(range(n_episodes), sla_per_ep, label=name,
                color=color, linewidth=lw, alpha=alpha)

    ax.axhline(y=95, color="orange", linestyle="--", linewidth=1.5, label="Target 95%")
    ax.set_xlabel("Episode"); ax.set_ylabel("SLA Rate (%)")
    ax.set_title("SLA Compliance Over Episodes", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plot_dir, "sla_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare DQN vs PPO vs Baselines")
    parser.add_argument("--dqn",      default="models/checkpoints/dqn_best.pth")
    parser.add_argument("--ppo",      default="models/checkpoints/ppo/ppo_best.zip")
    parser.add_argument("--episodes", type=int, default=COMPARE_CONFIG["n_episodes"])
    args = parser.parse_args()

    cfg = COMPARE_CONFIG
    os.makedirs(cfg["log_dir"],  exist_ok=True)
    os.makedirs(cfg["plot_dir"], exist_ok=True)

    env = EdgeCloudEnv(
        n_edge_nodes = cfg["n_edge_nodes"],
        max_steps    = cfg["max_steps"],
    )
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    all_data    = {}   # name → raw data dict
    metrics_all = {}   # name → metrics dict

    # ── DQN ────────────────────────────────────────────────────────────────
    if os.path.exists(args.dqn):
        print(f"[Load] DQN ← {args.dqn}")
        dqn = DQNAgent(obs_dim, n_actions)
        dqn.load(args.dqn)
        all_data["DQN"] = collect_episode_rewards(
            lambda obs: dqn.select_action(obs, greedy=True), env, args.episodes
        )
    else:
        print(f"[Skip] DQN model not found: {args.dqn}")

    # ── PPO ────────────────────────────────────────────────────────────────
    ppo_zip = args.ppo if args.ppo.endswith(".zip") else args.ppo + ".zip"
    if os.path.exists(ppo_zip):
        try:
            from stable_baselines3 import PPO as SB3PPO
            print(f"[Load] PPO ← {ppo_zip}")
            ppo_model = SB3PPO.load(ppo_zip)
            all_data["PPO"] = collect_episode_rewards(
                ppo_model.predict, env, args.episodes, is_sb3=True
            )
        except ImportError:
            print("[Skip] stable-baselines3 not installed")
    else:
        print(f"[Skip] PPO model not found: {ppo_zip}")

    # ── Baselines ──────────────────────────────────────────────────────────
    for policy in get_all_baselines(n_actions, cfg["n_edge_nodes"]):
        print(f"[Eval] {policy.name} ...")
        all_data[policy.name] = collect_episode_rewards(
            policy.select_action, env, args.episodes
        )

    # ── Compute metrics ────────────────────────────────────────────────────
    for name, data in all_data.items():
        metrics_all[name] = compute_metrics(data, cfg["n_edge_nodes"])

    # ── Statistical tests (RL vs each baseline) ────────────────────────────
    stat_results = {}
    for rl_name in ("DQN", "PPO"):
        if rl_name not in all_data:
            continue
        stat_results[rl_name] = {}
        for name, data in all_data.items():
            if name in ("DQN", "PPO"):
                continue
            stat_results[rl_name][name] = ttest_vs_baseline(
                all_data[rl_name]["ep_rewards"], data["ep_rewards"]
            )

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n{'='*95}")
    print(f"{'Policy':<22} {'Avg R':>8} {'±CI':>6} {'Lat(ms)':>9} "
          f"{'P95':>7} {'SLA%':>7} {'Cost':>9} {'Edge%':>7}")
    print(f"{'-'*95}")
    for name, m in metrics_all.items():
        marker = " ★" if name in ("DQN", "PPO") else ""
        print(
            f"{name:<22} {m['avg_reward']:>8.3f} "
            f"±{m['ci95_reward']:<5.2f} {m['avg_latency']:>9.1f} "
            f"{m['p95_latency']:>7.1f} {m['sla_rate']:>7.1f} "
            f"{m['avg_cost']:>9.5f} {m['edge_usage']:>7.1f}{marker}"
        )
    print(f"{'='*95}")

    # Statistical significance
    if stat_results:
        print(f"\n  Statistical significance (t-test, one-tailed, α=0.05):")
        for rl_name, comparisons in stat_results.items():
            print(f"\n  {rl_name} vs baselines:")
            for base_name, s in comparisons.items():
                sig = "✅ p<0.05" if s["significant"] else "❌ not sig."
                print(f"    vs {base_name:<22}: p={s['p_value']:.4f} "
                      f"| d={s['effect_size']:.3f} | {sig}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(cfg["log_dir"], "model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["policy", "avg_reward", "std_reward", "ci95_reward",
                      "avg_latency", "med_latency", "p95_latency", "p99_latency",
                      "sla_rate", "avg_cost", "edge_usage", "cloud_usage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in metrics_all.items():
            writer.writerow({"policy": name, **{k: round(v, 4) for k, v in m.items()}})

    # Stat test CSV
    stat_path = os.path.join(cfg["log_dir"], "statistical_tests.csv")
    with open(stat_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rl_model", "baseline", "t_stat", "p_value",
                         "effect_size", "significant"])
        for rl_name, comparisons in stat_results.items():
            for base_name, s in comparisons.items():
                writer.writerow([rl_name, base_name, s["t_stat"],
                                 s["p_value"], s["effect_size"], s["significant"]])

    print(f"\n[Log] Saved → {csv_path}")
    print(f"[Log] Saved → {stat_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_final_comparison(metrics_all, cfg["plot_dir"])
    plot_latency_boxplot(all_data, cfg["plot_dir"])
    plot_sla_over_time(all_data, args.episodes, cfg["plot_dir"])

    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()