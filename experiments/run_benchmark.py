"""
run_benchmark.py
================
Large-scale benchmark: 1000+ tasks, 3 scenarios, RL vs Baselines.
Đây là kết quả chính để đưa vào technical report (Tuần 4).

Scenarios:
  1. Constant load  – task đều đặn, deadline thoải mái
  2. Bursty load    – task tập trung, deadline gấp
  3. Mixed tasks    – kết hợp CPU-heavy + latency-sensitive

Chạy:
    python experiments/run_benchmark.py \
        --dqn models/checkpoints/dqn_best.pth \
        --ppo models/checkpoints/ppo/ppo_best.zip

Output:
    - results/benchmark_1000_tasks.csv   (raw data)
    - results/benchmark_summary.csv      (aggregated)
    - experiments/plots/cost_perf_scatter.png
    - experiments/plots/resource_heatmap.png
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_env.edge_cloud_env import EdgeCloudEnv
from rl_env.baseline_policies import get_all_baselines
from models.dqn_agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────────
# Scenario definitions – patch vào env để control workload
# ──────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    "constant_load": {
        "description":  "Constant load – task đều đặn, deadline thoải mái",
        "cpu_range":     (5, 35),
        "ram_range":     (5, 30),
        "deadline_range":(150, 500),
        "n_tasks":       350,
    },
    "bursty_load": {
        "description":  "Bursty load – tác vụ nặng, deadline gấp",
        "cpu_range":     (40, 85),
        "ram_range":     (30, 70),
        "deadline_range":(50, 150),
        "n_tasks":       350,
    },
    "mixed_tasks": {
        "description":  "Mixed – kết hợp CPU-heavy & latency-sensitive",
        "cpu_range":     (5, 80),
        "ram_range":     (5, 65),
        "deadline_range":(50, 500),
        "n_tasks":       350,
    },
}


class BenchmarkEnv(EdgeCloudEnv):
    """
    Subclass EdgeCloudEnv với task generator bị override
    để kiểm soát chính xác workload pattern.
    """

    def set_scenario(self, scenario_cfg: dict):
        self._cpu_range      = scenario_cfg["cpu_range"]
        self._ram_range      = scenario_cfg["ram_range"]
        self._deadline_range = scenario_cfg["deadline_range"]

    def _generate_task(self):
        self._task = {
            "cpu_req":  float(self.np_random.uniform(*self._cpu_range)),
            "ram_req":  float(self.np_random.uniform(*self._ram_range)),
            "deadline": float(self.np_random.uniform(*self._deadline_range)),
        }


# ──────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────

def run_scenario_benchmark(policy_fn, env: BenchmarkEnv,
                           scenario_name: str, scenario_cfg: dict,
                           is_sb3: bool = False, seed: int = 0) -> list:
    """
    Chạy benchmark cho 1 scenario với 1 policy.
    Trả về list of task records (raw per-task data).
    """
    env.set_scenario(scenario_cfg)
    n_tasks = scenario_cfg["n_tasks"]

    records  = []
    obs, _   = env.reset(seed=seed)
    task_idx = 0

    while task_idx < n_tasks:
        if is_sb3:
            action, _ = policy_fn(obs, deterministic=True)
            action = int(action)
        else:
            action = policy_fn(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        records.append({
            "scenario":   scenario_name,
            "task_id":    task_idx,
            "action":     action,
            "is_cloud":   int(info["is_cloud"]),
            "latency":    round(info["latency"],  2),
            "cost":       round(info["cost"],      6),
            "sla_met":    int(info["sla_met"]),
            "reward":     round(reward, 4),
        })
        task_idx += 1

        if terminated or truncated:
            obs, _ = env.reset(seed=seed + task_idx)

    return records


def summarize(records: list) -> dict:
    """Tính metrics từ raw task records."""
    latencies = [r["latency"] for r in records]
    costs     = [r["cost"]    for r in records]
    sla       = [r["sla_met"] for r in records]
    rewards   = [r["reward"]  for r in records]
    n         = len(records)

    return {
        "n_tasks":     n,
        "avg_reward":  round(np.mean(rewards), 4),
        "avg_latency": round(np.mean(latencies), 2),
        "med_latency": round(np.median(latencies), 2),
        "p95_latency": round(np.percentile(latencies, 95), 2),
        "p99_latency": round(np.percentile(latencies, 99), 2),
        "avg_cost":    round(np.mean(costs), 6),
        "total_cost":  round(np.sum(costs), 4),
        "sla_rate":    round(np.mean(sla) * 100, 2),
        "deadline_miss": n - sum(sla),
        "cloud_pct":   round(sum(r["is_cloud"] for r in records) / n * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────

def plot_cost_performance_scatter(results: dict, plot_dir: str):
    """Scatter: Cost vs SLA rate. RL models nên ở góc tốt nhất."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scenario_names = list(SCENARIOS.keys())

    colors = {
        "DQN": "#2196F3", "PPO": "#9C27B0",
        "Random": "#9E9E9E", "RoundRobin": "#FF9800",
        "LeastConnection": "#4CAF50", "EdgeOnly": "#00BCD4",
        "CloudOnly": "#F44336",
    }
    markers = {"DQN": "★", "PPO": "★"}

    for ax, sc in zip(axes, scenario_names):
        for policy_name, sc_data in results.items():
            if sc not in sc_data:
                continue
            m = sc_data[sc]
            c = colors.get(policy_name.split("(")[0].strip(), "#607D8B")
            size = 180 if policy_name in ("DQN", "PPO") else 80
            ax.scatter(m["avg_cost"], m["sla_rate"], c=c, s=size,
                       label=policy_name, zorder=5,
                       edgecolors="black" if policy_name in ("DQN", "PPO") else "none",
                       linewidths=1.5)
            ax.annotate(policy_name[:4], (m["avg_cost"], m["sla_rate"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax.set_xlabel("Avg Cost ↓")
        ax.set_ylabel("SLA Rate (%) ↑")
        ax.set_title(sc.replace("_", " ").title(), fontweight="bold")
        ax.grid(alpha=0.3)
        # "Ideal" corner annotation
        ax.annotate("ideal", xy=(ax.get_xlim()[0], 100),
                    fontsize=7, color="gray", style="italic")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.suptitle("Cost vs SLA Rate – All Scenarios", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(plot_dir, "cost_performance_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_resource_heatmap(results: dict, plot_dir: str):
    """
    Heatmap: policy × scenario → SLA rate
    (dễ thấy policy nào mạnh ở scenario nào)
    """
    policy_names   = list(results.keys())
    scenario_names = list(SCENARIOS.keys())

    matrix = np.array([
        [results[p].get(sc, {}).get("sla_rate", 0) for sc in scenario_names]
        for p in policy_names
    ])

    fig, ax = plt.subplots(figsize=(9, max(4, len(policy_names) * 0.7)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(scenario_names)))
    ax.set_xticklabels([s.replace("_", "\n") for s in scenario_names], fontsize=10)
    ax.set_yticks(range(len(policy_names)))
    ax.set_yticklabels(policy_names, fontsize=9)

    for i in range(len(policy_names)):
        for j in range(len(scenario_names)):
            v = matrix[i, j]
            color = "white" if v < 40 or v > 80 else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="SLA Rate (%)")
    ax.set_title("SLA Rate Heatmap – Policy × Scenario", fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(plot_dir, "resource_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_deadline_miss(results: dict, plot_dir: str):
    """Bar chart: số task miss deadline mỗi policy (càng thấp càng tốt)."""
    policy_names = list(results.keys())
    total_miss   = [
        sum(results[p].get(sc, {}).get("deadline_miss", 0) for sc in SCENARIOS)
        for p in policy_names
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_list = ["#2196F3" if p in ("DQN", "PPO") else "#BDBDBD"
                   for p in policy_names]
    bars = ax.bar(policy_names, total_miss, color=colors_list,
                  edgecolor="white", linewidth=0.8)

    for bar, p in zip(bars, policy_names):
        if p in ("DQN", "PPO"):
            bar.set_edgecolor("black")
            bar.set_linewidth(2)

    for bar, v in zip(bars, total_miss):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2, str(v),
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Policy")
    ax.set_ylabel("Total deadline misses")
    ax.set_title(f"Deadline Misses – Total across {sum(s['n_tasks'] for s in SCENARIOS.values())} tasks",
                 fontweight="bold")
    ax.set_xticklabels(policy_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, "deadline_miss_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn",  default="models/checkpoints/dqn_best.pth")
    parser.add_argument("--ppo",  default="models/checkpoints/ppo/ppo_best.zip")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("experiments/plots", exist_ok=True)

    env = BenchmarkEnv(n_edge_nodes=3, max_steps=99999)
    env.reset(seed=0)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Dựng dict policy_name → (policy_fn, is_sb3)
    policies = {}

    if os.path.exists(args.dqn):
        dqn = DQNAgent(obs_dim, n_actions)
        dqn.load(args.dqn)
        policies["DQN"] = (lambda o, dqn=dqn: dqn.select_action(o, greedy=True), False)

    ppo_zip = args.ppo if args.ppo.endswith(".zip") else args.ppo + ".zip"
    if os.path.exists(ppo_zip):
        try:
            from stable_baselines3 import PPO as SB3PPO
            ppo_model = SB3PPO.load(ppo_zip)
            policies["PPO"] = (ppo_model.predict, True)
        except ImportError:
            pass

    for p in get_all_baselines(n_actions, 3):
        policies[p.name] = (p.select_action, False)

    # ── Run benchmark ──────────────────────────────────────────────────────
    total_tasks = sum(s["n_tasks"] for s in SCENARIOS.values())
    print(f"\n{'='*70}")
    print(f"  Benchmark: {len(policies)} policies × {len(SCENARIOS)} scenarios × ~{total_tasks} tasks")
    print(f"{'='*70}\n")

    results      = {}   # policy_name → {scenario → summary}
    raw_records  = []   # all raw task records (for CSV)

    t0 = time.time()
    for policy_name, (policy_fn, is_sb3) in policies.items():
        results[policy_name] = {}
        print(f"  [{policy_name}]")
        for sc_name, sc_cfg in SCENARIOS.items():
            records = run_scenario_benchmark(
                policy_fn, env, sc_name, sc_cfg, is_sb3=is_sb3, seed=42
            )
            summary = summarize(records)
            results[policy_name][sc_name] = summary

            for r in records:
                r["policy"] = policy_name
            raw_records.extend(records)

            print(f"    {sc_name:<18}: SLA={summary['sla_rate']:5.1f}% | "
                  f"lat={summary['avg_latency']:6.1f}ms | "
                  f"miss={summary['deadline_miss']}")

    elapsed = time.time() - t0
    print(f"\n  ✅ Benchmark done in {elapsed:.1f}s")

    # ── Save raw CSV ────────────────────────────────────────────────────────
    raw_path = "results/benchmark_1000_tasks.csv"
    with open(raw_path, "w", newline="") as f:
        fieldnames = ["policy", "scenario", "task_id", "action", "is_cloud",
                      "latency", "cost", "sla_met", "reward"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_records)
    print(f"[CSV] Raw data → {raw_path} ({len(raw_records):,} rows)")

    # ── Save summary CSV ────────────────────────────────────────────────────
    summary_path = "results/benchmark_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["policy", "scenario", "n_tasks", "avg_reward", "avg_latency",
                      "p95_latency", "sla_rate", "deadline_miss", "avg_cost", "cloud_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pname, sc_results in results.items():
            for sc_name, m in sc_results.items():
                writer.writerow({"policy": pname, "scenario": sc_name, **m})
    print(f"[CSV] Summary → {summary_path}")

    # ── Print summary table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Policy':<22} {'Scenario':<18} {'SLA%':>6} {'Lat':>7} "
          f"{'P95':>7} {'Miss':>5} {'Cost':>9}")
    print(f"{'-'*80}")
    for pname, sc_results in results.items():
        for sc_name, m in sc_results.items():
            marker = " ★" if pname in ("DQN", "PPO") else ""
            print(f"{pname:<22} {sc_name:<18} {m['sla_rate']:>6.1f} "
                  f"{m['avg_latency']:>7.1f} {m['p95_latency']:>7.1f} "
                  f"{m['deadline_miss']:>5} {m['avg_cost']:>9.5f}{marker}")
    print(f"{'='*80}\n")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_cost_performance_scatter(results, "experiments/plots")
    plot_resource_heatmap(results, "experiments/plots")
    plot_deadline_miss(results, "experiments/plots")

    print("\n✅ All benchmark outputs saved!")
    print("   results/benchmark_1000_tasks.csv")
    print("   results/benchmark_summary.csv")
    print("   experiments/plots/cost_performance_scatter.png")
    print("   experiments/plots/resource_heatmap.png")
    print("   experiments/plots/deadline_miss_comparison.png")


if __name__ == "__main__":
    main()