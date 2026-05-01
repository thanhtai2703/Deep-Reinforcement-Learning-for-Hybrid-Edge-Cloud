"""
verify_action_distribution.py
==============================
Kiểm tra model sau training có bị collapse (chọn 1 action 100%) không.
Chạy 50 episodes greedy, đếm action distribution.

Usage:
    python scripts/verify_action_distribution.py --env uncalibrated --model models/checkpoints/dqn_uncalibrated/dqn_best.pth
    python scripts/verify_action_distribution.py --env calibrated   --model models/checkpoints/dqn_calibrated/dqn_best.pth
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_env.edge_cloud_env import EdgeCloudEnv
from models.dqn_agent import DQNAgent


def make_env(env_kind, n_edge_nodes=2, max_steps=200):
    if env_kind == "calibrated":
        from rl_env.edge_cloud_env_calibrated import EdgeCloudEnvCalibrated
        return EdgeCloudEnvCalibrated(n_edge_nodes=n_edge_nodes, max_steps=max_steps)
    return EdgeCloudEnv(n_edge_nodes=n_edge_nodes, max_steps=max_steps)


def verify(env_kind: str, model_path: str, n_episodes: int = 50, n_edge: int = 2):
    env = make_env(env_kind, n_edge_nodes=n_edge)
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]

    agent = DQNAgent(obs_dim, n_actions)
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0.0  # Greedy
    else:
        print(f"⚠ Model not found: {model_path} — using random agent")

    action_names = [f"edge_{i+1}" for i in range(n_edge)] + ["cloud", "reject"]
    action_counts = np.zeros(n_actions, dtype=int)
    total_reward = 0.0
    total_latency = 0.0
    sla_count = 0
    total_cost = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_counts[action] += 1
            total_reward += reward
            total_latency += info["latency"]
            total_cost += info["cost"]
            sla_count += int(info["sla_met"])
            total_steps += 1

    # Results
    print(f"\n{'='*60}")
    print(f"  VERIFICATION: {env_kind} env | {n_episodes} episodes | {total_steps} steps")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    print(f"\n  ACTION DISTRIBUTION:")
    collapsed = False
    for i in range(n_actions):
        pct = action_counts[i] / total_steps * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        flag = " ⚠ DOMINANT" if pct > 80 else (" ✗ NEVER" if pct == 0 else " ✓")
        print(f"    {action_names[i]:8s}: {action_counts[i]:5d} ({pct:5.1f}%) |{bar}|{flag}")
        if pct > 80:
            collapsed = True

    print(f"\n  PERFORMANCE:")
    print(f"    Avg reward     : {total_reward / n_episodes:.2f}")
    print(f"    Avg latency    : {total_latency / total_steps:.1f}ms")
    print(f"    SLA rate       : {sla_count / total_steps * 100:.1f}%")
    print(f"    Avg cost       : {total_cost / total_steps:.4f}")

    if collapsed:
        print(f"\n  ❌ COLLAPSED — model chọn 1 action > 80%. Cần review reward/training.")
    else:
        max_pct = action_counts.max() / total_steps * 100
        if max_pct > 60:
            print(f"\n  ⚠ BIASED — action max {max_pct:.1f}%. Chấp nhận được nhưng có thể cải thiện.")
        else:
            print(f"\n  ✅ DIVERSE — action distribution tốt, sẵn sàng deploy.")

    print()
    return not collapsed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=("uncalibrated", "calibrated"), default="uncalibrated")
    p.add_argument("--model", required=True, help="Path to dqn_best.pth")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--edges", type=int, default=2)
    args = p.parse_args()

    verify(args.env, args.model, args.episodes, args.edges)
