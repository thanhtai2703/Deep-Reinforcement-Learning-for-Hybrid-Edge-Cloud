"""
Comprehensive Model Evaluation - Kiểm tra Model hoạt động tốt như nào
Person 2: ML/RL Engineer - Model Performance Analysis
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_env.edge_cloud_env import EdgeCloudEnv
from rl_env.baseline_policies import (
    RoundRobinPolicy,
    LeastConnectionPolicy,
    EdgeOnlyPolicy,
    ThresholdPolicy,
)
from models.dqn_agent import DQNAgent

# obs layout (2 edge nodes, 12 dims total):
#  0: edge1_cpu  1: edge1_ram  2: edge1_lat
#  3: edge2_cpu  4: edge2_ram  5: edge2_lat
#  6: cloud_cpu  7: cloud_ram  8: cloud_lat
#  9: task_cpu  10: task_ram  11: task_deadline
_OBS_EDGE2_CPU  = 3
_OBS_EDGE2_RAM  = 4
_OBS_CLOUD_CPU  = 6
_OBS_CLOUD_RAM  = 7
_OBS_TASK_CPU   = 9
_OBS_TASK_DEAD  = 11

N_EDGE_NODES = 2
N_ACTIONS    = N_EDGE_NODES + 1   # 3
OBS_DIM      = N_EDGE_NODES * 3 + 3 + 3  # 12


def _load_dqn(model_path: str) -> DQNAgent:
    agent = DQNAgent(OBS_DIM, N_ACTIONS)
    agent.load(model_path)
    return agent


def test_model_vs_baselines(model_path="models/checkpoints/dqn_best.pth"):
    """So sánh model với tất cả baselines"""
    print("MODEL vs BASELINES COMPARISON")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Model không tồn tại: {model_path}")
        return None, None

    dqn_agent = _load_dqn(model_path)
    env = EdgeCloudEnv(n_edge_nodes=N_EDGE_NODES, max_steps=200)

    # (name, policy_or_None) – None = DQN
    policies = [
        ("Random",             None),
        ("Round Robin",        RoundRobinPolicy(N_ACTIONS, N_EDGE_NODES)),
        ("Least Connection",   LeastConnectionPolicy(N_ACTIONS, N_EDGE_NODES)),
        ("Edge Only",          EdgeOnlyPolicy(N_ACTIONS, N_EDGE_NODES)),
        ("Threshold",          ThresholdPolicy(N_ACTIONS, N_EDGE_NODES)),
        ("DQN (Trained)",      dqn_agent),
    ]

    results = {}
    detailed_results = []

    print("Testing policies (10 episodes each)...")

    for policy_name, policy in policies:
        print(f"\nTesting {policy_name}...")

        episode_rewards      = []
        episode_latencies    = []
        episode_miss_rates   = []
        action_distributions = []

        for episode in range(10):
            obs, _ = env.reset(seed=episode * 100)
            if hasattr(policy, "reset"):
                policy.reset()

            episode_reward = 0.0
            episode_actions = []
            latencies = []
            deadline_misses = 0
            total_steps = 0

            for _ in range(200):
                # Select action
                if policy is None:           # Random
                    action = env.action_space.sample()
                elif policy_name == "DQN (Trained)":
                    action = policy.select_action(obs, greedy=True)
                else:                        # Baseline
                    action = policy.select_action(obs)

                episode_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                total_steps += 1
                latencies.append(info["latency"])       # every step = one task
                if not info["sla_met"]:
                    deadline_misses += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_latencies.append(float(np.mean(latencies)) if latencies else 0.0)
            episode_miss_rates.append(deadline_misses / total_steps if total_steps > 0 else 0.0)

            action_counts = np.bincount(episode_actions, minlength=N_ACTIONS)
            action_distributions.append(action_counts / len(episode_actions) * 100)

        results[policy_name] = {
            "avg_reward":      float(np.mean(episode_rewards)),
            "std_reward":      float(np.std(episode_rewards)),
            "avg_latency":     float(np.mean(episode_latencies)),
            "avg_miss_rate":   float(np.mean(episode_miss_rates)),
            "avg_sla_rate":    1.0 - float(np.mean(episode_miss_rates)),
            "action_dist":     np.mean(action_distributions, axis=0),
        }

        detailed_results.append({
            "policy":       policy_name,
            "rewards":      episode_rewards,
            "latencies":    episode_latencies,
            "miss_rates":   episode_miss_rates,
        })

        r = results[policy_name]
        print(f"  Avg Reward : {r['avg_reward']:.1f}")
        print(f"  Avg Latency: {r['avg_latency']:.1f} ms")
        print(f"  SLA Rate   : {r['avg_sla_rate']:.1%}")
        print(f"  Miss Rate  : {r['avg_miss_rate']:.1%}")

    # Comparison table
    print(f"\n{'=' * 80}")
    print(f"{'Policy':<20} {'Reward':>10} {'Latency(ms)':>12} "
          f"{'SLA%':>8} {'MissRate%':>10}")
    print(f"{'-' * 80}")
    for pname, s in results.items():
        marker = " <--" if pname == "DQN (Trained)" else ""
        print(f"{pname:<20} {s['avg_reward']:>10.2f} {s['avg_latency']:>12.1f} "
              f"{s['avg_sla_rate']:>8.1%} {s['avg_miss_rate']:>10.1%}{marker}")
    print(f"{'=' * 80}")

    # Ranking
    sorted_policies = sorted(results.items(), key=lambda x: x[1]["avg_reward"], reverse=True)
    dqn_rank = next(i + 1 for i, (n, _) in enumerate(sorted_policies) if n == "DQN (Trained)")

    print(f"\nRANKING:")
    for i, (pname, s) in enumerate(sorted_policies, 1):
        print(f"  #{i}: {pname} - {s['avg_reward']:.1f} reward")

    dqn = results["DQN (Trained)"]
    rand = results["Random"]
    improvement = dqn["avg_reward"] - rand["avg_reward"]
    print(f"\nDQN ANALYSIS:")
    print(f"  Rank: #{dqn_rank} / {len(results)}")
    print(f"  Improvement over Random: +{improvement:.1f}")
    print(f"  Action dist: Edge1={dqn['action_dist'][0]:.0f}%, "
          f"Edge2={dqn['action_dist'][1]:.0f}%, "
          f"Cloud={dqn['action_dist'][2]:.0f}%")

    if dqn_rank == 1:
        print("  EXCELLENT: DQN is the best policy!")
    elif dqn_rank <= 3:
        print("  GOOD: DQN is among top 3")
    else:
        print("  NEEDS IMPROVEMENT: DQN underperforming")

    return results, detailed_results


def test_different_scenarios(model_path="models/checkpoints/dqn_best.pth"):
    """Test model trong các scenarios khác nhau"""
    print("\nSCENARIO TESTING")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Model không tồn tại: {model_path}")
        return {}

    dqn_agent = _load_dqn(model_path)

    scenarios = [
        ("Normal Load", 500, "normal"),
        ("High Load",   500, "high"),
        ("Burst Load",  100, "high"),
        ("Light Load", 1000, "normal"),
    ]

    scenario_results = {}

    for scenario_name, episode_length, load_type in scenarios:
        print(f"\nTesting {scenario_name}...")

        env = EdgeCloudEnv(n_edge_nodes=N_EDGE_NODES, max_steps=episode_length)

        rewards, latencies, miss_rates = [], [], []

        for episode in range(5):
            obs, _ = env.reset(seed=episode)

            # Simulate high load by warming up env state
            if load_type == "high":
                env._edge_cpu[:] = np.clip(
                    np.random.uniform(50, 85, N_EDGE_NODES), 0, 99
                )

            ep_reward = 0.0
            ep_latencies = []
            ep_misses = 0
            total = 0

            for _ in range(episode_length):
                action = dqn_agent.select_action(obs, greedy=True)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += reward
                total += 1
                ep_latencies.append(info["latency"])
                if not info["sla_met"]:
                    ep_misses += 1

                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            latencies.append(float(np.mean(ep_latencies)) if ep_latencies else 0.0)
            miss_rates.append(ep_misses / total if total > 0 else 0.0)

        scenario_results[scenario_name] = {
            "avg_reward":    float(np.mean(rewards)),
            "avg_latency":   float(np.mean(latencies)),
            "avg_miss_rate": float(np.mean(miss_rates)),
        }

        s = scenario_results[scenario_name]
        print(f"  Reward  : {s['avg_reward']:.1f}")
        print(f"  Latency : {s['avg_latency']:.1f} ms")
        print(f"  SLA Rate: {1 - s['avg_miss_rate']:.1%}")

    return scenario_results


def analyze_learning_behavior(model_path="models/checkpoints/dqn_best.pth"):
    """Phân tích behavior của model"""
    print("\nBEHAVIOR ANALYSIS")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Model không tồn tại: {model_path}")
        return []

    dqn_agent = _load_dqn(model_path)
    env = EdgeCloudEnv(n_edge_nodes=N_EDGE_NODES, max_steps=50)

    behaviors = []
    print("Testing decision-making in different situations...")

    for test_case in range(10):
        obs, _ = env.reset(seed=test_case)
        action = dqn_agent.select_action(obs, greedy=True)

        # Correct obs indices for 2 edge nodes
        edge1_load  = float(obs[0] + obs[1])          # cpu + ram
        edge2_load  = float(obs[_OBS_EDGE2_CPU] + obs[_OBS_EDGE2_RAM])
        cloud_load  = float(obs[_OBS_CLOUD_CPU] + obs[_OBS_CLOUD_RAM])
        task_demand = float(obs[_OBS_TASK_CPU])
        task_dead   = float(obs[_OBS_TASK_DEAD])

        if action == 0:
            rationale = "Less loaded edge" if edge1_load <= edge2_load else "Preferred edge despite load"
        elif action == 1:
            rationale = "Less loaded edge" if edge2_load < edge1_load else "Preferred edge despite load"
        else:
            if task_demand > 0.7:
                rationale = "High-demand task to cloud"
            elif edge1_load > 1.6 and edge2_load > 1.6:
                rationale = "Edge nodes overloaded"
            else:
                rationale = "Unexpected cloud choice"

        behavior = {
            "test_case":   test_case,
            "action":      action,
            "action_name": ["Edge1", "Edge2", "Cloud"][action],
            "edge1_load":  edge1_load,
            "edge2_load":  edge2_load,
            "cloud_load":  cloud_load,
            "task_demand": task_demand,
            "task_deadline": task_dead,
            "rationale":   rationale,
        }
        behaviors.append(behavior)
        print(f"  Case {test_case}: {behavior['action_name']} - {rationale}")

    action_counts = np.bincount([b["action"] for b in behaviors], minlength=N_ACTIONS)
    print("\nDecision Distribution:")
    for i, count in enumerate(action_counts):
        print(f"  {['Edge1','Edge2','Cloud'][i]}: {count}/10 ({count*10}%)")

    return behaviors


def generate_performance_report(model_path="models/checkpoints/dqn_best.pth"):
    """Tạo comprehensive performance report"""
    print("GENERATING COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    comparison_results, detailed_results = test_model_vs_baselines(model_path)
    if comparison_results is None:
        return None

    scenario_results  = test_different_scenarios(model_path)
    behavior_analysis = analyze_learning_behavior(model_path)

    dqn    = comparison_results["DQN (Trained)"]
    random = comparison_results["Random"]

    print(f"\nEXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"Model Avg Reward     : {dqn['avg_reward']:.1f}")
    print(f"Improvement vs Random: +{dqn['avg_reward'] - random['avg_reward']:.1f}")
    print(f"Average Latency      : {dqn['avg_latency']:.1f} ms")
    print(f"SLA Rate             : {dqn['avg_sla_rate']:.1%}")
    print(f"Miss Rate            : {dqn['avg_miss_rate']:.1%}")

    ratio = dqn["avg_reward"] / random["avg_reward"] if random["avg_reward"] != 0 else 1.0
    if ratio > 3.0:
        assessment = "EXCELLENT - Model significantly outperforms baselines"
    elif ratio > 2.0:
        assessment = "VERY GOOD - Model shows strong learning"
    elif ratio > 1.5:
        assessment = "GOOD - Model performs better than random"
    elif ratio > 1.1:
        assessment = "MARGINAL - Model shows slight improvement"
    else:
        assessment = "POOR - Model needs improvement"

    print(f"\nASSESSMENT: {assessment}")

    return {
        "comparison":  comparison_results,
        "scenarios":   scenario_results,
        "behavior":    behavior_analysis,
        "assessment":  assessment,
    }


if __name__ == "__main__":
    print("COMPREHENSIVE MODEL EVALUATION\n")

    model_path = "models/checkpoints/dqn_best.pth"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Train first: python -m rl_training.train_dqn")
        sys.exit(1)

    print("Starting evaluation...\n")
    try:
        report = generate_performance_report(model_path)
        if report:
            print("\nEVALUATION COMPLETE!")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
