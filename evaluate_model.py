"""
Comprehensive Model Evaluation - Kiểm tra Model hoạt động tốt như nào
Person 2: ML/RL Engineer - Model Performance Analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from rl_env.edge_cloud_env import EdgeCloudEnv
from rl_env.baseline_policies import (
    RoundRobinPolicy, 
    LeastLoadedPolicy, 
    GreedyLocalFirstPolicy,
    TaskAwarePolicy
)


def test_model_vs_baselines(model_path="models/dqn_100k.zip"):
    """So sánh model với tất cả baselines"""
    print("📊 MODEL vs BASELINES COMPARISON")
    print("=" * 60)
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"❌ Model không tồn tại: {model_path}")
        return
    
    model = DQN.load(model_path)
    env = EdgeCloudEnv(n_edge_nodes=2, max_steps=200)
    
    # Define policies to compare
    policies = {
        'Random': 'random',
        'Round Robin': RoundRobinPolicy(),
        'Least Loaded': LeastLoadedPolicy(),
        'Greedy Local': GreedyLocalFirstPolicy(),
        'Task Aware': TaskAwarePolicy(),
        'DQN (Trained)': model
    }
    
    results = {}
    detailed_results = []
    
    print("Testing policies (10 episodes each)...")
    
    for policy_name, policy in policies.items():
        print(f"\n🧪 Testing {policy_name}...")
        
        episode_rewards = []
        episode_latencies = []
        episode_completion_rates = []
        episode_miss_rates = []
        action_distributions = []
        
        for episode in range(10):
            obs, info = env.reset(seed=episode * 100)
            if hasattr(policy, 'reset'):
                policy.reset()
            
            episode_reward = 0
            episode_actions = []
            latencies = []
            total_tasks = 0
            completed_tasks = 0
            deadline_misses = 0
            
            for step in range(200):
                # Get action from policy
                if policy_name == 'Random':
                    action = env.action_space.sample()
                elif hasattr(policy, 'name'):  # Baseline policies
                    action = policy.predict(obs)
                else:  # DQN model
                    action, _ = policy.predict(obs, deterministic=True)
                    action = int(action)
                
                episode_actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                total_tasks += 1
                
                if info['task_completed']:
                    completed_tasks += 1
                    latencies.append(info['actual_latency'])
                
                if info['deadline_missed']:
                    deadline_misses += 1
                
                if terminated or truncated:
                    break
            
            # Calculate episode metrics
            episode_rewards.append(episode_reward)
            episode_latencies.append(np.mean(latencies) if latencies else 0)
            episode_completion_rates.append(completed_tasks / total_tasks if total_tasks > 0 else 0)
            episode_miss_rates.append(deadline_misses / total_tasks if total_tasks > 0 else 0)
            
            # Action distribution
            action_counts = np.bincount(episode_actions, minlength=3)
            action_percentages = action_counts / len(episode_actions) * 100
            action_distributions.append(action_percentages)
        
        # Calculate overall statistics
        results[policy_name] = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_latency': np.mean(episode_latencies),
            'avg_completion_rate': np.mean(episode_completion_rates),
            'avg_miss_rate': np.mean(episode_miss_rates),
            'action_dist': np.mean(action_distributions, axis=0)
        }
        
        # Store detailed results
        detailed_results.append({
            'policy': policy_name,
            'rewards': episode_rewards,
            'latencies': episode_latencies,
            'completion_rates': episode_completion_rates,
            'miss_rates': episode_miss_rates
        })
        
        print(f"  Avg Reward: {results[policy_name]['avg_reward']:.1f}")
        print(f"  Avg Latency: {results[policy_name]['avg_latency']:.1f}ms")
        print(f"  Completion Rate: {results[policy_name]['avg_completion_rate']:.1%}")
        print(f"  Miss Rate: {results[policy_name]['avg_miss_rate']:.1%}")
    
    # Print comparison table
    print(f"\n📋 COMPARISON TABLE")
    print("-" * 100)
    print(f"{'Policy':<15} {'Reward':<15} {'Latency':<15} {'Completion':<15} {'Miss Rate':<15}")
    print("-" * 100)
    
    for policy_name, stats in results.items():
        print(f"{policy_name:<15} "
              f"{stats['avg_reward']:8.1f}±{stats['std_reward']:4.1f}   "
              f"{stats['avg_latency']:8.1f}ms      "
              f"{stats['avg_completion_rate']:9.1%}      "
              f"{stats['avg_miss_rate']:9.1%}")
    
    # Find best policy
    best_policy = max(results.keys(), key=lambda k: results[k]['avg_reward'])
    dqn_rank = list(results.keys()).index('DQN (Trained)') + 1
    
    print(f"\n🏆 RANKING:")
    sorted_policies = sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    for i, (policy, stats) in enumerate(sorted_policies, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
        print(f"  {emoji} #{i}: {policy} - {stats['avg_reward']:.1f} reward")
    
    # DQN Analysis
    dqn_stats = results['DQN (Trained)']
    random_stats = results['Random']
    improvement = dqn_stats['avg_reward'] - random_stats['avg_reward']
    
    print(f"\n🤖 DQN ANALYSIS:")
    print(f"  Rank: #{dqn_rank} out of {len(results)}")
    print(f"  Improvement over Random: +{improvement:.1f} ({improvement/random_stats['avg_reward']*100:.0f}%)")
    print(f"  Action Distribution: Edge1={dqn_stats['action_dist'][0]:.0f}%, Edge2={dqn_stats['action_dist'][1]:.0f}%, Cloud={dqn_stats['action_dist'][2]:.0f}%")
    
    if dqn_rank == 1:
        print("  ✅ EXCELLENT: DQN is the best policy!")
    elif dqn_rank <= 3:
        print("  ✅ GOOD: DQN is among top 3 policies")
    else:
        print("  ⚠️ NEEDS IMPROVEMENT: DQN underperforming")
    
    return results, detailed_results


def test_different_scenarios(model_path="models/dqn_100k.zip"):
    """Test model trong các scenarios khác nhau"""
    print("\n🎭 SCENARIO TESTING")
    print("=" * 60)
    
    model = DQN.load(model_path)
    
    scenarios = [
        ("Normal Load", 500, 0.3),
        ("High Load", 500, 0.7), 
        ("Burst Load", 100, 0.9),
        ("Light Load", 1000, 0.1)
    ]
    
    scenario_results = {}
    
    for scenario_name, episode_length, load_factor in scenarios:
        print(f"\n🔬 Testing {scenario_name}...")
        
        env = EdgeCloudEnv(n_edge_nodes=2, max_steps=episode_length)
        
        # Modify environment for scenario
        if load_factor > 0.5:  # High load
            env.node_states['edge1']['cpu_util'] = np.random.uniform(0.4, 0.8) 
            env.node_states['edge2']['cpu_util'] = np.random.uniform(0.4, 0.8)
        
        rewards = []
        latencies = []
        completion_rates = []
        
        for episode in range(5):
            obs, info = env.reset(seed=episode)
            episode_reward = 0
            episode_latencies = []
            completed = 0
            total = 0
            
            for step in range(episode_length):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                
                episode_reward += reward
                total += 1
                
                if info['task_completed']:
                    completed += 1
                    episode_latencies.append(info['actual_latency'])
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            latencies.append(np.mean(episode_latencies) if episode_latencies else 0)
            completion_rates.append(completed / total if total > 0 else 0)
        
        scenario_results[scenario_name] = {
            'avg_reward': np.mean(rewards),
            'avg_latency': np.mean(latencies),
            'avg_completion': np.mean(completion_rates)
        }
        
        print(f"  Reward: {np.mean(rewards):.1f}")
        print(f"  Latency: {np.mean(latencies):.1f}ms")
        print(f"  Completion: {np.mean(completion_rates):.1%}")
    
    return scenario_results


def analyze_learning_behavior(model_path="models/dqn_100k.zip"):
    """Phân tích behavior của model"""
    print("\n🧠 BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    model = DQN.load(model_path)
    env = EdgeCloudEnv(n_edge_nodes=2, max_steps=50)
    
    # Test different states
    behaviors = []
    
    print("Testing decision-making in different situations...")
    
    for test_case in range(10):
        obs, info = env.reset(seed=test_case)
        
        # Get model's action and Q-values (if possible)
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        # Analyze current state
        edge1_load = obs[0] + obs[1]  # CPU + RAM
        edge2_load = obs[4] + obs[5]
        cloud_load = obs[8] + obs[9]
        task_demand = obs[11]
        task_deadline = obs[12]
        
        behavior = {
            'test_case': test_case,
            'action': action,
            'action_name': ['Edge1', 'Edge2', 'Cloud'][action],
            'edge1_load': edge1_load,
            'edge2_load': edge2_load,  
            'cloud_load': cloud_load,
            'task_demand': task_demand,
            'task_deadline': task_deadline,
            'decision_rationale': ''
        }
        
        # Analyze decision rationale
        if action == 0:  # Edge1
            if edge1_load < edge2_load:
                behavior['decision_rationale'] = "Chose less loaded edge node"
            else:
                behavior['decision_rationale'] = "Preferred edge despite higher load"
        elif action == 1:  # Edge2
            if edge2_load < edge1_load:
                behavior['decision_rationale'] = "Chose less loaded edge node"
            else:
                behavior['decision_rationale'] = "Preferred edge despite higher load"
        else:  # Cloud
            if task_demand > 0.7:
                behavior['decision_rationale'] = "High-demand task to cloud"
            elif edge1_load > 0.8 and edge2_load > 0.8:
                behavior['decision_rationale'] = "Edge nodes overloaded"
            else:
                behavior['decision_rationale'] = "Unexpected cloud choice"
        
        behaviors.append(behavior)
        
        print(f"Case {test_case}: {behavior['action_name']} - {behavior['decision_rationale']}")
    
    # Summary behavior analysis
    action_counts = [b['action'] for b in behaviors]
    action_distribution = np.bincount(action_counts, minlength=3)
    
    print(f"\nDecision Distribution:")
    for i, count in enumerate(action_distribution):
        node_name = ['Edge1', 'Edge2', 'Cloud'][i]
        print(f"  {node_name}: {count}/10 ({count*10}%)")
    
    return behaviors


def generate_performance_report():
    """Tạo comprehensive performance report"""
    print("📄 GENERATING COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 60)
    
    model_path = "models/dqn_100k.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Run all tests
    comparison_results, detailed_results = test_model_vs_baselines(model_path)
    scenario_results = test_different_scenarios(model_path)
    behavior_analysis = analyze_learning_behavior(model_path)
    
    # Generate summary
    dqn_stats = comparison_results['DQN (Trained)']
    random_stats = comparison_results['Random']
    
    print(f"\n📊 EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"✅ Model Performance: {dqn_stats['avg_reward']:.1f} avg reward")
    print(f"✅ Improvement over Random: +{(dqn_stats['avg_reward'] - random_stats['avg_reward']):.1f}")
    print(f"✅ Average Latency: {dqn_stats['avg_latency']:.1f}ms")
    print(f"✅ Task Completion Rate: {dqn_stats['avg_completion_rate']:.1%}")
    print(f"✅ Deadline Miss Rate: {dqn_stats['avg_miss_rate']:.1%}")
    
    # Performance assessment
    improvement_ratio = dqn_stats['avg_reward'] / random_stats['avg_reward']
    
    if improvement_ratio > 3.0:
        assessment = "🌟 EXCELLENT - Model significantly outperforms baselines"
    elif improvement_ratio > 2.0:
        assessment = "✅ VERY GOOD - Model shows strong learning"
    elif improvement_ratio > 1.5:
        assessment = "✅ GOOD - Model performs better than random"
    elif improvement_ratio > 1.1:
        assessment = "⚠️ MARGINAL - Model shows slight improvement"
    else:
        assessment = "❌ POOR - Model needs improvement"
    
    print(f"\n🎯 ASSESSMENT: {assessment}")
    
    return {
        'comparison': comparison_results,
        'scenarios': scenario_results,
        'behavior': behavior_analysis,
        'assessment': assessment
    }


if __name__ == "__main__":
    print("🚀 COMPREHENSIVE MODEL EVALUATION")
    print("\nThis will test your trained model thoroughly...")
    
    # Check if model exists
    if not os.path.exists("models/dqn_100k.zip"):
        print("❌ Model not found! Train a model first:")
        print("python -m rl_training.train_dqn")
        exit()
    
    print("\nStarting evaluation (takes ~2-3 minutes)...\n")
    
    # Run comprehensive evaluation
    try:
        report = generate_performance_report()
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print("Your model has been thoroughly tested across:")
        print("- ✅ Baseline comparisons")
        print("- ✅ Different load scenarios") 
        print("- ✅ Decision-making analysis")
        print("- ✅ Performance assessment")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()