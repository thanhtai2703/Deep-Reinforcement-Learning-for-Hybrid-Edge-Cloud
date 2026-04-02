"""
Validation Script - Kiểm tra DQN Training có hoạt động đúng không
Person 2: ML/RL Engineer - Training Validation
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from rl_env.edge_cloud_env import EdgeCloudEnv
from rl_env.baseline_policies import RoundRobinPolicy, LeastLoadedPolicy


def check_random_policy():
    """Kiểm tra random policy baseline performance"""
    print("🎲 Testing Random Policy...")
    
    env = EdgeCloudEnv(n_edge_nodes=2, max_steps=100)
    
    rewards = []
    completion_rates = []
    
    for episode in range(10):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        completed = 0
        total = 0
        
        for step in range(100):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            total += 1
            if info['task_completed']:
                completed += 1
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        completion_rates.append(completed / total if total > 0 else 0)
    
    avg_reward = np.mean(rewards)
    avg_completion = np.mean(completion_rates)
    
    print(f"Random Policy Results:")
    print(f"- Average reward: {avg_reward:.3f}")
    print(f"- Average completion rate: {avg_completion:.1%}")
    
    return avg_reward, avg_completion


def train_quick_model():
    """Train một model nhỏ để test"""
    print("\n🚂 Training Quick Model (10000 steps)...")
    
    env = EdgeCloudEnv(n_edge_nodes=2, max_steps=200)
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-3,             # Much higher learning rate
        buffer_size=20000,              # Smaller buffer for faster learning  
        learning_starts=200,            # Start learning earlier
        batch_size=64,                  # Larger batch size
        gamma=0.95,                     # Lower gamma for shorter horizon
        train_freq=1,                   # Update every step
        target_update_interval=200,     # More frequent target updates
        exploration_fraction=0.8,       # More exploration
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,      # Higher final exploration
        verbose=0
    )
    
    start_time = time.time()
    
    # Train
    model.learn(total_timesteps=10000, progress_bar=True)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    return model


def evaluate_learning_progress():
    """Kiểm tra model có học được không"""
    print("\n📈 Testing Learning Progress...")
    
    env = EdgeCloudEnv(n_edge_nodes=2, max_steps=100)
    
    # Train model
    model = train_quick_model()
    
    # Test at different training stages
    checkpoints = [0, 2000, 5000, 10000]
    results = []
    
    for checkpoint in checkpoints:
        if checkpoint == 0:
            # Random policy
            rewards = []
            for _ in range(5):
                obs, info = env.reset()
                episode_reward = 0
                for step in range(100):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    if terminated or truncated:
                        break
                rewards.append(episode_reward)
            
            avg_reward = np.mean(rewards)
            
        else:
            # Partially trained model (approximate)
            temp_model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                buffer_size=10000,
                learning_starts=min(500, checkpoint//2),
                verbose=0
            )
            temp_model.learn(total_timesteps=checkpoint, progress_bar=False)
            
            # Evaluate
            episode_rewards, _ = evaluate_policy(
                temp_model, env, n_eval_episodes=5, deterministic=True
            )
            avg_reward = np.mean(episode_rewards)
        
        results.append(avg_reward)
        print(f"Steps {checkpoint:4d}: Avg Reward = {avg_reward:.3f}")
    
    # Check if learning
    initial_reward = results[0]
    final_reward = results[-1]
    improvement = final_reward - initial_reward
    
    print(f"\n📊 Learning Analysis:")
    print(f"- Initial reward (random): {initial_reward:.3f}")
    print(f"- Final reward (10K steps): {final_reward:.3f}")
    print(f"- Improvement: {improvement:+.3f}")
    
    if improvement > 0.1:
        print("✅ Model is learning! Reward is improving")
        return True
    elif improvement > 0:
        print("⚠️ Model shows slight improvement")
        return True
    else:
        print("❌ Model may not be learning effectively")
        return False


def compare_with_baselines():
    """So sánh trained model với baselines"""
    print("\n⚖️ Comparing with Baselines...")
    
    env = EdgeCloudEnv(n_edge_nodes=2, max_steps=100)
    
    # Train quick model
    model = train_quick_model()
    
    # Test policies
    policies = {
        'Random': 'random',
        'Round Robin': RoundRobinPolicy(),
        'Least Loaded': LeastLoadedPolicy(),
        'DQN (10K steps)': model
    }
    
    results = {}
    
    for name, policy in policies.items():
        rewards = []
        
        for episode in range(5):
            obs, info = env.reset(seed=episode * 10)
            episode_reward = 0
            
            if hasattr(policy, 'reset'):
                policy.reset()
            
            for step in range(100):
                if name == 'Random':
                    action = env.action_space.sample()
                elif hasattr(policy, 'predict') and hasattr(policy, 'name'):
                    action = policy.predict(obs)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                    action = int(action)  # Convert numpy array to int
                    action = int(action)  # Convert numpy array to int
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
        
        results[name] = np.mean(rewards)
        print(f"{name:15}: {np.mean(rewards):7.3f}")
    
    # Check if DQN beats random
    dqn_reward = results['DQN (10K steps)']
    random_reward = results['Random']
    
    if dqn_reward > random_reward:
        print(f"\n✅ DQN beats random by {dqn_reward - random_reward:+.3f}")
        return True
    else:
        print(f"\n⚠️ DQN performance similar to random")
        return False


def run_full_validation():
    """Chạy full validation suite"""
    print("🔍 FULL TRAINING VALIDATION")
    print("=" * 50)
    
    # Check 1: Random baseline
    random_reward, random_completion = check_random_policy()
    
    # Check 2: Learning progress
    is_learning = evaluate_learning_progress()
    
    # Check 3: Baseline comparison
    beats_random = compare_with_baselines()
    
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY:")
    print(f"✓ Random baseline: {random_reward:.3f} reward, {random_completion:.1%} completion")
    print(f"✓ Learning progress: {'Yes' if is_learning else 'No'}")
    print(f"✓ Beats random: {'Yes' if beats_random else 'No'}")
    
    if is_learning and beats_random:
        print("\n🎉 SUCCESS: Training is working correctly!")
        print("✅ Your DQN model is learning and improving")
        return True
    else:
        print("\n⚠️ ISSUES DETECTED:")
        if not is_learning:
            print("- Model may not be learning effectively")
            print("- Try: increase learning rate, reduce complexity, check reward function")
        if not beats_random:
            print("- Model performance not better than random")
            print("- Try: longer training, better hyperparameters")
        return False


if __name__ == "__main__":
    print("🚀 DQN Training Validation Script")
    print("\nThis will test if your training setup works correctly...")
    print("(Takes ~2-3 minutes to complete)")
    
    input("\nPress Enter to start validation...")
    
    success = run_full_validation()
    
    if success:
        print("\n🚀 You're ready for full training!")
        print("Run: python -m rl_training.train_dqn")
    else:
        print("\n🔧 Fix issues before full training")
        print("Check hyperparameters and environment setup")