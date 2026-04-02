"""
Quick Learning Test - Check if training setup works
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from rl_env.edge_cloud_env import EdgeCloudEnv

def simple_learning_test():
    """Simple test để kiểm tra learning"""
    print("🧪 Simple Learning Test")
    
    env = EdgeCloudEnv(n_edge_nodes=2, max_steps=50)  # Shorter episodes
    
    # Test random policy trước
    print("\n1. Random Policy Baseline:")
    random_rewards = []
    for _ in range(3):
        obs, info = env.reset()
        episode_reward = 0
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(episode_reward)
    
    avg_random = np.mean(random_rewards)
    print(f"Random policy avg reward: {avg_random:.2f}")
    
    # Train model  
    print("\n2. Training DQN (15K steps):")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-2,          # Very high learning rate
        buffer_size=5000,            # Small buffer
        learning_starts=100,         # Start early
        batch_size=64,
        gamma=0.9,                   # Low gamma
        train_freq=1,                # Update every step
        target_update_interval=100,  # Frequent target updates
        exploration_fraction=0.9,    # Heavy exploration
        exploration_final_eps=0.05,  # More exploration
        verbose=1
    )
    
    model.learn(total_timesteps=15000, progress_bar=True)
    
    # Test trained model
    print("\n3. Testing Trained Model:")
    trained_rewards, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    avg_trained = np.mean(trained_rewards)
    
    print(f"Trained model avg reward: {avg_trained:.2f}")
    print(f"Improvement: {avg_trained - avg_random:+.2f}")
    
    # Action distribution
    print("\n4. Action Analysis:")
    actions = []
    obs, info = env.reset()
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            obs, info = env.reset()
    
    action_counts = np.bincount(actions, minlength=3)
    action_names = ['Edge1', 'Edge2', 'Cloud']
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        pct = count / len(actions) * 100
        print(f"{name}: {count}/{len(actions)} ({pct:.1f}%)")
    
    # Success criteria
    if avg_trained > avg_random + 1.0:  # Need meaningful improvement
        print(f"\n✅ SUCCESS: Model learned! (+{avg_trained - avg_random:.2f} improvement)")
        return True
    else:
        print(f"\n❌ FAIL: Model didn't learn effectively")
        return False

if __name__ == "__main__":
    simple_learning_test()