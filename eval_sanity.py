"""Quick sanity check: action distribution + sample decisions."""
import sys
sys.path.insert(0, '.')
from collections import Counter
import torch
from rl_env.edge_cloud_env_calibrated import EdgeCloudEnvCalibrated
from rl_env.edge_cloud_env import EdgeCloudEnv
from models.dqn_agent import DQNAgent, DQNConfig

CHECKPOINTS = [
    ("dqn_uncal", "models/checkpoints/dqn_uncalibrated/dqn_best.pth", EdgeCloudEnv),
    ("dqn_cal",   "models/checkpoints/dqn_calibrated/dqn_best.pth",   EdgeCloudEnvCalibrated),
]
ACTION_NAMES = {0: "edge_1", 1: "edge_2", 2: "cloud", 3: "reject"}

for name, ckpt, EnvCls in CHECKPOINTS:
    env = EnvCls(n_edge_nodes=2, max_steps=200)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, DQNConfig())
    agent.load(ckpt)

    counter = Counter()
    rewards, sla_count, n_steps = [], 0, 0
    for _ in range(20):
        obs, _ = env.reset()
        done = False
        while not done:
            a = agent.select_action(obs, greedy=True)
            counter[a] += 1
            obs, r, term, trunc, info = env.step(a)
            rewards.append(r)
            sla_count += int(info.get("sla_met", False))
            n_steps += 1
            done = term or trunc

    total = sum(counter.values())
    dist = {ACTION_NAMES[a]: f"{c/total*100:.1f}%" for a, c in counter.items()}
    print(f"\n{name}  (eval {n_steps} steps)")
    print(f"  Action dist: {dist}")
    print(f"  Avg reward/step: {sum(rewards)/len(rewards):.3f}")
    print(f"  SLA rate: {sla_count/n_steps*100:.1f}%")
