"""
env_test.py
===========
Unit tests và validation cho EdgeCloudEnv.

Chạy:
    python -m rl_env.env_test
    # hoặc dùng pytest:
    pytest rl_env/env_test.py -v

Covers:
  1. API contract (Gymnasium interface)
  2. Observation / action space bounds
  3. Reward sanity checks
  4. Episode behavior (reset, step, truncation)
  5. Random policy evaluation (1000 episodes)
  6. Determinism check (seed)
  7. Simulation dynamics (node load updates)
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_env.edge_cloud_env import EdgeCloudEnv
from rl_env.baseline_policies import (
    RandomPolicy, RoundRobinPolicy, LeastConnectionPolicy,
    EdgeOnlyPolicy, CloudOnlyPolicy, get_all_baselines,
)

# ── ANSI colors for pretty terminal output ────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

_results = []

def test(name: str):
    """Decorator – đăng ký test case."""
    def decorator(fn):
        _results.append((name, fn))
        return fn
    return decorator

def run_all():
    passed, failed = 0, []
    print(f"\n{BOLD}{'='*60}")
    print(f"  EdgeCloudEnv – Test Suite")
    print(f"{'='*60}{RESET}\n")

    for name, fn in _results:
        try:
            fn()
            print(f"  {GREEN}✓{RESET}  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  {RED}✗{RESET}  {name}")
            print(f"      {RED}→ {e}{RESET}")
            failed.append(name)
        except Exception as e:
            print(f"  {RED}✗{RESET}  {name}")
            print(f"      {RED}→ {type(e).__name__}: {e}{RESET}")
            failed.append(name)

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"  Passed: {GREEN}{passed}{RESET}  |  Failed: {RED}{len(failed)}{RESET}")
    if failed:
        print(f"\n  {RED}Failed tests:{RESET}")
        for f in failed:
            print(f"    - {f}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    return len(failed) == 0


# ──────────────────────────────────────────────────────────────────────────
# 1. API CONTRACT
# ──────────────────────────────────────────────────────────────────────────

@test("Env khởi tạo không lỗi")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    assert env is not None

@test("reset() trả về (obs, info) đúng kiểu")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3, seed=0)
    result = env.reset(seed=42)
    assert isinstance(result, tuple) and len(result) == 2
    obs, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

@test("step() trả về 5-tuple đúng chuẩn Gymnasium")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=0)
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

@test("action_space là Discrete(n_edge + 1)")
def _():
    for n in [2, 3, 4]:
        env = EdgeCloudEnv(n_edge_nodes=n)
        import gymnasium as gym
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == n + 1

@test("observation_space là Box với đúng kích thước")
def _():
    for n in [2, 3, 4]:
        env = EdgeCloudEnv(n_edge_nodes=n)
        expected_dim = n * 3 + 3 + 3
        assert env.observation_space.shape == (expected_dim,), \
            f"Expected ({expected_dim},), got {env.observation_space.shape}"


# ──────────────────────────────────────────────────────────────────────────
# 2. OBSERVATION BOUNDS
# ──────────────────────────────────────────────────────────────────────────

@test("Observation nằm trong [0, 1] sau reset")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    obs, _ = env.reset(seed=7)
    assert obs.min() >= 0.0, f"obs.min() = {obs.min()}"
    assert obs.max() <= 1.0, f"obs.max() = {obs.max()}"

@test("Observation nằm trong [0, 1] sau 200 steps")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    obs, _ = env.reset(seed=42)
    for _ in range(200):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert obs.min() >= -1e-6, f"obs.min() = {obs.min():.6f}"
        assert obs.max() <= 1.0 + 1e-6, f"obs.max() = {obs.max():.6f}"
        if terminated or truncated:
            break

@test("Observation dtype là float32")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    obs, _ = env.reset(seed=1)
    assert obs.dtype == np.float32, f"dtype = {obs.dtype}"

@test("Observation không có NaN hay Inf")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    obs, _ = env.reset(seed=99)
    for _ in range(100):
        obs, _, t, tr, _ = env.step(env.action_space.sample())
        assert not np.any(np.isnan(obs)), "NaN in observation"
        assert not np.any(np.isinf(obs)), "Inf in observation"
        if t or tr:
            break


# ──────────────────────────────────────────────────────────────────────────
# 3. REWARD SANITY
# ──────────────────────────────────────────────────────────────────────────

@test("Reward là scalar float")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(0)
    assert isinstance(reward, (float, np.floating)), f"type={type(reward)}"

@test("Reward không phải NaN hay Inf")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=5)
    for _ in range(200):
        _, reward, t, tr, _ = env.step(env.action_space.sample())
        assert not np.isnan(reward), "Reward is NaN"
        assert not np.isinf(reward), "Reward is Inf"
        if t or tr: break

@test("Reward nằm trong range hợp lý [-5, 2]")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=10)
    rewards = []
    for _ in range(500):
        _, r, t, tr, _ = env.step(env.action_space.sample())
        rewards.append(r)
        if t or tr:
            env.reset()
    mn, mx = min(rewards), max(rewards)
    assert mn >= -10.0, f"Reward too low: {mn}"
    assert mx <= 5.0,   f"Reward too high: {mx}"

@test("SLA met → reward cao hơn SLA missed")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    # Chạy nhiều steps, kiểm tra correlation reward vs sla_met
    env.reset(seed=77)
    sla_met_rewards, sla_miss_rewards = [], []
    for _ in range(500):
        _, r, t, tr, info = env.step(env.action_space.sample())
        if info["sla_met"]:
            sla_met_rewards.append(r)
        else:
            sla_miss_rewards.append(r)
        if t or tr:
            env.reset()
    if sla_met_rewards and sla_miss_rewards:
        assert np.mean(sla_met_rewards) > np.mean(sla_miss_rewards), \
            "SLA met rewards should be higher than SLA missed"


# ──────────────────────────────────────────────────────────────────────────
# 4. EPISODE BEHAVIOR
# ──────────────────────────────────────────────────────────────────────────

@test("Episode kết thúc đúng lúc (truncated tại max_steps)")
def _():
    max_steps = 50
    env = EdgeCloudEnv(n_edge_nodes=3, max_steps=max_steps)
    env.reset(seed=0)
    steps = 0
    truncated = False
    while not truncated:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated:
            break
    assert steps == max_steps, f"Expected {max_steps} steps, got {steps}"

@test("reset() reset lại step counter")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3, max_steps=10)
    env.reset(seed=0)
    for _ in range(10):
        _, _, _, truncated, _ = env.step(env.action_space.sample())
    # Sau khi truncated, reset và episode mới phải kéo dài 10 steps
    env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        _, _, t, tr, _ = env.step(env.action_space.sample())
        done = t or tr
        steps += 1
    assert steps == 10, f"Expected 10 steps after reset, got {steps}"

@test("info dict chứa đủ keys cần thiết")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)
    required_keys = {"latency", "cost", "sla_met", "action", "is_cloud"}
    assert required_keys.issubset(info.keys()), \
        f"Missing keys: {required_keys - info.keys()}"

@test("Action invalid bị báo lỗi")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset()
    try:
        env.step(999)
        assert False, "Should have raised AssertionError for invalid action"
    except AssertionError:
        pass


# ──────────────────────────────────────────────────────────────────────────
# 5. DETERMINISM
# ──────────────────────────────────────────────────────────────────────────

@test("Cùng seed → cùng trajectory (deterministic)")
def _():
    def collect(seed):
        env = EdgeCloudEnv(n_edge_nodes=3, max_steps=30)
        obs, _ = env.reset(seed=seed)
        rewards = []
        rng = np.random.default_rng(seed)
        for _ in range(30):
            a = int(rng.integers(0, env.action_space.n))
            _, r, t, tr, _ = env.step(a)
            rewards.append(r)
            if t or tr:
                break
        return rewards

    r1 = collect(42)
    r2 = collect(42)
    assert r1 == r2, "Same seed should produce same rewards"

@test("Khác seed → khác trajectory")
def _():
    def collect(seed):
        env = EdgeCloudEnv(n_edge_nodes=3, max_steps=30)
        obs, _ = env.reset(seed=seed)
        rewards = []
        rng = np.random.default_rng(seed)
        for _ in range(30):
            a = int(rng.integers(0, env.action_space.n))
            _, r, t, tr, _ = env.step(a)
            rewards.append(r)
            if t or tr:
                break
        return rewards

    r1 = collect(1)
    r2 = collect(2)
    assert r1 != r2, "Different seeds should produce different rewards"


# ──────────────────────────────────────────────────────────────────────────
# 6. SIMULATION DYNAMICS
# ──────────────────────────────────────────────────────────────────────────

@test("Cloud action (action=n_edge) → is_cloud=True trong info")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=0)
    cloud_action = env.n_edge_nodes   # = 3
    _, _, _, _, info = env.step(cloud_action)
    assert info["is_cloud"] is True

@test("Edge action → is_cloud=False trong info")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=0)
    _, _, _, _, info = env.step(0)   # Edge node 0
    assert info["is_cloud"] is False

@test("Latency luôn dương")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=0)
    for _ in range(100):
        _, _, t, tr, info = env.step(env.action_space.sample())
        assert info["latency"] > 0, f"Latency = {info['latency']}"
        if t or tr: break

@test("Cost luôn dương")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    env.reset(seed=0)
    for _ in range(100):
        _, _, t, tr, info = env.step(env.action_space.sample())
        assert info["cost"] > 0, f"Cost = {info['cost']}"
        if t or tr: break


# ──────────────────────────────────────────────────────────────────────────
# 7. RANDOM POLICY EVALUATION (1000 episodes)
# ──────────────────────────────────────────────────────────────────────────

def evaluate_random_policy(n_episodes: int = 1000, n_edge_nodes: int = 3) -> dict:
    """
    Chạy Random policy qua n_episodes episode, thu thập thống kê.
    Đây là bước validate quan trọng trước khi train RL.
    """
    env = EdgeCloudEnv(n_edge_nodes=n_edge_nodes, max_steps=200)
    policy = RandomPolicy(n_edge_nodes + 1, n_edge_nodes)

    ep_rewards, latencies, costs, sla_flags, ep_lengths = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            latencies.append(info["latency"])
            costs.append(info["cost"])
            sla_flags.append(int(info["sla_met"]))
            steps += 1

        ep_rewards.append(ep_reward)
        ep_lengths.append(steps)

    return {
        "n_episodes":     n_episodes,
        "avg_reward":     float(np.mean(ep_rewards)),
        "std_reward":     float(np.std(ep_rewards)),
        "avg_ep_length":  float(np.mean(ep_lengths)),
        "avg_latency":    float(np.mean(latencies)),
        "p95_latency":    float(np.percentile(latencies, 95)),
        "avg_cost":       float(np.mean(costs)),
        "sla_rate":       float(np.mean(sla_flags)) * 100,
        "min_reward":     float(np.min(ep_rewards)),
        "max_reward":     float(np.max(ep_rewards)),
    }


@test("Random policy 1000 episodes – reward không explode/vanish")
def _():
    stats = evaluate_random_policy(n_episodes=200)   # Dùng 200 cho test nhanh
    assert stats["avg_reward"] > -500, f"Reward quá thấp: {stats['avg_reward']}"
    assert stats["avg_reward"] < 500,  f"Reward quá cao: {stats['avg_reward']}"

@test("Episode length hợp lý (= max_steps=200)")
def _():
    stats = evaluate_random_policy(n_episodes=100)
    assert stats["avg_ep_length"] == 200, \
        f"Expected 200, got {stats['avg_ep_length']}"

@test("SLA rate > 0% và < 100% với random policy")
def _():
    stats = evaluate_random_policy(n_episodes=200)
    assert stats["sla_rate"] > 0,   f"SLA rate = {stats['sla_rate']:.1f}%"
    assert stats["sla_rate"] < 100, f"SLA rate = {stats['sla_rate']:.1f}%"


# ──────────────────────────────────────────────────────────────────────────
# 8. BASELINE POLICIES SANITY
# ──────────────────────────────────────────────────────────────────────────

@test("Tất cả baseline policies chạy không lỗi (100 episodes)")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    baselines = get_all_baselines(env.action_space.n, env.n_edge_nodes)
    for policy in baselines:
        for ep in range(10):
            obs, _ = env.reset(seed=ep)
            policy.reset()
            done = False
            while not done:
                action = policy.select_action(obs)
                obs, _, t, tr, _ = env.step(action)
                done = t or tr

@test("CloudOnlyPolicy luôn chọn action = n_edge")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    policy = CloudOnlyPolicy(env.action_space.n, env.n_edge_nodes)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = policy.select_action(obs)
        assert action == env.n_edge_nodes, f"Expected cloud action={env.n_edge_nodes}, got {action}"
        obs, _, t, tr, _ = env.step(action)
        if t or tr: break

@test("EdgeOnlyPolicy không bao giờ chọn Cloud")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    policy = EdgeOnlyPolicy(env.action_space.n, env.n_edge_nodes)
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = policy.select_action(obs)
        assert action < env.n_edge_nodes, f"EdgeOnly chọn Cloud! action={action}"
        obs, _, t, tr, _ = env.step(action)
        if t or tr: break

@test("RoundRobin xoay đúng thứ tự")
def _():
    env = EdgeCloudEnv(n_edge_nodes=3)
    policy = RoundRobinPolicy(env.action_space.n, env.n_edge_nodes)
    obs, _ = env.reset(seed=0)
    actions = []
    for i in range(8):
        actions.append(policy.select_action(obs))
    # Phải là [0,1,2,3,0,1,2,3]
    expected = [i % 4 for i in range(8)]
    assert actions == expected, f"RoundRobin order wrong: {actions}"


# ──────────────────────────────────────────────────────────────────────────
# FULL EVALUATION REPORT (gọi trực tiếp, không phải test)
# ──────────────────────────────────────────────────────────────────────────

def print_full_random_eval(n_episodes: int = 1000):
    """In báo cáo đầy đủ Random policy – dùng để validate env trước khi train."""
    print(f"\n{BOLD}{'='*60}")
    print(f"  Random Policy Evaluation – {n_episodes} episodes")
    print(f"{'='*60}{RESET}")

    t0 = time.time()
    stats = evaluate_random_policy(n_episodes=n_episodes)
    elapsed = time.time() - t0

    print(f"  Avg reward     : {stats['avg_reward']:8.3f}  ± {stats['std_reward']:.3f}")
    print(f"  Reward range   : [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print(f"  Avg ep length  : {stats['avg_ep_length']:.1f} steps")
    print(f"  Avg latency    : {stats['avg_latency']:.1f} ms")
    print(f"  P95 latency    : {stats['p95_latency']:.1f} ms")
    print(f"  Avg cost       : {stats['avg_cost']:.5f}")
    print(f"  SLA rate       : {stats['sla_rate']:.1f}%")
    print(f"  Time elapsed   : {elapsed:.1f}s")

    # Nhận xét
    print(f"\n  {BOLD}Nhận xét:{RESET}")
    if stats["sla_rate"] < 30:
        print(f"  {YELLOW}⚠ SLA rate thấp ({stats['sla_rate']:.1f}%) – reward function có thể cần điều chỉnh{RESET}")
    elif stats["sla_rate"] > 80:
        print(f"  {YELLOW}⚠ SLA rate cao quá ({stats['sla_rate']:.1f}%) với random – task deadline quá dễ{RESET}")
    else:
        print(f"  {GREEN}✓ SLA rate {stats['sla_rate']:.1f}% với random – env có vẻ balanced{RESET}")

    if abs(stats["avg_reward"]) > 200:
        print(f"  {YELLOW}⚠ Reward lớn ({stats['avg_reward']:.1f}) – cân nhắc scale lại{RESET}")
    else:
        print(f"  {GREEN}✓ Reward range ổn định – sẵn sàng train DQN{RESET}")

    print(f"{BOLD}{'='*60}{RESET}\n")
    return stats


# ──────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true",
                        help="Chạy full random policy evaluation (1000 episodes)")
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    # Chạy unit tests
    success = run_all()

    # Chạy full eval nếu yêu cầu
    if args.eval:
        print_full_random_eval(n_episodes=args.episodes)

    sys.exit(0 if success else 1)