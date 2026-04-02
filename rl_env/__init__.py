"""
rl_env/__init__.py
==================
Đăng ký môi trường EdgeCloudEnv với Gymnasium registry
để có thể gọi gym.make("EdgeCloud-v0").
"""

from gymnasium.envs.registration import register

register(
    id="EdgeCloud-v0",
    entry_point="rl_env.edge_cloud_env:EdgeCloudEnv",
    kwargs={
        "n_edge_nodes": 2,
        "use_prometheus": False,
        "max_steps": 200,
    },
    max_episode_steps=200,
)