"""
infra_config.py
===============
Cấu hình IP thực của hạ tầng Edge-Cloud.
Chỉnh sửa file này khi IP thay đổi.
"""

PROMETHEUS_URL = "http://localhost:9090"

# Map Prometheus instance (IP:port) → role trong dispatcher
INSTANCE_MAP = {
    "100.82.147.9:9100":  "edge_1",
    "100.69.169.33:9100": "edge_2",
    # Khớp với target trong monitoring/prometheus.yml job "cloud".
    # Khi đổi target ở đó, đổi ở đây.
    "host.docker.internal:9100": "cloud",
}

# HTTP worker endpoint trên mỗi node (task_worker.py)
WORKER_URLS = {
    "edge_1": "http://100.82.147.9:8765/task",
    "edge_2": "http://100.69.169.33:8765/task",
    "cloud":  "http://localhost:8765/task",
}
