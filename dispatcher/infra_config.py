"""
infra_config.py
===============
Cấu hình IP thực của hạ tầng Edge-Cloud.
Chỉnh sửa file này khi IP thay đổi.
"""

PROMETHEUS_URL = "http://localhost:9090"

# Map Prometheus instance (IP:port) → role trong dispatcher
# Key = instance label trong Prometheus (từ node_exporter)
INSTANCE_MAP = {
    "100.82.147.9:9100":  "edge_1",
    "100.69.169.33:9100": "edge_2",
    "<EC2_IP>:9100":      "cloud",   # ← thay bằng IP EC2 thật
}
