from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

# Cấu hình logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class PrometheusAPIError(Exception):
    """Raised when Prometheus returns an error or unexpected payload."""

@dataclass
class PrometheusConfig:
    base_url: str
    timeout_seconds: int = 10

class PrometheusClient:
    def __init__(self, config: PrometheusConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def instant_query(self, query: str, time: Optional[str] = None) -> Dict[str, Any]:
        """Execute an instant Prometheus query."""
        url = f"{self.config.base_url.rstrip('/')}/api/v1/query"
        data: Dict[str, Any] = {"query": query}
        if time:
            data["time"] = time

        try:
            response = self.session.post(
                url,
                data=data,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logging.error(f"HTTP error querying Prometheus: {exc}")
            raise PrometheusAPIError(f"HTTP error when querying Prometheus: {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            logging.error("Prometheus response is not valid JSON")
            raise PrometheusAPIError("Prometheus response is not valid JSON") from exc

        if payload.get("status") != "success":
            logging.error(f"Prometheus query failed: {payload}")
            raise PrometheusAPIError(f"Prometheus query failed: {payload}")

        return payload

    @staticmethod
    def vector_to_map(payload: Dict[str, Any]) -> Dict[str, float]:
        """Convert Prometheus instant vector response to dict."""
        result = payload.get("data", {}).get("result", [])
        output: Dict[str, float] = {}

        for item in result:
            metric = item.get("metric", {})
            instance = metric.get("instance", "unknown")
            value = item.get("value")

            if not value or len(value) < 2:
                continue

            try:
                output[instance] = round(float(value[1]), 2) # Làm tròn 2 chữ số thập phân cho đẹp
            except (ValueError, TypeError):
                continue

        return output

    def get_cpu_usage_pct(self) -> Dict[str, float]:
        query = '100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])))'
        return self.vector_to_map(self.instant_query(query))

    def get_ram_usage_pct(self) -> Dict[str, float]:
        query = '100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))'
        return self.vector_to_map(self.instant_query(query))

    def get_queue_length(self) -> Dict[str, float]:
        query = "app_queue_length"
        return self.vector_to_map(self.instant_query(query))

    def get_queue_proxy(self) -> Dict[str, float]:
        query = 'node_load1 / count by (instance) (node_cpu_seconds_total{mode="idle"})'
        return self.vector_to_map(self.instant_query(query))

    def get_network_latency_ms(self) -> Dict[str, float]:
        query = "edge_cloud_rtt_ms"
        return self.vector_to_map(self.instant_query(query))

    def get_dispatcher_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Hàm All-in-one: Gom tất cả metric lại và xử lý Fallback.
        Person 3 chỉ cần gọi đúng hàm này.
        """
        logging.info("Bắt đầu lấy dữ liệu metrics từ Prometheus...")
        cpu = self.get_cpu_usage_pct()
        ram = self.get_ram_usage_pct()
        queue_real = self.get_queue_length()
        queue_proxy = self.get_queue_proxy()
        latency = self.get_network_latency_ms()

        # Gom danh sách các máy (node) từ CPU và RAM
        instances = set(cpu.keys()).union(set(ram.keys()))
        result = {}

        for inst in instances:
            # Lọc bỏ các instance không phải là node_exporter (ví dụ bản thân tiến trình prometheus)
            if not inst.endswith(":9100"):
                continue

            # Xử lý fallback cho Queue: Nếu không có real queue thì lấy proxy
            q_val = queue_real.get(inst)
            if q_val is None:
                q_val = queue_proxy.get(inst)
            
            # Latency (Nếu không có thì để None -> tương đương null trong JSON)
            lat_val = latency.get(inst)

            result[inst] = {
                "cpu_usage_pct": cpu.get(inst),
                "ram_usage_pct": ram.get(inst),
                "queue_length": q_val,
                "network_latency_ms": lat_val
            }
        
        logging.info(f"Đã lấy thành công dữ liệu của {len(result)} nodes.")
        return result

def build_client() -> PrometheusClient:
    base_url = os.getenv("PROMETHEUS_BASE_URL", "http://localhost:9090")
    timeout = int(os.getenv("PROMETHEUS_TIMEOUT", "10"))
    return PrometheusClient(PrometheusConfig(base_url=base_url, timeout_seconds=timeout))

if __name__ == "__main__":
    client = build_client()
    import json
    
    print("\n=== KẾT QUẢ GỘP DÀNH CHO DISPATCHER (NGÀY 4) ===")
    final_data = client.get_dispatcher_metrics()
    print(json.dumps(final_data, indent=2))
