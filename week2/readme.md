**"Handoff Metrics API v0 cho Dispatcher:**

- **API Endpoint:** `http://100.82.147.9:8000/api/v1/metrics`
- **Tài liệu Swagger:** `http://100.82.147.9:8000/docs`
- **Status:** Hiện tại `cpu_usage_pct` và `ram_usage_pct` là dữ liệu THẬT lấy từ Central Prometheus (đã test load thành công). Các trường `queue_length` và `network_latency_ms` tạm thời trả về `null` trong bản v0 này cho đến khi có App thực tế.
