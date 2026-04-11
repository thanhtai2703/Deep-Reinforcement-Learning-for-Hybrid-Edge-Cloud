# Metrics API v0

## Base URL
http://<metrics-api-host>:8000

## Endpoints

### 1. Health check
GET /healthz

Response:
{
  "status": "ok"
}

### 2. Aggregated metrics
GET /api/v1/metrics

Response:
{
  "timestamp": "ISO-8601 UTC string",
  "cluster": "hybrid-lab",
  "nodes": {
    "<instance>": {
      "cpu_usage_pct": 0.0,
      "ram_usage_pct": 0.0,
      "queue_length": null,
      "network_latency_ms": null
    }
  }
}

## Notes
- cpu_usage_pct and ram_usage_pct are real metrics from central Prometheus
- queue_length is null unless app_queue_length is exported later
- network_latency_ms is null unless custom latency metric is exported later
## Current status
- cpu_usage_pct and ram_usage_pct are real metrics from central Prometheus
- sample workloads were deployed on edge-1, edge-2, and cloud-1
- metrics API successfully reflects node resource changes during test load
- queue_length is still null because app_queue_length has not been instrumented yet
- network_latency_ms is still null because no dedicated latency exporter/probe has been added yet
