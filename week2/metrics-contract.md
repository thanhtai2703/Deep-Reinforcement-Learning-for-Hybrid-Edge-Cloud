# Metrics Contract v0

## Purpose
This contract defines the structured metrics payload used by Dispatcher.

## Data source
Primary Prometheus endpoint: http://<EDGE_PROMETHEUS>:9090

## Response shape
{
  "timestamp": "ISO-8601 UTC string",
  "cluster": "hybrid-lab",
  "nodes": [
    {
      "instance": "100.82.147.9:9100",
      "role": "edge",
      "cpu_usage_pct": 42.1,
      "ram_usage_pct": 67.5,
      "queue_length": null,
      "network_latency_ms": null
    }
  ]
}

## Field meanings
- instance: Prometheus instance label
- role: edge | cloud
- cpu_usage_pct: real metric from Node Exporter
- ram_usage_pct: real metric from Node Exporter
- queue_length: null for v0 unless sample app exports app_queue_length
- network_latency_ms: null for v0 unless custom probe/exporter exists

## Notes
- v0 focuses on stable infrastructure metrics first
- extra state dimensions can be added later without breaking the structure
