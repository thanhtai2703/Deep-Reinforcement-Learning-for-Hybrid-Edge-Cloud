from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prometheus_client import build_client

class MetricsResponse(BaseModel):
    timestamp: str
    cluster: str
    nodes: Dict[str, Dict[str, Optional[float]]]

app = FastAPI(
    title="Hybrid Edge-Cloud Metrics API",
    version="0.1.0",
    description="REST API cung cấp metrics đã gộp từ Prometheus trung tâm cho Dispatcher",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

client = build_client()

@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/api/v1/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    try:
        logging.info("API /api/v1/metrics được gọi")

        # Gọi đúng hàm chúng ta đã viết ở Ngày 4
        nodes = client.get_dispatcher_metrics()

        return MetricsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cluster=os.getenv("CLUSTER_NAME", "hybrid-lab"),
            nodes=nodes,
        )

    except Exception as exc:
        logging.exception("Lỗi khi lấy metrics từ Prometheus")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
