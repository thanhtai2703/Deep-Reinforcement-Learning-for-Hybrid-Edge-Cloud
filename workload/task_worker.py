"""
task_worker.py
==============
HTTP worker chạy trên mỗi node (Edge1, Edge2, Cloud).
Nhận task từ Dispatcher, thực sự dùng CPU để chứng minh tác động thật.

Chạy trên mỗi node:
    pip install fastapi uvicorn
    python task_worker.py --port 8765

Test:
    curl -X POST http://localhost:8765/task \
         -H "Content-Type: application/json" \
         -d '{"task_id":"t1","cpu_requirement":60,"ram_requirement":30,"deadline_ms":300}'
"""

import argparse
import math
import socket
import time
import threading
from datetime import datetime, timezone

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise SystemExit("pip install fastapi uvicorn")

app = FastAPI(title="Task Worker")

_hostname = socket.gethostname()
_completed = 0
_lock = threading.Lock()


class TaskRequest(BaseModel):
    task_id: str
    cpu_requirement: float   # % CPU (5-90)
    ram_requirement: float   # % RAM (5-80)
    deadline_ms: float


class TaskResponse(BaseModel):
    task_id: str
    node: str
    status: str
    duration_ms: float
    cpu_requirement: float


@app.get("/health")
def health():
    return {"status": "ok", "node": _hostname, "completed": _completed}


@app.post("/task", response_model=TaskResponse)
def run_task(req: TaskRequest):
    """
    Thực sự dùng CPU tương ứng với cpu_requirement.
    cpu_requirement=60 → chạy ~0.6s CPU-intensive work.
    """
    t0 = time.perf_counter()

    # Tính thời gian chạy: cpu_req% → cpu_req/100 * 1.5 giây
    work_duration = (req.cpu_requirement / 100.0) * 1.5

    # CPU-intensive work để Node Exporter capture được
    deadline = time.perf_counter() + work_duration
    while time.perf_counter() < deadline:
        # Tính toán thực sự để dùng CPU
        _ = sum(math.sqrt(i) for i in range(2000))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    global _completed
    with _lock:
        _completed += 1

    sla_met = elapsed_ms <= req.deadline_ms
    print(
        f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
        f"{req.task_id} | cpu={req.cpu_requirement:.0f}% | "
        f"took={elapsed_ms:.0f}ms | deadline={req.deadline_ms:.0f}ms | "
        f"SLA={'OK' if sla_met else 'MISS'}"
    )

    return TaskResponse(
        task_id=req.task_id,
        node=_hostname,
        status="completed",
        duration_ms=round(elapsed_ms, 1),
        cpu_requirement=req.cpu_requirement,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"Worker starting on {_hostname}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
