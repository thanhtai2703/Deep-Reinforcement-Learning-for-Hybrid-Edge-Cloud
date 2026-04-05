"""
log_decisions.py
================
Database operations layer cho Smart Dispatcher.

Hỗ trợ SQLite (development) và PostgreSQL (production).
Schema tham chiếu: database/schema.sql

Chức năng:
  - Insert/update tasks
  - Log dispatch decisions
  - Query statistics
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("Database")

# Default database path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dispatcher.db"
)

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "schema.sql"
)


class DatabaseManager:
    """
    Quản lý kết nối database và CRUD operations.

    Parameters
    ----------
    db_path : đường dẫn file SQLite (hoặc PostgreSQL connection string)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_schema()
        logger.info("Database initialized: %s", db_path)

    @contextmanager
    def _connect(self):
        """Context manager cho SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_schema(self):
        """Tạo tables nếu chưa tồn tại."""
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()
        else:
            schema_sql = _FALLBACK_SCHEMA

        with self._connect() as conn:
            conn.executescript(schema_sql)

    # ------------------------------------------------------------------
    # Task operations
    # ------------------------------------------------------------------

    def insert_task(
        self,
        task_id: str,
        arrival_time: str,
        deadline_ms: int,
        cpu_requirement: float,
        ram_requirement_mb: int = 0,
        priority: str = "medium",
        payload_type: str = "compute",
    ) -> None:
        """Insert một task mới vào database."""
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO tasks
                   (id, arrival_time, deadline_ms, cpu_requirement,
                    ram_requirement_mb, priority, payload_type, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (task_id, arrival_time, deadline_ms, cpu_requirement,
                 ram_requirement_mb, priority, payload_type),
            )

    def update_task_status(
        self,
        task_id: str,
        status: str,
        assigned_node: Optional[str] = None,
        execution_latency_ms: Optional[float] = None,
        deadline_met: Optional[bool] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Cập nhật trạng thái task."""
        now = datetime.now(timezone.utc).isoformat()
        updates = ["status = ?", "updated_at = ?"]
        params: list = [status, now]

        if assigned_node is not None:
            updates.append("assigned_node = ?")
            params.append(assigned_node)
        if status == "dispatched":
            updates.append("dispatch_time = ?")
            params.append(now)
        if status == "running":
            updates.append("start_time = ?")
            params.append(now)
        if status == "completed":
            updates.append("completion_time = ?")
            params.append(now)
        if execution_latency_ms is not None:
            updates.append("execution_latency_ms = ?")
            params.append(execution_latency_ms)
        if deadline_met is not None:
            updates.append("deadline_met = ?")
            params.append(int(deadline_met))
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(task_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
                params,
            )

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def log_decision(
        self,
        task_id: str,
        policy_name: str,
        state_vector: list,
        action: int,
        selected_node: str,
        reward: Optional[float] = None,
        q_values: Optional[list] = None,
        inference_latency_ms: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Log một dispatch decision."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO decisions
                   (task_id, policy_name, state_vector, action,
                    selected_node, reward, q_values,
                    inference_latency_ms, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    policy_name,
                    json.dumps(state_vector),
                    action,
                    selected_node,
                    reward,
                    json.dumps(q_values) if q_values else None,
                    inference_latency_ms,
                    notes,
                ),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[dict]:
        """Lấy thông tin một task."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_recent_tasks(self, limit: int = 50) -> List[dict]:
        """Lấy N tasks gần nhất."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY arrival_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_decisions_for_task(self, task_id: str) -> List[dict]:
        """Lấy tất cả decisions cho một task."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE task_id = ? ORDER BY decision_time",
                (task_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_statistics(self, policy_name: Optional[str] = None) -> dict:
        """
        Thống kê tổng hợp: tổng tasks, SLA rate, avg latency, node distribution.
        """
        with self._connect() as conn:
            where = ""
            params = ()
            if policy_name:
                where = "WHERE d.policy_name = ?"
                params = (policy_name,)

            # Basic counts
            row = conn.execute(
                f"""SELECT
                        COUNT(*) as total_tasks,
                        SUM(CASE WHEN t.deadline_met = 1 THEN 1 ELSE 0 END) as sla_met,
                        AVG(t.execution_latency_ms) as avg_latency,
                        COUNT(CASE WHEN t.status = 'completed' THEN 1 END) as completed,
                        COUNT(CASE WHEN t.status = 'failed' THEN 1 END) as failed
                    FROM tasks t
                    LEFT JOIN decisions d ON t.id = d.task_id
                    {where}""",
                params,
            ).fetchone()

            total = row["total_tasks"] or 0
            sla_met = row["sla_met"] or 0

            # Node distribution
            node_rows = conn.execute(
                f"""SELECT t.assigned_node, COUNT(*) as cnt
                    FROM tasks t
                    LEFT JOIN decisions d ON t.id = d.task_id
                    {where}
                    GROUP BY t.assigned_node""",
                params,
            ).fetchall()
            node_dist = {r["assigned_node"]: r["cnt"] for r in node_rows if r["assigned_node"]}

            return {
                "total_tasks": total,
                "completed": row["completed"] or 0,
                "failed": row["failed"] or 0,
                "sla_rate": (sla_met / total * 100) if total > 0 else 0.0,
                "avg_latency_ms": round(row["avg_latency"] or 0, 2),
                "node_distribution": node_dist,
            }

    def count_tasks_by_status(self) -> Dict[str, int]:
        """Đếm tasks theo status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status"
            ).fetchall()
            return {r["status"]: r["cnt"] for r in rows}


# Fallback schema nếu schema.sql không tìm thấy
_FALLBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    arrival_time TIMESTAMP NOT NULL,
    deadline_ms INTEGER NOT NULL CHECK (deadline_ms > 0),
    cpu_requirement REAL NOT NULL CHECK (cpu_requirement >= 0),
    ram_requirement_mb INTEGER NOT NULL CHECK (ram_requirement_mb >= 0),
    priority TEXT NOT NULL DEFAULT 'medium',
    payload_type TEXT NOT NULL DEFAULT 'compute',
    assigned_node TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    dispatch_time TIMESTAMP,
    start_time TIMESTAMP,
    completion_time TIMESTAMP,
    execution_latency_ms REAL,
    deadline_met INTEGER,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    decision_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    policy_name TEXT NOT NULL,
    state_vector TEXT NOT NULL,
    action INTEGER NOT NULL,
    selected_node TEXT NOT NULL,
    reward REAL,
    q_values TEXT,
    inference_latency_ms REAL,
    notes TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks (id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_arrival_time ON tasks (arrival_time);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_node ON tasks (assigned_node);
CREATE INDEX IF NOT EXISTS idx_decisions_task_id ON decisions (task_id);
CREATE INDEX IF NOT EXISTS idx_decisions_policy_name ON decisions (policy_name);
CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions (decision_time);
"""
