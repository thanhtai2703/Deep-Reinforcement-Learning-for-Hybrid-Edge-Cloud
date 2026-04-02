-- Week 1 database schema for hybrid edge-cloud dispatcher
-- Compatible with SQLite and PostgreSQL (minor type differences are acceptable)

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
