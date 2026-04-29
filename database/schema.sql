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

-- Execution logs: denormalized table for sim calibration.
-- One row per dispatch+execution. Wide schema, JSON blobs for parametric
-- fields (node count varies). Designed for offline analysis (notebooks),
-- not for serving queries.
CREATE TABLE IF NOT EXISTS execution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Task spec (raw, for refit)
    cpu_requirement REAL,
    ram_requirement REAL,
    deadline_ms REAL,
    priority TEXT,
    payload_type TEXT,

    -- Dispatch decision
    policy_name TEXT,
    action INTEGER,
    selected_node TEXT,
    target_role TEXT,
    cpu_req_k8s TEXT,
    ram_req_k8s TEXT,

    -- Node state at dispatch (the X for fitting)
    metrics_summary_json TEXT,           -- full dict from state_builder
    selected_cpu REAL,                   -- denormalized for fast queries
    selected_ram REAL,
    selected_latency REAL,
    selected_queue REAL,

    -- Sim env's prediction at dispatch (for sim-vs-real comparison)
    est_latency_ms REAL,
    est_cost REAL,
    est_sla_met INTEGER,

    -- Real execution outcome (the Y)
    exec_backend TEXT,                   -- k8s / http / skipped
    exec_status TEXT,                    -- succeeded / failed / timeout / skipped
    total_ms INTEGER,
    submit_overhead_ms INTEGER,
    container_startup_ms INTEGER,
    exec_time_ms INTEGER,
    poll_overhead_ms INTEGER,
    pod_node TEXT,                       -- actual K8s node name

    sla_met_real INTEGER                 -- recomputed from total_ms vs deadline
);

CREATE INDEX IF NOT EXISTS idx_exec_logs_task ON execution_logs (task_id);
CREATE INDEX IF NOT EXISTS idx_exec_logs_policy ON execution_logs (policy_name);
CREATE INDEX IF NOT EXISTS idx_exec_logs_target ON execution_logs (target_role);
CREATE INDEX IF NOT EXISTS idx_exec_logs_status ON execution_logs (exec_status);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_arrival_time ON tasks (arrival_time);
CREATE INDEX IF NOT EXISTS idx_tasks_assigned_node ON tasks (assigned_node);
CREATE INDEX IF NOT EXISTS idx_decisions_task_id ON decisions (task_id);
CREATE INDEX IF NOT EXISTS idx_decisions_policy_name ON decisions (policy_name);
CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions (decision_time);
