# PHÂN CÔNG - PERSON 3: BACKEND/INTEGRATION ENGINEER

**Vai trò**: Xây dựng Smart Dispatcher, tích hợp RL model với K8s, phát triển toàn bộ backend

**Timeline**: 4 tuần

## 🎯 TRÁCH NHIỆM CHÍNH

- Thiết kế kiến trúc hệ thống tổng thể
- Xây dựng Task Generator (workload simulation)
- Phát triển Smart Dispatcher (core brain)
- Tích hợp RL model với production
- Tích hợp Kubernetes API
- Setup database, logging, documentation
- Viết deployment guide

---

## 📅 TUẦN 1: ARCHITECTURE & TASK GENERATOR

### ⛓️ Dependencies

**Cần từ team:**

- 📥 Person 2: State space definition (14 dimensions) để thiết kế architecture & database schema
  - **Timing**: Thứ 4 tuần 1 (Person 2 giao)
- 📥 Person 1: Kubeconfig files để access cluster
- 📥 Person 1: Prometheus endpoint URL

**Phải giao cho team:**

- ✅ Thứ 3 → Person 2: Architecture document (giúp Person 2 hiểu hệ thống)
- ✅ Cuối tuần → Person 2: Task format specification
- ✅ Cuối tuần → All: `README.md` với project structure

### Công việc

**1. Thiết kế System Architecture**

- Vẽ architecture diagram:
  - Task Queue → Dispatcher → K8s (Edge/Cloud)
  - Prometheus → Dispatcher (metrics)
  - Database ← Dispatcher (logs)
- Xác định data flow
- Define interfaces giữa components
- Decide technology stack
  **2. Thiết kế Database Schema**
- Table: `tasks`
  - id, arrival_time, deadline, cpu_requirement
  - assigned_node, completion_time, status
- Table: `decisions` (optional)
  - timestamp, state_vector, action, reward
- Chọn database: PostgreSQL hoặc SQLite

**3. Implement Task Generator**

- Generate synthetic tasks với:
  - Arrival time (Poisson process)
  - CPU requirement (random [0.1, 2.0] cores)
  - Deadline (relative: +5s, +10s, +20s)
  - Priority (optional: high/medium/low)
- Support multiple patterns:
  - Constant rate: 10 tasks/min
  - Bursty: 50 tasks/min trong 1 phút, sau đó nghỉ
  - Diurnal: Tăng giảm theo giờ

**4. Setup Git Repository**

- Initialize project structure
- Setup .gitignore
- Create README skeleton
- Define branch strategy (main, dev)

**5. Setup CI/CD Basics**

- GitHub Actions workflow
- Linting (flake8, black)
- Basic tests (pytest)

### Công cụ

- Python 3.10+
- PostgreSQL/SQLite
- Git, GitHub
- Draw.io hoặc Lucidchart (diagrams)

### Output

```
✅ File: docs/architecture.md với diagrams
✅ File: database/schema.sql
✅ File: workload/task_generator.py
   - generate_constant_load()
   - generate_bursty_load()
   - generate_diurnal_load()
✅ File: README.md với project structure
✅ File: .github/workflows/ci.yml
✅ Database: tasks table created
```

---

## 📅 TUẦN 2: DISPATCHER PROTOTYPE

### ⛓️ Dependencies

**Cần từ team:**

- 📥 **CRITICAL** Person 1: `monitoring/prometheus_client.py`
  - **Timing**: Giữa tuần 2 (cannot proceed without this)
  - **Action**: Hỏi Person 1 sớm nếu chưa nhận
- 📥 Person 2: State normalization format (phải match với Gymnasium env)
  - **Timing**: Cuối tuần 2
- 📥 Person 2: `rl_env/baseline_policies.py` để test

**Phải giao cho team:**

- ✅ Cuối tuần → Person 2: Feedback về state format (có đúng format không)

### Công việc

**1. Implement Dispatcher Core**

- Class: `SmartDispatcher`
- Constructor: Initialize policy (baseline hoặc rl)
- Method: `get_current_state()`
  - Query Prometheus (từ Person 1)
  - Build state vector [14 dimensions]
- Method: `make_decision(state)`
  - Load policy (baseline)
  - Return action (0/1/2)
- Method: `execute_action(action, task_id)`
  - Log decision vào database
  - (Chưa deploy pod, để tuần 3)
- Method: `run_loop()`
  - Main loop: query state → decide → execute

**2. Implement State Builder**

- Normalize metrics từ Prometheus
- CPU/RAM: [0, 100] → [0, 1]
- Latency: [0, 200ms] → [0, 1]
- Queue: [0, 20] → [0, 1]
- Handle missing data (use cached/default values)

**3. Integrate với Prometheus Client**

- Use `prometheus_client.py` từ Person 1
- Parse metrics response
- Build state vector format match Gymnasium env

**4. Implement Baseline Policy Testing**

- Load Round Robin policy
- Load Least Loaded policy
- Run 100 simulated tasks qua mỗi policy
- Log results vào database
- Compare avg latency

**5. Tạo CLI Tool**

- Command-line interface để test dispatcher
- `python dispatcher_cli.py --policy round_robin --num-tasks 50`
- Output: Statistics, logs

**6. Setup Database Layer**

- SQLAlchemy ORM hoặc raw SQL
- Insert/query tasks
- Log decisions

### Công cụ

- Python
- requests (HTTP client)
- SQLAlchemy (ORM)
- Click hoặc argparse (CLI)

### Output

```
✅ File: dispatcher/smart_dispatcher.py (~400 lines)
   - class SmartDispatcher
   - Main loop implementation
✅ File: dispatcher/state_builder.py
   - normalize_state()
   - handle_missing_data()
✅ File: dispatcher/dispatcher_cli.py
   - CLI tool
✅ File: database/log_decisions.py
   - Database operations
✅ Test results: 100 tasks qua Round Robin, 100 qua Least Loaded
✅ Logged to database: task_id, assigned_node, latency
```

---

## 📅 TUẦN 3: FULL INTEGRATION

### ⛓️ Dependencies

**Cần từ team:**

- 📥 **CRITICAL** Person 2: `models/best_model.zip`
  - **Timing**: Thứ 2 tuần 3 (cannot integrate RL without model)
  - **Action**: Coordinate với Person 2 để nhận sớm
- 📥 **CRITICAL** Person 1: `k8s_integration/pod_deployer.py`
  - **Timing**: Thứ 3 tuần 3 (need to deploy pods)
  - **Action**: Sync với Person 1 về interface
- 📥 Person 1: Docker image `task-processor:v1`
- 📥 Person 2: Model inference documentation

**Phải giao cho team:**

- ✅ Cuối tuần → Person 1: Dispatcher running (Person 1 cần để làm dashboard)
- ✅ Cuối tuần → Person 2: System ready để benchmark (Person 2 cần cho tuần 4)

### Công việc

**1. Integrate RL Model**

- Load trained model từ Person 2: `models/best_model.zip`
- Use Stable-Baselines3 để load: `PPO.load()`
- Implement: `model.predict(state)` → action
- Add model hot-reload (reload model without restart)

**2. Integrate Kubernetes Deployer**

- Use `pod_deployer.py` từ Person 1
- Connect Dispatcher decision với K8s deployment
- Call: `deploy_task_pod(task_id, target_node)`
- Handle deployment response

**3. Implement End-to-End Flow**

- Full workflow:
  1. Task arrives (from generator)
  2. Get system state (Prometheus)
  3. Normalize state
  4. RL model decides
  5. Deploy pod via K8s
  6. Wait for completion
  7. Log metrics to database
- Handle concurrent tasks (threading/async)

**4. Error Handling & Retry Logic**

- Prometheus timeout → use cached state
- K8s API failure → retry 3x with exponential backoff
- Model inference error → fallback to Least Loaded
- Database error → log to file

**5. Performance Profiling**

- Measure latency của từng step:
  - State query: ~50ms
  - Model inference: ~5ms
  - K8s API call: ~150ms
  - DB logging: ~20ms
- Optimize bottlenecks nếu cần

**6. Add REST API (Optional)**

- FastAPI hoặc Flask
- Endpoints:
  - `POST /submit_task`: Submit new task
  - `GET /task_status/{id}`: Query task status
  - `GET /stats`: Get statistics
- Useful cho external integration

**7. End-to-End Testing**

- Test 200 tasks end-to-end
- Mix of Edge & Cloud assignments
- Verify success rate >95%
- Verify decisions logged correctly

### Công cụ

- Stable-Baselines3 (load model)
- kubernetes Python client
- FastAPI (optional REST API)
- threading/asyncio (concurrency)

### Output

```
✅ File: dispatcher/smart_dispatcher_v2.py (full version)
   - RL model integrated
   - K8s integration
   - Error handling
✅ File: dispatcher/model_loader.py
   - Hot-reload logic
✅ File: dispatcher/error_handlers.py
   - Retry logic, fallbacks
✅ File: api/dispatcher_api.py (optional)
   - REST API endpoints
✅ File: tests/end_to_end_test.py
✅ Test results: 200 tasks, 98.5% success rate
✅ Performance report: Total latency ~225ms per decision
```

---

## 📅 TUẦN 4: DOCUMENTATION & DEPLOYMENT

### ⛓️ Dependencies

**Cần từ team:**

- 📥 Person 1: Production dashboard hoàn chỉnh
- 📥 Person 2: Technical report & benchmark results
- 📥 Person 1: Demo script

**Phải giao cho team:**

- ✅ Giữa tuần → All: Complete documentation suite
- ✅ Cuối tuần → All: Docker Compose setup (one-command deployment)
- ✅ Cuối tuần → All: Final codebase cleaned & tested

### Công việc

**1. Viết Comprehensive README**

- Project overview
- Architecture diagram
- Quick start guide (5 commands → running system)
- How to run demo
- Troubleshooting section

**2. API Documentation**

- Document Prometheus metrics API
- Document Dispatcher REST API (nếu có)
- Document Task Generator API
- Include examples, curl commands

**3. Troubleshooting Guide**

- Common errors & solutions:
  - "Dispatcher không connect được Prometheus"
  - "K8s pods stuck in pending"
  - "RL model inference slow"
  - "Database connection timeout"
- Debug tips

**4. Deployment Guide**

- Prerequisites checklist
- Step-by-step setup:
  - Setup infrastructure (reference Person 1)
  - Deploy monitoring
  - Setup database
  - Configure dispatcher
  - Run system
- Configuration templates
- Security considerations (API keys, network)

**5. Docker Compose Setup**

- Containerize all components:
  - Prometheus
  - Grafana
  - PostgreSQL
  - Dispatcher
  - Task Generator
- One-command deployment: `docker-compose up -d`
- Volume mounts cho data persistence

**6. Code Cleanup**

- Add docstrings to all functions
- Type hints (Python 3.10+)
- Remove debug print statements
- Consistent naming conventions
- Format code (black, flake8)

**7. Unit Tests**

- Test state_builder functions
- Test database operations
- Test error handlers
- Test CLI
- Target: >70% code coverage

**8. Final Integration Testing**

- Fresh install test (on new VM)
- Deploy với docker-compose
- Run 500 tasks end-to-end
- Verify dashboard updates correctly
- Verify logs readable
- Verify documentation accurate

### Công cụ

- Docker, Docker Compose
- pytest (testing)
- black, flake8 (code formatting)
- pytest-cov (coverage)
- Markdown (documentation)

### Output

```
✅ File: README.md (comprehensive, 200+ lines)
   - Quick start guide
   - Architecture overview
   - Links to other docs
✅ File: docs/API_DOCS.md
   - All API endpoints documented
   - Examples included
✅ File: docs/TROUBLESHOOTING.md
   - 10+ common issues & solutions
✅ File: docs/DEPLOYMENT_GUIDE.md
   - Step-by-step setup
   - Configuration templates
✅ File: docker-compose.yml
   - All services defined
   - Networks, volumes configured
✅ File: scripts/run_tests.sh
   - Run all tests
✅ Test coverage: >70%
✅ Code quality: flake8 passes
✅ Fresh install test: Success ✅
```

---

## 🔧 CÔNG CỤ TỔNG HỢP

### Backend Development

- **Python 3.10+**: Main language
- **SQLAlchemy**: ORM
- **FastAPI**: REST API (optional)
- **Click**: CLI framework

### Integration

- **kubernetes**: Python K8s client
- **requests**: HTTP client
- **stable-baselines3**: Load RL model

### Database

- **PostgreSQL**: Production database
- **SQLite**: Development/testing

### Testing

- **pytest**: Testing framework
- **pytest-cov**: Coverage
- **unittest.mock**: Mocking

### DevOps

- **Docker**: Containerization
- **Docker Compose**: Multi-container
- **GitHub Actions**: CI/CD

### Code Quality

- **black**: Code formatter
- **flake8**: Linter
- **mypy**: Type checking

---

## 📊 WEEKLY CHECKPOINTS

### Tuần 1 - Thứ 6

- [ ] Present: Architecture diagram
- [ ] Demo: Task generator producing 100 tasks
- [ ] Deliver: architecture.md, task_generator.py

### Tuần 2 - Thứ 6

- [ ] Demo: Dispatcher CLI với baseline policies
- [ ] Show: Database với logged decisions
- [ ] Deliver: smart_dispatcher.py

### Tuần 3 - Thứ 6

- [ ] Demo: End-to-end 200 tasks (RL → K8s)
- [ ] Show: Performance profiling results
- [ ] Deliver: smart_dispatcher_v2.py

### Tuần 4 - Thứ 6

- [ ] Demo: Docker Compose deployment
- [ ] Deliver: Complete documentation suite
- [ ] Fresh install test: Pass ✅

---

## ⚠️ THÁCH THỨC CÓ THỂ GẶP

| Issue                     | Solution                                          |
| ------------------------- | ------------------------------------------------- |
| Dispatcher quá chậm       | Profile code, optimize bottlenecks, cache metrics |
| K8s API timeout           | Increase timeout, implement retry logic           |
| Database bottleneck       | Add indexes, use connection pooling               |
| Model loading chậm        | Cache model in memory, lazy loading               |
| Concurrent tasks conflict | Use proper locking, queue-based processing        |
| Docker networking issues  | Use docker-compose networks, check ports          |

---

## ⛓️ DEPENDENCY TIMELINE TỔNG HỢP

```
TUẦN 1:
  Person 3 (bạn)
    │
    ├──← Nhận: State definition từ Person 2 (Thứ 4)
    ├──← Nhận: Kubeconfig từ Person 1
    ├──← Nhận: Prometheus URL từ Person 1
    │
    ├──→ Thứ 3: Giao Architecture doc → Person 2
    ├──→ Cuối tuần: Giao Task format → Person 2
    └──→ Cuối tuần: Giao README.md → All

TUẦN 2:
  Person 3 (bạn)
    │
    ├──← Nhận: prometheus_client.py từ Person 1 (Giữa tuần) [CRITICAL]
    ├──← Nhận: State normalization từ Person 2
    ├──← Nhận: baseline_policies.py từ Person 2
    │
    └──→ Cuối tuần: Feedback state format → Person 2

TUẦN 3:  [⚠️ NHIỀU DEPENDENCIES NHẤT]
  Person 3 (bạn)
    │
    ├──← Nhận: best_model.zip từ Person 2 (Thứ 2) [CRITICAL]
    ├──← Nhận: pod_deployer.py từ Person 1 (Thứ 3) [CRITICAL]
    ├──← Nhận: Docker image từ Person 1 (Giữa tuần)
    ├──← Nhận: Model inference docs từ Person 2
    │
    ├──→ Cuối tuần: Giao Dispatcher running → Person 1
    └──→ Cuối tuần: Giao System ready → Person 2

TUẦN 4:
  Person 3 (bạn)
    │
    ├──← Nhận: Dashboard từ Person 1
    ├──← Nhận: Technical report từ Person 2
    ├──← Nhận: Demo script từ Person 1
    │
    ├──→ Giữa tuần: Giao Complete docs → All
    └──→ Cuối tuần: Giao Docker Compose setup → All
```

---

## 🤝 COORDINATION VỚI TEAM

**Person 1 (Infrastructure)**

- Nhận: Prometheus client, K8s deployer, metrics format
- Cung cấp: Dispatcher requirements, API specs

**Person 2 (RL Engineer)**

- Nhận: Trained model, state format
- Cung cấp: Integration requirements, inference needs

**Daily Standup**: 9:00 AM - Sync on integration issues

---

## ✅ DEFINITION OF DONE

- [ ] Smart Dispatcher running stable
- [ ] RL model integrated và working
- [ ] K8s integration success rate >95%
- [ ] End-to-end test với 500 tasks: Pass
- [ ] Documentation complete (README, API, Troubleshooting, Deployment)
- [ ] Docker Compose deployment works
- [ ] Code coverage >70%
- [ ] Fresh install test: Pass
- [ ] All deliverables committed to Git

---

## 💡 TIPS

- **Start Simple**: Baseline policies trước, RL sau
- **Test Components Separately**: Unit test trước integration test
- **Mock External Dependencies**: Mock Prometheus, K8s khi test
- **Log Everything**: Logging giúp debug rất nhiều
- **Handle Failures Gracefully**: Always có fallback
- **Document as You Go**: Đừng để cuối project

---

## 📦 FINAL DELIVERABLES CHECKLIST

- [ ] dispatcher/smart_dispatcher_v2.py
- [ ] dispatcher/state_builder.py
- [ ] dispatcher/model_loader.py
- [ ] dispatcher/error_handlers.py
- [ ] workload/task_generator.py
- [ ] database/schema.sql
- [ ] database/log_decisions.py
- [ ] api/dispatcher_api.py (optional)
- [ ] tests/end_to_end_test.py
- [ ] docker-compose.yml
- [ ] README.md
- [ ] docs/API_DOCS.md
- [ ] docs/TROUBLESHOOTING.md
- [ ] docs/DEPLOYMENT_GUIDE.md
- [ ] docs/architecture.md

---

**Estimated Effort**: 40-50 giờ/tuần × 4 tuần = 160-200 giờ total
