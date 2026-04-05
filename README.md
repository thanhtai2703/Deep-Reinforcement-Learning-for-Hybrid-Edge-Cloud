# Hybrid Edge-Cloud Task Dispatching with Reinforcement Learning

Hệ thống điều phối tác vụ thông minh trong môi trường Hybrid Edge-Cloud,
sử dụng Reinforcement Learning (DQN / PPO) để tối ưu latency, chi phí
và SLA compliance.

## Architecture

```
Task Generator ──► Task Queue ──► Smart Dispatcher ──► K3s / EC2
                                       │                  │
                              Prometheus Metrics ◄─────────┘
                                       │
                                   Database
                                       │
                              Grafana Dashboard
```

- **2 Edge VM** (K3s) + **1 Cloud node** (AWS EC2)
- **Dispatcher** nhận task, query metrics, dùng RL model chọn node tối ưu
- **Prometheus** thu thập CPU/RAM/Latency realtime
- **Grafana** hiển thị dashboard giám sát

## Project Structure

```
Project_RL/
├── rl_env/                     # Gymnasium environment + baselines
│   ├── edge_cloud_env.py       #   EdgeCloudEnv (MDP simulation)
│   ├── baseline_policies.py    #   6 baseline policies
│   ├── env_test.py             #   Unit tests (17+ cases)
│   └── __init__.py             #   Gymnasium registry
│
├── models/                     # RL agents
│   └── dqn_agent.py            #   Custom DQN (QNetwork + ReplayBuffer)
│
├── rl_training/                # Training & evaluation pipelines
│   ├── train_dqn.py            #   DQN training loop (1000 episodes)
│   ├── train_ppo.py            #   PPO via Stable-Baselines3
│   ├── evaluate.py             #   DQN vs baselines comparison
│   └── compare_models.py       #   DQN vs PPO statistical analysis
│
├── dispatcher/                 # Smart Dispatcher (production)
│   ├── state_builder.py        #   Prometheus → normalized state vector
│   ├── model_loader.py         #   Load DQN/PPO with hot-reload
│   ├── error_handlers.py       #   Retry, fallback, circuit breaker
│   ├── smart_dispatcher.py     #   Core dispatcher logic
│   └── dispatcher_cli.py       #   CLI tool
│
├── workload/                   # Task generation
│   └── task_generator.py       #   3 patterns: constant, bursty, diurnal
│
├── database/                   # Persistence
│   ├── schema.sql              #   DDL (tasks + decisions tables)
│   └── log_decisions.py        #   SQLite/PostgreSQL operations
│
├── api/                        # REST API
│   └── dispatcher_api.py       #   FastAPI endpoints
│
├── experiments/                # Experiment scripts
│   ├── smart_dispatcher.py     #   Demo dispatcher (standalone)
│   └── run_benchmark.py        #   1000-task benchmark
│
├── tests/                      # Integration tests
│   └── end_to_end_test.py      #   E2E pipeline test
│
├── docs/                       # Documentation
│   ├── architecture.md         #   System architecture
│   ├── mdp-design.md           #   MDP formulation
│   ├── API_DOCS.md             #   API reference
│   ├── DEPLOYMENT_GUIDE.md     #   Setup & deploy
│   └── TROUBLESHOOTING.md      #   Common issues
│
├── scripts/
│   └── run_tests.sh            #   Run all tests
│
├── docker-compose.yml          #   One-command deployment
├── requirements.txt            #   Python dependencies
└── .github/workflows/ci.yml    #   CI pipeline
```

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd Project_RL
pip install -r requirements.txt

# 2. Validate environment
python rl_env/env_test.py

# 3. Generate sample workload
python workload/task_generator.py --pattern constant --count 100 --output workload/tasks.csv

# 4. Train DQN (quick test)
python -m rl_training.train_dqn

# 5. Evaluate
python -m rl_training.evaluate --model models/checkpoints/dqn_best.pth

# 6. Run dispatcher (demo mode)
python -m dispatcher.dispatcher_cli --policy dqn --num-tasks 50 --demo
```

## Team

| Person | Role | Scope |
|--------|------|-------|
| Person 1 | Infrastructure Lead | K3s, AWS, Prometheus, Grafana |
| Person 2 | ML/RL Engineer | MDP, Gymnasium env, DQN/PPO training |
| Person 3 | Backend/Integration | Dispatcher, API, DB, Docker, Docs |

## Tech Stack

- **Python 3.10+**, Gymnasium, Stable-Baselines3, PyTorch
- **K3s** (Edge), **AWS EC2** (Cloud)
- **Prometheus** + **Grafana** (Monitoring)
- **SQLite/PostgreSQL** (Persistence)
- **FastAPI** (REST API)
- **Docker Compose** (Deployment)

## License

Research project - Internal use only.
