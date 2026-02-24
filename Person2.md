# PHÂN CÔNG - PERSON 2: ML/RL ENGINEER

**Vai trò**: Thiết kế MDP, xây dựng môi trường Gymnasium, train RL model

**Timeline**: 4 tuần

---

## **Quick reminder**: Model training cần thời gian - bắt đầu sớm tuần 2!

## 🎯 TRÁCH NHIỆM CHÍNH

- Thiết kế bài toán RL (State, Action, Reward)
- Implement Gymnasium environment (simulation)
- Implement baseline policies để so sánh
- Train RL models (DQN, PPO)
- Evaluate và chọn model tốt nhất
- Viết technical report

---

## 📅 TUẦN 1: MDP DESIGN & SETUP

### ⛓️ Dependencies

**Cần từ team:**

- 📥 Person 1: Metrics format (CPU/RAM/Latency ranges) để thiết kế State space chính xác
  - **Timing**: Cuối tuần 1 (có thể làm song song, hỏi Person 1 sớm)

**Phải giao cho team:**

- ✅ Thứ 4 → Person 3: State space definition (14 dimensions format)
- ✅ Thứ 4 → Person 1: State features list (Person 1 cần biết query metrics gì)
- ✅ Cuối tuần → Person 3: `rl_env/baseline_policies.py`
- ✅ Cuối tuần → All: `docs/mdp-design.md` (toàn team nên đọc)

### Công việc

**1. Nghiên cứu Background**

- Đọc 3-5 papers về RL for task scheduling
- Đọc 2-3 papers về Edge-Cloud computing
- Note các approach phổ biến (DQN, PPO, A3C)
- Review Gymnasium documentation

**2. Thiết kế State Space**

- List tất cả features cần thiết (12-15 dimensions)
- Ví dụ: CPU_edge1, RAM_edge1, Queue_edge1, Latency_edge1, ...
- Xác định range cho mỗi feature
- Quyết định normalization strategy

**3. Thiết kế Action Space**

- Define discrete actions (3-5 actions)
- Action 0: Edge Node 1
- Action 1: Edge Node 2
- Action 2: Cloud
- (Optional) Action 3: Queue/wait

**4. Thiết kế Reward Function**

- Xác định các thành phần reward:
  - Task completion time (negative)
  - Resource cost (negative)
  - Deadline miss penalty (large negative)
  - Load balance bonus (positive)
- Define weights cho từng component
- Công thức: `R = w1*(-time) + w2*(-cost) + w3*penalty + w4*bonus`

**5. Implement Baseline Policies**

- Round Robin: Luân phiên giữa nodes
- Least Loaded: Chọn node có CPU thấp nhất
- Greedy Local-First: Ưu tiên Edge, fallback Cloud

**6. Define Evaluation Metrics**

- Avg task latency
- P95 latency
- SLA compliance rate (% tasks meet deadline)
- Resource utilization
- Cost per task

### Công cụ

- Python 3.10+
- Gymnasium
- NumPy, Pandas
- Papers: Google Scholar, arXiv

### Output

```
✅ File: docs/mdp-design.md (10+ trang)
   - State space definition
   - Action space definition
   - Reward function với explanation
   - Baseline algorithms description
✅ File: rl_env/baseline_policies.py
   - class RoundRobinPolicy
   - class LeastLoadedPolicy
   - class GreedyLocalFirstPolicy
✅ File: requirements.txt
   - gymnasium==0.29+
   - stable-baselines3==2.1+
   - torch, numpy, pandas, matplotlib
```

---

## 📅 TUẦN 2: GYMNASIUM ENVIRONMENT

### ⛓️ Dependencies

**Cần từ team:**

- ❌ Không phụ thuộc - có thể làm simulation độc lập
- 📝 (Optional) Person 1: Sample metrics data để validate simulation realistic

**Phải giao cho team:**

- ✅ Giữa tuần → Person 3: `rl_env/edge_cloud_env.py` (Person 3 cần hiểu interface)
- ✅ Cuối tuần → Person 3: State normalization format (cần match với Dispatcher)

### Công việc

**1. Implement Gymnasium Environment**

- Tạo class `EdgeCloudEnv(gym.Env)`
- Define `observation_space`: Box(14,) continuous
- Define `action_space`: Discrete(3)
- Implement `reset()`: Initialize state
- Implement `step(action)`:
  - Apply action
  - Simulate task execution
  - Calculate reward
  - Return (next_state, reward, done, truncated, info)

**2. Mô phỏng Task Execution**

- Task completion time = f(CPU_available, task_size)
- Network delay: Normal distribution (μ=80ms, σ=20ms)
- Queue delay: Linear với queue length
- Deadline check: Compare completion_time vs deadline

**3. Mô phỏng System Dynamics**

- CPU tăng khi task running, giảm khi completed
- RAM tương tự
- Queue length update
- Add noise để realistic (±5% fluctuation)

**4. Environment Validation**

- Test với random policy (1000 episodes)
- Verify episode length hợp lý (200-500 steps)
- Check reward không bị explode/vanish
- Visualize state transitions

**5. Initial Training**

- Setup Stable-Baselines3
- Train DQN model 100K steps
- Monitor learning curve với TensorBoard
- Verify agent đang học (reward tăng dần)

### Công cụ

- Gymnasium
- Stable-Baselines3
- PyTorch
- TensorBoard
- Matplotlib

### Output

```
✅ File: rl_env/edge_cloud_env.py (~300 lines)
   - class EdgeCloudEnv(gym.Env)
   - Complete MDP implementation
✅ File: rl_env/env_test.py
   - Unit tests
   - Random policy evaluation
✅ File: rl_training/train_dqn.py
   - Training script với hyperparameters
✅ Model: models/dqn_100k.zip
✅ Logs: logs/tensorboard/ (training curves)
✅ Validation report: Avg reward, episode length
```

---

## 📅 TUẦN 3: ADVANCED TRAINING & TUNING

### ⛓️ Dependencies

**Cần từ team:**

- ❌ Không phụ thuộc - training là độc lập

**Phải giao cho team:**

- ✅ Thứ 2 → Person 3: `models/best_model.zip` (Person 3 cần integrate vào Dispatcher)

### Công việc

**1. DQN Extended Training**

- Train DQN 500K steps
- Monitor convergence
- Checkpoint every 50K steps
- Log metrics: reward, SLA compliance, latency

**2. Hyperparameter Tuning**

- Experiment 3-5 reward weight combinations
- Tune learning_rate: [0.0001, 0.0003, 0.001]
- Tune gamma: [0.95, 0.99]
- Tune exploration schedule
- Pick best config based on validation

**3. PPO Training**

- Implement PPO training script
- Train 300K steps
- Compare với DQN performance
- PPO thường stable hơn cho continuous control

**4. Multi-Scenario Evaluation**

- Scenario 1: Constant load (10 tasks/min)
- Scenario 2: Bursty load (spike patterns)
- Scenario 3: Mixed tasks (CPU-heavy + latency-sensitive)
- Run 100 episodes per scenario
- Collect statistics

**5. Model Selection**

- Compare DQN vs PPO vs Baselines
- Metrics: Avg latency, SLA%, Cost
- Statistical significance test (t-test)
- Chọn model tốt nhất cho production

**6. Document Training Process**

- Training curves (rewards over time)
- Hyperparameter sensitivity analysis
- Comparison tables
- Ablation study (nếu có thời gian)

### Công cụ

- Stable-Baselines3 (DQN, PPO)
- TensorBoard (monitoring)
- Pandas, NumPy (analysis)
- Matplotlib, Seaborn (visualization)
- SciPy (statistical tests)

### Output

```
✅ Model: models/dqn_500k.zip
✅ Model: models/ppo_300k.zip ⭐
✅ Model: models/best_model.zip (symlink)
✅ File: docs/training-report.md
   - Training curves
   - Hyperparameter tuning results
   - Model comparison
✅ File: experiments/reward_tuning_results.csv
✅ File: experiments/multi_scenario_eval.csv
✅ Recommendation: "Use PPO for production"
```

---

## 📅 TUẦN 4: COMPREHENSIVE EVALUATION & REPORT

### ⛓️ Dependencies

**Cần từ team:**

- 📥 Person 3: Full system running (Dispatcher + K8s integration) để chạy benchmark 1000 tasks
- 📥 Person 1: Production infrastructure stable
- 📥 Person 3: Task Generator để tạo workload

**Phải giao cho team:**

- ✅ Giữa tuần → Person 1: Benchmark results (để visualize trên Grafana)
- ✅ Cuối tuần → All: Technical report và slides cho final presentation

### Công việc

**1. Large-Scale Benchmark**

- Run 1000+ tasks across 3 scenarios
- RL model vs 3 baselines
- Collect full metrics:
  - Latency (avg, median, P95, P99)
  - SLA compliance rate
  - Resource utilization (avg, peak)
  - Cost per task
  - Load balance score

**2. Statistical Analysis**

- Compute confidence intervals (95% CI)
- T-test: RL vs each baseline
- Effect size calculation
- Significance testing (p < 0.05)

**3. Visualization**

- Box plots: Latency distribution
- Line charts: SLA over time
- Bar charts: Policy comparison
- Heatmaps: Resource utilization patterns
- Scatter plots: Cost vs Performance

**4. Viết Technical Report**

- **Section 1**: Introduction (2 trang)
  - Problem statement
  - Motivation
  - Contributions
- **Section 2**: Related Work (3 trang)
  - RL for scheduling
  - Edge-Cloud computing
- **Section 3**: System Architecture (3 trang)
  - Overview diagram
  - Components description
- **Section 4**: MDP Formulation (3 trang)
  - State, Action, Reward chi tiết
  - Justification cho design choices
- **Section 5**: Training Methodology (3 trang)
  - Algorithms (DQN, PPO)
  - Hyperparameters
  - Training setup
- **Section 6**: Experimental Results (4 trang)
  - Benchmark tables
  - Statistical analysis
  - Visualizations
- **Section 7**: Discussion (2 trang)
  - Key findings
  - Limitations
  - Sim-to-real gap
- **Section 8**: Conclusion (1 trang)
  - Summary
  - Future work

**5. Tạo Presentation Slides**

- 30 slides cho demo
- Include: architecture, MDP, results, demo

### Công cụ

- Jupyter Notebook (analysis)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)
- SciPy (statistical tests)
- LaTeX hoặc Word (report)
- PowerPoint (slides)

### Output

```
✅ File: results/benchmark_1000_tasks.csv
✅ File: results/statistical_analysis.ipynb
✅ Folder: results/visualizations/ (10+ charts)
   - latency_boxplot.png
   - sla_comparison.png
   - cost_performance_scatter.png
   - resource_heatmap.png
✅ File: docs/technical-report.pdf (20 trang)
✅ File: docs/presentation-slides.pptx (30 slides)
✅ Summary: "RL giảm 25% latency, SLA tăng 15%"
```

---

## 🔧 CÔNG CỤ TỔNG HỢP

### Core Libraries

- **Gymnasium**: RL environment framework
- **Stable-Baselines3**: RL algorithms (DQN, PPO)
- **PyTorch**: Deep learning backend

### Data & Visualization

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualization

### Monitoring & Logging

- **TensorBoard**: Training monitoring
- **Weights & Biases**: Experiment tracking (optional)

### Analysis

- **SciPy**: Statistical tests
- **Jupyter Notebook**: Interactive analysis

### Documentation

- **Markdown**: Documentation
- **LaTeX/Word**: Technical report
- **PowerPoint**: Presentation

---

## 📊 WEEKLY CHECKPOINTS

### Tuần 1 - Thứ 6

- [ ] Present: MDP design document
- [ ] Demo: Baseline policies running
- [ ] Deliver: mdp-design.md

### Tuần 2 - Thứ 6

- [ ] Demo: Gymnasium env với random policy
- [ ] Show: Training curve (100K steps)
- [ ] Deliver: edge_cloud_env.py

### Tuần 3 - Thứ 6

- [ ] Present: Model comparison (DQN vs PPO)
- [ ] Show: Multi-scenario evaluation results
- [ ] Deliver: best_model.zip

### Tuần 4 - Thứ 6

- [ ] Present: Technical report + slides
- [ ] Show: Benchmark results (1000 tasks)
- [ ] Deliver: All deliverables

---

## ⚠️ THÁCH THỨC CÓ THỂ GẶP

| Issue             | Solution                                    |
| ----------------- | ------------------------------------------- |
| RL không converge | Tune reward weights, reduce learning rate   |
| Training quá chậm | Reduce state dimensions, use vectorized env |
| Reward bị explode | Clip rewards, normalize returns             |
| Overfitting       | Add noise, domain randomization             |
| Sim-to-real gap   | Model realistic dynamics, collect real data |

---

## ⛓️ DEPENDENCY TIMELINE TỔNG HỢP

```
TUẦN 1:
  Person 2 (bạn)
    │
    ├──← Nhận: Metrics format từ Person 1
    │
    ├──→ Thứ 4: Giao State definition → Person 3
    ├──→ Thứ 4: Giao State features list → Person 1
    ├──→ Cuối tuần: Giao baseline_policies.py → Person 3
    └──→ Cuối tuần: Giao mdp-design.md → All

TUẦN 2:
  Person 2 (bạn)  [LÀM ĐỘC LẬP - KHÔNG PHỤ THUỘC]
    │
    ├──→ Giữa tuần: Giao edge_cloud_env.py → Person 3
    └──→ Cuối tuần: Giao normalization format → Person 3

TUẦN 3:
  Person 2 (bạn)  [LÀM ĐỘC LẬP - TRAINING]
    │
    ├──→ Thứ 2: Giao best_model.zip → Person 3 [CRITICAL]
    ├──→ Giữa tuần: Giao Model inference docs → Person 3
    └──→ Cuối tuần: Giao training-report.md → All

TUẦN 4:
  Person 2 (bạn)
    │
    ├──← Nhận: Full system running từ Person 3
    ├──← Nhận: Infrastructure stable từ Person 1
    ├──← Nhận: Task Generator từ Person 3
    │
    ├──→ Giữa tuần: Giao Benchmark results → Person 1
    └──→ Cuối tuần: Giao Technical report + Slides → All
```

## ✅ DEFINITION OF DONE

- [ ] MDP design document approved
- [ ] Gymnasium environment validated (no bugs)
- [ ] Model trained và converge
- [ ] Model outperform baseline >10%
- [ ] Evaluation trên 1000+ tasks completed
- [ ] Statistical significance proven (p < 0.05)
- [ ] Technical report hoàn chỉnh (20 trang)
- [ ] Presentation slides ready (30 slides)

---

## 💡 TIPS

- **Start Simple**: Implement simple version trước, optimize sau
- **Log Everything**: TensorBoard is your friend
- **Validate Early**: Test env với random policy trước khi train
- **Compare Often**: Benchmark vs baselines thường xuyên
- **Document Why**: Giải thích mọi design decision

---

**Estimated Effort**: 40-50 giờ/tuần × 4 tuần = 160-200 giờ total
