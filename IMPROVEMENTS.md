# Danh sách cải tiến — Đồ án RL Edge-Cloud Task Offloading

> **Phạm vi**: Cải thiện để hoàn thiện đồ án ở mức tốt — bám sát thực tế, không hướng đến research mới lạ. Tất cả đề xuất đều dựa trên best-practice đã được công nhận trong literature (DQN/PPO chuẩn, scheduling thực tế).
>
> **Ưu tiên**: Phần [P0] bắt buộc fix trước khi train lại. Phần [P1] cần thiết để defend tốt. Phần [P2] làm nếu còn thời gian.

---

## P0 — Bug bắt buộc fix trước khi train lại

### B1. Bug logic trong `_simulate_node_metrics` ([rl_env/edge_cloud_env.py:139-159](rl_env/edge_cloud_env.py#L139-L159))

**Vấn đề**: Block `if not add_noise:` đặt SAU khi đã update `_edge_cpu`. Lần `reset()` đầu tiên, code áp dụng "noise" vào giá trị 0, rồi mới override bằng `uniform(20, 70)`. Logic bị đảo ngược → init không nhất quán giữa các giá trị.

**Cách fix**: Tách riêng init và step update:
```python
def _simulate_node_metrics(self, add_noise: bool = False):
    if not add_noise:
        # Init lần đầu (reset)
        self._edge_cpu = self.np_random.uniform(20, 70, self.n_edge_nodes)
        self._edge_ram = self.np_random.uniform(30, 80, self.n_edge_nodes)
        self._cloud_cpu = float(self.np_random.uniform(10, 50))
        self._cloud_ram = float(self.np_random.uniform(10, 50))
    else:
        # Step update với noise
        for i in range(self.n_edge_nodes):
            self._edge_cpu[i] = np.clip(self._edge_cpu[i] + self.np_random.uniform(-5, 5), 10, 90)
            self._edge_ram[i] = np.clip(self._edge_ram[i] + self.np_random.uniform(-5, 5), 10, 90)
        self._cloud_cpu = np.clip(self._cloud_cpu + self.np_random.uniform(-5, 5), 5, 80)
        self._cloud_ram = np.clip(self._cloud_ram + self.np_random.uniform(-5, 5), 5, 80)

    # Latency luôn re-compute từ CPU
    for i in range(self.n_edge_nodes):
        self._edge_latency[i] = np.clip(
            self.np_random.uniform(5, 30) + self._edge_cpu[i] * 0.3, 1, MAX_LATENCY
        )
    self._cloud_latency = float(self.np_random.uniform(30, 80) + self._cloud_cpu * 0.2)
```

### B2. INSTANCE_MAP fallback im lặng ([dispatcher/state_builder.py:250-258](dispatcher/state_builder.py#L250-L258))

**Vấn đề**: Khi IP trong Prometheus không khớp với `INSTANCE_MAP`, code im lặng dùng cache cũ (toàn 0) → dispatcher nghĩ là đang đọc real metrics nhưng thực ra là dùng zeros.

**Cách fix**: Log warning rõ ràng khi mismatch:
```python
def _fetch_prometheus_metrics(self):
    try:
        raw = self._prom_client.get_dispatcher_metrics()
    except Exception as e:
        logger.warning("Prometheus query failed: %s", e)
        return

    role_to_inst = {v: k for k, v in self._instance_map.items()}
    matched = 0
    for role in [f"edge_{i+1}" for i in range(self.n_edge_nodes)] + ["cloud"]:
        inst = role_to_inst.get(role)
        if inst not in raw:
            logger.warning("Role %s mapped to %s but not in Prometheus response", role, inst)
            continue
        matched += 1
        # ... update metrics
    if matched == 0:
        logger.error("Không match được instance nào — đang dùng zeros! Check INSTANCE_MAP.")
```

---

## P1 — Cải tiến cốt lõi cần có để defend tốt

### S1. Thêm Queue Length vào State Vector

**Lý do**: Đây là yếu tố quan trọng NHẤT trong scheduling thực tế nhưng đang thiếu. Một node CPU 30% nhưng có 10 task chờ vẫn chậm hơn node CPU 70% queue rỗng. Thiếu queue length là điểm yếu lớn khi defend — examiner sẽ hỏi.

`PrometheusClient.get_dispatcher_metrics()` đã trả về `queue_length` (real hoặc proxy từ `node_load1/cores`) nhưng StateBuilder không dùng.

**Cách fix**: Mở rộng state từ 12 dims → 15 dims (thêm 1 queue per node):
```
[e1_cpu, e1_ram, e1_lat, e1_queue,
 e2_cpu, e2_ram, e2_lat, e2_queue,
 cloud_cpu, cloud_ram, cloud_lat, cloud_queue,
 task_cpu, task_ram, task_deadline]
```

**Files cần sửa**:
- [rl_env/edge_cloud_env.py](rl_env/edge_cloud_env.py): thêm `_edge_queue`, `_cloud_queue`, sinh queue tăng khi dispatch và giảm khi step
- [dispatcher/state_builder.py](dispatcher/state_builder.py): thêm `queue_length` vào `NodeMetrics`, update `_compose_observation`
- Train lại model với obs_dim=15

### S2. Thêm Temporal Features

**Lý do**: Workload thực tế có pattern theo giờ (diurnal). Không có temporal feature → model không học được điều này → fail trên kịch bản thật.

**Cách fix**: Thêm 2 dims `sin(2π*hour/24), cos(2π*hour/24)`. Cyclic encoding chuẩn (không dùng `hour/24` vì 23h và 0h không gần nhau).

```python
import time
hour = (time.time() // 3600) % 24
obs.extend([np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)])
```

State vector cuối cùng: 17 dims.

### R1. Reward — SLA Continuous thay vì Discrete

**Vấn đề hiện tại** ([rl_env/edge_cloud_env.py:247-268](rl_env/edge_cloud_env.py#L247-L268)):
- Task xong sát deadline (199/200ms) cùng reward với task xong nhanh (10/200ms) — model không có incentive tối ưu thêm
- SLA bonus +1.0 áp đảo latency penalty max -0.6 → model học "miễn meet SLA là OK"

**Cách fix**:
```python
def _compute_reward(self, latency, cost, sla_met, deadline):
    latency_norm = latency / MAX_LATENCY
    cost_norm = cost / (CLOUD_COST_PER_UNIT * 110)

    # SLA continuous: tanh smoothing trong [-1, 1]
    slack = (deadline - latency) / deadline   # >0 nếu kịp, <0 nếu miss
    sla_signal = np.tanh(3.0 * slack)         # smooth, in [-1, 1]

    # Cân bằng scale: max latency penalty ≈ max sla bonus
    reward = -0.5 * latency_norm - 0.2 * cost_norm + 0.5 * sla_signal

    # Penalty thêm khi miss thực sự (cứng hơn để model học tránh)
    if not sla_met:
        reward -= 0.5

    return float(reward)
```

**Lưu ý cho defend**: Phải document weight chọn 0.5/0.2/0.5 + ablation tối thiểu 3 bộ weight (Phase C3).

### R2. Reward Ablation

**Lý do**: Magic numbers không có justification → bị hỏi khi defend.

**Cách làm**: Train với 3 bộ weight, so sánh trên cùng test workload:

| Config | latency | cost | SLA | Mục tiêu |
|--------|---------|------|-----|----------|
| **Balanced** (current) | 0.5 | 0.2 | 0.5 | Cân bằng |
| **Latency-priority** | 0.8 | 0.1 | 0.3 | Real-time apps |
| **Cost-priority** | 0.2 | 0.5 | 0.3 | Batch jobs |

Plot 3 đường training curve + bảng so sánh SLA% / avg latency / total cost. **Phải có** trong báo cáo.

### A1. Thêm Action "Reject"

**Lý do**: Trong thực tế, task đã miss deadline (arrival_time + estimated_processing > deadline) thì dispatch vô ích — chỉ tốn tài nguyên. Hiện tại buộc dispatch.

**Cách fix**: Mở rộng action space từ `Discrete(3)` → `Discrete(4)`:
```python
self.action_space = spaces.Discrete(self.n_edge_nodes + 2)  # +1 cloud, +1 reject

# Trong step():
if action == self.n_edge_nodes + 1:  # reject
    reward = -0.5  # penalty cố định, nhỏ hơn miss SLA (-2.0)
    return obs, reward, ...
```

Lưu ý: cần tune penalty reject. Quá nhỏ → model lười, reject hết. Quá lớn → không bao giờ reject.

### M1. Bật Double DQN

**Lý do**: Vanilla DQN có overestimation bias đã được literature chứng minh từ 2015 (Van Hasselt et al.). Double DQN là 2 dòng code.

**Cách fix** ([models/dqn_agent.py:223-226](models/dqn_agent.py#L223-L226)):
```python
# Cũ:
next_q = self.target_net(next_states_t).max(dim=1)[0]

# Mới (Double DQN):
with torch.no_grad():
    next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
    next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
```

### M2. Soft Target Update

**Lý do**: Hard update mỗi 100 steps gây "jolt" cho training. Soft update (Polyak averaging) mượt hơn, là default modern.

**Cách fix**:
```python
# Thay _update_target():
def _soft_update(self, tau=0.005):
    for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

# Gọi mỗi step thay vì mỗi 100 steps
```

### E1. Real Latency từ HTTP Probe

**Vấn đề**: `latency = 10 + cpu * 0.4` là synthetic, tạo correlation giả với CPU. Không có giá trị nếu defend Phase C2 (sim-to-real gap).

**Cách fix**: StateBuilder probe `WORKER_URLS[node]/health` trước khi build state, cache 10s.

```python
# dispatcher/state_builder.py
class StateBuilder:
    def __init__(self, ...):
        self._rtt_cache = {}     # {node: (rtt_ms, timestamp)}
        self._rtt_ttl = 10.0     # cache 10s

    def _measure_rtt(self, node: str) -> float:
        from dispatcher.infra_config import WORKER_URLS
        cached = self._rtt_cache.get(node)
        if cached and time.time() - cached[1] < self._rtt_ttl:
            return cached[0]
        url = WORKER_URLS[node].replace("/task", "/health")
        try:
            t0 = time.perf_counter()
            requests.get(url, timeout=1.0)
            rtt = (time.perf_counter() - t0) * 1000
        except Exception:
            rtt = 999.0  # node unreachable
        self._rtt_cache[node] = (rtt, time.time())
        return rtt
```

Thay synthetic latency trong `_fetch_prometheus_metrics` bằng `_measure_rtt`.

---

## P2 — Cải tiến nếu còn thời gian

### S3. Encode Task Priority & Payload Type

`TaskInfo` đã có `priority` (low/medium/high) và `payload_type` (compute/image/io) nhưng không được encode vào state.

```python
priority_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
payload_onehot = {"compute": [1,0,0], "image": [0,1,0], "io": [0,0,1]}
obs.extend([priority_map[task.priority]] + payload_onehot[task.payload_type])
```

### E2. Task Arrival theo Poisson

Hiện tại: 1 task/step (uniform). Thực tế: Poisson process với λ thay đổi.

```python
# Trong workload/task_generator.py
def diurnal_lambda(hour):
    """Workload cao 9-17h, thấp đêm."""
    return 5.0 + 10.0 * max(0, np.sin(np.pi * (hour - 6) / 12))

inter_arrival = np.random.exponential(1.0 / diurnal_lambda(current_hour))
```

### E3. Inject Node Failure trong Training

Để model robust với node failure (Phase D1):
```python
# Trong env.step():
if self.np_random.random() < 0.005:  # 0.5% chance per step
    failed_node = self.np_random.integers(self.n_edge_nodes)
    self._edge_cpu[failed_node] = 99.0   # mô phỏng down
```

### M3. Prioritized Experience Replay

Thay uniform sampling bằng PER. Convergence nhanh 2-3x. Đã có sẵn implementation tham khảo trong nhiều thư viện. Nếu không kịp tự viết → **bỏ DQN custom, dùng `stable-baselines3.DQN` với Double + Dueling + PER bật mặc định**.

---

## Documentation cần có cho báo cáo/defend

1. **Bảng so sánh State design**: cũ (12 dims) vs mới (17 dims) — argue tại sao thêm queue/temporal
2. **Reward ablation table**: 3 bộ weight × 3 metrics
3. **Convergence plot**: training curve DQN + PPO trên cùng env (Phase C1)
4. **Sim-to-real comparison**: Bảng SLA% / avg latency simulation vs Prometheus thật (Phase C2)
5. **Policy behavior heatmap**: với task light/heavy, model chọn node nào (Phase C5)
6. **Comparison vs baselines**: bảng cuối với t-test p-values

---

## Thứ tự thực hiện đề xuất

```
Tuần 1 (foundation):
  Day 1: B1, B2 (bug fixes)
  Day 2: S1 (queue length) + S2 (temporal)
  Day 3: R1 (reward continuous) + A1 (reject action)
  Day 4: M1 + M2 (Double DQN + soft update)
  Day 5: Train lại model + sanity check

Tuần 2 (rigor):
  Day 1-2: R2 (reward ablation, train 3 configs)
  Day 3: E1 (real latency probe) — cần infra ready
  Day 4: Sim-to-real comparison (Phase C2)
  Day 5: Policy visualization (Phase C5)

Tuần 3 (polish):
  - Báo cáo + plots
  - Demo D1 + D3 (cascading failure + live policy switch)
  - P2 nếu còn thời gian
```

**Tổng effort**: ~10-12 ngày làm việc cho mức "đồ án hoàn thiện tốt".

---

## Những gì KHÔNG làm (out of scope)

- Multi-task RL với context conditioning
- Pointer networks cho action space scaling
- Online continual learning
- Heavy-tailed latency distribution (log-normal/Pareto)
- Distributed training (Ray, IMPALA)
- Transformer-based state encoder
- Robust RL với adversarial training

Những thứ trên là research-grade, không cần cho đồ án. Nếu examiner hỏi → trả lời thẳng "out of scope, đề xuất cho future work".
