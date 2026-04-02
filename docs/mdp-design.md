# MDP Design for Hybrid Edge-Cloud Task Dispatching

## 1. Problem Statement

Goal: learn a dispatch policy that selects the best execution target for each incoming task in a hybrid system with two edge nodes and one cloud node.

Primary objectives:

- minimize end-to-end task latency
- maximize SLA compliance (meet deadline)
- control resource and compute cost

## 2. Agent, Environment, and Time Step

- Agent: scheduler policy (DQN or PPO)
- Environment: abstracted edge-cloud system state
- One step: one incoming task decision

## 3. Action Space

Discrete action space:

- action `0`: dispatch to `edge_1`
- action `1`: dispatch to `edge_2`
- action `2`: dispatch to `cloud`

Notes:

- This 3-action mapping is the target production mapping for 2-edge-1-cloud deployment.
- Simulator remains configurable by `n_edge_nodes`, but experiments for the report should use 2 edge nodes to match infrastructure.

## 4. State Space (14 dimensions)

State vector at step `t`:

1. `edge1_cpu` in [0, 1]
2. `edge1_ram` in [0, 1]
3. `edge1_queue` in [0, 1]
4. `edge1_latency` in [0, 1]
5. `edge2_cpu` in [0, 1]
6. `edge2_ram` in [0, 1]
7. `edge2_queue` in [0, 1]
8. `edge2_latency` in [0, 1]
9. `cloud_cpu` in [0, 1]
10. `cloud_ram` in [0, 1]
11. `cloud_latency` in [0, 1]
12. `task_cpu_demand` in [0, 1]
13. `task_ram_demand` in [0, 1]
14. `task_deadline` in [0, 1]

Normalization references:

- CPU, RAM: percent / 100
- queue length: value / queue_max (recommend queue_max=20)
- latency: ms / latency_max (recommend latency_max=200)
- deadline: ms / deadline_max (recommend deadline_max=500)

## 5. Transition Dynamics (High-level)

After action selection:

- selected node gets extra CPU/RAM load based on task demand
- queue length changes based on arrival and service
- observed latency follows current load and network condition
- non-selected nodes decay toward lower utilization over time

## 6. Reward Design

Use weighted multi-objective reward:

`R_t = w1 * (-latency_norm) + w2 * (-cost_norm) + w3 * SLA_term + w4 * balance_bonus`

Recommended initial weights:

- `w1 = 0.6` (latency)
- `w2 = 0.2` (cost)
- `w3 = 1.0` (SLA bonus, or -2.0 penalty if deadline miss)
- `w4 = 0.1` (load-balance optional)

SLA term example:

- if task meets deadline: `+1.0`
- else: `-2.0`

Reward clipping (optional): clip to [-10, 5] for training stability.

## 7. Baseline Policies for Comparison

Implemented in `rl_env/baseline_policies.py`:

- Random
- RoundRobin
- LeastConnection (least CPU load)
- EdgeOnly
- CloudOnly
- Threshold heuristic

These baselines provide non-RL references for report comparisons.

## 8. Evaluation Metrics

Mandatory metrics:

- average latency
- P95 latency
- SLA compliance rate
- average cost per task
- deadline miss rate

Additional metrics:

- policy decision distribution
- node utilization profile

## 9. Training Plan

Phase 1 (fast validation):

- train DQN first to validate environment and reward shaping

Phase 2 (model selection):

- train PPO and compare against DQN and baselines

Selection rule:

- choose best model by SLA first, then latency, then cost

## 10. Week 1 Deliverable Checklist

- [x] action/state/reward formalization documented
- [x] baseline policy set defined and implemented
- [x] evaluation metrics defined
- [x] training algorithm plan (DQN + PPO comparison)
