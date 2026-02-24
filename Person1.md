# PHÂN CÔNG - PERSON 1: INFRASTRUCTURE LEAD

**Vai trò**: Xây dựng & quản lý hạ tầng Edge-Cloud, Monitoring, DevOps

**Timeline**: 4 tuần

---

## 🔴 CRITICAL HANDOFF POINTS

## 🎯 TRÁCH NHIỆM CHÍNH

- Setup K3s cluster (Edge) + AWS EC2/EKS (Cloud)
- Deploy hệ thống giám sát Prometheus + Grafana
- Viết Python client để query metrics từ Prometheus
- Tích hợp Kubernetes API để deploy pods
- Tạo dashboard production và demo script

---

## 📅 TUẦN 1: CÀI ĐẶT HẠ TẦNG

### ⛓️ Dependencies

**Cần từ team:**

- ❌ Không phụ thuộc - có thể bắt đầu ngay

**Phải giao cho team:**

- ✅ Cuối tuần → Person 2: Metrics format (CPU, RAM, Latency ranges) để thiết kế State space
- ✅ Cuối tuần → Person 3: Kubeconfig files để access cluster
- ✅ Cuối tuần → Person 3: Prometheus endpoint URL

### Công việc

**1. Setup K3s Edge Cluster**

- Cài K3s trên 2 máy local/VM (hoặc Raspberry Pi)
- Test connectivity giữa các nodes
- Deploy sample workload (Nginx) để verify
- Export kubeconfig cho team

**2. Setup AWS Cloud**

- Tạo EC2 instance (t3.medium)
- Cài K3s trên EC2 (hoặc dùng EKS nếu budget cho phép)
- Cấu hình VPC, Security Group
- Setup VPN/tunneling để kết nối Edge ↔ Cloud

**3. Deploy Monitoring Stack**

- Deploy Prometheus trên cả Edge & Cloud
- Cài Prometheus Node Exporter (thu thập CPU/RAM)
- Deploy Grafana
- Tạo dashboard cơ bản: CPU, RAM, Network latency

### Công cụ

- K3s, Docker
- AWS Console/CLI
- Prometheus, Grafana
- kubectl, helm

### Output

```
✅ K3s Edge: 2 nodes running
✅ AWS Cloud: 1 node running
✅ Prometheus: scraping metrics every 15s
✅ Grafana Dashboard: hiển thị metrics 3 nodes
✅ File: infrastructure/k3s-setup.sh
✅ File: infrastructure/aws-setup.sh
✅ File: monitoring/prometheus-config.yml
✅ File: docs/setup-guide.md
```

---

## 📅 TUẦN 2: METRICS API & LOGGING

### ⛓️ Dependencies

**Cần từ team:**

- 📥 Person 2: State space definition (14 dimensions) để biết cần query metrics gì

**Phải giao cho team:**

- ✅ Giữa tuần → Person 3: `monitoring/prometheus_client.py` (Person 3 cần để integrate Dispatcher)
- ✅ Cuối tuần → Person 3: `monitoring/metrics_exporter.py`
- ✅ Cuối tuần → Person 3: Metrics API endpoint và documentation

### Công việc

**1. Viết Prometheus Query Client**

- Script Python query Prometheus API
- Lấy metrics: CPU, RAM, Queue length, Network latency
- Format output dạng JSON

**2. Tạo Metrics Exporter Service**

- REST API đơn giản expose metrics
- Endpoint: `/api/v1/metrics`
- Return structured data cho Dispatcher sử dụng

**3. Deploy Sample Workload**

- Deploy Nginx/sample app để test metrics collection
- Verify Prometheus đang track correctly

**4. Setup Logging**

- Deploy Loki hoặc simple file logging
- Cấu hình alert rules (CPU >90%, etc.)

### Công cụ

- Python (requests, flask/fastapi)
- Prometheus API
- Loki (optional)

### Output

```
✅ File: monitoring/prometheus_client.py
✅ File: monitoring/metrics_exporter.py
✅ Metrics API: http://monitor:9090/api/v1/metrics
✅ Sample workload running trên 3 nodes
✅ File: monitoring/alerting-rules.yml
```

---

## 📅 TUẦN 3: KUBERNETES INTEGRATION

### ⛓️ Dependencies

**Cần từ team:**

- ❌ Không phụ thuộc - có thể tiếp tục độc lập

**Phải giao cho team:**

- ✅ Thứ 3 → Person 3: `k8s_integration/pod_deployer.py` (Person 3 cần để integrate vào Dispatcher)
- ✅ Thứ 3 → Person 3: `k8s_templates/task-job.yaml`
- ✅ Giữa tuần → Person 3: Docker image `task-processor:v1` đã push lên registry

### Công việc

**1. Viết K8s Deployment Automation**

- Python script sử dụng Kubernetes client
- Function: `deploy_task_pod(task_id, target_node)`
- Tạo K8s Job với nodeSelector để chỉ định node

**2. Tạo K8s Templates**

- YAML template cho task Job
- Cấu hình resource limits (CPU, memory)
- NodeSelector labels (edge1, edge2, cloud)

**3. Tạo Test Workload**

- Build Docker image: simple Python script xử lý task
- Ví dụ: image processing, computation task
- Push image lên DockerHub/ECR

**4. Integration Testing**

- Deploy 50 pods qua script
- Track latency: dispatch → running → completed
- Log kết quả để phân tích

### Công cụ

- Python kubernetes client
- Docker, Dockerfile
- kubectl, K8s YAML

### Output

```
✅ File: k8s_integration/pod_deployer.py
✅ File: k8s_templates/task-job.yaml
✅ Docker image: task-processor:v1
✅ Integration test: 50 pods deployed thành công
✅ File: tests/integration_test.py
```

---

## 📅 TUẦN 4: DASHBOARD & DEMO

### ⛓️ Dependencies

**Cần từ team:**

- 📥 Person 3: Smart Dispatcher running (để dashboard có real data)
- 📥 Person 2: Benchmark results (để visualize trên dashboard)
- 📥 Person 3: Task Generator (để chạy demo 30 phút liên tục)

**Phải giao cho team:**

- ✅ Cuối tuần → All: Demo script và video để present

### Công việc

**1. Thiết kế Grafana Production Dashboard**

- Panel: Decision Distribution (Pie chart)
- Panel: Task Completion Latency (Time series)
- Panel: Node Resource Utilization (Gauge)
- Panel: SLA Compliance (Stat)
- Panel: Network Latency Heatmap

**2. Setup Alerting**

- Alert: CPU >90% for 5min
- Alert: SLA <80% for 10min
- Alert: Dispatcher down
- Tích hợp Slack/Email notification

**3. Tạo Demo Script**

- Script tự động chạy workload liên tục
- Switch giữa policies mỗi 10 phút
- Auto-capture screenshots
- Record demo video (screen recording)

**4. Tối ưu Performance**

- Optimize Prometheus queries
- Reduce scrape interval nếu cần
- Cleanup unused metrics

### Công cụ

- Grafana (dashboard design)
- Shell script
- Screen recording tool (OBS, etc.)

### Output

```
✅ Grafana Dashboard: 7 panels production-ready
✅ File: monitoring/grafana-dashboard-final.json
✅ File: monitoring/alerting-rules-final.yml
✅ File: demo/run_demo.sh
✅ Folder: demo/screenshots/
✅ File: demo/demo-video.mp4 (5-10 min)
```

---

## 🔧 CÔNG CỤ TỔNG HỢP

### Infrastructure

- **K3s**: Lightweight Kubernetes
- **Docker**: Containerization
- **AWS EC2**: Cloud compute

### Monitoring

- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Node Exporter**: System metrics
- **Loki**: Logging (optional)

### Development

- **Python 3.10+**: Scripting
- **kubectl**: K8s CLI
- **helm**: Package manager
- **AWS CLI**: Cloud management

### Testing

- **pytest**: Testing framework
- **Shell scripts**: Automation

---

---

## 📊 WEEKLY CHECKPOINTS

### Tuần 1 - Thứ 6

- [ ] Demo: kubectl get nodes (3 nodes ready)
- [ ] Demo: Grafana dashboard với live metrics
- [ ] Handoff: kubeconfig files cho team

### Tuần 2 - Thứ 6

- [ ] Demo: Python script query Prometheus
- [ ] Demo: Metrics API response
- [ ] Handoff: prometheus_client.py cho Person 3

### Tuần 3 - Thứ 6

- [ ] Demo: Deploy 50 pods via script
- [ ] Show: Latency breakdown report
- [ ] Handoff: pod_deployer.py cho Person 3

### Tuần 4 - Thứ 6

- [ ] Demo: Full dashboard với real-time data
- [ ] Demo: Run demo script 30 phút
- [ ] Deliver: Video demo

---

## ⚠️ BLOCKERS CÓ THỂ GẶP

| Issue                     | Solution                                    |
| ------------------------- | ------------------------------------------- |
| AWS quá đắt               | Dùng EC2 spot instances, tắt khi không dùng |
| K3s nodes không connect   | Check firewall, VPN config                  |
| Prometheus quá nhiều data | Tăng scrape interval, giảm retention        |
| Grafana query chậm        | Add index, optimize PromQL                  |
| K8s pod pending           | Check resource limits, node labels          |

---

## ⛓️ DEPENDENCY TIMELINE TỔNG HỢP

```
TUẦN 1:
  Person 1 (bạn)
    │
    ├──→ Cuối tuần: Giao Metrics format → Person 2
    ├──→ Cuối tuần: Giao Kubeconfig → Person 3
    └──→ Cuối tuần: Giao Prometheus URL → Person 3

TUẦN 2:
  Person 1 (bạn)
    │
    ├──← Nhận: State definition từ Person 2
    │
    ├──→ Giữa tuần: Giao prometheus_client.py → Person 3 [CRITICAL]
    └──→ Cuối tuần: Giao metrics_exporter.py → Person 3

TUẦN 3:
  Person 1 (bạn)
    │
    ├──→ Thứ 3: Giao pod_deployer.py → Person 3 [CRITICAL]
    ├──→ Giữa tuần: Giao Docker image → Person 3
    └──→ Cuối tuần: Giao K8s templates → Person 3

TUẦN 4:
  Person 1 (bạn)
    │
    ├──← Nhận: Dispatcher running từ Person 3
    ├──← Nhận: Benchmark results từ Person 2
    ├──← Nhận: Task Generator từ Person 3
    │
    └──→ Cuối tuần: Giao Demo script + Video → All
```

---

## ✅ DEFINITION OF DONE

- [ ] 3 K8s nodes running stable 24h+
- [ ] Prometheus collecting metrics without gaps
- [ ] Grafana dashboard accessible & updating real-time
- [ ] K8s deployer script works with 95%+ success rate
- [ ] Demo script chạy được trên fresh setup
- [ ] Documentation đầy đủ cho deployment
- [ ] Video demo hoàn chỉnh

---

**Estimated Effort**: 40-50 giờ/tuần × 4 tuần = 160-200 giờ total
