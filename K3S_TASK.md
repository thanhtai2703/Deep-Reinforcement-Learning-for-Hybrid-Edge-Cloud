# Nhiệm vụ — Real K3s Pod Deployment

> **Phụ trách**: TBD
> **Effort dự kiến**: 1-2 ngày
> **Tier**: P1 (cần thiết để defend đồ án nghiêm túc)

---

## Mục tiêu

Thay thế cơ chế "fake K8s" hiện tại (HTTP gọi `task_worker.py`) bằng deployment thật qua Kubernetes Job. Mỗi quyết định của RL agent sẽ tạo 1 Pod thật trên K3s cluster, đặt vào đúng node theo `nodeSelector`.

## Công dụng

| Hiện tại | Sau khi làm xong |
|----------|------------------|
| `task_worker.py` chạy như Python process trên VM | Mỗi task = 1 K8s Pod chạy trong container |
| Latency = chỉ có CPU burn time | Latency = pod scheduling + container startup + execution (giống thực tế) |
| Code có thể chạy trên VM thường, không cần K8s | Sử dụng K8s thật, defend được "đây là K8s" |
| Không có pod startup overhead | Có cold-start overhead — yếu tố quan trọng trong edge-cloud offloading |
| Không có resource isolation | Container cgroups giới hạn CPU/RAM thật |

---

## Các bước thực hiện

### Bước 1 — Build Docker image cho task processor
- Viết `task_processor.py`: đọc env var `CPU_REQUIREMENT`, burn CPU đúng theo yêu cầu, ghi log, exit code 0
- Viết `Dockerfile`: base `python:3.10-slim`, copy script
- Build và push lên DockerHub (hoặc registry nội bộ)
- Tag image: `<username>/task-processor:v1`

### Bước 2 — Setup K3s cluster
- Cài K3s master trên 1 trong 3 node (đề xuất: cloud EC2 vì stable nhất)
- Join 2 edge VM vào cluster với role agent
- Verify: `kubectl get nodes` thấy đủ 3 node, status Ready

### Bước 3 — Label nodes theo role
- `kubectl label node <edge1-hostname>  role=edge_1`
- `kubectl label node <edge2-hostname>  role=edge_2`
- `kubectl label node <cloud-hostname>  role=cloud`
- Verify: `kubectl get nodes --show-labels`

### Bước 4 — Viết `dispatcher/pod_deployer.py`
- Dùng thư viện `kubernetes` (Python client)
- Hàm chính: `deploy_task_pod(task_id, target_role, cpu_req, ram_req) → job_name`
- Tạo K8s Job với:
  - `nodeSelector: {role: target_role}`
  - Container image từ Bước 1
  - Env var: `CPU_REQUIREMENT`, `RAM_REQUIREMENT`
  - Resource limits: CPU/RAM theo yêu cầu task
  - `restartPolicy: Never`
- Hàm phụ: `wait_for_completion(job_name) → (status, duration_ms)`

### Bước 5 — Tích hợp vào dispatcher
- Sửa `dispatcher/smart_dispatcher.py` method `_execute_on_node`:
  - Thay HTTP POST tới `task_worker` bằng gọi `pod_deployer.deploy_task_pod()`
  - Đo latency từ lúc submit Job tới khi Pod `Completed`
- Giữ flag fallback về HTTP nếu cần debug nhanh

### Bước 6 — Verify end-to-end
- Chạy 1 task qua dispatcher → kiểm tra `kubectl get pods` thấy pod chạy đúng node
- Chạy 100 tasks → đo latency, so sánh với HTTP cũ
- Latency mới phải lớn hơn HTTP (do pod startup) — đây là kết quả ĐÚNG, không phải bug

### Bước 7 — Document
- Ghi lại hướng dẫn setup vào `infrastructure/K3S_SETUP.md`
- Ghi config K3s, kubectl commands, troubleshooting

---

## Tiêu chí hoàn thành (Definition of Done)

- [ ] `kubectl get nodes` hiển thị 3 node Ready với label đúng
- [ ] Docker image push được lên registry, kéo về node chạy được
- [ ] `pod_deployer.py` deploy được Job, đo được latency
- [ ] Dispatcher chạy 50 task không lỗi, mỗi task tạo Pod đúng node
- [ ] Document setup được người khác replicate được

---

## Tài liệu tham khảo

- K3s docs: https://docs.k3s.io
- Kubernetes Python client: https://github.com/kubernetes-client/python
- nodeSelector: https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/

## Liên hệ

Có vướng mắc về tích hợp với dispatcher (Bước 5) → liên hệ Person 3 (backend/integration).
