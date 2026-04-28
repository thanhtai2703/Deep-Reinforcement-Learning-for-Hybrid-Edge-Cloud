# Real K3s Pod Deployment

## 1. Mục tiêu

Task này nhằm thay thế cơ chế `"fake K8s"` hiện tại, tức là dispatcher gọi HTTP tới `task_worker.py`, bằng cơ chế triển khai thật qua Kubernetes Job trên K3s cluster.

Sau khi hoàn thành, mỗi quyết định của RL agent hoặc Smart Dispatcher sẽ tạo ra một Kubernetes Job thật. Mỗi Job sinh ra một Pod chạy container `task-processor`, được schedule vào đúng node bằng `nodeSelector`.

Mục tiêu chính:

- Mỗi task = 1 Kubernetes Pod thật.
- Pod chạy trong container, có CPU/RAM request và limit.
- Pod được đặt đúng node theo label:
  - `role=edge_1`
  - `role=edge_2`
  - `role=cloud`
- Đo latency thực từ lúc submit Kubernetes Job đến khi Job Completed.
- Tích hợp vào `SmartDispatcher` để thay thế HTTP worker cũ.
- Giữ fallback về HTTP để debug khi cần.

---

## 2. Công dụng

| Trước khi làm                                         | Sau khi hoàn thành                                           |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| `task_worker.py` chạy như Python process trên VM      | Mỗi task chạy thành Kubernetes Pod thật                      |
| Dispatcher gọi HTTP tới worker                        | Dispatcher tạo Kubernetes Job qua Python Kubernetes Client   |
| Latency chỉ gồm thời gian xử lý giả lập hoặc CPU burn | Latency gồm pod scheduling, container startup và execution   |
| Không có pod startup overhead                         | Có cold-start overhead giống hệ thống thật                   |
| Không có resource isolation rõ ràng                   | Có CPU/RAM request và limit qua Kubernetes/cgroups           |
| Khó chứng minh là K8s thật                            | Có thể chứng minh bằng `kubectl get jobs` và `kubectl get pods -o wide` |

---

## 3. Kiến trúc cluster sau khi hoàn thành

K3s cluster gồm 3 node:

| Node      | Vai trò Kubernetes | Label         | IP              |
| --------- | ------------------ | ------------- | --------------- |
| `cloud-1` | control-plane      | `role=cloud`  | `172.31.18.213` |
| `edge-1`  | worker/agent       | `role=edge_1` | `100.82.147.9`  |
| `edge-2`  | worker/agent       | `role=edge_2` | `100.69.169.33` |

Kiểm tra cluster:

```bash
kubectl get nodes -o wide
kubectl get nodes -L role
```

Kết quả mong muốn:

```text
NAME      STATUS   ROLES           INTERNAL-IP
cloud-1   Ready    control-plane    172.31.18.213
edge-1    Ready    <none>           100.82.147.9
edge-2    Ready    <none>           100.69.169.33
```

---

# Các bước thực hiện

---

## Bước 1 — Build Docker image cho task processor

### 1.1. Mục tiêu

Tạo Docker image cho task processor. Container này sẽ:

- Đọc biến môi trường:
  - `TASK_ID`
  - `TARGET_ROLE`
  - `CPU_REQUIREMENT`
  - `RAM_REQUIREMENT`
  - `TASK_DURATION_SECONDS`
- Burn CPU theo yêu cầu.
- Allocate một phần RAM để thể hiện resource usage.
- Ghi log.
- Exit code `0` khi hoàn thành.

### 1.2. Docker image đã sử dụng

Image được build và push lên DockerHub:

```text
ovapil/task-processor:v1
```

### 1.3. Các biến môi trường của container

| Env var                 | Ý nghĩa             |
| ----------------------- | ------------------- |
| `TASK_ID`               | ID của task         |
| `TARGET_ROLE`           | Node role được chọn |
| `CPU_REQUIREMENT`       | CPU request/limit   |
| `RAM_REQUIREMENT`       | RAM request/limit   |
| `TASK_DURATION_SECONDS` | Thời gian burn CPU  |

### 1.4. Test image bằng Kubernetes Pod thủ công

Ví dụ chạy Pod vào `edge-1`:

```bash
kubectl run test-task-processor \
  --image=ovapil/task-processor:v1 \
  --restart=Never \
  --overrides='
{
  "spec": {
    "nodeSelector": {
      "role": "edge_1"
    },
    "containers": [
      {
        "name": "test-task-processor",
        "image": "ovapil/task-processor:v1",
        "env": [
          {"name": "TASK_ID", "value": "test-k3s-001"},
          {"name": "TARGET_ROLE", "value": "edge_1"},
          {"name": "CPU_REQUIREMENT", "value": "500m"},
          {"name": "RAM_REQUIREMENT", "value": "128Mi"},
          {"name": "TASK_DURATION_SECONDS", "value": "2"}
        ]
      }
    ],
    "restartPolicy": "Never"
  }
}'
```

Kiểm tra Pod:

```bash
kubectl get pods -o wide
```

Kết quả đã đạt:

```text
test-task-processor   Completed   NODE=edge-1
test-task-processor   Completed   NODE=edge-2
test-task-processor   Completed   NODE=cloud-1
```

Log khi chạy trên `cloud-1`:

```text
============================================================
[task_processor] START task_id=test-k3s-003
[task_processor] target_role=cloud
[task_processor] CPU_REQUIREMENT=500m parsed=0.5 cores
[task_processor] RAM_REQUIREMENT=128Mi parsed=128 MB
[task_processor] TASK_DURATION_SECONDS=2.0
[task_processor] allocated_memory=32 MB
[task_processor] completed duration_ms=2014
[task_processor] END task_id=test-k3s-003
============================================================
```

Kết luận Bước 1:

- Docker image `ovapil/task-processor:v1` đã chạy được trên K3s.
- Node có thể pull image từ DockerHub.
- Container chạy xong và exit code `0`.

---

## Bước 2 — Setup K3s cluster

### 2.1. Mục tiêu

Thiết lập một K3s cluster thật gồm:

- `cloud-1`: control-plane/server.
- `edge-1`: agent/worker.
- `edge-2`: agent/worker.

### 2.2. Trạng thái ban đầu

Ban đầu có 2 cluster riêng:

```text
Cluster edge:
- edge-1 control-plane
- edge-2 worker

Cluster cloud:
- cloud-1 control-plane
```

Để đáp ứng yêu cầu task, đã gộp lại thành một cluster duy nhất:

```text
cloud-1 = control-plane
edge-1  = worker/agent
edge-2  = worker/agent
```

### 2.3. Join edge vào cloud cluster

Trên `cloud-1`, lấy token:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

Lưu ý:

- Không commit token thật lên GitHub.
- Token phải bắt đầu bằng `K10`.
- Không được có dấu nháy hoặc khoảng trắng đầu dòng.

Trên `edge-1`, join vào cluster:

```bash
curl -sfL https://get.k3s.io | sudo INSTALL_K3S_EXEC="agent \
  --server https://100.97.43.77:6443 \
  --token-file /etc/rancher/k3s/node-token \
  --node-name edge-1 \
  --node-ip 100.82.147.9" sh -
```

Trên `edge-2`, join vào cluster:

```bash
curl -sfL https://get.k3s.io | sudo INSTALL_K3S_EXEC="agent \
  --server https://100.97.43.77:6443 \
  --token-file /etc/rancher/k3s/node-token \
  --node-name edge-2 \
  --node-ip 100.69.169.33" sh -
```

### 2.4. Verify cluster

Trên `cloud-1`:

```bash
kubectl get nodes -o wide
```

Kết quả đã đạt:

```text
NAME      STATUS   ROLES           AGE   VERSION        INTERNAL-IP
cloud-1   Ready    control-plane   33d   v1.34.5+k3s1   172.31.18.213
edge-1    Ready    <none>          19h   v1.34.6+k3s1   100.82.147.9
edge-2    Ready    <none>          19h   v1.34.6+k3s1   100.69.169.33
```

Kết luận Bước 2:

- Cluster K3s 3 node đã hoạt động.
- `cloud-1` là control-plane.
- `edge-1` và `edge-2` là worker nodes.
- Tất cả node ở trạng thái `Ready`.

---

## Bước 3 — Label nodes theo role

### 3.1. Mục tiêu

Label node để Kubernetes có thể schedule Pod đúng vị trí bằng `nodeSelector`.

### 3.2. Lệnh label

```bash
kubectl label node cloud-1 role=cloud --overwrite
kubectl label node edge-1 role=edge_1 --overwrite
kubectl label node edge-2 role=edge_2 --overwrite
```

Verify:

```bash
kubectl get nodes -L role
```

Kết quả mong muốn:

```text
NAME      STATUS   ROLES           ROLE
cloud-1   Ready    control-plane    cloud
edge-1    Ready    <none>           edge_1
edge-2    Ready    <none>           edge_2
```

### 3.3. Ý nghĩa

Khi tạo Job/Pod, chỉ cần khai báo:

```yaml
nodeSelector:
  role: edge_1
```

thì Kubernetes sẽ schedule Pod vào `edge-1`.

Tương tự:

```yaml
nodeSelector:
  role: edge_2
```

sẽ schedule Pod vào `edge-2`.

```yaml
nodeSelector:
  role: cloud
```

sẽ schedule Pod vào `cloud-1`.

Kết luận Bước 3:

- Node đã được label đúng.
- `nodeSelector` đã hoạt động đúng khi test Pod thủ công.

---

## Bước 4 — Viết `dispatcher/pod_deployer.py`

### 4.1. Mục tiêu

Viết module Python dùng Kubernetes Python Client để tạo Kubernetes Job thật.

File chính:

```text
dispatcher/pod_deployer.py
```

Các hàm chính:

```python
deploy_task_pod(task_id, target_role, cpu_req, ram_req) -> job_name
wait_for_completion(job_name) -> (status, duration_ms)
deploy_and_wait(...) -> (job_name, status, total_duration_ms)
```

### 4.2. Chức năng chính

`pod_deployer.py` tạo Kubernetes Job với:

- `nodeSelector` theo `target_role`.
- Image: `ovapil/task-processor:v1`.
- Env var:
  - `TASK_ID`
  - `TARGET_ROLE`
  - `CPU_REQUIREMENT`
  - `RAM_REQUIREMENT`
  - `TASK_DURATION_SECONDS`
- Resource requests/limits:
  - CPU theo task.
  - RAM theo task.
- `restartPolicy: Never`.
- `backoffLimit: 0`.
- `ttlSecondsAfterFinished: 300`.

### 4.3. Test thủ công

Test vào `edge-1`:

```bash
PYTHONPATH=$PWD python3 dispatcher/pod_deployer.py \
  --task-id test-edge-1 \
  --target-role edge_1 \
  --cpu 500m \
  --ram 128Mi \
  --duration 2
```

Kết quả:

```text
[pod_deployer] Created Job: task-test-edge-1-1777349772736
[pod_deployer] target_role=edge_1, cpu=500m, ram=128Mi
[pod_deployer] Job succeeded: task-test-edge-1-1777349772736, node=edge-1, duration_ms=10060
[pod_deployer] deploy_and_wait result: job=task-test-edge-1-1777349772736, status=succeeded, total_duration_ms=10758
```

Test vào `edge-2`:

```bash
PYTHONPATH=$PWD python3 dispatcher/pod_deployer.py \
  --task-id test-edge-2 \
  --target-role edge_2 \
  --cpu 500m \
  --ram 128Mi \
  --duration 2
```

Kết quả:

```text
[pod_deployer] Created Job: task-test-edge-2-1777349830207
[pod_deployer] target_role=edge_2, cpu=500m, ram=128Mi
[pod_deployer] Job succeeded: task-test-edge-2-1777349830207, node=edge-2, duration_ms=7046
[pod_deployer] deploy_and_wait result: job=task-test-edge-2-1777349830207, status=succeeded, total_duration_ms=7289
```

Test vào `cloud`:

```bash
PYTHONPATH=$PWD python3 dispatcher/pod_deployer.py \
  --task-id test-cloud \
  --target-role cloud \
  --cpu 500m \
  --ram 128Mi \
  --duration 2
```

Kết quả:

```text
[pod_deployer] Created Job: task-test-cloud-1777349860329
[pod_deployer] target_role=cloud, cpu=500m, ram=128Mi
[pod_deployer] Job succeeded: task-test-cloud-1777349860329, node=cloud-1, duration_ms=5036
[pod_deployer] deploy_and_wait result: job=task-test-cloud-1777349860329, status=succeeded, total_duration_ms=5166
```

### 4.4. Kiểm tra Job/Pod

```bash
kubectl get jobs
kubectl get pods -o wide
```

Ví dụ kết quả:

```text
task-test-edge-1-...   Complete   1/1
task-test-edge-2-...   Complete   1/1
task-test-cloud-...    Complete   1/1
```

Pod chạy đúng node:

```text
task-test-edge-1-...   Completed   edge-1
task-test-edge-2-...   Completed   edge-2
task-test-cloud-...    Completed   cloud-1
```

Kết luận Bước 4:

- `pod_deployer.py` deploy được Kubernetes Job thật.
- Job sinh Pod thật.
- Pod chạy đúng node theo `nodeSelector`.
- Đo được latency thực từ submit Job đến Job Completed.

---

## Bước 5 — Tích hợp vào dispatcher

### 5.1. Mục tiêu

Tích hợp `pod_deployer.py` vào:

```text
dispatcher/smart_dispatcher.py
```

Cụ thể là sửa method:

```python
_execute_on_node(self, task, node_name)
```

Trước đây method này gọi HTTP worker:

```python
requests.post(url, json=payload, timeout=10)
```

Sau khi sửa, backend mặc định sẽ gọi Kubernetes Job thật thông qua:

```python
from dispatcher.pod_deployer import deploy_and_wait
```

### 5.2. Backend mới

Backend mặc định:

```bash
EXECUTION_BACKEND=k8s
```

Fallback về HTTP worker cũ:

```bash
EXECUTION_BACKEND=http
```

Khi chạy `--demo` nhưng vẫn muốn tạo Kubernetes Job thật, dùng:

```bash
FORCE_EXECUTE=true
```

### 5.3. Lý do cần `FORCE_EXECUTE=true`

Trong `dispatch()`, demo mode ban đầu chỉ chạy simulation và không gọi `_execute_on_node()`.

Đã sửa logic để:

```python
force_execute = os.getenv("FORCE_EXECUTE", "false").lower() in ("1", "true", "yes")

if (not self.demo_mode) or force_execute:
    self._execute_on_node(task, node_name)
```

Nhờ vậy, có thể dùng demo metrics nhưng vẫn tạo Kubernetes Job thật.

### 5.4. Test dispatcher 6 task

Lệnh chạy:

```bash
PYTHONPATH=$PWD EXECUTION_BACKEND=k8s FORCE_EXECUTE=true python3 dispatcher/dispatcher_cli.py \
  --policy round_robin \
  --num-tasks 6 \
  --edges 2 \
  --demo
```

Kết quả:

```text
[pod_deployer] Job succeeded: task-task-000006-1777370673846, node=cloud-1, duration_ms=5041
[pod_deployer] deploy_and_wait result: job=task-task-000006-1777370673846, status=succeeded, total_duration_ms=5583
SmartDispatcher: Task task_000006 → cloud | backend=k8s job=task-task-000006-1777370673846 status=succeeded latency_ms=5583 total_ms=5583
```

Tổng thời gian:

```text
Total time: 33.80s
```

Điều này chứng minh dispatcher đã gọi K8s backend thật, vì nếu chỉ simulation thì 6 task chỉ mất khoảng 0.1s.

### 5.5. Kiểm tra Job/Pod sau test 6 task

```bash
kubectl get jobs | tail -20
kubectl get pods -o wide | tail -20
```

Kết quả:

```text
task-task-000001-...   Complete   1/1
task-task-000002-...   Complete   1/1
task-task-000003-...   Complete   1/1
task-task-000004-...   Complete   1/1
task-task-000005-...   Complete   1/1
task-task-000006-...   Complete   1/1
```

Pod chạy đúng node:

```text
task_000001 -> edge-1
task_000002 -> edge-2
task_000003 -> cloud-1
task_000004 -> edge-1
task_000005 -> edge-2
task_000006 -> cloud-1
```

Kết luận Bước 5:

- `SmartDispatcher` đã tích hợp K8s backend.
- Dispatcher tạo Kubernetes Job thật.
- Pod chạy đúng node.
- Fallback HTTP vẫn được giữ lại bằng `EXECUTION_BACKEND=http`.

---

## Bước 6 — Verify end-to-end

### 6.1. Test 50 Kubernetes Jobs độc lập

Đã tạo file test:

```text
tests/test_50_k8s_jobs.py
```

Chạy:

```bash
PYTHONPATH=$PWD python3 tests/test_50_k8s_jobs.py
```

Kết quả file:

```text
k8s_50_tasks_results.csv
```

Thống kê:

```text
Total tasks: 50
Succeeded: 50
Failed: 0
Average latency_ms: 6287.66
edge_1 17
edge_2 17
cloud 16
```

Kết luận:

- 50/50 Job succeeded.
- Mỗi task tạo Kubernetes Job/Pod thật.
- Task được phân phối đều vào `edge_1`, `edge_2`, `cloud`.
- Latency thực trung bình khoảng `6287.66 ms`.

### 6.2. Test dispatcher 50 task

Lệnh chạy:

```bash
PYTHONPATH=$PWD EXECUTION_BACKEND=k8s FORCE_EXECUTE=true python3 dispatcher/dispatcher_cli.py \
  --policy round_robin \
  --num-tasks 50 \
  --edges 2 \
  --demo 2>&1 | tee dispatcher_k8s_50_tasks.log
```

Kết quả summary:

```text
Dispatcher Summary (50 tasks, policy=round_robin)
SLA Rate      : 100.0%
Avg Latency   : 69.9 ms
P95 Latency   : 156.0 ms
Avg Cost      : 1.37470
Cloud Usage   : 32.0%
Edge Usage    : 68.0%
Total time    : 279.09s
```

Kiểm tra log:

```bash
grep "status=succeeded" dispatcher_k8s_50_tasks.log | wc -l
```

Kết quả:

```text
50
```

Kiểm tra Kubernetes Job:

```bash
kubectl get jobs | grep task-task | tail -20
```

Kết quả:

```text
task-task-000031-...   Complete   1/1
task-task-000032-...   Complete   1/1
task-task-000033-...   Complete   1/1
...
task-task-000050-...   Complete   1/1
```

Kiểm tra Pod:

```bash
kubectl get pods -o wide | grep task-task | tail -20
```

Kết quả:

```text
task-task-000031-...   Completed   edge-1
task-task-000032-...   Completed   edge-2
task-task-000033-...   Completed   cloud-1
task-task-000034-...   Completed   edge-1
task-task-000035-...   Completed   edge-2
task-task-000036-...   Completed   cloud-1
...
task-task-000050-...   Completed   edge-2
```

Kết luận Bước 6:

- Dispatcher chạy 50 task thành công.
- 50/50 task có `status=succeeded`.
- Mỗi task tạo Kubernetes Job thật.
- Mỗi Job tạo Pod thật.
- Pod chạy đúng node theo `nodeSelector`.
- Tổng thời gian chạy thực tế: `279.09s`.
- Latency thực tế lớn hơn HTTP cũ là đúng kỳ vọng vì có pod scheduling và container startup overhead.

![image-20260428172023954](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20260428172023954.png)



![image-20260428172031393](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20260428172031393.png)



![image-20260428172039558](C:\Users\pc\AppData\Roaming\Typora\typora-user-images\image-20260428172039558.png)

---

## Bước 7 — Document

Tài liệu này ghi lại toàn bộ quá trình setup, các lệnh verify, kết quả test và troubleshooting.

Các file/sản phẩm chính sau khi hoàn thành:

| File / Artifact                  | Ý nghĩa                                     |
| -------------------------------- | ------------------------------------------- |
| `dispatcher/pod_deployer.py`     | Module deploy Kubernetes Job bằng Python    |
| `dispatcher/smart_dispatcher.py` | Dispatcher đã tích hợp K8s backend          |
| `ovapil/task-processor:v1`       | Docker image task processor                 |
| `tests/test_50_k8s_jobs.py`      | Script test 50 Kubernetes Jobs độc lập      |
| `k8s_50_tasks_results.csv`       | Kết quả test 50 Jobs độc lập                |
| `dispatcher_k8s_50_tasks.log`    | Log dispatcher chạy 50 task qua K8s backend |

---

# Kết quả tổng hợp

## 1. Cluster

```text
cloud-1   Ready   control-plane
edge-1    Ready   worker
edge-2    Ready   worker
```

## 2. Node labels

```text
cloud-1   role=cloud
edge-1    role=edge_1
edge-2    role=edge_2
```

## 3. Docker image

```text
ovapil/task-processor:v1
```

## 4. K8s Job

Mỗi task tạo một Kubernetes Job:

```text
task-task-000031-...
task-task-000032-...
task-task-000033-...
```

Mỗi Job tạo một Pod:

```text
task-task-000031-...   Completed   edge-1
task-task-000032-...   Completed   edge-2
task-task-000033-...   Completed   cloud-1
```

## 5. Dispatcher

Dispatcher đã gọi K8s backend thật:

```text
SmartDispatcher: Task task_000006 → cloud | backend=k8s job=task-task-000006-... status=succeeded latency_ms=5583
```

---

# Lưu ý về latency

Trong output của `dispatcher_cli.py` có hai loại latency:

## 1. Latency estimate của dispatcher

Ví dụ:

```text
Avg Latency: 69.9 ms
```

Đây là latency estimate dùng cho thuật toán/simulation, được tính trong `_estimate_execution()`.

## 2. Latency thực của Kubernetes Job

Ví dụ:

```text
[pod_deployer] Job succeeded: ..., duration_ms=5041
[pod_deployer] deploy_and_wait result: ..., total_duration_ms=5583
```

Đây là latency thật từ lúc submit Kubernetes Job đến khi Job Completed.

Khi báo cáo, cần nói rõ:

> Dispatcher vẫn giữ latency estimate để phục vụ thuật toán và dashboard, còn latency thực của K8s execution được đo bởi `pod_deployer.py`.

---

# Troubleshooting

## 1. Lỗi token: `invalid token CA hash length`

Nguyên nhân:

- Token copy sai.
- Có dấu nháy ở đầu.
- Có khoảng trắng trước `K10`.
- Token bị thiếu đoạn `::server:...`.

Kiểm tra token file:

```bash
sudo awk '{print "len=" length($0), "prefix=" substr($0,1,3), "has_double_colon_pos=" index($0,"::")}' /etc/rancher/k3s/node-token
```

Đúng:

```text
prefix=K10
```

Sai:

```text
prefix=' K
```

Cách sửa:

- Mở token file.
- Xóa dấu nháy hoặc khoảng trắng.
- Đảm bảo token bắt đầu ngay bằng `K10`.

## 2. Lỗi `not authorized` khi join agent

Nguyên nhân thường là token sai hoặc token lấy từ cluster khác.

Cách xử lý:

- Lấy token trực tiếp từ `cloud-1`:

```bash
sudo cat /var/lib/rancher/k3s/server/node-token
```

- Ghi token vào:

```text
/etc/rancher/k3s/node-token
```

- Join lại agent.

## 3. Edge node hiện IP LAN `192.168.x.x`

Nếu `edge-1` hoặc `edge-2` hiện IP LAN không truy cập được từ cloud, `kubectl logs` hoặc API proxy có thể lỗi.

Cách xử lý:

Join agent với `--node-ip` là IP reachable từ cloud:

```bash
--node-ip 100.82.147.9
```

hoặc:

```bash
--node-ip 100.69.169.33
```

## 4. `kubectl logs` tới edge bị `502 Bad Gateway`

Kiểm tra cloud có gọi được kubelet port không:

```bash
nc -vz 100.82.147.9 10250
nc -vz 100.69.169.33 10250
```

Nếu kết quả là:

```text
succeeded
```

thì network đã thông.

Kiểm tra kubelet healthz:

```bash
curl -k https://100.82.147.9:10250/healthz
```

Nếu trả về:

```text
Unauthorized
```

thì kubelet reachable nhưng yêu cầu xác thực. Điều này không chặn Job chạy thành công.

## 5. Python không đọc được kubeconfig

Copy kubeconfig cho user:

```bash
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER:$USER ~/.kube/config
chmod 600 ~/.kube/config
echo 'export KUBECONFIG=$HOME/.kube/config' >> ~/.bashrc
source ~/.bashrc
```

Kiểm tra:

```bash
kubectl get nodes -o wide
```

## 6. Python thiếu thư viện

Một số lỗi đã gặp:

```text
ModuleNotFoundError: No module named 'kubernetes'
ModuleNotFoundError: No module named 'numpy'
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'gymnasium'
```

Cách xử lý:

```bash
python3 -m pip install kubernetes
python3 -m pip install numpy
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install gymnasium
```

---

# Definition of Done

| Tiêu chí                                                     | Trạng thái |
| ------------------------------------------------------------ | ---------- |
| `kubectl get nodes` hiển thị 3 node Ready với label đúng     | Hoàn thành |
| Docker image push được lên registry, kéo về node chạy được   | Hoàn thành |
| `pod_deployer.py` deploy được Job, đo được latency           | Hoàn thành |
| Dispatcher chạy 50 task không lỗi, mỗi task tạo Pod đúng node | Hoàn thành |
| Document setup được người khác replicate được                | Hoàn thành |

---

# Kết luận

Task **Real K3s Pod Deployment** đã hoàn thành.

Hệ thống hiện tại đã chuyển từ cơ chế giả lập HTTP worker sang Kubernetes Job thật. Mỗi task được triển khai thành một Pod thật trên K3s, chạy đúng node thông qua `nodeSelector`, có CPU/RAM resource limit, có đo latency thực từ submit Job đến Job Completed.

Kết quả test quan trọng nhất:

```text
50 Kubernetes Jobs độc lập:
- Total tasks: 50
- Succeeded: 50
- Failed: 0
- Average latency_ms: 6287.66

Dispatcher 50 tasks:
- status=succeeded: 50
- SLA Rate: 100.0%
- Total time: 279.09s
- Pod chạy đúng edge-1, edge-2, cloud-1
```

Kết quả này đủ để defend rằng hệ thống đã sử dụng Kubernetes/K3s thật, không còn chỉ là fake K8s hoặc HTTP worker simulation.