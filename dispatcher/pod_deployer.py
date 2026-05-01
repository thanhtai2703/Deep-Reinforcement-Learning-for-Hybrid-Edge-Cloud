import argparse
import re
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from kubernetes import client, config
from kubernetes.client.rest import ApiException


@dataclass
class TimingBreakdown:
    """
    Per-task timing components captured from K8s for sim calibration.

    K8s timestamps have 1s resolution. Use only for distribution-level fitting,
    not per-sample millisecond accuracy.

    Fields (ms, None if not available):
    - submit_overhead_ms : job_created -> pod_scheduled (API + scheduler)
    - container_startup_ms : pod_scheduled -> container_started (image pull + init)
    - exec_time_ms       : container_started -> container_finished (compute)
    - total_ms           : dispatcher-side wall clock, submit -> wait_complete
    """
    total_ms: int = 0
    submit_overhead_ms: Optional[int] = None
    container_startup_ms: Optional[int] = None
    exec_time_ms: Optional[int] = None
    poll_overhead_ms: Optional[int] = None
    node_name: Optional[str] = None
    job_created_at: Optional[str] = None
    pod_scheduled_at: Optional[str] = None
    container_started_at: Optional[str] = None
    container_finished_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


IMAGE_NAME = "taithanh/task-processor:v1"
NAMESPACE = "default"
KUBECONFIG_PATH = "/home/ubuntu/.kube/config"


def _safe_name(value: str) -> str:
    """
    Convert any task id to a Kubernetes-safe name.
    Kubernetes Job names must be lowercase DNS-compatible names.
    """
    value = value.lower()
    value = re.sub(r"[^a-z0-9-]", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:45] or "task"


def load_k8s_config() -> None:
    """
    Load Kubernetes config.
    On cloud-1, K3s kubeconfig is usually at /etc/rancher/k3s/k3s.yaml.
    """
    try:
        config.load_kube_config(config_file=KUBECONFIG_PATH)
    except Exception:
        config.load_incluster_config()


def deploy_task_pod(
    task_id: str,
    target_role: str,
    cpu_req: str,
    ram_req: str,
    duration_seconds: float = 2.0,
    namespace: str = NAMESPACE,
    image: str = IMAGE_NAME,
    cpu_intensity: float = 30.0,
    ram_intensity: float = 30.0,
) -> str:
    """
    Deploy one task as a real Kubernetes Job.

    Parameters:
    - task_id: unique task id
    - target_role: one of edge_1, edge_2, cloud
    - cpu_req: CPU request/limit, e.g. "500m", "1"
    - ram_req: RAM request/limit, e.g. "128Mi", "512Mi"
    - duration_seconds: CPU burn duration inside the container

    Returns:
    - job_name
    """
    load_k8s_config()
    batch_api = client.BatchV1Api()

    safe_task_id = _safe_name(task_id)
    timestamp = int(time.time() * 1000)
    job_name = f"task-{safe_task_id}-{timestamp}"[:63].rstrip("-")

    # Bóp hiệu năng của Edge để tạo sự chênh lệch phần cứng (Throttling)
    # Cloud chạy tẹt ga (theo request), Edge bị giới hạn cứng ở 200m (0.2 CPU)
    limit_cpu = cpu_req
    if target_role.startswith("edge"):
        limit_cpu = "200m"  # Ép task trên Edge chạy chậm hơn Cloud gấp nhiều lần!

    container = client.V1Container(
        name="task-processor",
        image=image,
        image_pull_policy="IfNotPresent",
        env=[
            client.V1EnvVar(name="TASK_ID", value=task_id),
            client.V1EnvVar(name="TARGET_ROLE", value=target_role),
            client.V1EnvVar(name="CPU_REQUIREMENT", value=cpu_req),
            client.V1EnvVar(name="RAM_REQUIREMENT", value=ram_req),
            client.V1EnvVar(name="TASK_DURATION_SECONDS", value=str(duration_seconds)),
            client.V1EnvVar(name="CPU_INTENSITY", value=str(cpu_intensity)),
            client.V1EnvVar(name="RAM_INTENSITY", value=str(ram_intensity)),
        ],
        resources=client.V1ResourceRequirements(
            requests={
                "cpu": limit_cpu,  # Sửa ở đây: requests không được vượt quá limits
                "memory": ram_req,
            },
            limits={
                "cpu": limit_cpu,
                "memory": ram_req,
            },
        ),
    )

    pod_template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={
                "app": "task-processor",
                "task-id": safe_task_id,
                "job-name": job_name,
            }
        ),
        spec=client.V1PodSpec(
            restart_policy="Never",
            node_selector={
                "role": target_role,
            },
            containers=[container],
        ),
    )

    job_spec = client.V1JobSpec(
        template=pod_template,
        backoff_limit=0,
        ttl_seconds_after_finished=300,
    )

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(
            name=job_name,
            labels={
                "app": "task-processor",
                "target-role": target_role,
            },
        ),
        spec=job_spec,
    )

    try:
        batch_api.create_namespaced_job(namespace=namespace, body=job)
        print(f"[pod_deployer] Created Job: {job_name}")
        print(f"[pod_deployer] target_role={target_role}, cpu={cpu_req}, ram={ram_req}")
        return job_name
    except ApiException as e:
        print(f"[pod_deployer] Failed to create Job: {e}")
        raise


def get_job_pod_node(job_name: str, namespace: str = NAMESPACE) -> Optional[str]:
    """
    Return the node name where the Job's Pod is running.
    """
    load_k8s_config()
    core_api = client.CoreV1Api()

    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"job-name={job_name}",
    )

    if not pods.items:
        return None

    return pods.items[0].spec.node_name


def _ms_between(t_start, t_end) -> Optional[int]:
    """Return (t_end - t_start) in ms, or None if either is missing."""
    if t_start is None or t_end is None:
        return None
    return int((t_end - t_start).total_seconds() * 1000)


def fetch_k8s_timings(
    job_name: str,
    namespace: str = NAMESPACE,
) -> TimingBreakdown:
    """
    Fetch all K8s server-side timestamps for a finished Job and compute
    component-level timing breakdown. K8s timestamps have 1-second resolution,
    so per-sample error can be ±1s. Suitable for distribution fitting, not
    for high-precision benchmarking.
    """
    load_k8s_config()
    batch_api = client.BatchV1Api()
    core_api = client.CoreV1Api()

    breakdown = TimingBreakdown()

    try:
        job = batch_api.read_namespaced_job(name=job_name, namespace=namespace)
        job_created = job.metadata.creation_timestamp
        breakdown.job_created_at = job_created.isoformat() if job_created else None
    except ApiException:
        job_created = None

    try:
        pods = core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=f"job-name={job_name}",
        )
    except ApiException:
        return breakdown

    if not pods.items:
        return breakdown

    pod = pods.items[0]
    breakdown.node_name = pod.spec.node_name

    pod_scheduled = pod.status.start_time
    breakdown.pod_scheduled_at = pod_scheduled.isoformat() if pod_scheduled else None

    container_started = None
    container_finished = None
    if pod.status.container_statuses:
        cs = pod.status.container_statuses[0]
        if cs.state and cs.state.terminated:
            container_started = cs.state.terminated.started_at
            container_finished = cs.state.terminated.finished_at
        elif cs.state and cs.state.running:
            container_started = cs.state.running.started_at

    breakdown.container_started_at = (
        container_started.isoformat() if container_started else None
    )
    breakdown.container_finished_at = (
        container_finished.isoformat() if container_finished else None
    )

    breakdown.submit_overhead_ms = _ms_between(job_created, pod_scheduled)
    breakdown.container_startup_ms = _ms_between(pod_scheduled, container_started)
    breakdown.exec_time_ms = _ms_between(container_started, container_finished)

    return breakdown


def wait_for_completion(
    job_name: str,
    namespace: str = NAMESPACE,
    timeout_seconds: int = 180,
    poll_interval: float = 1.0,
) -> Tuple[str, TimingBreakdown]:
    """
    Wait until the Kubernetes Job completes or fails, then fetch K8s timing
    breakdown.

    Returns:
    - status: "succeeded", "failed", or "timeout"
    - timings: TimingBreakdown with component-level latencies
    """
    load_k8s_config()
    batch_api = client.BatchV1Api()

    start = time.perf_counter()
    final_status = None

    while True:
        elapsed_seconds = time.perf_counter() - start

        if elapsed_seconds > timeout_seconds:
            final_status = "timeout"
            break

        job = batch_api.read_namespaced_job_status(
            name=job_name,
            namespace=namespace,
        )

        status = job.status

        if status.succeeded and status.succeeded >= 1:
            final_status = "succeeded"
            break

        if status.failed and status.failed >= 1:
            final_status = "failed"
            break

        time.sleep(poll_interval)

    duration_ms = int((time.perf_counter() - start) * 1000)
    timings = fetch_k8s_timings(job_name, namespace)
    timings.total_ms = duration_ms

    if timings.exec_time_ms is not None and timings.container_startup_ms is not None and timings.submit_overhead_ms is not None:
        accounted = timings.submit_overhead_ms + timings.container_startup_ms + timings.exec_time_ms
        timings.poll_overhead_ms = max(0, duration_ms - accounted)

    print(
        f"[pod_deployer] Job {final_status}: {job_name}, "
        f"node={timings.node_name}, total_ms={duration_ms}, "
        f"submit={timings.submit_overhead_ms}, "
        f"startup={timings.container_startup_ms}, "
        f"exec={timings.exec_time_ms}, "
        f"poll={timings.poll_overhead_ms}"
    )

    return final_status, timings


def delete_job(job_name: str, namespace: str = NAMESPACE) -> None:
    """
    Optional cleanup function.
    Deletes the Job and its Pod.
    """
    load_k8s_config()
    batch_api = client.BatchV1Api()

    try:
        batch_api.delete_namespaced_job(
            name=job_name,
            namespace=namespace,
            propagation_policy="Background",
        )
        print(f"[pod_deployer] Deleted Job: {job_name}")
    except ApiException as e:
        if e.status == 404:
            print(f"[pod_deployer] Job already deleted: {job_name}")
        else:
            raise


def deploy_and_wait(
    task_id: str,
    target_role: str,
    cpu_req: str,
    ram_req: str,
    duration_seconds: float = 2.0,
    cleanup: bool = False,
    cpu_intensity: float = 30.0,
    ram_intensity: float = 30.0,
) -> Tuple[str, str, TimingBreakdown]:
    """
    Helper for dispatcher integration.

    Returns:
    - job_name
    - status: "succeeded", "failed", or "timeout"
    - timings: TimingBreakdown with submit_overhead, container_startup,
               exec_time, poll_overhead, total_ms, plus raw timestamps.
               Use .total_ms for the dispatcher-side wall-clock latency
               (overrides the previous int return).
    """
    submit_start = time.perf_counter()

    job_name = deploy_task_pod(
        task_id=task_id,
        target_role=target_role,
        cpu_req=cpu_req,
        ram_req=ram_req,
        duration_seconds=duration_seconds,
        cpu_intensity=cpu_intensity,
        ram_intensity=ram_intensity,
    )

    status, timings = wait_for_completion(job_name)

    total_duration_ms = int((time.perf_counter() - submit_start) * 1000)
    timings.total_ms = total_duration_ms

    print(
        f"[pod_deployer] deploy_and_wait result: "
        f"job={job_name}, status={status}, total_ms={total_duration_ms}, "
        f"breakdown=submit:{timings.submit_overhead_ms} "
        f"startup:{timings.container_startup_ms} "
        f"exec:{timings.exec_time_ms} poll:{timings.poll_overhead_ms}"
    )

    if cleanup:
        delete_job(job_name)

    return job_name, status, timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy task as Kubernetes Job")
    parser.add_argument("--task-id", default="manual-test-001")
    parser.add_argument("--target-role", default="edge_1", choices=["edge_1", "edge_2", "cloud"])
    parser.add_argument("--cpu", default="500m")
    parser.add_argument("--ram", default="128Mi")
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--cleanup", action="store_true")

    args = parser.parse_args()

    deploy_and_wait(
        task_id=args.task_id,
        target_role=args.target_role,
        cpu_req=args.cpu,
        ram_req=args.ram,
        duration_seconds=args.duration,
        cleanup=args.cleanup,
    )