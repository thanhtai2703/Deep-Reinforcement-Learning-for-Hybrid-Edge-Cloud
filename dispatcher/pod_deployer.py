import argparse
import re
import time
from typing import Optional, Tuple

from kubernetes import client, config
from kubernetes.client.rest import ApiException


IMAGE_NAME = "ovapil/task-processor:v1"
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
        ],
        resources=client.V1ResourceRequirements(
            requests={
                "cpu": cpu_req,
                "memory": ram_req,
            },
            limits={
                "cpu": cpu_req,
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


def wait_for_completion(
    job_name: str,
    namespace: str = NAMESPACE,
    timeout_seconds: int = 180,
    poll_interval: float = 1.0,
) -> Tuple[str, int]:
    """
    Wait until the Kubernetes Job completes or fails.

    Returns:
    - status: "succeeded", "failed", or "timeout"
    - duration_ms: latency from submit/wait start to final status
    """
    load_k8s_config()
    batch_api = client.BatchV1Api()

    start = time.perf_counter()

    while True:
        elapsed_seconds = time.perf_counter() - start

        if elapsed_seconds > timeout_seconds:
            duration_ms = int(elapsed_seconds * 1000)
            print(f"[pod_deployer] Job timeout: {job_name}, duration_ms={duration_ms}")
            return "timeout", duration_ms

        job = batch_api.read_namespaced_job_status(
            name=job_name,
            namespace=namespace,
        )

        status = job.status

        if status.succeeded and status.succeeded >= 1:
            duration_ms = int((time.perf_counter() - start) * 1000)
            node_name = get_job_pod_node(job_name, namespace)
            print(
                f"[pod_deployer] Job succeeded: {job_name}, "
                f"node={node_name}, duration_ms={duration_ms}"
            )
            return "succeeded", duration_ms

        if status.failed and status.failed >= 1:
            duration_ms = int((time.perf_counter() - start) * 1000)
            node_name = get_job_pod_node(job_name, namespace)
            print(
                f"[pod_deployer] Job failed: {job_name}, "
                f"node={node_name}, duration_ms={duration_ms}"
            )
            return "failed", duration_ms

        time.sleep(poll_interval)


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
) -> Tuple[str, str, int]:
    """
    Helper for dispatcher integration.

    Returns:
    - job_name
    - status
    - duration_ms
    """
    submit_start = time.perf_counter()

    job_name = deploy_task_pod(
        task_id=task_id,
        target_role=target_role,
        cpu_req=cpu_req,
        ram_req=ram_req,
        duration_seconds=duration_seconds,
    )

    status, wait_duration_ms = wait_for_completion(job_name)

    total_duration_ms = int((time.perf_counter() - submit_start) * 1000)

    print(
        f"[pod_deployer] deploy_and_wait result: "
        f"job={job_name}, status={status}, total_duration_ms={total_duration_ms}"
    )

    if cleanup:
        delete_job(job_name)

    return job_name, status, total_duration_ms


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