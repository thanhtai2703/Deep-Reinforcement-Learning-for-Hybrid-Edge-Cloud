import os
import time
import math
import hashlib


def parse_cpu_to_cores(value: str) -> float:
    """
    Parse CPU requirement.
    Examples:
    - "500m" -> 0.5 core
    - "1"    -> 1.0 core
    - "2"    -> 2.0 cores
    """
    if not value:
        return 0.5

    value = value.strip().lower()

    if value.endswith("m"):
        return max(float(value[:-1]) / 1000.0, 0.1)

    return max(float(value), 0.1)


def parse_memory_to_mb(value: str) -> int:
    """
    Parse RAM requirement.
    Examples:
    - "128Mi" -> 128 MB
    - "512Mi" -> 512 MB
    - "1Gi"   -> 1024 MB
    """
    if not value:
        return 64

    value = value.strip().lower()

    if value.endswith("mi"):
        return int(float(value[:-2]))

    if value.endswith("gi"):
        return int(float(value[:-2]) * 1024)

    if value.endswith("m"):
        return int(float(value[:-1]))

    return int(float(value))


def burn_cpu(duration_seconds: float, cpu_cores: float):
    """
    CPU burn workload. Cố định khối lượng tính toán thay vì thời gian.
    Nếu K8s bóp băng thông CPU (limits), quá trình này sẽ tự nhiên mất nhiều thời gian hơn.
    """
    # Trung bình 1 giây 1 nhân CPU có thể giải 1,000,000 mã băm SHA256
    total_hashes = int(1_000_000 * duration_seconds * max(cpu_cores, 0.1))
    
    payload = b"edge-cloud-task"
    
    # Giữ nguyên cấu trúc chia nhỏ chunk để log không bị đổi format
    chunk_size = max(int(cpu_cores * 10000), 1000)
    iterations_needed = max(1, total_hashes // chunk_size)
    
    for _ in range(iterations_needed):
        data = payload
        for _ in range(chunk_size):
            data = hashlib.sha256(data).digest()
            
    return iterations_needed


def allocate_memory(memory_mb: int, ram_intensity: float):
    """
    Allocate memory proportional to ram_intensity (0-100 scale).
    safe_mb = ram_intensity% of memory_mb (clamp ≤ 80% to avoid OOMKilled).
    """
    fraction = min(max(ram_intensity / 100.0, 0.1), 0.8)
    safe_mb = max(int(memory_mb * fraction), 8)
    block = bytearray(safe_mb * 1024 * 1024)

    for i in range(0, len(block), 4096):
        block[i] = 1

    return block, safe_mb


def main():
    task_id = os.getenv("TASK_ID", "unknown-task")
    target_role = os.getenv("TARGET_ROLE", "unknown-role")
    cpu_requirement = os.getenv("CPU_REQUIREMENT", "500m")
    ram_requirement = os.getenv("RAM_REQUIREMENT", "128Mi")
    duration_seconds = float(os.getenv("TASK_DURATION_SECONDS", "2.0"))
    # Raw intensity (0-100) drives actual workload size, separate from K8s limits.
    cpu_intensity = float(os.getenv("CPU_INTENSITY", "30"))
    ram_intensity = float(os.getenv("RAM_INTENSITY", "30"))

    # Heavy task = burns longer. Reference cpu_intensity=30 keeps original duration.
    actual_duration = duration_seconds * (cpu_intensity / 30.0)

    cpu_cores = parse_cpu_to_cores(cpu_requirement)
    memory_mb = parse_memory_to_mb(ram_requirement)

    print("=" * 60, flush=True)
    print(f"[task_processor] START task_id={task_id}", flush=True)
    print(f"[task_processor] target_role={target_role}", flush=True)
    print(f"[task_processor] CPU_REQUIREMENT={cpu_requirement} parsed={cpu_cores} cores", flush=True)
    print(f"[task_processor] RAM_REQUIREMENT={ram_requirement} parsed={memory_mb} MB", flush=True)
    print(f"[task_processor] TASK_DURATION_SECONDS={duration_seconds}", flush=True)
    print(f"[task_processor] CPU_INTENSITY={cpu_intensity} RAM_INTENSITY={ram_intensity}", flush=True)
    print(f"[task_processor] actual_burn_duration={actual_duration:.2f}s", flush=True)

    start = time.time()

    mem_block, allocated_mb = allocate_memory(memory_mb, ram_intensity)
    print(f"[task_processor] allocated_memory={allocated_mb} MB", flush=True)

    iterations = burn_cpu(actual_duration, cpu_cores)

    elapsed_ms = int((time.time() - start) * 1000)

    print(f"[task_processor] iterations={iterations}", flush=True)
    print(f"[task_processor] completed duration_ms={elapsed_ms}", flush=True)
    print(f"[task_processor] END task_id={task_id}", flush=True)
    print("=" * 60, flush=True)

    # Keep reference so memory allocation is not optimized away.
    _ = mem_block

    exit(0)


if __name__ == "__main__":
    main()