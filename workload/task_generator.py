"""Task workload generator for Week 1 deliverable.

Supported patterns:
- constant load
- bursty load
- diurnal load

Usage examples:
    python workload/task_generator.py --pattern constant --count 100 --output workload/tasks_constant.csv
    python workload/task_generator.py --pattern bursty --count 300 --output workload/tasks_bursty.csv
    python workload/task_generator.py --pattern diurnal --count 500 --output workload/tasks_diurnal.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List
import math
import random


PRIORITIES = ("low", "medium", "high")
PAYLOAD_TYPES = ("compute", "image", "io")


@dataclass
class Task:
    task_id: str
    arrival_time: str
    cpu_requirement: float
    ram_requirement_mb: int
    deadline_ms: int
    priority: str
    payload_type: str


def _make_task(task_index: int, arrival_dt: datetime, rng: random.Random) -> Task:
    cpu_req = round(rng.uniform(0.1, 2.0), 3)
    ram_req = rng.randint(128, 2048)
    deadline_ms = rng.choice((5000, 10000, 20000))
    priority = rng.choices(PRIORITIES, weights=(0.2, 0.6, 0.2), k=1)[0]
    payload_type = rng.choice(PAYLOAD_TYPES)

    return Task(
        task_id=f"task_{task_index:06d}",
        arrival_time=arrival_dt.isoformat(),
        cpu_requirement=cpu_req,
        ram_requirement_mb=ram_req,
        deadline_ms=deadline_ms,
        priority=priority,
        payload_type=payload_type,
    )


def generate_constant_load(
    count: int,
    rate_per_minute: float = 10.0,
    start_time: datetime | None = None,
    seed: int = 42,
) -> List[Task]:
    """
    Generate tasks với Poisson process — inter-arrival time exponential.
    Rate cố định λ tasks/phút.
    """
    if count <= 0:
        return []

    rng = random.Random(seed)
    start = start_time or datetime.now(timezone.utc)
    mean_interval_sec = 60.0 / max(rate_per_minute, 0.1)

    tasks: List[Task] = []
    current = start
    for i in range(count):
        if i > 0:
            # Exponential inter-arrival → Poisson process
            interval = rng.expovariate(1.0 / mean_interval_sec)
            current = current + timedelta(seconds=max(0.01, interval))
        tasks.append(_make_task(i + 1, current, rng))
    return tasks


def generate_bursty_load(
    count: int,
    burst_rate_per_minute: float = 50.0,
    idle_seconds: float = 30.0,
    burst_size: int = 25,
    start_time: datetime | None = None,
    seed: int = 42,
) -> List[Task]:
    """Generate tasks in short bursts separated by idle gaps."""
    if count <= 0:
        return []

    rng = random.Random(seed)
    start = start_time or datetime.now(timezone.utc)
    burst_interval = 60.0 / burst_rate_per_minute

    tasks: List[Task] = []
    current = start
    created = 0
    while created < count:
        n_this_burst = min(burst_size, count - created)
        for _ in range(n_this_burst):
            if created > 0:
                current = current + timedelta(seconds=max(0.02, burst_interval + rng.uniform(-0.1, 0.1)))
            created += 1
            tasks.append(_make_task(created, current, rng))

        # Idle phase after each burst.
        current = current + timedelta(seconds=idle_seconds + rng.uniform(-5.0, 5.0))

    return tasks


def diurnal_lambda(hour: float, base: float = 5.0, peak_extra: float = 10.0) -> float:
    """
    Rate λ tasks/phút theo giờ trong ngày.
    Cao nhất quanh 12h trưa, thấp nhất ban đêm.

    λ(hour) = base + peak_extra * max(0, sin(π * (hour - 6) / 12))
      hour 6-18: nửa chu kỳ sin → workload cao ban ngày
      hour 18-6: max(0, sin<0) = 0 → chỉ có baseline
    """
    return base + peak_extra * max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))


def generate_diurnal_load(
    count: int,
    base_rate_per_minute: float = 5.0,
    peak_extra_per_minute: float = 10.0,
    start_time: datetime | None = None,
    seed: int = 42,
) -> List[Task]:
    """
    Poisson process với rate λ(t) thay đổi theo giờ trong ngày.
    Inter-arrival time = exponential(1/λ(current_hour)) — non-homogeneous Poisson.
    """
    if count <= 0:
        return []

    rng = random.Random(seed)
    start = start_time or datetime.now(timezone.utc)

    tasks: List[Task] = []
    current = start
    for i in range(count):
        if i > 0:
            hour = current.hour + current.minute / 60.0
            rate = diurnal_lambda(hour, base_rate_per_minute, peak_extra_per_minute)
            mean_interval_sec = 60.0 / max(rate, 0.1)
            interval = rng.expovariate(1.0 / mean_interval_sec)
            current = current + timedelta(seconds=max(0.01, interval))
        tasks.append(_make_task(i + 1, current, rng))

    return tasks


def save_tasks_to_csv(tasks: List[Task], output_path: str) -> None:
    if not tasks:
        raise ValueError("No tasks to write")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(tasks[0]).keys()))
        writer.writeheader()
        for task in tasks:
            writer.writerow(asdict(task))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic task workload")
    parser.add_argument("--pattern", choices=("constant", "bursty", "diurnal"), default="constant")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="workload/tasks.csv")
    args = parser.parse_args()

    start_time = datetime.now(timezone.utc)

    if args.pattern == "constant":
        tasks = generate_constant_load(count=args.count, start_time=start_time, seed=args.seed)
    elif args.pattern == "bursty":
        tasks = generate_bursty_load(count=args.count, start_time=start_time, seed=args.seed)
    else:
        tasks = generate_diurnal_load(count=args.count, start_time=start_time, seed=args.seed)

    save_tasks_to_csv(tasks, args.output)
    print(f"Generated {len(tasks)} tasks using pattern='{args.pattern}'")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
