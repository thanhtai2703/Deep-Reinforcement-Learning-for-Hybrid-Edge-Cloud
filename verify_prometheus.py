"""
verify_prometheus.py
====================
Chứng minh hệ thống đang đọc metrics THẬT từ Prometheus,
không phải fallback sang simulation.

Chạy:
    python verify_prometheus.py
    python verify_prometheus.py --prometheus http://localhost:9090
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {BLUE}→{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")
def sep(): print("─" * 60)


def check_prometheus_reachable(url: str) -> bool:
    header("BƯỚC 1 — Kết nối Prometheus")
    try:
        import requests
        r = requests.get(f"{url.rstrip('/')}/-/healthy", timeout=5)
        if r.status_code == 200:
            ok(f"Prometheus phản hồi tại {url}")
            return True
        else:
            fail(f"Prometheus trả về HTTP {r.status_code}")
            return False
    except Exception as e:
        fail(f"Không kết nối được Prometheus: {e}")
        info("Kiểm tra: Prometheus có đang chạy không? (docker ps / systemctl status prometheus)")
        return False


def query_raw_instances(url: str) -> dict:
    """Truy vấn thẳng Prometheus, trả về {instance: {cpu, ram, latency}}"""
    from week2.prometheus_client import PrometheusClient, PrometheusConfig
    config = PrometheusConfig(base_url=url, timeout_seconds=10)
    client = PrometheusClient(config)

    header("BƯỚC 2 — Query thô từ Prometheus")

    # CPU
    try:
        cpu_data = client.get_cpu_usage_pct()
        if cpu_data:
            ok(f"CPU query thành công — tìm thấy {len(cpu_data)} instance(s)")
            for inst, val in cpu_data.items():
                info(f"  [{inst}]  CPU = {val:.1f}%")
        else:
            warn("CPU query trả về rỗng — chưa có node_exporter nào scrape được")
        ok_cpu = bool(cpu_data)
    except Exception as e:
        fail(f"CPU query lỗi: {e}")
        cpu_data = {}
        ok_cpu = False

    # RAM
    try:
        ram_data = client.get_ram_usage_pct()
        if ram_data:
            ok(f"RAM query thành công — {len(ram_data)} instance(s)")
            for inst, val in ram_data.items():
                info(f"  [{inst}]  RAM = {val:.1f}%")
        else:
            warn("RAM query trả về rỗng")
        ok_ram = bool(ram_data)
    except Exception as e:
        fail(f"RAM query lỗi: {e}")
        ram_data = {}
        ok_ram = False

    # Latency (custom metric — thường không có)
    try:
        lat_data = client.get_network_latency_ms()
        if lat_data:
            ok(f"Latency (edge_cloud_rtt_ms) query thành công — {len(lat_data)} instance(s)")
            for inst, val in lat_data.items():
                info(f"  [{inst}]  Latency = {val:.1f}ms")
        else:
            warn("Latency (edge_cloud_rtt_ms) trả về rỗng — sẽ dùng synthetic (CPU-based)")
            info("→ Đây là bình thường nếu bạn chưa cài custom exporter")
    except Exception as e:
        warn(f"Latency query không thành công: {e} — sẽ dùng synthetic")
        lat_data = {}

    return {"cpu": cpu_data, "ram": ram_data, "latency": lat_data}


def check_instance_map(raw: dict) -> dict:
    """So sánh instances từ Prometheus với INSTANCE_MAP trong infra_config.py"""
    header("BƯỚC 3 — Kiểm tra INSTANCE_MAP (infra_config.py)")

    try:
        from dispatcher.infra_config import INSTANCE_MAP
        info(f"INSTANCE_MAP hiện tại:")
        for ip, role in INSTANCE_MAP.items():
            print(f"    {ip!r:30s} → {role!r}")
    except ImportError:
        warn("Không import được infra_config.py")
        return {}

    cpu_instances = set(raw["cpu"].keys())
    ram_instances = set(raw["ram"].keys())
    prom_instances = cpu_instances | ram_instances

    mapped_instances = set(INSTANCE_MAP.keys())

    print()
    matched = prom_instances & mapped_instances
    unmatched_prom = prom_instances - mapped_instances
    unmatched_config = mapped_instances - prom_instances

    if matched:
        ok(f"Matched ({len(matched)}): {matched}")
    if unmatched_prom:
        fail(f"Có trong Prometheus nhưng KHÔNG có trong INSTANCE_MAP: {unmatched_prom}")
        info("→ Thêm các instance này vào infra_config.py")
    if unmatched_config:
        warn(f"Có trong INSTANCE_MAP nhưng Prometheus chưa scrape: {unmatched_config}")
        info("→ Kiểm tra node_exporter đang chạy và prometheus.yml có scrape đúng target không")

    return {"matched": matched, "unmatched_prom": unmatched_prom, "unmatched_config": unmatched_config}


def check_dispatcher_state(url: str) -> None:
    """Chạy StateBuilder thật và so sánh output"""
    header("BƯỚC 4 — StateBuilder thực tế")

    try:
        from dispatcher.infra_config import INSTANCE_MAP
        from dispatcher.state_builder import StateBuilder, TaskInfo

        sb_real = StateBuilder(
            n_edge_nodes=2,
            prometheus_url=url,
            use_prometheus=True,
            instance_map=INSTANCE_MAP,
        )
        sb_sim = StateBuilder(n_edge_nodes=2, use_prometheus=False)
        sb_sim.reset_simulation(seed=99)

        dummy_task = TaskInfo(
            task_id="diag_001",
            cpu_requirement=30.0,
            ram_requirement=25.0,
            deadline_ms=200.0,
        )

        state_real = sb_real.build_state(dummy_task)
        state_sim  = sb_sim.build_state(dummy_task)

        diff = abs(state_real - state_sim)

        info("State vector (12 chiều = [e1_cpu,e1_ram,e1_lat, e2_cpu,..., cloud_..., task_...]):")
        labels = [
            "edge1_cpu","edge1_ram","edge1_lat",
            "edge2_cpu","edge2_ram","edge2_lat",
            "cloud_cpu","cloud_ram","cloud_lat",
            "task_cpu","task_ram","task_deadline",
        ]

        print(f"\n  {'Dim':<16} {'REAL':>8} {'SIM':>8} {'|diff|':>8}")
        print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8}")
        node_dims_match = 0
        for i, (lbl, r, s, d) in enumerate(zip(labels, state_real, state_sim, diff)):
            marker = "  "
            if i < 9:  # node dims only, not task dims
                if d > 0.02:
                    node_dims_match += 1
                    marker = f"{GREEN}✓{RESET} "
                else:
                    marker = f"{YELLOW}~{RESET} "
            print(f"  {marker}{lbl:<14} {r:>8.4f} {s:>8.4f} {d:>8.4f}")

        print()
        metrics_real = sb_real.get_current_metrics_summary()
        info("Raw metrics từ StateBuilder (sau khi fetch Prometheus):")
        for k, v in metrics_real.items():
            print(f"    {k}: {v}")

        print()
        if node_dims_match >= 3:
            ok(f"REAL state khác SIM state ở {node_dims_match}/9 node dims → đang đọc dữ liệu THẬT")
        else:
            warn(f"REAL state gần giống SIM state ({node_dims_match}/9 dims khác) — "
                 "có thể instance_map không khớp hoặc Prometheus không có dữ liệu")
            info("→ Xem BƯỚC 3 để debug")

    except Exception as e:
        fail(f"StateBuilder gặp lỗi: {e}")
        import traceback
        traceback.print_exc()


def live_watch(url: str, duration: int = 10) -> None:
    """Poll Prometheus liên tục để xem metrics thay đổi theo thời gian thật"""
    header(f"BƯỚC 5 — Live watch ({duration}s) — CPU phải dao động")

    try:
        from week2.prometheus_client import PrometheusClient, PrometheusConfig
        client = PrometheusClient(PrometheusConfig(base_url=url, timeout_seconds=5))

        readings = []
        for i in range(duration):
            cpu = client.get_cpu_usage_pct()
            ts = time.strftime("%H:%M:%S")
            line = f"  [{ts}]"
            for inst, val in cpu.items():
                line += f"  {inst}: {val:.1f}%"
            print(line if cpu else f"  [{ts}]  (no data)")
            readings.append(cpu)
            time.sleep(1)

        if len(readings) >= 2 and readings[0] and readings[-1]:
            all_instances = set(readings[0].keys()) | set(readings[-1].keys())
            for inst in all_instances:
                vals = [r.get(inst, 0) for r in readings if inst in r]
                variance = max(vals) - min(vals) if vals else 0
                if variance > 0.5:
                    ok(f"{inst}: CPU dao động {min(vals):.1f}% → {max(vals):.1f}% (delta={variance:.1f}%) — DỮ LIỆU SỐNG")
                else:
                    warn(f"{inst}: CPU gần như không đổi (delta={variance:.2f}%) — có thể là giá trị cache")
        else:
            warn("Không đủ dữ liệu để phân tích biến động")

    except Exception as e:
        fail(f"Live watch lỗi: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus", default="http://localhost:9090",
                        help="Prometheus URL (default: http://localhost:9090)")
    parser.add_argument("--watch", type=int, default=0,
                        help="Live watch N giây (0 = bỏ qua, recommend 10)")
    args = parser.parse_args()

    url = args.prometheus

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  PROMETHEUS VERIFICATION TOOL{RESET}")
    print(f"  Target: {url}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Bước 1: Kiểm tra kết nối
    reachable = check_prometheus_reachable(url)
    if not reachable:
        sep()
        print(f"\n{RED}Dừng: Prometheus không phản hồi.{RESET}")
        print("Khởi động với: docker compose up -d prometheus")
        sys.exit(1)

    # Bước 2: Query thô
    raw = query_raw_instances(url)

    # Bước 3: Kiểm tra mapping
    mapping_result = check_instance_map(raw)

    # Bước 4: StateBuilder end-to-end
    check_dispatcher_state(url)

    # Bước 5 (optional): Live watch
    if args.watch > 0:
        live_watch(url, args.watch)

    # Tổng kết
    header("KẾT LUẬN")
    has_cpu = bool(raw["cpu"])
    has_ram = bool(raw["ram"])
    has_match = bool(mapping_result.get("matched"))

    if has_cpu and has_ram and has_match:
        ok("HỆ THỐNG ĐANG ĐỌC METRICS THẬT — Prometheus có dữ liệu và INSTANCE_MAP khớp")
    elif has_cpu and has_ram and not has_match:
        warn("Prometheus có dữ liệu NHƯNG INSTANCE_MAP không khớp IP")
        info("→ Cập nhật infra_config.py với đúng IP instance từ Bước 2")
        info("→ Dispatcher đang fallback sang simulation mode!")
    elif not has_cpu:
        fail("Prometheus không có CPU metrics — node_exporter chưa được scrape")
        info("→ Kiểm tra prometheus.yml targets và node_exporter đang chạy trên các node")
    sep()


if __name__ == "__main__":
    main()
