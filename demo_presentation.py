"""
demo_presentation.py
====================
Script demo tự động cho buổi báo cáo đồ án.

Chạy:
    python demo_presentation.py --model models/checkpoints/ppo/ppo_best.zip

Kịch bản:
    Màn 1 (60s)  - Round Robin   : baseline tệ
    Màn 2 (90s)  - PPO           : AI tốt hơn rõ rệt
    Màn 3 (90s)  - PPO + Overload: AI thích nghi khi Edge1 quá tải

Lưu ý:
    - Mở Grafana (localhost:3000) trước khi chạy script này
    - Để script chạy, thuyết trình theo từng màn
    - Nhấn Ctrl+C để dừng sớm
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from demo_runner import PolicyRunner, PROM_AVAILABLE, start_http_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Demo")

SEPARATOR = "=" * 65


def banner(text: str):
    print(f"\n{SEPARATOR}")
    print(f"  {text}")
    print(f"{SEPARATOR}\n")


def countdown(seconds: int, label: str = ""):
    for remaining in range(seconds, 0, -10):
        logger.info("⏱  %s còn %ds...", label, remaining)
        time.sleep(min(10, remaining))


def inject_stress_edge1(edge1_ip: str, duration: int = 90):
    """
    SSH vào Edge1, chạy lệnh stress để giả lập CPU overload.
    Cần cài: sudo apt install stress (trên Edge1)
    """
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"ubuntu@{edge1_ip}",
        f"nohup stress --cpu 4 --timeout {duration} &>/dev/null &"
    ]
    try:
        subprocess.Popen(cmd)
        logger.warning("🔥 STRESS injected vào Edge1 (%s) — %ds", edge1_ip, duration)
    except FileNotFoundError:
        logger.warning("ssh không tìm thấy — inject stress thủ công trên Edge1:")
        logger.warning("  ssh ubuntu@%s 'stress --cpu 4 --timeout %d &'", edge1_ip, duration)


def run_scene(label: str, policy: str, model_path: str | None,
              duration: int, prometheus: str, pattern: str = "constant"):
    banner(f"MÀN: {label}")
    runner = PolicyRunner(
        policy_name=policy,
        model_path=model_path,
        tasks_per_sec=3.0,
        pattern=pattern,
        demo_mode=(prometheus == "demo"),
        prometheus_url=prometheus if prometheus != "demo" else "http://localhost:9090",
    )
    t = threading.Thread(target=runner.run, daemon=True)
    t.start()
    countdown(duration, label)
    runner.stop()
    t.join(timeout=5)

    s = runner.dispatcher.get_summary()
    print(f"\n  📊 Kết quả {label}:")
    print(f"     SLA Rate   : {s['sla_rate']:.1f}%")
    print(f"     Avg Latency: {s['avg_latency_ms']:.0f} ms")
    print(f"     P95 Latency: {s['p95_latency_ms']:.0f} ms")
    print(f"     Cloud Usage: {s['cloud_usage_pct']:.1f}%")
    print()
    return s


def main():
    parser = argparse.ArgumentParser(description="Demo Presentation Script")
    parser.add_argument("--model", default="models/checkpoints/ppo/ppo_best.zip",
                        help="PPO model path")
    parser.add_argument("--prometheus", default="http://localhost:9090",
                        help="Prometheus URL (dùng 'demo' để chạy simulation)")
    parser.add_argument("--edge1-ip", default="100.82.147.9",
                        help="Tailscale IP của Edge1 (để inject stress)")
    parser.add_argument("--scene", type=int, default=0,
                        help="Chạy màn cụ thể: 1, 2, 3 (default: 0 = tất cả)")
    parser.add_argument("--duration", type=int, default=90,
                        help="Giây mỗi màn (default: 90)")
    args = parser.parse_args()

    if PROM_AVAILABLE:
        start_http_server(8000)
        logger.info("Metrics endpoint: http://localhost:8000/metrics")

    results = {}

    # ── Màn 1: Round Robin ──────────────────────────────────────────────────
    if args.scene in (0, 1):
        print("\n" + "=" * 65)
        print("  HƯỚNG DẪN THUYẾT TRÌNH MÀN 1")
        print("  → Giải thích: 'Đây là Round Robin — thuật toán truyền thống,")
        print("    phân phối task lần lượt không quan tâm đến tải node.'")
        print("  → Nhìn Grafana: SLA thấp, latency không ổn định.")
        print("=" * 65)
        input("\n  [ENTER để bắt đầu Màn 1 — Round Robin]\n")

        results["round_robin"] = run_scene(
            "Màn 1 — Round Robin (Baseline)",
            policy="round_robin", model_path=None,
            duration=args.duration, prometheus=args.prometheus,
            pattern="mixed",
        )

    # ── Màn 2: PPO ─────────────────────────────────────────────────────────
    if args.scene in (0, 2):
        print("=" * 65)
        print("  HƯỚNG DẪN THUYẾT TRÌNH MÀN 2")
        print("  → Giải thích: 'Bây giờ bật AI (PPO). Model được train")
        print("    500K episodes trong môi trường simulation.'")
        print("  → Nhìn Grafana: SLA tăng, latency giảm rõ rệt.")
        print("  → Chỉ vào Decision Distribution: AI chọn node thông minh.")
        print("=" * 65)
        input("\n  [ENTER để bắt đầu Màn 2 — PPO]\n")

        results["ppo"] = run_scene(
            "Màn 2 — PPO (AI)",
            policy="ppo", model_path=args.model,
            duration=args.duration, prometheus=args.prometheus,
            pattern="mixed",
        )

    # ── Màn 3: PPO + Stress ─────────────────────────────────────────────────
    if args.scene in (0, 3):
        print("=" * 65)
        print("  HƯỚNG DẪN THUYẾT TRÌNH MÀN 3")
        print("  → Giải thích: 'Giờ ta giả lập Edge1 bị quá tải đột ngột.'")
        print("  → Script sẽ SSH vào Edge1 và chạy stress --cpu 4.")
        print("  → Nhìn Grafana: Edge1 CPU tăng vọt.")
        print("  → AI tự chuyển task sang Edge2 và Cloud — không cần cấu hình.")
        print("=" * 65)
        input("\n  [ENTER để bắt đầu Màn 3 — PPO + Edge Overload]\n")

        # Start stress sau 15s để kịp giới thiệu
        def delayed_stress():
            time.sleep(15)
            inject_stress_edge1(args.edge1_ip, duration=args.duration)

        threading.Thread(target=delayed_stress, daemon=True).start()

        results["ppo_stress"] = run_scene(
            "Màn 3 — PPO + Edge1 Overload",
            policy="ppo", model_path=args.model,
            duration=args.duration, prometheus=args.prometheus,
            pattern="heavy",
        )

    # ── Tổng kết ────────────────────────────────────────────────────────────
    banner("KẾT QUẢ TỔNG HỢP")
    print(f"  {'Policy':<25} {'SLA%':>8} {'Avg Lat(ms)':>12} {'P95 Lat(ms)':>12}")
    print(f"  {'-' * 60}")
    for name, s in results.items():
        marker = " ← AI" if "ppo" in name else ""
        print(f"  {name:<25} {s['sla_rate']:>8.1f} {s['avg_latency_ms']:>12.0f} "
              f"{s['p95_latency_ms']:>12.0f}{marker}")
    print()
    logger.info("Demo hoàn tất.")


if __name__ == "__main__":
    main()
