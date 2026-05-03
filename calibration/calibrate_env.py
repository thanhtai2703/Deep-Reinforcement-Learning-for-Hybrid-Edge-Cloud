"""
calibrate_env.py
================
Sim calibration via decomposed formula.

Instead of fitting total_ms ~ cpu_during_exec (R² ≈ 0 because total_ms is
dominated by K8s overhead, not CPU contention), we decompose:

  total_ms = submit_overhead[role]      (K8s API + scheduling)
           + container_startup[role]    (image pull + init)
           + exec_time                  (CPU burn — the part env can model)
           + poll_overhead              (status polling, 1Hz fixed)

Of these, only `exec_time` is workload-dependent. We fit:

  exec_time_ms = α[role] * workload_proxy_ms * (1 + β[role] * cpu_during/100)

where workload_proxy_ms mirrors task-processor.py:
  duration_s = clamp(deadline_ms/1000 * 0.5, 1, 15)
  workload_proxy_ms = duration_s * (cpu_requirement / 30) * 1000

α[role]  : hardware-specific scaling (1.0 = node burns at reference rate)
β[role]  : contention slope. β=2 means 100% CPU doubles exec_time.

Linearization:
  Let u = exec_time / workload_proxy = α * (1 + β * cpu/100)
       u = α + (α*β/100) * cpu
  OLS u ~ a + b*cpu  =>  α=a, β=100b/a.
"""

# %% imports
from __future__ import annotations
import sys
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "database" / "dispatcher.db"
OUT_DIR = ROOT / "calibration"
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

UNCAL = dict(
    edge_base_latency_ms=17.5,
    cloud_base_latency_ms=55.0,
    load_coef=2.0,
    edge_cost_per_unit=0.01,
    cloud_cost_per_unit=0.05,
)


# %% load data
def load_logs(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM execution_logs
            WHERE exec_status='succeeded'
              AND selected_cpu IS NOT NULL
              AND total_ms IS NOT NULL
              AND exec_time_ms IS NOT NULL
              AND target_role IS NOT NULL
              AND cpu_requirement IS NOT NULL
              AND deadline_ms IS NOT NULL
            """,
            conn,
        )
    return df


df = load_logs(DB_PATH)
print(f"Loaded {len(df)} rows")


# %% workload proxy (mirrors task-processor.py)
def workload_proxy_ms(cpu_req: float, deadline_ms: float) -> float:
    duration_s = max(1.0, min(deadline_ms / 1000.0 * 0.5, 15.0))
    return duration_s * (cpu_req / 30.0) * 1000.0


df["workload_proxy_ms"] = df.apply(
    lambda r: workload_proxy_ms(r["cpu_requirement"], r["deadline_ms"]),
    axis=1,
)

# Drop degenerate rows
df = df[df["workload_proxy_ms"] > 100].copy()
df["u"] = df["exec_time_ms"] / df["workload_proxy_ms"]

# Trim outliers (3σ on u to remove broken K8s timing samples)
for role, sub in df.groupby("target_role"):
    mu = sub["u"].mean()
    sd = sub["u"].std()
    if not np.isfinite(sd) or sd == 0:
        continue
    mask = (df["target_role"] == role) & ((df["u"] - mu).abs() > 3 * sd)
    df = df[~mask]

print(f"After cleaning: {len(df)} rows")


# %% fit exec_time per role: u = α + (α*β/100)*cpu
# We fit on selected_cpu (snapshot at dispatch) instead of cpu_during_exec
# because the env passes the snapshot value to predict_total_ms during
# training/inference. Fitting on cpu_during_exec would create a train-test
# mismatch (post-hoc avg vs at-decision snapshot).
exec_calibration = {}
for role, sub in df.groupby("target_role"):
    if len(sub) < 5:
        print(f"[skip] {role}: only {len(sub)} samples")
        continue

    cpu = sub["selected_cpu"].values
    u = sub["u"].values

    X = np.column_stack([np.ones_like(cpu), cpu])
    coef, *_ = np.linalg.lstsq(X, u, rcond=None)
    a, b = coef
    alpha = float(a)
    # Clamp α to be physically meaningful (>= small positive)
    if alpha < 0.05:
        # Fallback: use mean(u) as α, β=0 (no contention slope captured)
        alpha = float(max(np.mean(u), 0.05))
        beta = 0.0
        print(f"[warn] {role}: linear fit gave α={float(a):.3f} (≤0); "
              f"falling back to mean α={alpha:.3f}, β=0")
    else:
        beta = float(100.0 * b / a)

    pred_u = alpha * (1.0 + beta * cpu / 100.0)
    rss = float(np.sum((u - pred_u) ** 2))
    tss = float(np.sum((u - u.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0

    pred_exec = pred_u * sub["workload_proxy_ms"].values
    rmse_ms = float(np.sqrt(np.mean(
        (sub["exec_time_ms"].values - pred_exec) ** 2
    )))

    exec_calibration[role] = dict(
        alpha=alpha, beta=beta,
        r2=r2, rmse_ms=rmse_ms, n=len(sub),
    )
    print(f"{role}: α={alpha:.2f}  β={beta:.2f}  R²={r2:.3f}  "
          f"exec_RMSE={rmse_ms:.0f}ms  n={len(sub)}")


# %% fit per-role overhead constants
overhead_calibration = {}
for role, sub in df.groupby("target_role"):
    overhead_calibration[role] = dict(
        submit_overhead_ms=float(sub["submit_overhead_ms"].fillna(0).mean()),
        container_startup_ms=float(sub["container_startup_ms"].fillna(0).mean()),
        poll_overhead_ms=float(sub["poll_overhead_ms"].fillna(1000).mean()),
    )
    o = overhead_calibration[role]
    print(f"{role} overhead: submit={o['submit_overhead_ms']:.0f}ms  "
          f"startup={o['container_startup_ms']:.0f}ms  "
          f"poll={o['poll_overhead_ms']:.0f}ms")


# %% predictions
def predict_uncal(row: pd.Series) -> float:
    base = (UNCAL["cloud_base_latency_ms"] if row["target_role"] == "cloud"
            else UNCAL["edge_base_latency_ms"])
    return base * (1.0 + row["selected_cpu"] / 100.0 * UNCAL["load_coef"])


def predict_cal(row: pd.Series) -> float:
    role = row["target_role"]
    if role not in exec_calibration:
        return predict_uncal(row)
    ec = exec_calibration[role]
    oh = overhead_calibration[role]
    exec_pred = (ec["alpha"] * row["workload_proxy_ms"]
                 * (1.0 + ec["beta"] * row["selected_cpu"] / 100.0))
    return (oh["submit_overhead_ms"] + oh["container_startup_ms"]
            + exec_pred + oh["poll_overhead_ms"])


df["pred_uncal_ms"] = df.apply(predict_uncal, axis=1)
df["pred_cal_ms"] = df.apply(predict_cal, axis=1)


# %% sim-to-real gap metrics
def compute_gap(df: pd.DataFrame, pred_col: str) -> dict:
    real = df["total_ms"].values
    pred = df[pred_col].values
    mae = float(np.mean(np.abs(real - pred)))
    rmse = float(np.sqrt(np.mean((real - pred) ** 2)))
    mape = float(np.mean(np.abs((real - pred) / real)) * 100)

    real_sorted = np.sort(real)
    pred_sorted = np.sort(pred)
    grid = np.unique(np.concatenate([real_sorted, pred_sorted]))
    cdf_real = np.searchsorted(real_sorted, grid, side="right") / len(real_sorted)
    cdf_pred = np.searchsorted(pred_sorted, grid, side="right") / len(pred_sorted)
    ks = float(np.max(np.abs(cdf_real - cdf_pred)))

    return dict(mae_ms=mae, rmse_ms=rmse, mape_pct=mape, ks=ks)


metrics = {
    "uncalibrated": compute_gap(df, "pred_uncal_ms"),
    "calibrated":   compute_gap(df, "pred_cal_ms"),
}
print("\n=== Sim-to-real gap ===")
for k, v in metrics.items():
    print(f"{k:14s}  MAE={v['mae_ms']:6.0f}ms  RMSE={v['rmse_ms']:6.0f}ms  "
          f"MAPE={v['mape_pct']:5.1f}%  KS={v['ks']:.3f}")


# %% plots
def plot_distribution(df: pd.DataFrame, out: Path):
    roles = sorted(df["target_role"].unique())
    fig, axes = plt.subplots(1, len(roles), figsize=(5 * len(roles), 4))
    if len(roles) == 1:
        axes = [axes]
    for ax, role in zip(axes, roles):
        sub = df[df["target_role"] == role]
        bins = np.linspace(0, sub["total_ms"].max() * 1.1, 30)
        ax.hist(sub["total_ms"],      bins=bins, alpha=0.45,
                label=f"Real (n={len(sub)})", color="C0")
        ax.hist(sub["pred_uncal_ms"], bins=bins, alpha=0.45,
                label="Uncalibrated", color="C1")
        ax.hist(sub["pred_cal_ms"],   bins=bins, alpha=0.45,
                label="Calibrated", color="C2")
        ax.set_title(role); ax.set_xlabel("Latency (ms)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Latency distribution: real vs sim_uncal vs sim_cal",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_scatter(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, col, title in [
        (axes[0], "pred_uncal_ms", "Uncalibrated"),
        (axes[1], "pred_cal_ms", "Calibrated"),
    ]:
        for role, sub in df.groupby("target_role"):
            ax.scatter(sub["total_ms"], sub[col], alpha=0.5, s=20, label=role)
        lim = max(df["total_ms"].max(),
                  df["pred_cal_ms"].max(),
                  df["pred_uncal_ms"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Real total_ms"); ax.set_ylabel("Sim prediction (ms)")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Sim prediction vs real (closer to y=x is better)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_exec_fit(df: pd.DataFrame, out: Path):
    """Diagnostic: u = exec_time/workload_proxy vs cpu_during_exec per role."""
    roles = sorted(df["target_role"].unique())
    fig, axes = plt.subplots(1, len(roles), figsize=(5 * len(roles), 4),
                             sharey=True)
    if len(roles) == 1:
        axes = [axes]
    for ax, role in zip(axes, roles):
        sub = df[df["target_role"] == role]
        ax.scatter(sub["selected_cpu"], sub["u"], alpha=0.5, s=20)
        if role in exec_calibration:
            ec = exec_calibration[role]
            xx = np.linspace(0, 100, 50)
            yy = ec["alpha"] * (1 + ec["beta"] * xx / 100)
            ax.plot(xx, yy, "r-",
                    label=f"α={ec['alpha']:.2f}, β={ec['beta']:.2f}")
            ax.legend()
        ax.set_title(f"{role} (R²={exec_calibration.get(role, {}).get('r2', 0):.3f})")
        ax.set_xlabel("CPU at dispatch / snapshot (%)")
        ax.set_ylabel("u = exec_time / workload_proxy")
        ax.grid(alpha=0.3)
    fig.suptitle("Per-role contention fit: u = α(1 + β·cpu_snapshot/100)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


plot_distribution(df, PLOT_DIR / "distribution_comparison.png")
plot_scatter(df, PLOT_DIR / "scatter_pred_vs_real.png")
plot_exec_fit(df, PLOT_DIR / "exec_fit_per_role.png")


# %% export
def export_constants(path: Path):
    lines = [
        '"""',
        "calibrated_constants.py",
        "Auto-generated by calibration/calibrate_env.py",
        "",
        "Decomposed formula:",
        "  total_ms = submit_overhead + container_startup",
        "           + α * workload_proxy * (1 + β * cpu_during/100)",
        "           + poll_overhead",
        "",
        "  workload_proxy_ms = clamp(deadline_ms/1000 * 0.5, 1, 15)",
        "                    * (cpu_requirement / 30) * 1000",
        '"""',
        "",
        f"EXEC_CALIBRATION = {repr(exec_calibration)}",
        "",
        f"OVERHEAD_CALIBRATION = {repr(overhead_calibration)}",
        "",
        "",
        "def workload_proxy_ms(cpu_requirement: float, deadline_ms: float) -> float:",
        "    duration_s = max(1.0, min(deadline_ms / 1000.0 * 0.5, 15.0))",
        "    return duration_s * (cpu_requirement / 30.0) * 1000.0",
        "",
        "",
        "def predict_total_ms(role: str, cpu_requirement: float,",
        "                     deadline_ms: float, cpu_during_exec: float) -> float:",
        "    ec = EXEC_CALIBRATION.get(role)",
        "    oh = OVERHEAD_CALIBRATION.get(role)",
        "    if ec is None or oh is None:",
        "        return 50.0  # fallback",
        "    wp = workload_proxy_ms(cpu_requirement, deadline_ms)",
        "    exec_pred = ec['alpha'] * wp * (1 + ec['beta'] * cpu_during_exec / 100)",
        "    return (oh['submit_overhead_ms'] + oh['container_startup_ms']",
        "            + exec_pred + oh['poll_overhead_ms'])",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {path}")


export_constants(OUT_DIR / "calibrated_constants.py")


# %% report
def write_report(path: Path):
    lines = [
        "# Sim Calibration Report (decomposed model)",
        "",
        f"- Data: `{DB_PATH}` table `execution_logs`",
        f"- Successful rows used: {len(df)}",
        "",
        "## Per-role exec_time fit",
        "",
        "  exec_time = α × workload_proxy × (1 + β × cpu_during/100)",
        "",
        "| Role | n | α | β | R² | exec RMSE (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for role, c in exec_calibration.items():
        lines.append(
            f"| {role} | {c['n']} | {c['alpha']:.2f} | {c['beta']:.2f} | "
            f"{c['r2']:.3f} | {c['rmse_ms']:.0f} |"
        )
    lines += [
        "",
        "## Per-role overhead constants (avg)",
        "",
        "| Role | submit (ms) | startup (ms) | poll (ms) |",
        "|---|---|---|---|",
    ]
    for role, o in overhead_calibration.items():
        lines.append(
            f"| {role} | {o['submit_overhead_ms']:.0f} | "
            f"{o['container_startup_ms']:.0f} | {o['poll_overhead_ms']:.0f} |"
        )

    lines += [
        "",
        "## Sim-to-real gap on total_ms",
        "",
        "| Variant | MAE (ms) | RMSE (ms) | MAPE (%) | KS distance |",
        "|---|---|---|---|---|",
    ]
    for k, v in metrics.items():
        lines.append(
            f"| {k} | {v['mae_ms']:.0f} | {v['rmse_ms']:.0f} | "
            f"{v['mape_pct']:.1f} | {v['ks']:.3f} |"
        )

    if metrics["uncalibrated"]["mae_ms"] > 0:
        improvement_mae = (
            (metrics["uncalibrated"]["mae_ms"] - metrics["calibrated"]["mae_ms"])
            / metrics["uncalibrated"]["mae_ms"] * 100
        )
        lines += [
            "",
            f"**Calibration reduces MAE by {improvement_mae:.1f}%** "
            f"({metrics['uncalibrated']['mae_ms']:.0f}ms "
            f"→ {metrics['calibrated']['mae_ms']:.0f}ms).",
        ]
    if metrics["uncalibrated"]["ks"] > 0:
        improvement_ks = (
            (metrics["uncalibrated"]["ks"] - metrics["calibrated"]["ks"])
            / metrics["uncalibrated"]["ks"] * 100
        )
        lines += [
            f"**KS distance reduced by {improvement_ks:.1f}%** "
            f"({metrics['uncalibrated']['ks']:.3f} "
            f"→ {metrics['calibrated']['ks']:.3f}).",
        ]
    lines += [
        "",
        "## Output files",
        "",
        "- `calibration/calibrated_constants.py`",
        "- `calibration/plots/distribution_comparison.png`",
        "- `calibration/plots/scatter_pred_vs_real.png`",
        "- `calibration/plots/exec_fit_per_role.png`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {path}")


write_report(OUT_DIR / "calibration_report.md")
print("\nDone.")
