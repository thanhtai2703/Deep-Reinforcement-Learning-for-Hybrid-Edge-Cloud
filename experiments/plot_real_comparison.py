import sys
import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# Color & label mapping
# ────────────────────────────────────────────────────────────────
COLORS = {
    "dqn":              "#1976D2",
    "dqn-cal":          "#1976D2",
    "dqn-uncal":        "#90CAF9",
    "ppo":              "#7B1FA2",
    "random":           "#9E9E9E",
    "round_robin":      "#FF9800",
    "least_connection": "#4CAF50",
    "edge_only":        "#00BCD4",
    "cloud_only":       "#F44336",
}

PRETTY_NAME = {
    "dqn":              "DQN",
    "dqn-cal":          "DQN (calibrated)",
    "dqn-uncal":        "DQN (uncalibrated)",
    "random":           "Random",
    "round_robin":      "Round-Robin",
    "least_connection": "Least-Conn",
    "edge_only":        "Edge-Only",
    "cloud_only":       "Cloud-Only",
}

def get_edge_columns(df: pd.DataFrame):
    """Lấy danh sách các cột edge_N_pct một cách chính xác."""
    pattern = re.compile(r"^edge_(\d+)_pct$")
    cols = [c for c in df.columns if pattern.match(c)]
    # Sắp xếp theo số thứ tự node (1, 2, 3...)
    cols.sort(key=lambda x: int(pattern.match(x).group(1)))
    return cols

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["policy"], keep="last").reset_index(drop=True)
    df["policy_pretty"] = df["policy"].map(lambda x: PRETTY_NAME.get(x, x))
    df["color"]         = df["policy"].map(lambda x: COLORS.get(x, "#607D8B"))
    return df

# ────────────────────────────────────────────────────────────────
# Plot 1: 4-panel comparison
# ────────────────────────────────────────────────────────────────
def plot_4panel(df: pd.DataFrame, out: Path):
    edge_cols = get_edge_columns(df)
    # Tính tổng % sử dụng edge một cách an toàn
    df["edge_total_pct"] = df[edge_cols].sum(axis=1)

    panels = [
        ("sla_rate",       "SLA Rate (%) ↑",         100),
        ("lat_mean",       "Mean Latency (ms) ↓",    None),
        ("cost_avg",       "Avg Cost / task ↓",      None),
        ("edge_total_pct", "Edge Usage (%) ↑",       100),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Real K8s Comparison — DQN vs Baselines", fontsize=14, fontweight="bold")

    for ax, (col, title, ymax) in zip(axes.flatten(), panels):
        bars = ax.bar(df["policy_pretty"], df[col], color=df["color"],
                     edgecolor="white", linewidth=0.8)
        
        for i, name in enumerate(df["policy"]):
            if "dqn" in name.lower() or "ppo" in name.lower():
                bars[i].set_edgecolor("black")
                bars[i].set_linewidth(2.0)
                
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.tick_params(axis='x', rotation=25)
        
        if ymax:
            ax.set_ylim(0, ymax * 1.1)
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (df[col].max() * 0.02),
                    f"{v:.1f}" if v >= 1 else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")

# ────────────────────────────────────────────────────────────────
# Plot 2: Stacked action distribution
# ────────────────────────────────────────────────────────────────
def plot_action_distribution(df: pd.DataFrame, out: Path):
    edge_cols = get_edge_columns(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(df))
    # Tạo bảng màu cho các node edge
    edge_colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(edge_cols)))
    
    for i, col in enumerate(edge_cols):
        ax.bar(df["policy_pretty"], df[col], bottom=bottom,
               label=f"Edge {i+1}", color=edge_colors[i],
               edgecolor="white", linewidth=0.5)
        bottom += df[col]

    ax.bar(df["policy_pretty"], df["cloud_pct"], bottom=bottom,
           label="Cloud", color="#2196F3", edgecolor="white", linewidth=0.5)
    bottom += df["cloud_pct"]

    ax.bar(df["policy_pretty"], df["rejected_pct"], bottom=bottom,
           label="Reject", color="#F44336", edgecolor="white", linewidth=0.5)

    ax.set_title("Action Distribution per Policy (Real K8s)", fontweight="bold", fontsize=13)
    ax.set_ylabel("% of dispatched actions")
    ax.set_ylim(0, 105)
    ax.tick_params(axis='x', rotation=25)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")

# ────────────────────────────────────────────────────────────────
# Plot 3: Latency profile
# ────────────────────────────────────────────────────────────────
def plot_latency_profile(df: pd.DataFrame, out: Path):
    metrics = ["lat_median", "lat_mean", "lat_p95", "lat_p99"]
    labels  = ["Median", "Mean", "P95", "P99"]

    x = np.arange(len(df))
    bar_w = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        ax.bar(x + i * bar_w, df[m], bar_w, label=lbl, edgecolor="white", linewidth=0.5)

    ax.set_title("Latency Profile per Policy (Real K8s)", fontweight="bold", fontsize=13)
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x + bar_w * 1.5)
    ax.set_xticklabels(df["policy_pretty"], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {out}")

# ────────────────────────────────────────────────────────────────
# Markdown table
# ────────────────────────────────────────────────────────────────
def write_markdown_table(df: pd.DataFrame, out: Path):
    edge_cols = get_edge_columns(df)
    
    headers = ["Policy", "SLA %", "Mean (ms)", "P95 (ms)", "Cost"]
    headers += [f"E{i+1} %" for i in range(len(edge_cols))]
    headers += ["Cloud %", "Reject %"]

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        vals = [
            str(row["policy_pretty"]),
            f"{row['sla_rate']:.1f}",
            f"{row['lat_mean']:.0f}",
            f"{row['lat_p95']:.0f}",
            f"{row['cost_avg']:.4f}",
        ]
        for col in edge_cols:
            vals.append(f"{row[col]:.1f}")
        vals.append(f"{row['cloud_pct']:.1f}")
        vals.append(f"{row['rejected_pct']:.1f}")
        lines.append("| " + " | ".join(vals) + " |")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Saved] {out}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiments/plot_real_comparison.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}")
        sys.exit(1)

    plot_dir = Path("experiments/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(csv_path)
    print(f"[Loaded] {len(df)} policies from {csv_path}")

    plot_4panel(df,                plot_dir / "real_comparison_4panel.png")
    plot_action_distribution(df,   plot_dir / "real_comparison_actions.png")
    plot_latency_profile(df,       plot_dir / "real_comparison_latency.png")
    write_markdown_table(df,       plot_dir / "real_comparison_table.md")

if __name__ == "__main__":
    main()