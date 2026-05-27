from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from zone2_simulation import DT_MINUTES, STRESS_WARNING_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = BASE_DIR / "sim_outputs" / "app_scenarios"
KPI_FILE = SCENARIOS_DIR / "app_scenarios_kpi.csv"

LABELS = {
    "immediate_plug_in": "Immediate\n(no app)",
    "unmanaged_evening": "Evening peak\n(no app)",
    "smart_flat_spread": "Smart app:\nflat spread",
    "smart_price_aware": "Smart app:\nprice-optimized",
    "smart_grid_aware": "Smart app:\ngrid + price",
}
ORDER = [
    "unmanaged_evening",
    "immediate_plug_in",
    "smart_flat_spread",
    "smart_price_aware",
    "smart_grid_aware",
]
COLORS = ["#dc2626", "#f97316", "#3b82f6", "#059669", "#7c3aed"]


def load_stress_duration_table() -> pd.DataFrame:
    """Minutes at stress >= 0.95 and overload minutes from scenario timeseries."""
    rows = []
    for scenario in ORDER:
        path = SCENARIOS_DIR / f"{scenario}_timeseries.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run: python zone2_app_charging_sim.py"
            )
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        stress = frame["Stress_Ratio"]
        rows.append(
            {
                "scenario": scenario,
                "minutes_above_stress_095": int(
                    (stress >= STRESS_WARNING_THRESHOLD).sum() * DT_MINUTES
                ),
                "overload_minutes": int(frame["Bottleneck"].sum() * DT_MINUTES),
            }
        )
    return pd.DataFrame(rows)


def plot_grid_stress_duration(df: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(df))
    x_labels = [LABELS[s] for s in df["scenario"]]
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_high = ax.bar(
        x - width / 2,
        df["minutes_above_stress_095"],
        width=width,
        color="#fca5a5",
        edgecolor="white",
        label=f"Minutes stress ≥ {STRESS_WARNING_THRESHOLD}",
    )
    bars_over = ax.bar(
        x + width / 2,
        df["overload_minutes"],
        width=width,
        color="#ea580c",
        edgecolor="white",
        label="Minutes in overload (stress ≥ 1.0)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel("Minutes per day")
    ax.set_xlabel("Charging behaviour")
    ax.set_title(
        "Zone 2: Grid stress duration by charging behaviour",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_ylim(0, max(df["minutes_above_stress_095"].max() * 1.15, 30))

    for bar in list(bars_high) + list(bars_over):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 4,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    evening = df.loc[df["scenario"] == "unmanaged_evening"].iloc[0]
    ax.text(
        0.02,
        0.98,
        f"Evening peak habit: {int(evening['overload_minutes'])} min overload, "
        f"{int(evening['minutes_above_stress_095'])} min high stress\n"
        f"Each step = {DT_MINUTES} minutes",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="#fef2f2", edgecolor="#fecaca"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_money_saved(out_path: Path) -> None:
    df = pd.read_csv(KPI_FILE).set_index("scenario").loc[ORDER].reset_index()
    x = np.arange(len(df))
    x_labels = [LABELS[s] for s in df["scenario"]]
    savings_k = df["annual_savings_eur"] / 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    money_colors = ["#9ca3af", "#9ca3af", "#60a5fa", "#34d399", "#a78bfa"]
    bars = ax.bar(x, savings_k, color=money_colors, width=0.65, edgecolor="white")
    ax.set_ylabel("Annual savings (thousand EUR)")
    ax.set_xlabel("Charging behaviour")
    ax.set_title(
        "Zone 2: Money saved with smart charging app\n(vs full power when plugged in)",
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    for bar, eur in zip(bars, df["annual_savings_eur"]):
        label = "Reference" if eur <= 0 else f"EUR {eur / 1000:.0f}k\nper year"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if eur > 0 else "normal",
        )
    ax.text(
        0.98,
        0.97,
        f"Fleet: 5,000 EVs | 25 kWh/day\n"
        f"Best: EUR {df['annual_savings_eur'].max() / 1e6:.2f}M saved/year",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#ecfdf5", edgecolor="#a7f3d0"),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    stress_df = load_stress_duration_table()
    congestion_path = SCENARIOS_DIR / "graph_congestion_reduction.png"
    plot_grid_stress_duration(stress_df, congestion_path)
    plot_money_saved(SCENARIOS_DIR / "graph_money_saved.png")
    print("Saved:", congestion_path)
    print("Saved:", SCENARIOS_DIR / "graph_money_saved.png")
    print(stress_df.to_string(index=False))


if __name__ == "__main__":
    main()
