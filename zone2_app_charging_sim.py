from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from zone2_simulation import (
    CONGESTION_INDEX_BASE,
    DT_HOURS,
    EV_DAILY_MWH,
    EV_KWH_PER_DAY,
    N_EVS,
    enrich_simulation_frame,
    load_eindhoven_congestion_levels,
    load_mean_hourly_price_profile,
    load_zone2_15min_demand,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "sim_outputs" / "app_scenarios"

CHARGER_KW = 11
FLEET_MAX_MW = N_EVS * CHARGER_KW / 1_000
PLUG_IN_HOUR = 18
READY_BY_HOUR = 7
UNMANAGED_START = 17
UNMANAGED_END = 21


def plug_in_window_mask(index: pd.DatetimeIndex) -> pd.Series:
    return (index.hour >= PLUG_IN_HOUR) | (index.hour < READY_BY_HOUR)


def charging_window_slots(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Chronological order: 18:00 through 06:45 on the profile day."""
    in_window = plug_in_window_mask(index)
    evening = index[in_window & (index.hour >= PLUG_IN_HOUR)]
    morning = index[in_window & (index.hour < READY_BY_HOUR)]
    return list(evening) + list(morning)


def scenario_immediate_plug_in(index: pd.DatetimeIndex) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    remaining_mwh = EV_DAILY_MWH
    for slot in charging_window_slots(index):
        if remaining_mwh <= 0:
            break
        add_mwh = min(remaining_mwh, FLEET_MAX_MW * DT_HOURS)
        ev.loc[slot] = add_mwh / DT_HOURS
        remaining_mwh -= add_mwh
    return ev


def scenario_unmanaged_evening(index: pd.DatetimeIndex) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    mask = (index.hour >= UNMANAGED_START) & (index.hour < UNMANAGED_END)
    n_slots = int(mask.sum())
    if n_slots:
        ev.loc[mask] = EV_DAILY_MWH / (n_slots * DT_HOURS)
    return ev


def scenario_smart_flat_spread(index: pd.DatetimeIndex) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    mask = plug_in_window_mask(index)
    n_slots = int(mask.sum())
    if n_slots:
        ev.loc[mask] = EV_DAILY_MWH / (n_slots * DT_HOURS)
    return ev


def scenario_smart_price_aware(
    index: pd.DatetimeIndex, price: pd.Series
) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    window = plug_in_window_mask(index)
    remaining_mwh = EV_DAILY_MWH
    for slot in price.loc[window].sort_values().index:
        if remaining_mwh <= 0:
            break
        add_mwh = min(remaining_mwh, FLEET_MAX_MW * DT_HOURS)
        ev.loc[slot] = add_mwh / DT_HOURS
        remaining_mwh -= add_mwh
    return ev


def scenario_smart_grid_aware(
    grid_load: pd.Series,
    index: pd.DatetimeIndex,
    price: pd.Series,
    zone_capacity_mw: float,
) -> pd.Series:
    ev = pd.Series(0.0, index=index, name="EV_Load_MW")
    window = plug_in_window_mask(index)
    remaining_mwh = EV_DAILY_MWH
    total_load = grid_load.copy()
    for slot in price.loc[window].sort_values().index:
        if remaining_mwh <= 0:
            break
        headroom_mw = zone_capacity_mw - total_load.loc[slot] - ev.loc[slot]
        if headroom_mw <= 0:
            continue
        add_mwh = min(remaining_mwh, FLEET_MAX_MW * DT_HOURS, headroom_mw * DT_HOURS)
        if add_mwh <= 0:
            continue
        ev.loc[slot] += add_mwh / DT_HOURS
        remaining_mwh -= add_mwh
        total_load.loc[slot] = grid_load.loc[slot] + ev.loc[slot]
    return ev


def annual_ev_energy_cost(ev_load: pd.Series, price: pd.Series) -> float:
    return float((ev_load * DT_HOURS * price).sum() * 365)


def build_kpi_row(
    label: str,
    frame: pd.DataFrame,
    annual_cost_eur: float,
    baseline_annual_cost_eur: float,
) -> dict:
    return {
        "scenario": label,
        "peak_total_load_mw": frame["Total_Load_MW"].max(),
        "peak_stress_ratio": frame["Stress_Ratio"].max(),
        "bottleneck_intervals": int(frame["Bottleneck"].sum()),
        "mean_congestion_index": frame["Congestion_Index"].mean(),
        "annual_ev_energy_cost_eur": annual_cost_eur,
        "annual_savings_eur": baseline_annual_cost_eur - annual_cost_eur,
    }


def plot_charging_profiles(
    frames: dict[str, pd.DataFrame], out_path: Path
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = {
        "immediate_plug_in": "#dc2626",
        "unmanaged_evening": "#ea580c",
        "smart_flat_spread": "#2563eb",
        "smart_price_aware": "#059669",
        "smart_grid_aware": "#7c3aed",
    }
    for name, frame in frames.items():
        c = colors.get(name, None)
        axes[0].plot(frame.index, frame["EV_Load_MW"], label=name, color=c, alpha=0.9)
        axes[1].plot(frame.index, frame["Total_Load_MW"], label=name, color=c, alpha=0.9)
        axes[2].plot(frame.index, frame["Stress_Ratio"], label=name, color=c, alpha=0.9)
    cap = next(iter(frames.values()))["Zone_Capacity_MW"].iloc[0]
    axes[1].axhline(cap, color="#6b7280", linestyle="--", label="Zone capacity")
    axes[2].axhline(1.0, color="#6b7280", linestyle="--", label="Stress = 1.0")
    axes[0].set_ylabel("EV load (MW)")
    axes[0].set_title("Zone Z2: APP smart charging scenarios")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Total load (MW)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Stress ratio")
    axes[2].set_xlabel("Time of day")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_peak_comparison(kpi_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(kpi_df))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        kpi_df["peak_total_load_mw"],
        width=width,
        label="Peak total load (MW)",
        color="#2563eb",
    )
    ax2 = ax.twinx()
    ax2.bar(
        [i + width / 2 for i in x],
        kpi_df["peak_stress_ratio"],
        width=width,
        label="Peak stress ratio",
        color="#dc2626",
        alpha=0.85,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(kpi_df["scenario"], rotation=25, ha="right")
    ax.set_ylabel("Peak total load (MW)")
    ax2.set_ylabel("Peak stress ratio")
    ax.set_title("Zone Z2 APP scenarios: peak load and stress")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_load = load_zone2_15min_demand()
    price = load_mean_hourly_price_profile()
    _, eindhoven_pc6 = load_eindhoven_congestion_levels()
    index = grid_load.index

    ev_immediate = scenario_immediate_plug_in(index)
    zone_capacity_mw = float((grid_load + ev_immediate).max())

    scenarios: dict[str, pd.Series] = {
        "immediate_plug_in": ev_immediate,
        "unmanaged_evening": scenario_unmanaged_evening(index),
        "smart_flat_spread": scenario_smart_flat_spread(index),
        "smart_price_aware": scenario_smart_price_aware(index, price),
        "smart_grid_aware": scenario_smart_grid_aware(
            grid_load, index, price, zone_capacity_mw
        ),
    }

    frames: dict[str, pd.DataFrame] = {}
    kpi_rows: list[dict] = []
    baseline_cost = annual_ev_energy_cost(scenarios["immediate_plug_in"], price)

    for name, ev_load in scenarios.items():
        frame = enrich_simulation_frame(
            grid_load, ev_load, price, zone_capacity_mw
        )
        frames[name] = frame
        annual_cost = annual_ev_energy_cost(ev_load, price)
        kpi_rows.append(
            build_kpi_row(name, frame, annual_cost, baseline_cost)
        )
        frame.to_csv(OUTPUT_DIR / f"{name}_timeseries.csv", index_label="timestamp")

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(OUTPUT_DIR / "app_scenarios_kpi.csv", index=False)

    plot_charging_profiles(frames, OUTPUT_DIR / "app_charging_profiles.png")
    plot_peak_comparison(kpi_df, OUTPUT_DIR / "app_peak_comparison.png")

    print("=== Zone Z2 APP smart charging simulation ===")
    print(f"EVs: {N_EVS}, {EV_KWH_PER_DAY} kWh/day -> {EV_DAILY_MWH:.1f} MWh/day")
    print(f"Fleet max charging power: {FLEET_MAX_MW:.2f} MW ({CHARGER_KW} kW/EV)")
    print(f"Zone capacity (MW): {zone_capacity_mw:.2f}")
    print(f"Eindhoven PC6 rows: {len(eindhoven_pc6)}")
    print()
    print(kpi_df.to_string(index=False))
    print()
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
