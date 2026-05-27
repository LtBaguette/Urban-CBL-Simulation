from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_Set" / "Data_Set"
DS5_DIR = next(DATA_DIR.glob("Dataset 5*"))
DS6_DIR = next(DATA_DIR.glob("Dataset 6*"))
DS7_DIR = next(DATA_DIR.glob("Dataset 7*"))

ZONAL_LOAD_FILE = DS6_DIR / "eindhoven_zonal_load.csv"
DISTRICTS_FILE = DS6_DIR / "eindhoven_districts.csv"
TENNET_CONGESTION_FILE = DS5_DIR / "tennetcongestie.csv"
CONGESTION_PC6_FILE = DS5_DIR / "congestie_pc6.csv"
PRICE_FILE = (
    DS7_DIR
    / "european_wholesale_electricity_price_data_hourly"
    / "Netherlands.csv"
)

OUTPUT_DIR = BASE_DIR / "sim_outputs"
FOCUS_ZONE = "Z2"
EINDHOVEN_PC_PREFIXES = ("561", "562", "563", "564")
REFERENCE_DAY = pd.Timestamp("2025-01-01")

N_EVS = 5_000
EV_KWH_PER_DAY = 25
EV_DAILY_MWH = N_EVS * EV_KWH_PER_DAY / 1_000
PEAK_HOUR_START = 17
PEAK_HOUR_END = 21
OFFPEAK_HOUR_END = 6

CONGESTION_INDEX_BASE = 19_641.34
INTERVENTION_PCTS = (10, 15, 20)
REFERENCE_ANNUAL_SAVINGS_EUR = {10: 170_596, 15: 255_894, 20: 341_192}
DT_HOURS = 0.25
DT_MINUTES = 15
STRESS_WARNING_THRESHOLD = 0.95


def parse_eu_decimal(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def load_zone2_15min_demand() -> pd.Series:
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"]).copy()

    zone2 = load_df.loc[load_df["zone_id"] == FOCUS_ZONE].sort_values("timestamp")
    if zone2.empty:
        raise ValueError(f"No records found for zone {FOCUS_ZONE} in {ZONAL_LOAD_FILE}")

    resampled_days: list[pd.Series] = []
    for _, day_df in zone2.groupby(zone2["timestamp"].dt.date):
        day_start = pd.Timestamp(day_df["timestamp"].dt.date.iloc[0])
        day_index = pd.date_range(day_start, periods=96, freq="15min")
        hourly = day_df.set_index("timestamp")["demand_MW"]
        quarter_hourly = (
            hourly.reindex(day_index.union(hourly.index))
            .sort_index()
            .interpolate(method="time")
            .reindex(day_index)
        )
        quarter_hourly.index = quarter_hourly.index.strftime("%H:%M")
        resampled_days.append(quarter_hourly)

    profile = pd.concat(resampled_days, axis=1).mean(axis=1)
    profile.index = pd.date_range(REFERENCE_DAY, periods=96, freq="15min")
    profile.name = "demand_MW"
    return profile


def load_mean_hourly_price_profile() -> pd.Series:
    price_df = pd.read_csv(PRICE_FILE)
    price_df["timestamp"] = pd.to_datetime(
        price_df["Datetime (Local)"], errors="coerce"
    )
    price_df = price_df.dropna(subset=["timestamp"])
    hourly = price_df.groupby(price_df["timestamp"].dt.hour)["Price (EUR/MWhe)"].mean()
    hourly = hourly.reindex(range(24)).interpolate()
    index = pd.date_range(REFERENCE_DAY, periods=96, freq="15min")
    return pd.Series([hourly.loc[ts.hour] for ts in index], index=index, name="price_eur_mwh")


def load_eindhoven_congestion_levels() -> tuple[pd.DataFrame, pd.DataFrame]:
    tennet_df = pd.read_csv(TENNET_CONGESTION_FILE, sep=";", encoding="utf-8")
    tennet_df.columns = [col.strip().strip(",") for col in tennet_df.columns]
    tennet_df["afname"] = parse_eu_decimal(tennet_df["afname"])
    tennet_df["opwek"] = parse_eu_decimal(tennet_df["opwek"])
    eindhoven_stations = tennet_df.loc[
        tennet_df["tennet_id"].astype(str).str.contains("Eindhoven", case=False, na=False)
    ].copy()

    pc6_chunks = []
    for chunk in pd.read_csv(
        CONGESTION_PC6_FILE, sep=";", encoding="utf-8", chunksize=200_000
    ):
        chunk.columns = [col.strip().strip(",") for col in chunk.columns]
        mask = chunk["postcode"].astype(str).str.startswith(EINDHOVEN_PC_PREFIXES)
        if mask.any():
            filtered = chunk.loc[mask].copy()
            filtered["afname"] = parse_eu_decimal(filtered["afname"])
            filtered["opwek"] = parse_eu_decimal(filtered["opwek"])
            pc6_chunks.append(filtered)
    eindhoven_pc6 = (
        pd.concat(pc6_chunks, ignore_index=True) if pc6_chunks else pd.DataFrame()
    )
    return eindhoven_stations, eindhoven_pc6


def build_baseline_ev_load(index: pd.DatetimeIndex) -> pd.Series:
    ev = pd.Series(0.0, index=index)
    peak_mask = (index.hour >= PEAK_HOUR_START) & (index.hour < PEAK_HOUR_END)
    ev.loc[peak_mask] = EV_DAILY_MWH / (peak_mask.sum() * DT_HOURS)
    return ev.rename("EV_Load_MW")


def apply_smart_charging_shift(
    grid_load: pd.Series,
    ev_load: pd.Series,
    shift_fraction: float,
    zone_capacity_mw: float,
    price: pd.Series,
) -> pd.Series:
    shifted = ev_load.copy()
    peak_mask = (shifted.index.hour >= PEAK_HOUR_START) & (
        shifted.index.hour < PEAK_HOUR_END
    )
    offpeak_mask = shifted.index.hour < OFFPEAK_HOUR_END
    shift_mwh = shift_fraction * EV_DAILY_MWH
    n_peak = int(peak_mask.sum())
    shifted.loc[peak_mask] -= shift_mwh / (n_peak * DT_HOURS)

    remaining_mwh = shift_mwh
    total_load = grid_load + shifted
    for slot in price.loc[offpeak_mask].sort_values().index:
        if remaining_mwh <= 0:
            break
        headroom_mw = zone_capacity_mw - total_load.loc[slot]
        if headroom_mw <= 0:
            continue
        add_mwh = min(remaining_mwh, headroom_mw * DT_HOURS)
        shifted.loc[slot] += add_mwh / DT_HOURS
        remaining_mwh -= add_mwh
        total_load.loc[slot] = grid_load.loc[slot] + shifted.loc[slot]
    return shifted.rename("EV_Load_MW")


def enrich_simulation_frame(
    grid_load: pd.Series,
    ev_load: pd.Series,
    price: pd.Series,
    zone_capacity_mw: float,
) -> pd.DataFrame:
    total_load = grid_load + ev_load
    stress = total_load / zone_capacity_mw
    return pd.DataFrame(
        {
            "Baseline_Grid_Load_MW": grid_load,
            "EV_Load_MW": ev_load,
            "Total_Load_MW": total_load,
            "Price_EUR_per_MWh": price,
            "Zone_Capacity_MW": zone_capacity_mw,
            "Stress_Ratio": stress,
            "Bottleneck": stress >= 1.0,
            "Congestion_Index": CONGESTION_INDEX_BASE * stress,
        }
    )


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


def plot_baseline_vs_intervention(
    baseline: pd.DataFrame,
    interventions: dict[int, pd.DataFrame],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(
        baseline.index,
        baseline["Total_Load_MW"],
        label="Baseline total load",
        color="#dc2626",
        linewidth=2,
    )
    for pct, frame in interventions.items():
        axes[0].plot(
            frame.index,
            frame["Total_Load_MW"],
            label=f"Intervention {pct}%",
            linewidth=1.5,
            alpha=0.85,
        )
    axes[0].axhline(
        baseline["Zone_Capacity_MW"].iloc[0],
        color="#6b7280",
        linestyle="--",
        label="Zone capacity",
    )
    axes[0].set_ylabel("Load (MW)")
    axes[0].set_title(f"Zone {FOCUS_ZONE}: Baseline vs Smart Charging Interventions")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(
        baseline.index,
        baseline["Stress_Ratio"],
        label="Baseline stress",
        color="#dc2626",
        linewidth=2,
    )
    for pct, frame in interventions.items():
        axes[1].plot(
            frame.index,
            frame["Stress_Ratio"],
            label=f"Intervention {pct}%",
            linewidth=1.5,
            alpha=0.85,
        )
    axes[1].axhline(1.0, color="#6b7280", linestyle="--", label="Stress = 1.0")
    axes[1].set_ylabel("Stress ratio")
    axes[1].set_xlabel("Time of day")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid_load = load_zone2_15min_demand()
    price = load_mean_hourly_price_profile()
    districts = pd.read_csv(DISTRICTS_FILE)
    eindhoven_stations, eindhoven_pc6 = load_eindhoven_congestion_levels()

    ev_baseline = build_baseline_ev_load(grid_load.index)
    zone_capacity_mw = float((grid_load + ev_baseline).max())
    baseline_frame = enrich_simulation_frame(
        grid_load, ev_baseline, price, zone_capacity_mw
    )
    baseline_annual_cost = annual_ev_energy_cost(ev_baseline, price)

    intervention_frames: dict[int, pd.DataFrame] = {}
    kpi_rows: list[dict] = []

    baseline_kpi = build_kpi_row(
        "baseline",
        baseline_frame,
        baseline_annual_cost,
        baseline_annual_cost,
    )
    baseline_kpi["reference_annual_savings_eur"] = 0.0
    baseline_kpi["savings_vs_reference_pct"] = 0.0
    kpi_rows.append(baseline_kpi)

    for pct in INTERVENTION_PCTS:
        ev_shifted = apply_smart_charging_shift(
            grid_load, ev_baseline, pct / 100, zone_capacity_mw, price
        )
        frame = enrich_simulation_frame(
            grid_load, ev_shifted, price, zone_capacity_mw
        )
        intervention_frames[pct] = frame
        annual_cost = annual_ev_energy_cost(ev_shifted, price)
        row = build_kpi_row(
            f"intervention_{pct}pct",
            frame,
            annual_cost,
            baseline_annual_cost,
        )
        ref = REFERENCE_ANNUAL_SAVINGS_EUR[pct]
        row["reference_annual_savings_eur"] = ref
        row["savings_vs_reference_pct"] = (
            (row["annual_savings_eur"] - ref) / ref * 100 if ref else 0.0
        )
        kpi_rows.append(row)
        frame.to_csv(
            OUTPUT_DIR / f"zone2_timeseries_intervention_{pct}pct.csv",
            index_label="timestamp",
        )

    baseline_frame.to_csv(
        OUTPUT_DIR / "zone2_timeseries_baseline.csv", index_label="timestamp"
    )
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(OUTPUT_DIR / "zone2_kpi_comparison.csv", index=False)

    plot_baseline_vs_intervention(
        baseline_frame,
        intervention_frames,
        OUTPUT_DIR / "zone2_baseline_vs_intervention.png",
    )

    print("=== Zone Z2 smart charging simulation ===")
    print(f"Districts metadata rows: {len(districts)}")
    print(f"Eindhoven TenneT stations: {len(eindhoven_stations)}")
    print(f"Eindhoven PC6 rows: {len(eindhoven_pc6)}")
    print(f"Zone capacity (MW): {zone_capacity_mw:.2f}")
    print(f"Baseline peak stress ratio: {baseline_frame['Stress_Ratio'].max():.4f}")
    print()
    print(kpi_df.to_string(index=False))
    print()
    print("=== Validation vs reference annual savings (EUR) ===")
    for pct in INTERVENTION_PCTS:
        row = kpi_df.loc[kpi_df["scenario"] == f"intervention_{pct}pct"].iloc[0]
        print(
            f"{pct}%: simulated={row['annual_savings_eur']:.0f}, "
            f"reference={row['reference_annual_savings_eur']:.0f}, "
            f"delta={row['savings_vs_reference_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
