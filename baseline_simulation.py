from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# DATA PATHS (Dataset 5 & 6)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data_Set" / "Data_Set"
DS5_DIR = next(DATA_DIR.glob("Dataset 5*"))
DS6_DIR = next(DATA_DIR.glob("Dataset 6*"))

ZONAL_LOAD_FILE = DS6_DIR / "eindhoven_zonal_load.csv"
TENNET_CONGESTION_FILE = DS5_DIR / "tennetcongestion.csv"
CONGESTION_PC6_FILE = DS5_DIR / "congestion_pc6.csv"

FOCUS_ZONE = "Z2"
EINDHOVEN_PC_PREFIXES = ("561", "562", "563", "564")


def parse_eu_decimal(series: pd.Series) -> pd.Series:
    """Dataset 5 uses comma decimals (e.g. '1,0' -> 1.0)."""
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def load_zone2_hourly_demand() -> pd.Series:
    """Dataset 6: average hourly demand (MW) for Zone Z2 across available days."""
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"]).copy()

    zone2 = load_df.loc[load_df["zone_id"] == FOCUS_ZONE].copy()
    if zone2.empty:
        raise ValueError(f"No records found for zone {FOCUS_ZONE} in {ZONAL_LOAD_FILE}")

    zone2["hour"] = zone2["timestamp"].dt.hour
    hourly = (
        zone2.groupby("hour", as_index=True)["demand_MW"]
        .mean()
        .reindex(range(24))
        .interpolate()
    )
    return hourly


def load_eindhoven_congestion_levels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dataset 5: TenneT station ratings for Eindhoven + postal-code congestion
    in Eindhoven area (561–564).
    """
    tennet_df = pd.read_csv(TENNET_CONGESTION_FILE, sep=";", encoding="utf-8")
    tennet_df.columns = [col.strip().strip(",") for col in tennet_df.columns]
    tennet_df = tennet_df.rename(columns={"tennet_id": "station_id"})
    tennet_df["consumption"] = parse_eu_decimal(tennet_df["consumption"])
    tennet_df["generation"] = parse_eu_decimal(tennet_df["generation"])

    eindhoven_stations = tennet_df.loc[
        tennet_df["station_id"].astype(str).str.contains("Eindhoven", case=False, na=False)
    ].copy()

    pc6_chunks = []
    for chunk in pd.read_csv(
        CONGESTION_PC6_FILE, sep=";", encoding="utf-8", chunksize=200_000
    ):
        chunk.columns = [col.strip().strip(",") for col in chunk.columns]
        mask = chunk["postal_code"].astype(str).str.startswith(EINDHOVEN_PC_PREFIXES)
        if mask.any():
            filtered = chunk.loc[mask].copy()
            filtered["consumption"] = parse_eu_decimal(filtered["consumption"])
            filtered["generation"] = parse_eu_decimal(filtered["generation"])
            pc6_chunks.append(filtered)

    eindhoven_pc6 = (
        pd.concat(pc6_chunks, ignore_index=True) if pc6_chunks else pd.DataFrame()
    )

    return eindhoven_stations, eindhoven_pc6


def build_simulation_space(hourly_demand_mw: pd.Series) -> pd.DataFrame:
    """Build 15-minute index and map Dataset 6 hourly demand onto it."""
    sample_day = pd.Timestamp("2025-01-01")
    time_steps = pd.date_range(
        start=sample_day, periods=96, freq="15min"
    )
    sim_space = pd.DataFrame(index=time_steps)
    sim_space["hour"] = sim_space.index.hour

    sim_space["Baseline_Grid_Load_MW"] = sim_space["hour"].map(hourly_demand_mw)
    return sim_space


def main() -> None:
    hourly_demand = load_zone2_hourly_demand()
    eindhoven_stations, eindhoven_pc6 = load_eindhoven_congestion_levels()
    sim_space = build_simulation_space(hourly_demand)

    station_consumption = float(eindhoven_stations["consumption"].mean())
    station_generation = float(eindhoven_stations["generation"].mean())
    if not eindhoven_pc6.empty:
        pc6_consumption = float(eindhoven_pc6["consumption"].mean())
        pc6_generation = float(eindhoven_pc6["generation"].mean())
    else:
        pc6_consumption = pc6_generation = float("nan")

    peak_mw = sim_space["Baseline_Grid_Load_MW"].max()
    peak_time = sim_space["Baseline_Grid_Load_MW"].idxmax().strftime("%H:%M")

    print("=== Data sources ===")
    print(f"Load profile: {ZONAL_LOAD_FILE.name} (zone {FOCUS_ZONE})")
    print(f"Grid constraints: {TENNET_CONGESTION_FILE.name}, {CONGESTION_PC6_FILE.name}")
    print(f"Eindhoven TenneT stations: {len(eindhoven_stations)}")
    print(f"Eindhoven postal codes (561–564): {len(eindhoven_pc6)}")
    print()
    print("=== Zone 2 baseline congestion (demand proxy, MW) ===")
    print(f"Peak demand: {peak_mw:.2f} MW at {peak_time}")
    print(f"Min demand: {sim_space['Baseline_Grid_Load_MW'].min():.2f} MW")
    print(f"Mean demand: {sim_space['Baseline_Grid_Load_MW'].mean():.2f} MW")
    print()
    print("=== Dataset 5 congestion ratings (0=none, 3=blocked) ===")
    print(f"Eindhoven stations – avg consumption: {station_consumption:.2f}")
    print(f"Eindhoven stations – avg generation: {station_generation:.2f}")
    if pd.notna(pc6_consumption):
        print(f"Eindhoven PC6 – avg consumption: {pc6_consumption:.2f}")
        print(f"Eindhoven PC6 – avg generation: {pc6_generation:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(
        sim_space.index,
        sim_space["Baseline_Grid_Load_MW"],
        label=f"Zone {FOCUS_ZONE} demand (Dataset 6)",
        color="red",
        linewidth=2,
    )
    plt.fill_between(
        sim_space.index,
        sim_space["Baseline_Grid_Load_MW"],
        color="red",
        alpha=0.1,
    )

    plt.xlabel("Time of Day")
    plt.ylabel("Grid demand (MW)")
    plt.title(
        "Focus Area 2: Baseline 24-Hour Grid Congestion (Dataset 5 & 6, No Intervention)"
    )
    plt.legend(loc="upper left")
    subtitle = (
        f"Dataset 5 – Eindhoven station congestion (avg): "
        f"consumption {station_consumption:.1f}, generation {station_generation:.1f} | "
        f"PC6 codes: {len(eindhoven_pc6)}"
    )
    plt.gcf().text(0.5, 0.01, subtitle, ha="center", fontsize=9, color="#374151")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.show()


if __name__ == "__main__":
    main()
