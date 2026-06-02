from __future__ import annotations

import pandas as pd

from sim.config import SimConfig
from sim.paths import (
    CONGESTION_PC6_FILE,
    EINDHOVEN_PC_PREFIXES,
    PRICE_FILE,
    TENNET_CONGESTION_FILE,
    ZONAL_LOAD_FILE,
)


def parse_eu_decimal(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def load_zone2_15min_demand(cfg: SimConfig) -> pd.Series:
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"]).copy()

    zone2 = load_df.loc[load_df["zone_id"] == cfg.focus_zone].sort_values("timestamp")
    if zone2.empty:
        raise ValueError(f"No records for zone {cfg.focus_zone} in {ZONAL_LOAD_FILE.name}")

    reference_day = pd.Timestamp(cfg.reference_day)
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
    profile.index = pd.date_range(reference_day, periods=96, freq="15min")
    profile.name = "demand_MW"
    return profile


def load_mean_hourly_price_profile(cfg: SimConfig) -> pd.Series:
    price_df = pd.read_csv(PRICE_FILE)
    price_df["timestamp"] = pd.to_datetime(
        price_df["Datetime (Local)"], errors="coerce"
    )
    price_df = price_df.dropna(subset=["timestamp"])
    hourly = price_df.groupby(price_df["timestamp"].dt.hour)["Price (EUR/MWhe)"].mean()
    hourly = hourly.reindex(range(24)).interpolate()
    index = pd.date_range(pd.Timestamp(cfg.reference_day), periods=96, freq="15min")
    return pd.Series(
        [hourly.loc[ts.hour] for ts in index], index=index, name="price_eur_mwh"
    )


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
