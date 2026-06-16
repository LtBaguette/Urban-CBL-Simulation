from __future__ import annotations

import pandas as pd

from sim.config import SimConfig
from sim.data_loaders import parse_eu_decimal
from sim.paths import TENNET_CONGESTION_FILE, TENNET_GEBIEDEN_FILE, ZONAL_LOAD_FILE


def load_noord_brabant_transport_capacity_mw(cfg: SimConfig) -> float:
    df = pd.read_csv(TENNET_GEBIEDEN_FILE, sep=";", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    row = df.loc[df["congestiegebied"] == cfg.capacity_region]
    if row.empty:
        raise ValueError(f"No row for {cfg.capacity_region!r} in tennetgebieden.csv")
    cap = parse_eu_decimal(row["aanwezige_transportcapaciteit_afname"]).iloc[0]
    if pd.isna(cap) or cap <= 0:
        raise ValueError(f"Invalid regional afname capacity: {cap}")
    return float(cap)


def load_eindhoven_peak_demands_mw(focus_zone: str) -> tuple[float, float]:
    load_df = pd.read_csv(ZONAL_LOAD_FILE)
    load_df["timestamp"] = pd.to_datetime(load_df["timestamp"], errors="coerce")
    load_df = load_df.dropna(subset=["timestamp"])
    wide = load_df.pivot(index="timestamp", columns="zone_id", values="demand_MW")
    city = wide.sum(axis=1)
    if focus_zone not in wide.columns:
        raise ValueError(f"Zone {focus_zone} missing from zonal load file")
    return float(city.max()), float(wide[focus_zone].max())


def load_congestion_derate_factor(cfg: SimConfig) -> float:
    tennet_df = pd.read_csv(TENNET_CONGESTION_FILE, sep=";", encoding="utf-8")
    tennet_df.columns = [col.strip().strip(",") for col in tennet_df.columns]
    tennet_df["afname"] = parse_eu_decimal(tennet_df["afname"])
    eindhoven = tennet_df.loc[
        tennet_df["tennet_id"].astype(str).str.contains("Eindhoven", case=False, na=False)
    ]
    if eindhoven.empty:
        return 0.0
    mean_rating = float(eindhoven["afname"].mean())
    return min(
        cfg.max_congestion_derate,
        cfg.derate_per_rating_point * mean_rating,
    )


def resolve_zone_capacity_mw(
    grid_load: pd.Series, cfg: SimConfig
) -> tuple[float, dict]:
    """
    Fixed capacity for all scenarios: Dataset 6 grid peak with Dataset 5 derating.
    """
    cap_nb = load_noord_brabant_transport_capacity_mw(cfg)
    city_peak, z2_peak_raw = load_eindhoven_peak_demands_mw(cfg.focus_zone)
    grid_peak = float(grid_load.max())
    derate = load_congestion_derate_factor(cfg)
    zone_capacity_mw = grid_peak * (1.0 - derate)
    alloc_nb = cap_nb * (z2_peak_raw / city_peak) if city_peak > 0 else cap_nb

    meta = {
        "cap_noord_brabant_mw": cap_nb,
        "city_peak_mw": city_peak,
        "z2_peak_raw_mw": z2_peak_raw,
        "grid_profile_peak_mw": grid_peak,
        "congestion_derate_pct": round(derate * 100, 2),
        "cap_z2_nb_proportional_mw": round(alloc_nb, 2),
        "capacity_source": "dataset6_grid_peak_with_dataset5_derating",
    }
    return zone_capacity_mw, meta
