from __future__ import annotations

import pandas as pd

from sim.config import SimConfig
from sim.dso_value import annual_dso_savings_eur


def enrich_simulation_frame(
    grid_load: pd.Series,
    ev_load: pd.Series,
    price: pd.Series,
    zone_capacity_mw: float,
    cfg: SimConfig,
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
            "Congestion_Index": cfg.congestion_index_base * stress,
        }
    )


def annual_ev_energy_cost(ev_load: pd.Series, price: pd.Series, cfg: SimConfig) -> float:
    return float((ev_load * cfg.dt_hours * price).sum() * 365)


def grid_stress_minutes(frame: pd.DataFrame, cfg: SimConfig) -> tuple[int, int]:
    stress = frame["Stress_Ratio"]
    return (
        int(frame["Bottleneck"].sum() * cfg.dt_minutes),
        int((stress >= cfg.stress_warning_threshold).sum() * cfg.dt_minutes),
    )


def congestion_stress_integral_saved(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    cfg: SimConfig,
) -> float:
    """Daily sum of stress reductions vs reference on high-stress slots (dimensionless)."""
    mask = reference_frame["Stress_Ratio"] >= cfg.stress_warning_threshold
    ref = reference_frame.loc[mask, "Stress_Ratio"]
    scen = frame.loc[mask, "Stress_Ratio"]
    return float((ref - scen).clip(lower=0).sum())


def build_intervention_kpi_row(
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
        "zone_capacity_mw": float(frame["Zone_Capacity_MW"].iloc[0]),
        "annual_ev_energy_cost_eur": annual_cost_eur,
        "annual_savings_eur": baseline_annual_cost_eur - annual_cost_eur,
    }


def build_app_kpi_row(
    label: str,
    frame: pd.DataFrame,
    annual_cost_eur: float,
    baseline_annual_cost_eur: float,
    cfg: SimConfig,
    reference_overload_minutes: int,
    reference_high_stress_minutes: int,
    reference_peak_stress_ratio: float,
    reference_frame: pd.DataFrame | None = None,
) -> dict:
    row = build_intervention_kpi_row(
        label, frame, annual_cost_eur, baseline_annual_cost_eur
    )
    overload_minutes, high_stress_minutes = grid_stress_minutes(frame, cfg)
    customer_savings = row["annual_savings_eur"]
    peak_stress = float(frame["Stress_Ratio"].max())
    stress_integral_saved = 0.0
    if reference_frame is not None and peak_stress <= reference_peak_stress_ratio:
        stress_integral_saved = congestion_stress_integral_saved(
            frame, reference_frame, cfg
        )
    dso_savings = annual_dso_savings_eur(
        overload_minutes,
        high_stress_minutes,
        float(frame["Stress_Ratio"].max()),
        reference_overload_minutes,
        reference_high_stress_minutes,
        reference_peak_stress_ratio,
        cfg,
        congestion_stress_integral_saved=stress_integral_saved,
    )
    row["overload_minutes"] = overload_minutes
    row["minutes_above_stress_095"] = high_stress_minutes
    row["congestion_stress_integral_saved"] = stress_integral_saved
    row["annual_customer_savings_eur"] = customer_savings
    row["annual_dso_savings_eur"] = dso_savings
    row["annual_total_savings_eur"] = customer_savings + dso_savings
    row["dso_savings_warning"] = dso_savings < 0
    return row
