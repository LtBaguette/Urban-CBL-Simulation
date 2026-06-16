"""Monetize grid-stress deltas for DSO-facing KPIs."""

from __future__ import annotations

from sim.config import SimConfig


def annual_dso_savings_eur(
    overload_minutes: int,
    high_stress_minutes: int,
    peak_stress_ratio: float,
    reference_overload_minutes: int,
    reference_high_stress_minutes: int,
    reference_peak_stress_ratio: float,
    cfg: SimConfig,
    congestion_stress_integral_saved: float = 0.0,
    *,
    peak_total_load_mw: float | None = None,
    reference_peak_total_load_mw: float | None = None,
) -> float:
    """Grid-operator value vs immediate plug-in reference (annual EUR)."""
    overload_saved = reference_overload_minutes - overload_minutes
    stress_saved = reference_high_stress_minutes - high_stress_minutes
    peak_stress_saved = reference_peak_stress_ratio - peak_stress_ratio
    # Rate is EUR/year per 0.01 stress ratio (one daily peak index, not ×365).
    peak_stress_centiles = peak_stress_saved / 0.01
    annual_peak_stress_eur = peak_stress_centiles * cfg.dso_eur_per_peak_stress_point_year

    peak_mw_saved = 0.0
    if peak_total_load_mw is not None and reference_peak_total_load_mw is not None:
        peak_mw_saved = reference_peak_total_load_mw - peak_total_load_mw
    annual_peak_mw_eur = peak_mw_saved * cfg.dso_eur_per_peak_mw_reduction_year

    daily_eur = (
        overload_saved * cfg.dso_eur_per_overload_minute_year
        + stress_saved * cfg.dso_eur_per_high_stress_minute_year
        + congestion_stress_integral_saved
        * cfg.dso_eur_per_congestion_stress_integral_day_year
    )
    return float(daily_eur * 365 + annual_peak_stress_eur + annual_peak_mw_eur)


def dso_rate_bounds(cfg: SimConfig) -> dict[str, tuple[float, float]]:
    """Low/high annual rates for ±sensitivity_pct sensitivity analysis."""
    factor_low = 1.0 - cfg.dso_sensitivity_pct / 100.0
    factor_high = 1.0 + cfg.dso_sensitivity_pct / 100.0
    return {
        "overload": (
            cfg.dso_eur_per_overload_minute_year * factor_low,
            cfg.dso_eur_per_overload_minute_year * factor_high,
        ),
        "high_stress": (
            cfg.dso_eur_per_high_stress_minute_year * factor_low,
            cfg.dso_eur_per_high_stress_minute_year * factor_high,
        ),
        "peak_stress": (
            cfg.dso_eur_per_peak_stress_point_year * factor_low,
            cfg.dso_eur_per_peak_stress_point_year * factor_high,
        ),
        "peak_mw": (
            cfg.dso_eur_per_peak_mw_reduction_year * factor_low,
            cfg.dso_eur_per_peak_mw_reduction_year * factor_high,
        ),
    }
