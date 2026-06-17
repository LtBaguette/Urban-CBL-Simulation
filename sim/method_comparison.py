"""
Side-by-side comparison of all EV charging methods on identical KPI fields.

Reference for savings & DSO value: immediate_plug_in (deterministic APP).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim.config import SimConfig, load_config
from sim.customer_savings import per_ev_monthly
from sim.dso_value import annual_dso_savings_eur
from sim.metrics import (
    congestion_stress_integral_saved,
    daily_ev_energy_cost,
    daily_ev_energy_cost_slots,
    enrich_simulation_frame,
    grid_stress_minutes,
)
from sim.grid_oriented_optimizer import optimize_ev_load_grid_oriented
from sim.price_oriented_optimizer import optimize_ev_load_price_oriented
from sim.residuals import load_residuals_config

REFERENCE_METHOD = "immediate_plug_in"
EL_DIABLO_DISPLAY_NAME = "el diablo"

METHOD_ORDER: list[tuple[str, str]] = [
    ("immediate_plug_in", "Immediate plug-in"),
    ("unmanaged_evening", "Baseline (evening peak)"),
    ("smart_flat_spread", "Smart (flat spread)"),
    ("smart_price_aware", "Smart (price-aware)"),
    ("smart_grid_aware", "Simulation Algorithm"),
    ("simulation_v5", EL_DIABLO_DISPLAY_NAME),
    ("price_oriented_baseline", "Price-oriented"),
    ("grid_oriented_baseline", "Grid-oriented"),
]

# Omitted from graph_method_comparison.png (still in CSV / full pipeline).
CHART_EXCLUDED_METHOD_KEYS = frozenset({
    "unmanaged_evening",
    "smart_flat_spread",
    "smart_price_aware",
})

def _display_name(method_key: str) -> str:
    return next(name for key, name in METHOD_ORDER if key == method_key)


STOCHASTIC_METHOD_KEYS = frozenset({
    "simulation_v5",
    "price_oriented_baseline",
    "grid_oriented_baseline",
})

METHOD_COLORS = {
    "immediate_plug_in": "#dc2626",
    "unmanaged_evening": "#ea580c",
    "smart_flat_spread": "#2563eb",
    "smart_price_aware": "#059669",
    "smart_grid_aware": "#7c3aed",
    "simulation_v5": "#378ADD",
    "price_oriented_baseline": "#D85A30",
    "grid_oriented_baseline": "#1D9E75",
}


UNIFIED_COLUMNS = [
    "method_key",
    "display_name",
    "model_type",
    "reference_method",
    "n_evs",
    "kwh_per_day",
    "app_adoption_rate",
    "zone_capacity_mw",
    "peak_total_load_mw",
    "peak_reduction_mw_vs_reference",
    "peak_stress_ratio",
    "bottleneck_intervals_per_day",
    "overload_minutes_per_day",
    "minutes_above_stress_095",
    "mean_congestion_index",
    "ev_energy_mwh_per_day",
    "daily_ev_energy_cost_eur",
    "annual_ev_energy_cost_eur",
    "annual_customer_savings_eur",
    "monthly_savings_per_ev_eur",
    "annual_dso_savings_eur",
    "annual_total_savings_eur",
    "congestion_stress_integral_saved",
    "dso_savings_warning",
    "rank_by_total_savings",
]


@dataclass
class ReferenceContext:
    peak_total_load_mw: float
    peak_stress_ratio: float
    overload_minutes: int
    high_stress_minutes: int
    annual_ev_energy_cost_eur: float
    daily_ev_energy_cost_eur: float
    zone_capacity_mw: float
    frame: pd.DataFrame


def _load_reference_context(cfg: SimConfig) -> ReferenceContext:
    path = cfg.app_scenarios_dir / f"{REFERENCE_METHOD}_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/run_app_scenarios.py")
    ts = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    ref_overload, ref_high = grid_stress_minutes(ts, cfg)
    daily_cost = daily_ev_energy_cost(ts["EV_Load_MW"], ts["Price_EUR_per_MWh"], cfg)
    return ReferenceContext(
        peak_total_load_mw=float(ts["Total_Load_MW"].max()),
        peak_stress_ratio=float(ts["Stress_Ratio"].max()),
        overload_minutes=ref_overload,
        high_stress_minutes=ref_high,
        annual_ev_energy_cost_eur=daily_cost * 365,
        daily_ev_energy_cost_eur=daily_cost,
        zone_capacity_mw=float(ts["Zone_Capacity_MW"].iloc[0]),
        frame=ts,
    )


def _app_adoption_rate(cfg: SimConfig) -> float:
    _, res_cfg = load_residuals_config(cfg)
    return res_cfg.app_adoption_rate


def _base_row(method_key: str, display_name: str, model_type: str, cfg: SimConfig) -> dict:
    return {
        "method_key": method_key,
        "display_name": display_name,
        "model_type": model_type,
        "reference_method": REFERENCE_METHOD,
        "n_evs": cfg.fleet.n_evs,
        "kwh_per_day": cfg.fleet.kwh_per_day,
        "app_adoption_rate": _app_adoption_rate(cfg),
    }


def _load_scenario_timeseries(scenario: str, cfg: SimConfig) -> pd.DataFrame:
    path = cfg.app_scenarios_dir / f"{scenario}_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/run_app_scenarios.py")
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")


def _blended_frame(
    method_ts: pd.DataFrame,
    ref_ts: pd.DataFrame,
    adoption: float,
    cfg: SimConfig,
) -> pd.DataFrame:
    """Mix fleet: (1-adoption) on reference EV schedule, adoption on method schedule."""
    ev_blended = (1.0 - adoption) * ref_ts["EV_Load_MW"] + adoption * method_ts["EV_Load_MW"]
    return enrich_simulation_frame(
        method_ts["Baseline_Grid_Load_MW"],
        ev_blended,
        method_ts["Price_EUR_per_MWh"],
        float(method_ts["Zone_Capacity_MW"].iloc[0]),
        cfg,
    )


def _finalize_row(row: dict, ref: ReferenceContext, cfg: SimConfig) -> dict:
    row["peak_reduction_mw_vs_reference"] = (
        ref.peak_total_load_mw - row["peak_total_load_mw"]
    )
    row["monthly_savings_per_ev_eur"] = per_ev_monthly(
        row["annual_customer_savings_eur"] / 12, cfg
    )
    row["annual_total_savings_eur"] = (
        row["annual_customer_savings_eur"] + row["annual_dso_savings_eur"]
    )
    return row


def _row_from_deterministic_frame(
    method_key: str,
    display_name: str,
    frame: pd.DataFrame,
    ref: ReferenceContext,
    cfg: SimConfig,
) -> dict:
    row = _base_row(method_key, display_name, "deterministic", cfg)
    overload, high_stress = grid_stress_minutes(frame, cfg)
    daily_cost = daily_ev_energy_cost(frame["EV_Load_MW"], frame["Price_EUR_per_MWh"], cfg)
    annual_cost = daily_cost * 365
    customer_savings = ref.annual_ev_energy_cost_eur - annual_cost
    peak_stress = float(frame["Stress_Ratio"].max())
    stress_integral = 0.0
    if peak_stress <= ref.peak_stress_ratio:
        stress_integral = congestion_stress_integral_saved(frame, ref.frame, cfg)
    dso_savings = annual_dso_savings_eur(
        overload,
        high_stress,
        peak_stress,
        ref.overload_minutes,
        ref.high_stress_minutes,
        ref.peak_stress_ratio,
        cfg,
        congestion_stress_integral_saved=stress_integral,
        peak_total_load_mw=float(frame["Total_Load_MW"].max()),
        reference_peak_total_load_mw=ref.peak_total_load_mw,
    )
    row.update(
        {
            "zone_capacity_mw": float(frame["Zone_Capacity_MW"].iloc[0]),
            "peak_total_load_mw": float(frame["Total_Load_MW"].max()),
            "peak_stress_ratio": peak_stress,
            "bottleneck_intervals_per_day": int(frame["Bottleneck"].sum()),
            "overload_minutes_per_day": overload,
            "minutes_above_stress_095": high_stress,
            "mean_congestion_index": float(frame["Congestion_Index"].mean()),
            "ev_energy_mwh_per_day": float(
                frame["EV_Load_MW"].sum() * cfg.dt_hours
            ),
            "daily_ev_energy_cost_eur": daily_cost,
            "annual_ev_energy_cost_eur": annual_cost,
            "annual_customer_savings_eur": customer_savings,
            "annual_dso_savings_eur": dso_savings,
            "congestion_stress_integral_saved": stress_integral,
            "dso_savings_warning": dso_savings < 0,
        }
    )
    return _finalize_row(row, ref, cfg)


def _row_from_deterministic_scenario(
    method_key: str,
    display_name: str,
    ref: ReferenceContext,
    cfg: SimConfig,
) -> dict:
    adoption = _app_adoption_rate(cfg)
    method_ts = _load_scenario_timeseries(method_key, cfg)
    frame = _blended_frame(method_ts, ref.frame, adoption, cfg)
    return _row_from_deterministic_frame(method_key, display_name, frame, ref, cfg)


def _row_from_v5(cfg: SimConfig, ref: ReferenceContext) -> dict:
    v5_dir = cfg.output_dir / "simulation_v5"
    runs = sorted(v5_dir.glob("run_*_15min.csv"))
    if not runs:
        raise FileNotFoundError(
            f"Missing el diablo runs in {v5_dir}; run scripts/run_simulation_v5.py"
        )

    per_run: list[dict] = []
    ref_stress = ref.frame["Stress_Ratio"].values
    for run_path in runs:
        df = pd.read_csv(run_path)
        scen_stress = df["managed_stress"].values
        frame = pd.DataFrame(
            {
                "EV_Load_MW": df["managed_ev_mw"].values,
                "Total_Load_MW": df["managed_total_mw"].values,
                "Stress_Ratio": scen_stress,
                "Bottleneck": scen_stress >= 1.0,
                "Congestion_Index": cfg.congestion_index_base * scen_stress,
            }
        )
        overload, high_stress = grid_stress_minutes(frame, cfg)
        daily_cost = daily_ev_energy_cost_slots(
            df["managed_ev_mw"].values, df["price_eur_mwh"].values
        )
        annual_cost = daily_cost * 365
        customer_savings = ref.annual_ev_energy_cost_eur - annual_cost
        peak_stress = float(scen_stress.max())
        stress_integral = 0.0
        if peak_stress <= ref.peak_stress_ratio:
            mask = ref_stress >= cfg.stress_warning_threshold
            stress_integral = float(
                (ref_stress[mask] - scen_stress[mask]).clip(min=0).sum()
            )
        dso_savings = annual_dso_savings_eur(
            overload,
            high_stress,
            peak_stress,
            ref.overload_minutes,
            ref.high_stress_minutes,
            ref.peak_stress_ratio,
            cfg,
            congestion_stress_integral_saved=stress_integral,
            peak_total_load_mw=float(df["managed_total_mw"].max()),
            reference_peak_total_load_mw=ref.peak_total_load_mw,
        )
        per_run.append(
            {
                "peak_total_load_mw": float(df["managed_total_mw"].max()),
                "peak_stress_ratio": peak_stress,
                "bottleneck_intervals_per_day": int((scen_stress >= 1.0).sum()),
                "overload_minutes_per_day": overload,
                "minutes_above_stress_095": high_stress,
                "mean_congestion_index": float(frame["Congestion_Index"].mean()),
                "ev_energy_mwh_per_day": float(df["managed_ev_mw"].sum() / 4),
                "daily_ev_energy_cost_eur": daily_cost,
                "annual_ev_energy_cost_eur": annual_cost,
                "annual_customer_savings_eur": customer_savings,
                "annual_dso_savings_eur": dso_savings,
                "congestion_stress_integral_saved": stress_integral,
                "dso_savings_warning": dso_savings < 0,
            }
        )

    avg = pd.DataFrame(per_run).mean(numeric_only=True)
    row = _base_row("simulation_v5", _display_name("simulation_v5"), "stochastic_v5", cfg)
    row.update(
        {
            "zone_capacity_mw": ref.zone_capacity_mw,
            "peak_total_load_mw": float(avg["peak_total_load_mw"]),
            "peak_stress_ratio": float(avg["peak_stress_ratio"]),
            "bottleneck_intervals_per_day": float(avg["bottleneck_intervals_per_day"]),
            "overload_minutes_per_day": float(avg["overload_minutes_per_day"]),
            "minutes_above_stress_095": float(avg["minutes_above_stress_095"]),
            "mean_congestion_index": float(avg["mean_congestion_index"]),
            "ev_energy_mwh_per_day": float(avg["ev_energy_mwh_per_day"]),
            "daily_ev_energy_cost_eur": float(avg["daily_ev_energy_cost_eur"]),
            "annual_ev_energy_cost_eur": float(avg["annual_ev_energy_cost_eur"]),
            "annual_customer_savings_eur": float(avg["annual_customer_savings_eur"]),
            "annual_dso_savings_eur": float(avg["annual_dso_savings_eur"]),
            "congestion_stress_integral_saved": float(
                avg["congestion_stress_integral_saved"]
            ),
            "dso_savings_warning": bool((pd.DataFrame(per_run)["dso_savings_warning"]).any()),
        }
    )
    return _finalize_row(row, ref, cfg)


def _row_from_price_oriented(cfg: SimConfig, ref: ReferenceContext) -> dict:
    """Price-only block scheduler on the same stochastic days as el diablo."""
    from sim.simulation_v5 import _apply_runtime_config, build_arrival_weights

    sim_cfg, res_cfg = load_residuals_config(cfg)
    _apply_runtime_config(sim_cfg, res_cfg)
    arrival_weights = build_arrival_weights()
    fleet_mwh = cfg.fleet.daily_mwh

    v5_dir = cfg.output_dir / "simulation_v5"
    runs = sorted(v5_dir.glob("run_*_15min.csv"))
    if not runs:
        raise FileNotFoundError(
            f"Missing el diablo runs in {v5_dir}; run scripts/run_simulation_v5.py"
        )

    per_run: list[dict] = []
    ref_stress = ref.frame["Stress_Ratio"].values
    for run_path in runs:
        run_id = int(run_path.stem.split("_")[1])
        app_seed = res_cfg.random_seed * 100 + run_id
        df = pd.read_csv(run_path)
        base_load = pd.Series(df["base_load_mw"].values, index=range(len(df)))
        prices = df["price_eur_mwh"].values
        opt_ev_load, _, _ = optimize_ev_load_price_oriented(
            arrival_weights,
            prices,
            app_seed,
            n_evs=cfg.fleet.n_evs,
            ev_kwh_per_day=cfg.fleet.kwh_per_day,
            total_ev_mwh=fleet_mwh,
            app_adoption_rate=res_cfg.app_adoption_rate,
            charger_kw={
                "Home": res_cfg.charger_kw["Home"],
                "Fast": res_cfg.charger_kw["Fast"],
                "Super": res_cfg.charger_kw["Super"],
            },
        )
        managed_total = base_load.values + opt_ev_load.values
        scen_stress = managed_total / ref.zone_capacity_mw
        frame = pd.DataFrame(
            {
                "EV_Load_MW": opt_ev_load.values,
                "Total_Load_MW": managed_total,
                "Stress_Ratio": scen_stress,
                "Bottleneck": scen_stress >= 1.0,
                "Congestion_Index": cfg.congestion_index_base * scen_stress,
            }
        )
        overload, high_stress = grid_stress_minutes(frame, cfg)
        daily_cost = daily_ev_energy_cost_slots(opt_ev_load.values, prices, slots_per_hour=4)
        annual_cost = daily_cost * 365
        customer_savings = ref.annual_ev_energy_cost_eur - annual_cost
        peak_stress = float(scen_stress.max())
        stress_integral = 0.0
        if peak_stress <= ref.peak_stress_ratio:
            mask = ref_stress >= cfg.stress_warning_threshold
            stress_integral = float(
                (ref_stress[mask] - scen_stress[mask]).clip(min=0).sum()
            )
        dso_savings = annual_dso_savings_eur(
            overload,
            high_stress,
            peak_stress,
            ref.overload_minutes,
            ref.high_stress_minutes,
            ref.peak_stress_ratio,
            cfg,
            congestion_stress_integral_saved=stress_integral,
            peak_total_load_mw=float(managed_total.max()),
            reference_peak_total_load_mw=ref.peak_total_load_mw,
        )
        per_run.append(
            {
                "peak_total_load_mw": float(managed_total.max()),
                "peak_stress_ratio": peak_stress,
                "bottleneck_intervals_per_day": int((scen_stress >= 1.0).sum()),
                "overload_minutes_per_day": overload,
                "minutes_above_stress_095": high_stress,
                "mean_congestion_index": float(frame["Congestion_Index"].mean()),
                "ev_energy_mwh_per_day": float(opt_ev_load.sum() / 4),
                "daily_ev_energy_cost_eur": daily_cost,
                "annual_ev_energy_cost_eur": annual_cost,
                "annual_customer_savings_eur": customer_savings,
                "annual_dso_savings_eur": dso_savings,
                "congestion_stress_integral_saved": stress_integral,
                "dso_savings_warning": dso_savings < 0,
            }
        )

    avg = pd.DataFrame(per_run).mean(numeric_only=True)
    row = _base_row(
        "price_oriented_baseline",
        _display_name("price_oriented_baseline"),
        "price_oriented_stochastic",
        cfg,
    )
    row.update(
        {
            "zone_capacity_mw": ref.zone_capacity_mw,
            "peak_total_load_mw": float(avg["peak_total_load_mw"]),
            "peak_stress_ratio": float(avg["peak_stress_ratio"]),
            "bottleneck_intervals_per_day": float(avg["bottleneck_intervals_per_day"]),
            "overload_minutes_per_day": float(avg["overload_minutes_per_day"]),
            "minutes_above_stress_095": float(avg["minutes_above_stress_095"]),
            "mean_congestion_index": float(avg["mean_congestion_index"]),
            "ev_energy_mwh_per_day": float(avg["ev_energy_mwh_per_day"]),
            "daily_ev_energy_cost_eur": float(avg["daily_ev_energy_cost_eur"]),
            "annual_ev_energy_cost_eur": float(avg["annual_ev_energy_cost_eur"]),
            "annual_customer_savings_eur": float(avg["annual_customer_savings_eur"]),
            "annual_dso_savings_eur": float(avg["annual_dso_savings_eur"]),
            "congestion_stress_integral_saved": float(
                avg["congestion_stress_integral_saved"]
            ),
            "dso_savings_warning": bool((pd.DataFrame(per_run)["dso_savings_warning"]).any()),
        }
    )
    return _finalize_row(row, ref, cfg)


def _row_from_grid_oriented(cfg: SimConfig, ref: ReferenceContext) -> dict:
    """Grid-only block scheduler on the same stochastic days as el diablo."""
    from sim.simulation_v5 import _apply_runtime_config, build_arrival_weights

    sim_cfg, res_cfg = load_residuals_config(cfg)
    _apply_runtime_config(sim_cfg, res_cfg)
    arrival_weights = build_arrival_weights()

    v5_dir = cfg.output_dir / "simulation_v5"
    runs = sorted(v5_dir.glob("run_*_15min.csv"))
    if not runs:
        raise FileNotFoundError(
            f"Missing el diablo runs in {v5_dir}; run scripts/run_simulation_v5.py"
        )

    per_run: list[dict] = []
    ref_stress = ref.frame["Stress_Ratio"].values
    charger_kw = {
        "Home": res_cfg.charger_kw["Home"],
        "Fast": res_cfg.charger_kw["Fast"],
        "Super": res_cfg.charger_kw["Super"],
    }
    for run_path in runs:
        run_id = int(run_path.stem.split("_")[1])
        app_seed = res_cfg.random_seed * 100 + run_id
        df = pd.read_csv(run_path)
        base_load = pd.Series(df["base_load_mw"].values, index=range(len(df)))
        prices = df["price_eur_mwh"].values
        opt_ev_load, _, _ = optimize_ev_load_grid_oriented(
            base_load,
            arrival_weights,
            sim_cfg.grid_load_penalty_eur_per_mw,
            app_seed,
            n_evs=cfg.fleet.n_evs,
            total_ev_mwh=cfg.fleet.daily_mwh,
            app_adoption_rate=res_cfg.app_adoption_rate,
            charger_kw=charger_kw,
            zone_capacity_mw=ref.zone_capacity_mw,
            stress_warning_threshold=cfg.stress_warning_threshold,
        )
        managed_total = base_load.values + opt_ev_load.values
        scen_stress = managed_total / ref.zone_capacity_mw
        frame = pd.DataFrame(
            {
                "EV_Load_MW": opt_ev_load.values,
                "Total_Load_MW": managed_total,
                "Stress_Ratio": scen_stress,
                "Bottleneck": scen_stress >= 1.0,
                "Congestion_Index": cfg.congestion_index_base * scen_stress,
            }
        )
        overload, high_stress = grid_stress_minutes(frame, cfg)
        daily_cost = daily_ev_energy_cost_slots(opt_ev_load.values, prices, slots_per_hour=4)
        annual_cost = daily_cost * 365
        customer_savings = ref.annual_ev_energy_cost_eur - annual_cost
        peak_stress = float(scen_stress.max())
        stress_integral = 0.0
        if peak_stress <= ref.peak_stress_ratio:
            mask = ref_stress >= cfg.stress_warning_threshold
            stress_integral = float(
                (ref_stress[mask] - scen_stress[mask]).clip(min=0).sum()
            )
        dso_savings = annual_dso_savings_eur(
            overload,
            high_stress,
            peak_stress,
            ref.overload_minutes,
            ref.high_stress_minutes,
            ref.peak_stress_ratio,
            cfg,
            congestion_stress_integral_saved=stress_integral,
            peak_total_load_mw=float(managed_total.max()),
            reference_peak_total_load_mw=ref.peak_total_load_mw,
        )
        per_run.append(
            {
                "peak_total_load_mw": float(managed_total.max()),
                "peak_stress_ratio": peak_stress,
                "bottleneck_intervals_per_day": int((scen_stress >= 1.0).sum()),
                "overload_minutes_per_day": overload,
                "minutes_above_stress_095": high_stress,
                "mean_congestion_index": float(frame["Congestion_Index"].mean()),
                "ev_energy_mwh_per_day": float(opt_ev_load.sum() / 4),
                "daily_ev_energy_cost_eur": daily_cost,
                "annual_ev_energy_cost_eur": annual_cost,
                "annual_customer_savings_eur": customer_savings,
                "annual_dso_savings_eur": dso_savings,
                "congestion_stress_integral_saved": stress_integral,
                "dso_savings_warning": dso_savings < 0,
            }
        )

    avg = pd.DataFrame(per_run).mean(numeric_only=True)
    row = _base_row(
        "grid_oriented_baseline",
        _display_name("grid_oriented_baseline"),
        "grid_oriented_stochastic",
        cfg,
    )
    row.update(
        {
            "zone_capacity_mw": ref.zone_capacity_mw,
            "peak_total_load_mw": float(avg["peak_total_load_mw"]),
            "peak_stress_ratio": float(avg["peak_stress_ratio"]),
            "bottleneck_intervals_per_day": float(avg["bottleneck_intervals_per_day"]),
            "overload_minutes_per_day": float(avg["overload_minutes_per_day"]),
            "minutes_above_stress_095": float(avg["minutes_above_stress_095"]),
            "mean_congestion_index": float(avg["mean_congestion_index"]),
            "ev_energy_mwh_per_day": float(avg["ev_energy_mwh_per_day"]),
            "daily_ev_energy_cost_eur": float(avg["daily_ev_energy_cost_eur"]),
            "annual_ev_energy_cost_eur": float(avg["annual_ev_energy_cost_eur"]),
            "annual_customer_savings_eur": float(avg["annual_customer_savings_eur"]),
            "annual_dso_savings_eur": float(avg["annual_dso_savings_eur"]),
            "congestion_stress_integral_saved": float(
                avg["congestion_stress_integral_saved"]
            ),
            "dso_savings_warning": bool((pd.DataFrame(per_run)["dso_savings_warning"]).any()),
        }
    )
    return _finalize_row(row, ref, cfg)


def build_method_comparison(cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ref = _load_reference_context(cfg)
    rows: list[dict] = []
    for method_key, display_name in METHOD_ORDER:
        if method_key == "simulation_v5":
            rows.append(_row_from_v5(cfg, ref))
        elif method_key == "price_oriented_baseline":
            rows.append(_row_from_price_oriented(cfg, ref))
        elif method_key == "grid_oriented_baseline":
            rows.append(_row_from_grid_oriented(cfg, ref))
        else:
            rows.append(_row_from_deterministic_scenario(method_key, display_name, ref, cfg))

    df = pd.DataFrame(rows)
    rank_map = (
        df.sort_values("annual_total_savings_eur", ascending=False)["method_key"]
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "rank_by_total_savings", "method_key": "method_key"})
    )
    rank_map["rank_by_total_savings"] += 1
    df = df.merge(rank_map, on="method_key")
    order = {k: i for i, (k, _) in enumerate(METHOD_ORDER)}
    df["_order"] = df["method_key"].map(order)
    return (
        df.sort_values("_order")
        .drop(columns="_order")[UNIFIED_COLUMNS]
        .reset_index(drop=True)
    )


def save_method_comparison(cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = build_method_comparison(cfg)
    out = cfg.output_dir / "method_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _chart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Subset and re-rank methods shown on graph_method_comparison.png."""
    chart_df = df[~df["method_key"].isin(CHART_EXCLUDED_METHOD_KEYS)].copy()
    rank_map = (
        chart_df.sort_values("annual_total_savings_eur", ascending=False)["method_key"]
        .reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "rank_by_total_savings", "method_key": "method_key"})
    )
    rank_map["rank_by_total_savings"] += 1
    chart_df = chart_df.drop(columns=["rank_by_total_savings"]).merge(
        rank_map, on="method_key"
    )
    order = {k: i for i, (k, _) in enumerate(METHOD_ORDER)}
    chart_df["_order"] = chart_df["method_key"].map(order)
    return chart_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def plot_method_comparison(df: pd.DataFrame, out_path: Path, cfg: SimConfig) -> None:
    """Bar chart of total savings + table of all unified metrics."""
    df = _chart_dataframe(df)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.4], hspace=0.28)

    ax_bar = fig.add_subplot(gs[0])
    names = df["display_name"].tolist()
    x = np.arange(len(names))
    cust_k = df["annual_customer_savings_eur"] / 1000
    dso_k = df["annual_dso_savings_eur"] / 1000
    width = 0.35
    ax_bar.bar(x - width / 2, cust_k, width, label="Customer", color="#22c55e")
    ax_bar.bar(x + width / 2, dso_k, width, label="DSO", color="#6366f1")
    ax_bar.axhline(0, color="#374151", linewidth=0.8)
    ax_bar.set_ylabel("Annual EUR (thousands)")
    adoption_pct = int(_app_adoption_rate(cfg) * 100)
    _, res_cfg = load_residuals_config(cfg)
    v5_runs = res_cfg.v5_simulation_repeats
    ax_bar.set_title(
        f"All charging methods at {adoption_pct}% app adoption vs immediate plug-in",
        fontweight="bold",
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax_bar.legend(loc="upper left")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    for i, (_, row) in enumerate(df.iterrows()):
        total_k = row["annual_total_savings_eur"] / 1000
        ax_bar.text(
            i,
            max(cust_k.iloc[i], dso_k.iloc[i], 0) + 40,
            f"#{int(row['rank_by_total_savings'])} total {total_k:.0f}k",
            ha="center",
            fontsize=7,
            fontweight="bold",
        )

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    table_cols = [
        "display_name",
        "rank_by_total_savings",
        "annual_total_savings_eur",
        "monthly_savings_per_ev_eur",
        "peak_reduction_mw_vs_reference",
        "annual_customer_savings_eur",
        "annual_dso_savings_eur",
        "app_adoption_rate",
    ]
    tbl = df[table_cols].copy()
    tbl["annual_total_savings_eur"] = tbl["annual_total_savings_eur"].map(lambda v: f"{v:,.0f}")
    tbl["monthly_savings_per_ev_eur"] = tbl["monthly_savings_per_ev_eur"].map(lambda v: f"{v:.1f}")
    tbl["peak_reduction_mw_vs_reference"] = tbl["peak_reduction_mw_vs_reference"].map(
        lambda v: f"{v:.1f}"
    )
    tbl["annual_customer_savings_eur"] = tbl["annual_customer_savings_eur"].map(
        lambda v: f"{v:,.0f}"
    )
    tbl["annual_dso_savings_eur"] = tbl["annual_dso_savings_eur"].map(lambda v: f"{v:,.0f}")
    tbl["app_adoption_rate"] = tbl["app_adoption_rate"].map(lambda v: f"{v:.0%}")
    tbl.columns = [
        "Method",
        "Rank",
        "Total EUR/yr",
        "EUR/EV/mo",
        "Peak cut MW",
        "Customer EUR/yr",
        "DSO EUR/yr",
        "App %",
    ]
    table = ax_tbl.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    ax_tbl.set_title("Unified KPI summary (full data in method_comparison.csv)", fontsize=10)
    fig.text(
        0.5,
        0.01,
        f"{adoption_pct}% of fleet on each method's schedule; "
        f"{100 - adoption_pct}% remain on immediate plug-in (reference). "
        f"{EL_DIABLO_DISPLAY_NAME}: mean of {v5_runs} stochastic runs "
        f"(seeds {res_cfg.random_seed + 1}–{res_cfg.random_seed + v5_runs}).",
        ha="center",
        fontsize=9,
        style="italic",
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _v5_run_paths(cfg: SimConfig) -> list[Path]:
    v5_dir = cfg.output_dir / "simulation_v5"
    runs = sorted(v5_dir.glob("run_*_15min.csv"))
    if not runs:
        raise FileNotFoundError(
            f"Missing el diablo runs in {v5_dir}; run scripts/run_simulation_v5.py"
        )
    return runs


def _profile_frame_from_arrays(
    ev: np.ndarray,
    total: np.ndarray,
    index: pd.DatetimeIndex,
    zone_capacity_mw: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EV_Load_MW": ev,
            "Total_Load_MW": total,
            "Stress_Ratio": total / zone_capacity_mw,
            "Zone_Capacity_MW": zone_capacity_mw,
        },
        index=index,
    )


def _avg_v5_managed_profile(cfg: SimConfig, ref: ReferenceContext) -> pd.DataFrame:
    evs: list[np.ndarray] = []
    totals: list[np.ndarray] = []
    for run_path in _v5_run_paths(cfg):
        df = pd.read_csv(run_path)
        evs.append(df["managed_ev_mw"].values)
        totals.append(df["managed_total_mw"].values)
    return _profile_frame_from_arrays(
        np.mean(evs, axis=0),
        np.mean(totals, axis=0),
        ref.frame.index,
        ref.zone_capacity_mw,
    )


def _avg_optimizer_profile(
    cfg: SimConfig,
    ref: ReferenceContext,
    optimizer: str,
) -> pd.DataFrame:
    from sim.simulation_v5 import _apply_runtime_config, build_arrival_weights

    sim_cfg, res_cfg = load_residuals_config(cfg)
    _apply_runtime_config(sim_cfg, res_cfg)
    arrival_weights = build_arrival_weights()
    charger_kw = {
        "Home": res_cfg.charger_kw["Home"],
        "Fast": res_cfg.charger_kw["Fast"],
        "Super": res_cfg.charger_kw["Super"],
    }
    evs: list[np.ndarray] = []
    totals: list[np.ndarray] = []
    for run_path in _v5_run_paths(cfg):
        run_id = int(run_path.stem.split("_")[1])
        app_seed = res_cfg.random_seed * 100 + run_id
        df = pd.read_csv(run_path)
        base_load = pd.Series(df["base_load_mw"].values, index=range(len(df)))
        prices = df["price_eur_mwh"].values
        if optimizer == "price_oriented_baseline":
            opt_ev, _, _ = optimize_ev_load_price_oriented(
                arrival_weights,
                prices,
                app_seed,
                n_evs=cfg.fleet.n_evs,
                ev_kwh_per_day=cfg.fleet.kwh_per_day,
                total_ev_mwh=cfg.fleet.daily_mwh,
                app_adoption_rate=res_cfg.app_adoption_rate,
                charger_kw=charger_kw,
            )
        elif optimizer == "grid_oriented_baseline":
            opt_ev, _, _ = optimize_ev_load_grid_oriented(
                base_load,
                arrival_weights,
                sim_cfg.grid_load_penalty_eur_per_mw,
                app_seed,
                n_evs=cfg.fleet.n_evs,
                total_ev_mwh=cfg.fleet.daily_mwh,
                app_adoption_rate=res_cfg.app_adoption_rate,
                charger_kw=charger_kw,
                zone_capacity_mw=ref.zone_capacity_mw,
                stress_warning_threshold=cfg.stress_warning_threshold,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        managed_total = base_load.values + opt_ev.values
        evs.append(opt_ev.values)
        totals.append(managed_total)
    return _profile_frame_from_arrays(
        np.mean(evs, axis=0),
        np.mean(totals, axis=0),
        ref.frame.index,
        ref.zone_capacity_mw,
    )


def build_all_method_profile_frames(cfg: SimConfig | None = None) -> dict[str, pd.DataFrame]:
    """Daily load profiles for every method in METHOD_ORDER (same basis as KPI table)."""
    cfg = cfg or load_config()
    ref = _load_reference_context(cfg)
    adoption = _app_adoption_rate(cfg)
    frames: dict[str, pd.DataFrame] = {}

    for method_key, display_name in METHOD_ORDER:
        if method_key in STOCHASTIC_METHOD_KEYS:
            if method_key == "simulation_v5":
                frame = _avg_v5_managed_profile(cfg, ref)
            else:
                frame = _avg_optimizer_profile(cfg, ref, method_key)
        elif method_key == REFERENCE_METHOD:
            frame = ref.frame
        else:
            method_ts = _load_scenario_timeseries(method_key, cfg)
            frame = _blended_frame(method_ts, ref.frame, adoption, cfg)
        frames[display_name] = frame
    return frames


def plot_all_methods_profiles(
    frames: dict[str, pd.DataFrame],
    out_path: Path,
    cfg: SimConfig,
) -> None:
    """Three-panel EV / total load / stress chart for all comparison methods."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    cap = next(iter(frames.values()))["Zone_Capacity_MW"].iloc[0]
    adoption_pct = int(_app_adoption_rate(cfg) * 100)

    for display_name, frame in frames.items():
        method_key = next(k for k, d in METHOD_ORDER if d == display_name)
        color = METHOD_COLORS[method_key]
        axes[0].plot(frame.index, frame["EV_Load_MW"], label=display_name, color=color, alpha=0.9)
        axes[1].plot(
            frame.index, frame["Total_Load_MW"], label=display_name, color=color, alpha=0.9
        )
        axes[2].plot(
            frame.index, frame["Stress_Ratio"], label=display_name, color=color, alpha=0.9
        )

    axes[1].axhline(cap, color="#6b7280", linestyle="--", label="Zone capacity")
    axes[2].axhline(1.0, color="#6b7280", linestyle="--", label="Stress = 1.0")
    axes[0].set_ylabel("EV load (MW)")
    axes[0].set_title(
        f"Zone {cfg.focus_zone}: all_methods_profiles "
        f"({adoption_pct}% app adoption; stochastic = mean AR(1) day)",
        fontweight="bold",
    )
    axes[1].set_ylabel("Total load (MW)")
    axes[2].set_ylabel("Stress ratio")
    axes[2].set_xlabel("Time of day")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=6, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_methods_profiles(cfg: SimConfig | None = None) -> Path:
    cfg = cfg or load_config()
    frames = build_all_method_profile_frames(cfg)
    out_path = cfg.ensure_graphs_dir() / "all_methods_profiles.png"
    plot_all_methods_profiles(frames, out_path, cfg)
    return out_path


def generate_method_comparison(cfg: SimConfig | None = None) -> tuple[pd.DataFrame, Path]:
    cfg = cfg or load_config()
    df = save_method_comparison(cfg)
    chart = cfg.ensure_graphs_dir() / "graph_method_comparison.png"
    plot_method_comparison(df, chart, cfg)
    return df, chart
