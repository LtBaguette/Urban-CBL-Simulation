from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sim.capacity import resolve_zone_capacity_mw
from sim.config import SimConfig, load_config
from sim.data_loaders import (
    load_eindhoven_congestion_levels,
    load_mean_hourly_price_profile,
    load_zone2_15min_demand,
)
from sim.energy import assert_load_invariants
from sim.ev_scenarios import (
    apply_smart_charging_shift,
    build_baseline_ev_load,
    scenario_immediate_plug_in,
    scenario_smart_flat_spread,
    scenario_smart_grid_aware,
    scenario_smart_price_aware,
    scenario_unmanaged_evening,
)
from sim.metrics import (
    annual_ev_energy_cost,
    build_app_kpi_row,
    build_intervention_kpi_row,
    enrich_simulation_frame,
    grid_stress_minutes,
)
from sim.paths import DISTRICTS_FILE
from sim.customer_savings import save_monthly_customer_savings
from sim.validate import summarize_reference_gap, validate_reference_savings


def _plot_charging_profiles(
    frames: dict[str, pd.DataFrame], out_path: Path, cfg: SimConfig
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
        c = colors.get(name)
        axes[0].plot(frame.index, frame["EV_Load_MW"], label=name, color=c, alpha=0.9)
        axes[1].plot(frame.index, frame["Total_Load_MW"], label=name, color=c, alpha=0.9)
        axes[2].plot(frame.index, frame["Stress_Ratio"], label=name, color=c, alpha=0.9)
    cap = next(iter(frames.values()))["Zone_Capacity_MW"].iloc[0]
    axes[1].axhline(cap, color="#6b7280", linestyle="--", label="Zone capacity")
    axes[2].axhline(1.0, color="#6b7280", linestyle="--", label="Stress = 1.0")
    axes[0].set_ylabel("EV load (MW)")
    axes[0].set_title(f"Zone {cfg.focus_zone}: APP smart charging scenarios")
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


def run_zone2_interventions(cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_load = load_zone2_15min_demand(cfg)
    price = load_mean_hourly_price_profile(cfg)
    districts = pd.read_csv(DISTRICTS_FILE)
    eindhoven_stations, eindhoven_pc6 = load_eindhoven_congestion_levels()

    zone_capacity_mw, capacity_meta = resolve_zone_capacity_mw(grid_load, cfg)
    ev_baseline = build_baseline_ev_load(grid_load.index, cfg)
    assert_load_invariants(ev_baseline, cfg, scenario="baseline_ev")

    baseline_frame = enrich_simulation_frame(
        grid_load, ev_baseline, price, zone_capacity_mw, cfg
    )
    baseline_annual_cost = annual_ev_energy_cost(ev_baseline, price, cfg)

    intervention_frames: dict[int, pd.DataFrame] = {}
    kpi_rows: list[dict] = []

    baseline_kpi = build_intervention_kpi_row(
        "baseline",
        baseline_frame,
        baseline_annual_cost,
        baseline_annual_cost,
    )
    baseline_kpi["reference_annual_savings_eur"] = 0.0
    baseline_kpi["savings_vs_reference_pct"] = 0.0
    kpi_rows.append(baseline_kpi)

    for pct in cfg.intervention_pcts:
        ev_shifted = apply_smart_charging_shift(
            grid_load, ev_baseline, pct / 100, zone_capacity_mw, price, cfg
        )
        assert_load_invariants(ev_shifted, cfg, scenario=f"intervention_{pct}pct")
        frame = enrich_simulation_frame(
            grid_load, ev_shifted, price, zone_capacity_mw, cfg
        )
        intervention_frames[pct] = frame
        annual_cost = annual_ev_energy_cost(ev_shifted, price, cfg)
        row = build_intervention_kpi_row(
            f"intervention_{pct}pct",
            frame,
            annual_cost,
            baseline_annual_cost,
        )
        ref = cfg.reference_annual_savings_eur[pct]
        row["reference_annual_savings_eur"] = ref
        row["savings_vs_reference_pct"] = (
            (row["annual_savings_eur"] - ref) / ref * 100 if ref else 0.0
        )
        kpi_rows.append(row)
        frame.to_csv(
            out_dir / f"zone2_timeseries_intervention_{pct}pct.csv",
            index_label="timestamp",
        )

    baseline_frame.to_csv(
        out_dir / "zone2_timeseries_baseline.csv", index_label="timestamp"
    )
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(out_dir / "zone2_kpi_comparison.csv", index=False)

    print("=== Zone Z2 smart charging simulation ===")
    print(f"Districts metadata rows: {len(districts)}")
    print(f"Eindhoven TenneT stations: {len(eindhoven_stations)}")
    print(f"Eindhoven PC6 rows: {len(eindhoven_pc6)}")
    print(f"Zone capacity (MW): {zone_capacity_mw:.2f} [{capacity_meta['capacity_source']}]")
    print(
        f"  Noord-Brabant cap: {capacity_meta['cap_noord_brabant_mw']:.0f} MW, "
        f"derating: {capacity_meta['congestion_derate_pct']:.1f}%"
    )
    print(f"Baseline peak stress ratio: {baseline_frame['Stress_Ratio'].max():.4f}")
    print()
    print(kpi_df.to_string(index=False))
    print()
    print("=== Validation vs reference annual savings (EUR) ===")
    for line in validate_reference_savings(kpi_df, cfg, fail=False):
        print(line)
    print()
    for line in summarize_reference_gap(kpi_df, cfg):
        print(line)

    return kpi_df


def run_app_scenarios(cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    out_dir = cfg.app_scenarios_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_load = load_zone2_15min_demand(cfg)
    price = load_mean_hourly_price_profile(cfg)
    _, eindhoven_pc6 = load_eindhoven_congestion_levels()
    index = grid_load.index

    zone_capacity_mw, capacity_meta = resolve_zone_capacity_mw(grid_load, cfg)

    scenarios: dict[str, pd.Series] = {
        "immediate_plug_in": scenario_immediate_plug_in(index, price, cfg),
        "unmanaged_evening": scenario_unmanaged_evening(index, price, cfg),
        "smart_flat_spread": scenario_smart_flat_spread(index, price, cfg),
        "smart_price_aware": scenario_smart_price_aware(index, price, cfg),
        "smart_grid_aware": scenario_smart_grid_aware(
            grid_load, index, price, zone_capacity_mw, cfg
        ),
    }

    frames: dict[str, pd.DataFrame] = {}
    kpi_rows: list[dict] = []
    baseline_cost = annual_ev_energy_cost(scenarios["immediate_plug_in"], price, cfg)

    ref_frame = enrich_simulation_frame(
        grid_load, scenarios["immediate_plug_in"], price, zone_capacity_mw, cfg
    )
    ref_overload, ref_high_stress = grid_stress_minutes(ref_frame, cfg)
    ref_peak_stress = float(ref_frame["Stress_Ratio"].max())

    for name, ev_load in scenarios.items():
        assert_load_invariants(ev_load, cfg, scenario=name)
        frame = enrich_simulation_frame(
            grid_load, ev_load, price, zone_capacity_mw, cfg
        )
        frames[name] = frame
        annual_cost = annual_ev_energy_cost(ev_load, price, cfg)
        kpi_rows.append(
            build_app_kpi_row(
                name,
                frame,
                annual_cost,
                baseline_cost,
                cfg,
                ref_overload,
                ref_high_stress,
                ref_peak_stress,
                reference_frame=ref_frame,
            )
        )
        frame.to_csv(out_dir / f"{name}_timeseries.csv", index_label="timestamp")

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(out_dir / "app_scenarios_kpi.csv", index=False)
    monthly_path = save_monthly_customer_savings(cfg)
    print(f"Monthly customer savings: {monthly_path}")

    graphs_dir = cfg.ensure_graphs_dir()
    _plot_charging_profiles(frames, graphs_dir / "app_charging_profiles.png", cfg)

    print("=== Zone Z2 APP smart charging simulation ===")
    print(
        f"EVs: {cfg.fleet.n_evs}, {cfg.fleet.kwh_per_day} kWh/day -> "
        f"{cfg.fleet.daily_mwh:.1f} MWh/day"
    )
    print(
        f"Fleet max charging power: {cfg.fleet.fleet_max_mw:.2f} MW "
        f"({cfg.fleet.charger_kw} kW/EV)"
    )
    print(f"Zone capacity (MW): {zone_capacity_mw:.2f} [{capacity_meta['capacity_source']}]")
    print(f"Eindhoven PC6 rows: {len(eindhoven_pc6)}")
    grid_eq_price = scenarios["smart_grid_aware"].equals(scenarios["smart_price_aware"])
    print(f"smart_grid_aware == smart_price_aware schedules: {grid_eq_price}")
    if grid_eq_price:
        print("  WARNING: grid-aware should differ from price-only when capacity binds.")
    warned = kpi_df.loc[kpi_df["dso_savings_warning"] == True, "scenario"].tolist()
    if warned:
        print(f"  DSO savings warning (grid worse than immediate plug-in): {warned}")
    print()
    print(kpi_df.to_string(index=False))
    print()
    print(f"Outputs written to: {out_dir}")

    return kpi_df
