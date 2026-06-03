"""Customer bill savings vs immediate plug-in (daily → monthly, fleet and per EV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sim.config import SimConfig, load_config
from sim.metrics import daily_ev_energy_cost, days_per_month

REFERENCE_SCENARIO = "immediate_plug_in"

SMART_SCENARIOS = (
    "smart_flat_spread",
    "smart_price_aware",
)

APP_SCENARIO_ORDER = [
    "immediate_plug_in",
    "unmanaged_evening",
    "smart_flat_spread",
    "smart_price_aware",
    "smart_grid_aware",
]


def monthly_savings_from_daily(daily_savings_eur: float, cfg: SimConfig) -> float:
    """Scale one day's fleet savings to an average calendar month."""
    return float(daily_savings_eur * days_per_month(cfg))


def per_ev_monthly(monthly_fleet_eur: float, cfg: SimConfig) -> float:
    return float(monthly_fleet_eur / cfg.fleet.n_evs)


def compute_monthly_customer_savings(
    cfg: SimConfig | None = None,
    *,
    scenarios: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Monthly customer savings vs immediate plug-in from simulated EV schedules.

    Uses the same energy × price logic as annual KPIs: one reference day scaled
    to a month (365/12) and divided by fleet size for per-EV figures.
    """
    cfg = cfg or load_config()
    scenarios = scenarios or tuple(APP_SCENARIO_ORDER)
    app_dir = cfg.app_scenarios_dir

    ref_path = app_dir / f"{REFERENCE_SCENARIO}_timeseries.csv"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Missing {ref_path}; run scripts/run_app_scenarios.py first."
        )

    ref = pd.read_csv(ref_path, parse_dates=["timestamp"])
    ref_cost = daily_ev_energy_cost(
        ref.set_index("timestamp")["EV_Load_MW"],
        ref.set_index("timestamp")["Price_EUR_per_MWh"],
        cfg,
    )

    rows: list[dict] = []
    for name in scenarios:
        path = app_dir / f"{name}_timeseries.csv"
        if not path.exists():
            continue
        ts = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
        scen_cost = daily_ev_energy_cost(ts["EV_Load_MW"], ts["Price_EUR_per_MWh"], cfg)
        daily_saved = ref_cost - scen_cost
        monthly_fleet = monthly_savings_from_daily(daily_saved, cfg)
        rows.append(
            {
                "scenario": name,
                "reference_scenario": REFERENCE_SCENARIO,
                "daily_fleet_savings_eur": daily_saved,
                "monthly_fleet_savings_eur": monthly_fleet,
                "monthly_savings_per_ev_eur": per_ev_monthly(monthly_fleet, cfg),
                "annual_customer_savings_eur": daily_saved * 365,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {s: i for i, s in enumerate(APP_SCENARIO_ORDER)}
    df["_order"] = df["scenario"].map(order)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def save_monthly_customer_savings(
    cfg: SimConfig | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = compute_monthly_customer_savings(cfg)
    out = path or (cfg.app_scenarios_dir / "customer_monthly_savings.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_monthly_customer_savings(cfg: SimConfig | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = cfg.app_scenarios_dir / "customer_monthly_savings.csv"
    if path.exists():
        return pd.read_csv(path)
    return save_monthly_customer_savings(cfg)
