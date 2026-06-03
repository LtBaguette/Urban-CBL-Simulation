"""
Zone Z2 intervention simulation — thin wrapper around sim.deterministic.

Prefer: python scripts/run_zone2.py
"""

from sim.config import load_config
from sim.deterministic import run_zone2_interventions

_cfg = load_config()
N_EVS = _cfg.fleet.n_evs
EV_KWH_PER_DAY = _cfg.fleet.kwh_per_day
EV_DAILY_MWH = _cfg.fleet.daily_mwh
FLEET_MAX_MW = _cfg.fleet.fleet_max_mw
DT_HOURS = _cfg.dt_hours
DT_MINUTES = _cfg.dt_minutes
STRESS_WARNING_THRESHOLD = _cfg.stress_warning_threshold
INTERVENTION_PCTS = _cfg.intervention_pcts
REFERENCE_ANNUAL_SAVINGS_EUR = _cfg.reference_annual_savings_eur
CONGESTION_INDEX_BASE = _cfg.congestion_index_base

# Re-export for backward compatibility
from sim.data_loaders import (  # noqa: E402
    load_eindhoven_congestion_levels,
    load_mean_hourly_price_profile,
    load_zone2_15min_demand,
)
from sim.ev_scenarios import apply_smart_charging_shift, build_baseline_ev_load  # noqa: E402
from sim.metrics import (  # noqa: E402
    annual_ev_energy_cost,
    build_intervention_kpi_row as build_kpi_row,
    enrich_simulation_frame,
)

if __name__ == "__main__":
    run_zone2_interventions()
