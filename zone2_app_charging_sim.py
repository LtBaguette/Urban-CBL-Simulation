"""
Zone Z2 APP charging scenarios — thin wrapper around sim.deterministic.

Prefer: python scripts/run_app_scenarios.py
"""

from sim.config import load_config
from sim.deterministic import run_app_scenarios

_cfg = load_config()
N_EVS = _cfg.fleet.n_evs
EV_KWH_PER_DAY = _cfg.fleet.kwh_per_day
EV_DAILY_MWH = _cfg.fleet.daily_mwh
FLEET_MAX_MW = _cfg.fleet.fleet_max_mw
DT_HOURS = _cfg.dt_hours
DT_MINUTES = _cfg.dt_minutes

from sim.metrics import annual_ev_energy_cost, enrich_simulation_frame  # noqa: E402
from sim.ev_scenarios import (  # noqa: E402
    scenario_immediate_plug_in,
    scenario_smart_flat_spread,
    scenario_smart_grid_aware,
    scenario_smart_price_aware,
    scenario_unmanaged_evening,
)

if __name__ == "__main__":
    run_app_scenarios()
