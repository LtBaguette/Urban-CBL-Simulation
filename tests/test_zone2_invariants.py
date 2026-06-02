from __future__ import annotations

import pytest

from sim.capacity import resolve_zone_capacity_mw
from sim.config import load_config
from sim.data_loaders import load_mean_hourly_price_profile, load_zone2_15min_demand
from sim.energy import assert_energy_balance, assert_load_invariants
from sim.ev_scenarios import (
    scenario_immediate_plug_in,
    scenario_smart_grid_aware,
    scenario_smart_price_aware,
)
from sim.validate import validate_capacity_constant
from sim.metrics import enrich_simulation_frame


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def grid_price_cap(cfg):
    grid = load_zone2_15min_demand(cfg)
    price = load_mean_hourly_price_profile(cfg)
    cap, _ = resolve_zone_capacity_mw(grid, cfg)
    return grid, price, cap


def test_zone_capacity_fixed_across_scenarios(grid_price_cap, cfg):
    grid, price, cap = grid_price_cap
    ev_a = scenario_immediate_plug_in(grid.index, price, cfg)
    ev_b = scenario_smart_grid_aware(grid, grid.index, price, cap, cfg)
    assert cap == pytest.approx(resolve_zone_capacity_mw(grid, cfg)[0])
    frames = [
        enrich_simulation_frame(grid, ev_a, price, cap, cfg),
        enrich_simulation_frame(grid, ev_b, price, cap, cfg),
    ]
    validate_capacity_constant(frames, cap)


def test_energy_and_load_invariants(grid_price_cap, cfg):
    grid, price, cap = grid_price_cap
    for name, ev in [
        ("immediate", scenario_immediate_plug_in(grid.index, price, cfg)),
        ("price", scenario_smart_price_aware(grid.index, price, cfg)),
        ("grid", scenario_smart_grid_aware(grid, grid.index, price, cap, cfg)),
    ]:
        assert_energy_balance(ev, cfg.fleet.daily_mwh, cfg, scenario=name)
        assert_load_invariants(ev, cfg, scenario=name)


def test_smart_grid_differs_from_price_aware(grid_price_cap, cfg):
    grid, price, cap = grid_price_cap
    ev_price = scenario_smart_price_aware(grid.index, price, cfg)
    ev_grid = scenario_smart_grid_aware(grid, grid.index, price, cap, cfg)
    assert not ev_price.equals(ev_grid)
